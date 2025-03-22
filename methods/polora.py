import torch
import torch.nn as nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from einops.layers.torch import Reduce

import logging
import numpy as np
from tqdm import tqdm
from sklearn.cluster import KMeans

from methods.base import BaseLearner
from utils.toolkit import tensor2numpy, accuracy
from models.sinet_polora import SiNet
from timm_monet.models.monet import PolyBlock_LoRA, Downsample_LoRA
from timm_monet.layers.mlp import PolyMlp_SkipLoRA, PolyMlp_SkipOneLoRA, PolyMlp_LoRA_Old, Channel_Projection_LoRA_Old, Channel_Projection_OneLoRA
from copy import deepcopy
from utils.schedulers import CosineSchedule
from utils.toolkit import count_parameters
import math

class Polora(BaseLearner):

    def __init__(self, args):
        super().__init__(args)
        if args["net_type"] == "sip":
            self._network = SiNet(args)
        else:
            raise ValueError('Unknown net: {}.'.format(args["net_type"]))
        
        for module in self._network.modules():
            # if isinstance(module, PolyBlock_LoRA):
            #     module.init_param()
            if isinstance(module, PolyMlp_LoRA_Old):
                module.init_weights()
            # if isinstance(module, Channel_Projection_LoRA):
            #     module.init_param()
            if isinstance(module, Channel_Projection_LoRA_Old):
                module.init_param()
            if isinstance(module, PolyMlp_SkipLoRA):
                module.init_param()
            if isinstance(module, PolyMlp_SkipOneLoRA):
                module.init_param()
            if isinstance(module, Channel_Projection_OneLoRA):
                module.init_param()
            if isinstance(module, Downsample_LoRA):
                module.init_param()


        self.args = args
        self.optim = args["optim"]
        self.EPSILON = args["EPSILON"]
        self.init_epoch = args["init_epoch"]
        self.init_lr = args["init_lr"]
        self.init_lr_decay = args["init_lr_decay"]
        self.init_weight_decay = args["init_weight_decay"]
        self.epochs = args["epochs"]
        self.lrate = args["lrate"]
        self.lrate_decay = args["lrate_decay"]
        self.batch_size = args["batch_size"]
        self.weight_decay = args["weight_decay"]
        self.num_workers = args["num_workers"]
        self.lamb = args["lamb"]
        self.lame = args["lame"]
        self.alpha_gpm = args["alpha_gpm"]
        self.total_sessions = args["total_sessions"]
        self.dataset = args["dataset"]
        self.finetune = args["finetune"]
        self.test = args["test"]
        self.use_gpm = args["use_gpm"]
        self.use_dualgpm = args["use_dualgpm"]
        self.update_gpm = self.use_gpm or self.use_dualgpm

        self.topk = 1  # origin is 5
        self.class_num = self._network.class_num
        self.debug = False

        self.all_keys = []
        self.feature_list = []
        self.project_type = []

    def after_task(self):
        # self._old_network = self._network.copy().freeze()
        self._known_classes = self._total_classes
        logging.info('Exemplar size: {}'.format(self.exemplar_size))

    def incremental_train(self, data_manager):
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
        self._network.update_fc(self._total_classes)

        logging.info('Learning on {}-{}'.format(self._known_classes, self._total_classes))

        train_dataset = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes), source='train', mode='train')
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True,
                                       num_workers=self.num_workers)
        test_dataset = data_manager.get_dataset(np.arange(0, self._total_classes), source='test', mode='test')
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False,
                                      num_workers=self.num_workers)

        if len(self._multiple_gpus) > 1:
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        self._train(self.train_loader, self.test_loader)
        self.clustering(self.train_loader)
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

    def incremental_test(self, data_manager, tasktest):
        for _ in range(tasktest + 1):
            self._known_classes = self._total_classes
            self._cur_task += 1
            self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
            self._network.update_fc(self._total_classes)

        logging.info('Learning on {}-{}'.format(self._known_classes, self._total_classes))

        train_dataset = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes), source='train', mode='train')
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True,
                                       num_workers=self.num_workers)
        test_dataset = data_manager.get_dataset(np.arange(0, self._total_classes), source='test', mode='test')
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False,
                                      num_workers=self.num_workers)

        if len(self._multiple_gpus) > 1:
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module
        self._network.to(self._device)

    def _train(self, train_loader, test_loader):
        self._network.to(self._device)
        # if self._old_network is not None:
        #     self._old_network.to(self._device)

        for name, param in self._network.named_parameters():
            if 'full' not in self.finetune:
                param.requires_grad_(False)
            try:
                if "classifier_pool" + "." + str(self._network.module.numtask - 1) in name:
                    param.requires_grad_(True)
                # if "lora_B" + "." + str(self._network.module.numtask - 1) in name:
                #     param.requires_grad_(True)
                if 'lora_B' in self.finetune:
                    if "lora_B" in name and name.endswith("." + str(self._network.module.numtask - 1) + ".weight"):
                        param.requires_grad_(True)
                if 'lora_A' in self.finetune:
                    if "lora_A" in name and name.endswith("." + str(self._network.module.numtask - 1) + ".weight"):
                        param.requires_grad_(True)
                if 'norm' in self.finetune and 'norm' in name:
                    param.requires_grad_(True)
            except:
                if "classifier_pool" + "." + str(self._network.numtask - 1) in name:
                    param.requires_grad_(True)
                # if "lora_B" + "." + str(self._network.numtask - 1) in name:
                #     param.requires_grad_(True)
                if 'lora_B' in self.finetune:    
                    if "lora_B" in name and name.endswith("." + str(self._network.numtask - 1) + ".weight"):
                        param.requires_grad_(True)
                if 'lora_A' in self.finetune:    
                    if "lora_A" in name and name.endswith("." + str(self._network.numtask - 1) + ".weight"):
                        param.requires_grad_(True)
                if 'norm' in self.finetune and 'norm' in name:
                    param.requires_grad_(True)

        # Double check
        enabled = set()
        for name, param in self._network.named_parameters():
            if param.requires_grad:
                enabled.add(name)
        print(f"Parameters to be updated: {enabled}")
        logging.info('All params: {}'.format(count_parameters(self._network)))
        logging.info('Trainable params: {}'.format(count_parameters(self._network, True)))
        logging.info('Tuning ratio: {}'.format(count_parameters(self._network, True) / count_parameters(self._network)))

        with torch.no_grad():
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                self._network(inputs, get_cur_feat=self.update_gpm)
                # if i > 3: break

            if self._cur_task == 0:
                for module in self._network.modules():
                    if self.update_gpm:
                        if isinstance(module, PolyBlock_LoRA) or isinstance(module, PolyMlp_SkipLoRA) or isinstance(module, Downsample_LoRA) or isinstance(module, Channel_Projection_LoRA_Old):
                            cur_matrix = module.cur_matrix
                            U, _, _ = torch.linalg.svd(cur_matrix)
                            try:
                                module.lora_A[self._cur_task].weight.data.copy_(U[:,:module.rank].T/self.alpha_gpm)
                            except Exception as e:
                                import ipdb; ipdb.set_trace()
                            module.cur_matrix.zero_()
                            module.n_cur_matrix = 0
            else:

                kk = 0
                for module in self._network.modules():
                    if self.update_gpm:
                        if isinstance(module, PolyBlock_LoRA) or isinstance(module, PolyMlp_SkipLoRA) or isinstance(module, Downsample_LoRA) or isinstance(module, Channel_Projection_LoRA_Old):
                            cur_matrix = module.cur_matrix
                            try:
                                if self.use_gpm:
                                    cur_matrix = cur_matrix - torch.mm(self.feature_mat[kk],cur_matrix)
                                    # cur_matrix = torch.mm(self.feature_mat[kk],cur_matrix)
                                if self.use_dualgpm:
                                    if self.project_type[kk] == 'remove':
                                        cur_matrix = cur_matrix - torch.mm(self.feature_mat[kk],cur_matrix)
                                    else:
                                        assert self.project_type[kk] == 'retain'
                                        cur_matrix = torch.mm(self.feature_mat[kk],cur_matrix)
                            except Exception as e:
                                import ipdb; ipdb.set_trace()
                            cU, cS, cV = torch.linalg.svd(cur_matrix, full_matrices=False)
                            module.lora_A[self._cur_task].weight.data.copy_(cU[:,:module.rank].T/self.alpha_gpm)
                            module.cur_matrix.zero_()
                            module.n_cur_matrix = 0
                            kk += 1

        if self._cur_task==0:
            if self.optim == 'sgd':
                optimizer = optim.SGD(self._network.parameters(), momentum=0.9,lr=self.init_lr,weight_decay=self.init_weight_decay)
                scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer,T_max=self.init_epoch)
            elif self.optim == 'adam':
                optimizer = optim.Adam(self._network.parameters(),lr=self.init_lr,weight_decay=self.init_weight_decay, betas=(0.9,0.999))
                scheduler = CosineSchedule(optimizer=optimizer,K=self.init_epoch)
            else:
                raise Exception
            self.run_epoch = self.init_epoch
            self.train_function(train_loader,test_loader,optimizer,scheduler)
        else:
            if self.optim == 'sgd':
                optimizer = optim.SGD(self._network.parameters(), momentum=0.9,lr=self.lrate,weight_decay=self.weight_decay)
                scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer,T_max=self.epochs)
            elif self.optim == 'adam':
                optimizer = optim.Adam(self._network.parameters(),lr=self.lrate,weight_decay=self.weight_decay, betas=(0.9,0.999))
                scheduler = CosineSchedule(optimizer=optimizer,K=self.epochs)
            else:
                raise Exception
            self.run_epoch = self.epochs
            self.train_function(train_loader, test_loader, optimizer, scheduler)

        with torch.no_grad():
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                self._network(inputs, get_cur_feat=self.update_gpm)

            mat_list = []
            for module in self._network.modules():
                if self.update_gpm:
                    if isinstance(module, PolyBlock_LoRA) or isinstance(module, PolyMlp_SkipLoRA) or isinstance(module, Downsample_LoRA) or isinstance(module, Channel_Projection_LoRA_Old):
                        mat_list.append(deepcopy(module.cur_matrix))
                        module.cur_matrix.zero_()
                        module.n_cur_matrix = 0
            if self.use_gpm:
                self.update_GPM(mat_list)
            if self.use_dualgpm:
                self.update_DualGPM(mat_list)

            # Projection Matrix Precomputation
            self.feature_mat = []
            for p in range(len(self.feature_list)):
                Uf=torch.Tensor(np.dot(self.feature_list[p],self.feature_list[p].transpose()))
                print('Layer {} - Projection Matrix shape: {}'.format(p+1,Uf.shape))
                self.feature_mat.append(Uf)

        return

    def train_function(self, train_loader, test_loader, optimizer, scheduler):
        prog_bar = tqdm(range(self.run_epoch))
        for _, epoch in enumerate(prog_bar):
            self._network.eval()
            losses = 0.
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):

                inputs, targets = inputs.to(self._device), targets.to(self._device)
                mask = (targets >= self._known_classes).nonzero().view(-1)
                inputs = torch.index_select(inputs, 0, mask)
                targets = torch.index_select(targets, 0, mask)-self._known_classes

                logits = self._network(inputs)['logits']
                loss = F.cross_entropy(logits, targets)

                optimizer.zero_grad()
                loss.backward()

                optimizer.step()
                losses += loss.item()

                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)
                if self.debug and i > 10: break

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)

            info = 'Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}'.format(
                self._cur_task, epoch + 1, self.run_epoch, losses / len(train_loader), train_acc)
            prog_bar.set_description(info)

        logging.info(info)


    def clustering(self, dataloader):
        features = []
        for i, (_, inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(self._device), targets.to(self._device)
            mask = (targets >= self._known_classes).nonzero().view(-1)
            inputs = torch.index_select(inputs, 0, mask)
            with torch.no_grad():
                if isinstance(self._network, nn.DataParallel):
                    feature = self._network.module.extract_vector(inputs)
                else:
                    feature = self._network.extract_vector(inputs)
            feature = Reduce('b c h w -> b c', 'mean')(feature)
            feature = feature / feature.norm(dim=-1, keepdim=True)
            features.append(feature)
        features = torch.cat(features, 0).cpu().detach().numpy()
        clustering = KMeans(n_clusters=5, random_state=0).fit(features)
        self.all_keys.append(torch.tensor(clustering.cluster_centers_).to(feature.device))

    def _evaluate(self, y_pred, y_true):
        ret = {}
        print(f'length of y pred: {len(y_pred)}, length of y true: {len(y_true)}')
        grouped = accuracy(y_pred, y_true, self._known_classes, self.class_num)
        ret['grouped'] = grouped
        ret['top1'] = grouped['total']
        return ret

    def _eval_cnn(self, loader):
        self._network.eval()
        y_pred, y_true = [], []
        y_pred_with_task = []
        y_pred_given_task = []
        y_pred_task, y_true_task = [], []
        for _, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            targets = targets.to(self._device)
            with torch.no_grad():
                y_true_task.append((torch.div(targets, self.class_num, rounding_mode='floor')).cpu())

                if isinstance(self._network, nn.DataParallel):
                    outputs = self._network.module.interface(inputs)
                else:
                    outputs = self._network.interface(inputs)

            # # Split the tensor into 4 chunks of 10 values each along the second dimension
            # output_chunks = torch.chunk(outputs, 10, dim=1)

            # # Apply softmax to each chunk independently and concatenate the results back
            # softmax_chunks = [F.softmax(chunk, dim=1) for chunk in output_chunks]

            # # Concatenate the softmax outputs to get the final tensor
            # output_softmax = torch.cat(softmax_chunks, dim=1)
            
            predicts = torch.topk(outputs, k=self.topk, dim=1, largest=True, sorted=True)[1].view(-1)  # [bs, topk]
            y_pred_task.append((torch.div(predicts, self.class_num, rounding_mode='floor')).cpu())
            # y_pred_task.append((predicts//self.class_num).cpu())

            outputs_with_task = torch.zeros_like(outputs)[:,:self.class_num]
            for idx, i in enumerate(torch.div(targets, self.class_num, rounding_mode='floor')):
                en, be = self.class_num*i, self.class_num*(i+1)
                outputs_with_task[idx] = outputs[idx, en:be]
            predicts_with_task = outputs_with_task.argmax(dim=1)
            predicts_with_task = predicts_with_task + (torch.div(targets, self.class_num, rounding_mode='floor'))*self.class_num

            if self.test:
                # Get unique values in the target tensor
                unique_values = torch.unique(targets)
                input_list, target_list, output_given_task_list = [], [], []
                for value in unique_values:
                    # Get indices where out is equal to the current unique value
                    indices = (targets == value).nonzero(as_tuple=True)[0]  # Get the indices of matching values
                    input_list.append(inputs[indices])  # Append the grouped inp
                    target_list.append(targets[indices])  # Create a tensor for the out value and append

                # del outputs
                # torch.cuda.empty_cache()
                # Predict given task, using just lora branch up to task index

                # if target_list[0][0] > 10:    
                #     import ipdb; ipdb.set_trace()
                
                for inputl, targetl in zip(input_list, target_list):
                    task_idx = targetl[0] // self.class_num
                    # en, be = self.class_num*task_idx, self.class_num*(task_idx+1)
                    if isinstance(self._network, nn.DataParallel):
                        output_given_task = self._network.module.interface_task(inputl, task_idx)
                    else:
                        output_given_task = self._network.interface_task(inputl, task_idx)
                    output_given_task_list.append(output_given_task)
                try:
                    outputs_given_task = torch.cat(output_given_task_list, dim=0)
                except Exception as e:
                    import ipdb; ipdb.set_trace()
                predicts_given_task = outputs_given_task.argmax(dim=1)
                predicts_given_task += self.class_num*task_idx
            else:
                # predicts_given_task = deepcopy(predicts_with_task)
                predicts_given_task = predicts_with_task

            # print(predicts.shape)
            y_pred.append(predicts.cpu().numpy())
            y_pred_with_task.append(predicts_with_task.cpu().numpy())
            y_pred_given_task.append(predicts_given_task.cpu().numpy())
            y_true.append(targets.cpu().numpy())

        return np.concatenate(y_pred), np.concatenate(y_pred_with_task), np.concatenate(y_pred_given_task), np.concatenate(y_true), torch.cat(y_pred_task), torch.cat(y_true_task)  # [N, topk]
    
    def _eval_cnn_test(self, loader, args):
        self._network.eval()
        y_pred, y_true = [], []
        y_pred_softmax = []
        y_pred_with_task = []
        y_pred_given_task = []
        y_pred_task, y_true_task = [], []
        for _, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            targets = targets.to(self._device)
            with torch.no_grad():
                y_true_task.append((torch.div(targets, self.class_num, rounding_mode='floor')).cpu())

                if isinstance(self._network, nn.DataParallel):
                    outputs = self._network.module.interface(inputs)
                else:
                    outputs = self._network.interface(inputs)

            # Split the tensor into 4 chunks of 10 values each along the second dimension
            output_chunks = torch.chunk(outputs, args['total_sessions'], dim=1)

            # # Apply softmax to each chunk independently and concatenate the results back
            # softmax_chunks = [F.softmax(chunk, dim=1) for chunk in output_chunks]
            # # Concatenate the softmax outputs to get the final tensor
            # outputs_softmax = torch.cat(softmax_chunks, dim=1)
            # predicts_softmax = torch.topk(outputs_softmax, k=self.topk, dim=1, largest=True, sorted=True)[1].view(-1)  # [bs, topk]

            # Calculate entropy for each chunk
            entropies = []

            for chunk in output_chunks:
                # Apply softmax to get probability distributions (normalize across dimension 1)
                probs = F.softmax(chunk / 20, dim=1)
                # Calculate log probabilities
                log_probs = torch.log(probs + 1e-8)  # Adding a small value to avoid log(0)
                # Calculate entropy
                entropy = -torch.sum(probs * log_probs, dim=1)
                entropies.append(entropy)

            # for chunk in output_chunks:
            #     # Apply softmax to get probability distributions (normalize across dimension 1)
            #     max_indices = torch.argmax(chunk, dim=1, keepdim=True)
            #     # Create a mask to filter out the maximum value
            #     mask = torch.ones_like(chunk, dtype=torch.bool)
            #     mask.scatter_(1, max_indices, False)

            #     # Apply the mask to get a new tensor of shape (16, 9)
            #     chunk = chunk[mask].view(16, 9)
            #     probs = F.softmax(chunk, dim=1)
            #     # Calculate log probabilities
            #     log_probs = torch.log(probs + 1e-8)  # Adding a small value to avoid log(0)
            #     # Calculate entropy
            #     entropy = -torch.sum(probs * log_probs, dim=1)
            #     entropies.append(entropy)

            # Stack entropies to form a tensor of shape (16, 10), where each column corresponds to a chunk
            entropies_stack = torch.stack(entropies, dim=1)
            y_pred_task_softmax = torch.argmin(entropies_stack, dim=1)
            outputs_with_task_softmax = torch.zeros_like(outputs)[:,:self.class_num]
            
            for idx, i in enumerate(y_pred_task_softmax):
                # try:
                #     en, be = self.class_num*i, self.class_num*(i+1)
                #     outputs_with_task_softmax[idx] = outputs[idx, en:be]
                # except Exception as e:
                #     pass
                en, be = self.class_num*i, self.class_num*(i+1)
                outputs_with_task_softmax[idx] = outputs[idx, en:be]
            predicts_with_task_softmax = outputs_with_task_softmax.argmax(dim=1)
            predicts_softmax = predicts_with_task_softmax + y_pred_task_softmax*self.class_num


            
            predicts = torch.topk(outputs, k=self.topk, dim=1, largest=True, sorted=True)[1].view(-1)  # [bs, topk]
            y_pred_task.append((torch.div(predicts, self.class_num, rounding_mode='floor')).cpu())
            # y_pred_task.append((predicts//self.class_num).cpu())

            outputs_with_task = torch.zeros_like(outputs)[:,:self.class_num]
            for idx, i in enumerate(torch.div(targets, self.class_num, rounding_mode='floor')):
                en, be = self.class_num*i, self.class_num*(i+1)
                outputs_with_task[idx] = outputs[idx, en:be]
            predicts_with_task = outputs_with_task.argmax(dim=1)
            predicts_with_task = predicts_with_task + (torch.div(targets, self.class_num, rounding_mode='floor'))*self.class_num

            # Get unique values in the target tensor
            unique_values = torch.unique(targets)
            input_list, target_list, output_given_task_list = [], [], []
            for value in unique_values:
                # Get indices where out is equal to the current unique value
                indices = (targets == value).nonzero(as_tuple=True)[0]  # Get the indices of matching values
                input_list.append(inputs[indices])  # Append the grouped inp
                target_list.append(targets[indices])  # Create a tensor for the out value and append

            # del outputs
            # torch.cuda.empty_cache()
            # Predict given task, using just lora branch up to task index

            # if target_list[0][0] > 10:    
            #     import ipdb; ipdb.set_trace()
            
            for inputl, targetl in zip(input_list, target_list):
                task_idx = targetl[0] // self.class_num
                # en, be = self.class_num*task_idx, self.class_num*(task_idx+1)
                if isinstance(self._network, nn.DataParallel):
                    output_given_task = self._network.module.interface_task(inputl, task_idx)
                else:
                    output_given_task = self._network.interface_task(inputl, task_idx)
                output_given_task_list.append(output_given_task)
            try:
                outputs_given_task = torch.cat(output_given_task_list, dim=0)
            except Exception as e:
                import ipdb; ipdb.set_trace()
            predicts_given_task = outputs_given_task.argmax(dim=1)
            predicts_given_task += self.class_num*task_idx

            # print(predicts.shape)
            y_pred.append(predicts.cpu().numpy())
            y_pred_softmax.append(predicts_softmax.cpu().numpy())
            y_pred_with_task.append(predicts_with_task.cpu().numpy())
            y_pred_given_task.append(predicts_given_task.cpu().numpy())
            y_true.append(targets.cpu().numpy())

        return np.concatenate(y_pred), np.concatenate(y_pred_softmax), np.concatenate(y_pred_with_task), np.concatenate(y_pred_given_task), np.concatenate(y_true), torch.cat(y_pred_task), torch.cat(y_true_task)  # [N, topk]

    def _compute_accuracy_domain(self, model, loader):
        model.eval()
        correct, total = 0, 0
        for i, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                outputs = model(inputs)['logits']

            predicts = torch.max(outputs, dim=1)[1]
            correct += ((predicts % self.class_num).cpu() == (targets % self.class_num)).sum()
            total += len(targets)

        return np.around(tensor2numpy(correct) * 100 / total, decimals=2)

    def update_DualGPM (self, mat_list):
        threshold = (self.lame - self.lamb)*self._cur_task/self.total_sessions + self.lamb
        print ('Threshold: ', threshold) 
        if len(self.feature_list) == 0:
            # After First Task 
            for i in range(len(mat_list)):
                activation = mat_list[i]
                U,S,_ = np.linalg.svd(activation, full_matrices=False)
                # criteria (Eq-5)
                sval_total = (S**2).sum()
                sval_ratio = (S**2)/sval_total
                r = np.sum(np.cumsum(sval_ratio)<threshold) #+1  
                if r < (activation.shape[0]/2):
                    self.feature_list.append(U[:,0:max(r,1)])
                    self.project_type.append('remove')
                else:
                    self.feature_list.append(U[:,0:max(r,1)])
                    self.project_type.append('retain')
        else:
            for i in range(len(mat_list)):
                if self.project_type[i] == 'remove':
                    activation = mat_list[i]
                    U1,S1,Vh1=np.linalg.svd(activation, full_matrices=False)
                    sval_total = (S1**2).sum()
                    # Projected Representation (Eq-8)
                    act_hat = activation - np.dot(np.dot(self.feature_list[i],self.feature_list[i].transpose()),activation)
                    U,S,_ = np.linalg.svd(act_hat, full_matrices=False)
                    # criteria (Eq-9)
                    sval_hat = (S**2).sum()
                    sval_ratio = (S**2)/sval_total               
                    accumulated_sval = (sval_total-sval_hat)/sval_total
            
                    r = 0
                    for ii in range (sval_ratio.shape[0]):
                        if accumulated_sval < threshold:
                            accumulated_sval += sval_ratio[ii]
                            r += 1
                        else:
                            break
                    if r == 0:
                        print ('Skip Updating DualGPM for layer: {}'.format(i+1)) 
                        continue
                    # update GPM
                    Ui=np.hstack((self.feature_list[i],U[:,0:r]))  
                    if Ui.shape[1] > Ui.shape[0] :
                        self.feature_list[i]=Ui[:,0:Ui.shape[0]]
                    else:
                        self.feature_list[i]=Ui
                else:
                    assert self.project_type[i] == 'retain'
                    activation = mat_list[i]
                    U1,S1,_=np.linalg.svd(activation, full_matrices=False)
                    sval_total = (S1**2).sum()
                    # Projected Representation (Eq-8)
                    act_hat = np.dot(np.dot(self.feature_list[i],self.feature_list[i].transpose()),activation)
                    U,S,_ = np.linalg.svd(act_hat, full_matrices=False)
                    # criteria (Eq-9)
                    sval_hat = (S**2).sum()
                    sval_ratio = (S**2)/sval_total               
                    accumulated_sval = sval_hat/sval_total

                    r = 0
                    for ii in range (sval_ratio.shape[0]):
                        if accumulated_sval >= (1-threshold):
                            accumulated_sval -= sval_ratio[ii]
                            r += 1
                        else:
                            break
                    if r == 0:
                        print ('Skip Updating DualGPM for layer: {}'.format(i+1)) 
                        continue

                    # update GPM by Projected Representation (Eq-8)
                    act_feature = self.feature_list[i] - np.dot(np.dot(U[:,0:r],U[:,0:r].transpose()),self.feature_list[i])
                    Ui, _, _ = np.linalg.svd(act_feature)
                    self.feature_list[i]=Ui[:,:self.feature_list[i].shape[1]-r]

        print('-'*40)
        print('Gradient Constraints Summary')
        print('-'*40)
        if self.use_dualgpm:
            for i in range(len(self.feature_list)):
                if self.project_type[i]=='remove' and (self.feature_list[i].shape[1] > (self.feature_list[i].shape[0]/2)):
                    feature = self.feature_list[i]
                    U, S, V = np.linalg.svd(feature)
                    new_feature = U[:,feature.shape[1]:]
                    self.feature_list[i] = new_feature
                    self.project_type[i] = 'retain'
                elif self.project_type[i]=='retain':
                    try:
                        assert self.feature_list[i].shape[1] <= (self.feature_list[i].shape[0]/2)
                    except Exception as e:
                        pass
                        # import ipdb; ipdb.set_trace()
                print ('Layer {} : {}/{} type {}'.format(i+1,self.feature_list[i].shape[1], self.feature_list[i].shape[0], self.project_type[i]))
            print('-'*40)


    def update_GPM (self, mat_list):
        threshold = (self.lame - self.lamb)*self._cur_task/self.total_sessions + self.lamb
        print ('Threshold: ', threshold) 
        if len(self.feature_list) == 0:
            # After First Task 
            for i in range(len(mat_list)):
                activation = mat_list[i]
                U,S,_ = np.linalg.svd(activation, full_matrices=False)
                # criteria (Eq-5)
                sval_total = (S**2).sum()
                sval_ratio = (S**2)/sval_total
                r = np.sum(np.cumsum(sval_ratio)<threshold) #+1  
                self.feature_list.append(U[:,0:max(r,1)])
        else:
            for i in range(len(mat_list)):
                activation = mat_list[i]
                _,S1,_=np.linalg.svd(activation, full_matrices=False)
                sval_total = (S1**2).sum()
                # Projected Representation (Eq-8)
                act_hat = activation - np.dot(np.dot(self.feature_list[i],self.feature_list[i].transpose()),activation)
                U,S,_ = np.linalg.svd(act_hat, full_matrices=False)
                # criteria (Eq-9)
                sval_hat = (S**2).sum()
                sval_ratio = (S**2)/sval_total               
                accumulated_sval = (sval_total-sval_hat)/sval_total
            
                r = 0
                for ii in range (sval_ratio.shape[0]):
                    if accumulated_sval < threshold:
                        accumulated_sval += sval_ratio[ii]
                        r += 1
                    else:
                        break
                if r == 0:
                    print ('Skip Updating GPM for layer: {}'.format(i+1)) 
                    continue
                # update GPM
                Ui=np.hstack((self.feature_list[i],U[:,0:r]))  
                if Ui.shape[1] > Ui.shape[0] :
                    self.feature_list[i]=Ui[:,0:Ui.shape[0]]
                else:
                    self.feature_list[i]=Ui
    
        print('-'*40)
        print('Gradient Constraints Summary')
        print('-'*40)
        for i in range(len(self.feature_list)):
            print('Layer {} : {}/{}'.format(i+1,self.feature_list[i].shape[1], self.feature_list[i].shape[0]))
        print('-'*40)  
