import os
import os.path
import sys
import logging
import copy
import time
import torch
from utils import factory
from utils.data_manager import DataManager
from utils.toolkit import count_parameters


def evaluate(args):
    seed_list = copy.deepcopy(args['seed'])
    device = copy.deepcopy(args['device'])
    device = device.split(',')

    for seed in seed_list:
        args['seed'] = seed
        args['device'] = device
        _evaluate(args)

    myseed = 42069  # set a random seed for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(myseed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(myseed)


def _evaluate(args):
    logdir = f"logs/{args['dataset']}/{args['init_cls']}_{args['increment']}_{args['net_type']}/{args['model_name']}/alpha_{args['alpha_lora']}/r_{args['rank']}/{args['lamb']}_{args['lame']}-{args['lrate']}/{args['model']}/{args['exp']}"

    if not os.path.exists(logdir):
        os.makedirs(logdir)
    logfilename = os.path.join(logdir, '{}'.format(args['seed']))
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(filename)s] => %(message)s',
        handlers=[
            logging.FileHandler(filename=logfilename + 'evaluate.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    if not os.path.exists(logfilename):
        os.makedirs(logfilename)
    print(logfilename)
    _set_random(args)
    _set_device(args)
    print_args(args)
    data_manager = DataManager(args['dataset'], args['shuffle'], args['seed'], args['init_cls'], args['increment'], args)
    args['class_order'] = data_manager._class_order
    model = factory.get_model(args['model_name'], args)    
    task = args['tasktest']
    if args['test']:
        checkpoint_path = f"{logdir}/{args['seed']}/task_{task}.pth"
        # checkpoint_path = args['initial_checkpoint']
        # Ensure checkpoint is on the correct device
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        # Load the state_dict into model._network
        model._network.load_state_dict(checkpoint, strict=True)
        # for layer_name in checkpoint.keys():
        #     if 'lora' not in layer_name:
        #         print(layer_name)
        # import ipdb; ipdb.set_trace()

    cnn_curve, cnn_curve_softmax, cnn_curve_with_task, cnn_curve_given_task, nme_curve, cnn_curve_task = {'top1': []}, {'top1': []}, {'top1': []}, {'top1': []}, {'top1': []}, {'top1': []}
    print(f'\nTask {task}')
    model.incremental_test(data_manager, task)
    cnn_accy, cnn_accy_softmax, cnn_accy_with_task, cnn_accy_given_task, nme_accy, cnn_accy_task = model.eval_tasktest(args)

    logging.info('Accuracy: {}'.format(cnn_accy['grouped']))
    logging.info('Accuracy softmax: {}'.format(cnn_accy_softmax['grouped']))
    logging.info('Accuracy with tasks: {}'.format(cnn_accy_with_task['grouped']))
    logging.info('Accuracy given tasks: {}'.format(cnn_accy_given_task['grouped']))
    cnn_curve['top1'].append(cnn_accy['top1'])
    cnn_curve_softmax['top1'].append(cnn_accy_softmax['top1'])
    cnn_curve_with_task['top1'].append(cnn_accy_with_task['top1'])
    cnn_curve_given_task['top1'].append(cnn_accy_given_task['top1'])
    cnn_curve_task['top1'].append(cnn_accy_task)
    logging.info('Accuracy top1 curve: {}'.format(cnn_curve['top1']))
    logging.info('Accuracy softmax top1 curve: {}'.format(cnn_curve_softmax['top1']))
    logging.info('Accuracy top1 with task curve: {}'.format(cnn_curve_with_task['top1']))
    logging.info('Accuracy top1 given task curve: {}'.format(cnn_curve_given_task['top1']))
    logging.info('Accuracy top1 task curve: {}'.format(cnn_curve_task['top1']))


def _set_device(args):
    device_type = args['device']
    gpus = []

    for device in device_type:
        if device_type == -1:
            device = torch.device('cpu')
        else:
            device = torch.device('cuda:{}'.format(device))

        gpus.append(device)

    args['device'] = gpus


def _set_random(args):
    torch.manual_seed(args['seed'])
    torch.cuda.manual_seed(args['seed'])
    torch.cuda.manual_seed_all(args['seed'])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def print_args(args):
    for key, value in args.items():
        logging.info('{}: {}'.format(key, value))

