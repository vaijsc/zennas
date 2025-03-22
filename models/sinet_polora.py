import torch
import torch.nn as nn
from einops.layers.torch import Reduce
import copy

from timm_monet.models import create_model


def _create_image_encoder(args, pretrained=False, **kwargs):
    if kwargs.get('features_only', None):
        raise RuntimeError('features_only not implemented for Vision Transformer models.')

    # NOTE this extra code to support handling of repr size for in21k pretrained models
    # pretrained_cfg = resolve_pretrained_cfg(variant, kwargs=kwargs)
    pretrained_cfg = {
        'url': 'https://storage.googleapis.com/vit_models/augreg/B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0.npz',
        'num_classes': 21843,
        'input_size': (3, 224, 224),
        'pool_size': None,
        'crop_pct': 0.9,
        'interpolation': 'bicubic',
        'fixed_input_size': True,
        'mean': (0.5, 0.5, 0.5),
        'std': (0.5, 0.5, 0.5),
        'first_conv': 'patch_embed.proj',
        'classifier': 'head'
    }
    default_num_classes = pretrained_cfg['num_classes']
    num_classes = kwargs.get('num_classes', default_num_classes)
    repr_size = kwargs.pop('representation_size', None)
    if repr_size is not None and num_classes != default_num_classes:
        repr_size = None

    model = create_model(
        args['model'],
        pretrained=False,
        in_chans=args['in_chans'],
        num_classes=args['num_classes'],
        drop_rate=args['drop'],
        drop_path_rate=None,
        drop_block_rate=None,
        global_pool=None,
        bn_momentum=None,
        bn_eps=None,
        scriptable=args['torchscript'],
        # checkpoint_path=args['initial_checkpoint'],
        checkpoint_path=None,
        **kwargs,
    )

    if not args['test']:
        checkpoint_path = args['initial_checkpoint']
        # Ensure checkpoint is on the correct device
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        # Load the checkpoint weights into the model
        model.load_state_dict(checkpoint, strict=False)  # Adjust key if needed based on the saved checkpoint format

    # model2 = create_model(
    #     args['model'],
    #     pretrained=False,
    #     in_chans=args['in_chans'],
    #     num_classes=args['num_classes'],
    #     drop_rate=args['drop'],
    #     drop_path_rate=None,
    #     drop_block_rate=None,
    #     global_pool=None,
    #     bn_momentum=None,
    #     bn_eps=None,
    #     scriptable=args['torchscript'],
    #     checkpoint_path=args['initial_checkpoint'],
    #     **args['model_kwargs'],
    # )
    # model2.load_state_dict(model.state_dict())
    # checkpoint_path = './checkpoints/MONet_S.pth'

    # # Ensure checkpoint is on the correct device
    # checkpoint = torch.load(checkpoint_path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

    # # Load the checkpoint weights into the model
    # model.load_state_dict(checkpoint)  # Adjust key if needed based on the saved checkpoint format
    # # Retrieve state dictionaries of both models
    # model_dict = model.state_dict()
    # model2_dict = model2.state_dict()

    # # Check if the keys (parameter names) match between the two models
    # if model_dict.keys() != model2_dict.keys():
    #     print("Models have different parameter keys.")
    # else:
    #     # Compare each parameter value
    #     identical = True
    #     for key in model_dict:
    #         if not torch.equal(model_dict[key], model2_dict[key]):
    #             print(f"Parameter mismatch found at: {key}")
    #             identical = False
    #             break

    #     if identical:
    #         print("All parameters are identical between model and model2.")
    #     else:
    #         print("Models do not have the same parameters.")

    return model



class SiNet(nn.Module):

    def __init__(self, args):
        super(SiNet, self).__init__()

        model_kwargs = dict(n_tasks=args["total_sessions"], rank=args["rank"], alpha_lora=args['alpha_lora'])
        self.image_encoder =_create_image_encoder(args, pretrained=True, **model_kwargs)

        self.class_num = args["init_cls"]
        # self.classifier_pool = nn.ModuleList([
        #     nn.Linear(args["embd_dim"], self.class_num, bias=True)
        #     for i in range(args["total_sessions"])
        # ])

        self.classifier_pool = nn.ModuleList([
            nn.Sequential(
                Reduce('b c h w -> b c', 'mean'),
                nn.Linear(args["embd_dim"], self.class_num, bias=True)
            ) for i in range(args["total_sessions"])
        ])

        self.numtask = 0

    @property
    def feature_dim(self):
        return self.image_encoder.out_dim

    def extract_vector(self, image, task=None):
        if task == None:
            image_features = self.image_encoder(image, self.numtask-1)
        else:
            image_features = self.image_encoder(image, task)
        # image_features = image_features[:,0,:]
        # image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        return image_features

    def forward(self, image, get_feat=False, get_cur_feat=False, fc_only=False):
        if fc_only:
            fc_outs = []
            for ti in range(self.numtask):
                fc_out = self.classifier_pool[ti](image)
                fc_outs.append(fc_out)
            return torch.cat(fc_outs, dim=1)

        logits = []
        image_features = self.image_encoder(image, task_id=self.numtask-1, get_feat=get_feat, get_cur_feat=get_cur_feat)
        for classifier in [self.classifier_pool[self.numtask-1]]:
            logits.append(classifier(image_features))

        return {
            'logits': torch.cat(logits, dim=1),
            'features': image_features
        }

    def interface(self, image, task_id = None):
        image_features = self.image_encoder(image, task_id=self.numtask-1 if task_id is None else task_id)

        logits = []
        for classifier in self.classifier_pool[:self.numtask]:
            logits.append(classifier(image_features))
        logits = torch.cat(logits,1)
        return logits
    
    def interface_task(self, image, task_id = None):
        image_features = self.image_encoder(image, task_id=task_id)

        logits = []
        logits.append(self.classifier_pool[task_id](image_features))
        logits = torch.cat(logits,1)
        return logits

    def update_fc(self, nb_classes):
        self.numtask +=1

    def classifier_backup(self, task_id):
        self.classifier_pool_backup[task_id].load_state_dict(self.classifier_pool[task_id].state_dict())

    def classifier_recall(self):
        self.classifier_pool.load_state_dict(self.old_state_dict)

    def copy(self):
        return copy.deepcopy(self)

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        self.eval()

        return self
