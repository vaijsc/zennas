import os
import numpy as np
from torchvision import datasets, transforms
from utils.toolkit import split_images_labels
from utils.datautils.core50data import CORE50
import ipdb
import yaml
from PIL import Image
from shutil import move, rmtree
import torch
from sklearn.model_selection import train_test_split

class iData(object):
    train_trsf = []
    test_trsf = []
    common_trsf = []
    class_order = None

def build_transform(is_train, args):
    input_size = 224
    resize_im = input_size > 32
    if is_train:
        scale = (0.05, 1.0)
        ratio = (3. / 4., 4. / 3.)
        
        transform = [
            transforms.RandomResizedCrop(input_size, scale=scale, ratio=ratio),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
        ]
        return transform

    t = []
    if resize_im:
        size = int((256 / 224) * input_size)
        t.append(
            transforms.Resize(size, interpolation=3),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(input_size))
    t.append(transforms.ToTensor())
    
    # return transforms.Compose(t)
    return t

class iCUB(iData):
    use_path = True
    
    train_trsf=[
            transforms.RandomResizedCrop(224, scale=(0.05, 1.0), ratio=(3./4., 4./3.)),
            transforms.RandomHorizontalFlip(p=0.5),
            ]
    test_trsf=[
        transforms.Resize(256, interpolation=3), 
        transforms.CenterCrop(224),
        ]
    common_trsf = [transforms.ToTensor(),
                   transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]

    class_order = np.arange(200).tolist()

    def __init__(self, args):
        self.args = args
        class_order = np.arange(200).tolist()
        self.class_order = class_order

    def download_data(self):
        # assert 0, "You should specify the folder of your dataset"
        train_dir = "data/cub/train/"
        test_dir = "data/cub/test/"

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)

class iCIFAR10(iData):
    use_path = False
    train_trsf = [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ]

    test_trsf = [
        transforms.Resize(224),
        transforms.ToTensor()
        ]
    common_trsf = [
        transforms.Normalize(
            mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0)
        ),
    ]

    class_order = np.arange(10).tolist()

    def __init__(self, args):
        self.args = args
        class_order = np.arange(10).tolist()
        self.class_order = class_order

    def download_data(self):
        train_dataset = datasets.cifar.CIFAR10(self.args['data_path'], train=True, download=True)
        test_dataset = datasets.cifar.CIFAR10(self.args['data_path'], train=False, download=True)
        self.train_data, self.train_targets = train_dataset.data, np.array(
            train_dataset.targets
        )
        self.test_data, self.test_targets = test_dataset.data, np.array(
            test_dataset.targets
        )
        import ipdb; ipdb.set_trace()


class iCIFAR100(iData):
    use_path = False
    train_trsf = [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
    ]

    test_trsf = [
        transforms.Resize(224),
        ]
    
    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0)
        ),
    ]

    class_order = np.arange(100).tolist()

    def __init__(self, args):
        self.args = args
        class_order = np.arange(100).tolist()
        self.class_order = class_order

    def download_data(self):
        train_dataset = datasets.cifar.CIFAR100(self.args['data_path'], train=True, download=True)
        test_dataset = datasets.cifar.CIFAR100(self.args['data_path'], train=False, download=True)
        self.train_data, self.train_targets = train_dataset.data, np.array(
            train_dataset.targets
        )
        self.test_data, self.test_targets = test_dataset.data, np.array(
            test_dataset.targets
        )

class iIMAGENET_R(iData):
    use_path = True
    train_trsf = [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ]

    test_trsf = [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor()
        ]
    common_trsf = [
        transforms.Normalize(
            mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0)
        ),
    ]

    class_order = np.arange(200).tolist()

    def __init__(self, args):
        self.args = args
        class_order = np.arange(200).tolist()
        self.class_order = class_order

    def download_data(self):
        # load splits from config file
        if not os.path.exists(os.path.join(self.args['data_path'], 'train')) and not os.path.exists(os.path.join(self.args['data_path'], 'train')):
            self.dataset = datasets.ImageFolder(self.args['data_path'], transform=None)
            
            train_size = int(0.8 * len(self.dataset))
            val_size = len(self.dataset) - train_size
            
            train, val = torch.utils.data.random_split(self.dataset, [train_size, val_size])
            train_idx, val_idx = train.indices, val.indices
    
            self.train_file_list = [self.dataset.imgs[i][0] for i in train_idx]
            self.test_file_list = [self.dataset.imgs[i][0] for i in val_idx]

            self.split()

        train_data_config = datasets.ImageFolder(os.path.join(self.args['data_path'], 'train')).samples
        test_data_config = datasets.ImageFolder(os.path.join(self.args['data_path'], 'test')).samples
        self.train_data = np.array([config[0] for config in train_data_config])
        self.train_targets = np.array([config[1] for config in train_data_config])
        self.test_data = np.array([config[0] for config in test_data_config])
        self.test_targets = np.array([config[1] for config in test_data_config])


    def split(self):
        train_folder = os.path.join(self.args['data_path'], 'train')
        test_folder = os.path.join(self.args['data_path'], 'test')

        if os.path.exists(train_folder):
            rmtree(train_folder)
        if os.path.exists(test_folder):
            rmtree(test_folder)
        os.mkdir(train_folder)
        os.mkdir(test_folder)

        for c in self.dataset.classes:
            if not os.path.exists(os.path.join(train_folder, c)):
                os.mkdir(os.path.join(os.path.join(train_folder, c)))
            if not os.path.exists(os.path.join(test_folder, c)):
                os.mkdir(os.path.join(os.path.join(test_folder, c)))
        
        for path in self.train_file_list:
            if '\\' in path:
                path = path.replace('\\', '/')
            src = path
            dst = os.path.join(train_folder, '/'.join(path.split('/')[-2:]))
            move(src, dst)

        for path in self.test_file_list:
            if '\\' in path:
                path = path.replace('\\', '/')
            src = path
            dst = os.path.join(test_folder, '/'.join(path.split('/')[-2:]))
            move(src, dst)
        
        for c in self.dataset.classes:
            path = os.path.join(self.args['data_path'], c)
            rmtree(path)


class iIMAGENET_A(iData):
    use_path = True
    train_trsf=[
            transforms.RandomResizedCrop(224, scale=(0.05, 1.0), ratio=(3./4., 4./3.)),
            transforms.RandomHorizontalFlip(p=0.5),
            ]
    test_trsf=[
        transforms.Resize(256, interpolation=3), 
        transforms.CenterCrop(224),
        ]
    common_trsf = [transforms.ToTensor()]

    class_order = np.arange(200).tolist()

    def __init__(self, args):
        self.args = args
        class_order = np.arange(200).tolist()
        self.class_order = class_order

    def download_data(self):
        # assert 0, "You should specify the folder of your dataset"
        train_dir = "data/imagenet-a/train/"
        test_dir = "data/imagenet-a/test/"

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)


class iDomainNet(iData):

    use_path = True
    train_trsf = [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
    ]
    test_trsf = [
        transforms.Resize(256),
        transforms.CenterCrop(224),
    ]
    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0]),
    ]

    def __init__(self, args):
        self.args = args
        class_order = np.arange(345).tolist()
        self.class_order = class_order
        self.domain_names = ["clipart", "infograph", "painting", "quickdraw", "real", "sketch", ]

    def download_data(self):
        # load splits from config file
        train_data_config = yaml.load(open('dataloaders/splits/domainnet_train.yaml', 'r'), Loader=yaml.Loader)
        test_data_config = yaml.load(open('dataloaders/splits/domainnet_test.yaml', 'r'), Loader=yaml.Loader)
        self.train_data = np.array(train_data_config['data'])
        self.train_targets = np.array(train_data_config['targets'])
        self.test_data = np.array(test_data_config['data'])
        self.test_targets = np.array(test_data_config['targets'])
        import ipdb; ipdb.set_trace()

# class iOfficeHome(iData):
#     use_path = True
#     train_trsf = [
#         transforms.RandomResizedCrop(224),
#         transforms.RandomHorizontalFlip(),
#     ]
#     test_trsf = [
#         transforms.Resize(256),
#         transforms.CenterCrop(224),
#     ]
#     common_trsf = [
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0]),
#     ]

#     def __init__(self, args):
#         self.args = args
#         class_order = np.arange(65).tolist()  # Office-Home has 65 classes
#         self.class_order = class_order
#         self.domain_names = ["Art", "Clipart", "Product", "Real_World"]

#     def download_data(self):
#         data_path = self.args['data_path']
        
#         # Collect images and labels from each domain
#         self.train_data, self.train_targets = [], []
#         self.test_data, self.test_targets = [], []

#         for domain in self.domain_names:
#             domain_path = os.path.join(data_path, domain)
#             for label, category in enumerate(sorted(os.listdir(domain_path))):
#                 category_path = os.path.join(domain_path, category)
#                 images = [os.path.join(category_path, img) for img in os.listdir(category_path) if img.endswith('.jpg')]

#                 # Split images into train and test
#                 train_imgs, test_imgs = train_test_split(images, test_size=0.2, random_state=42)
                
#                 # Assign to train and test lists
#                 self.train_data.extend(train_imgs)
#                 self.train_targets.extend([label] * len(train_imgs))
#                 self.test_data.extend(test_imgs)
#                 self.test_targets.extend([label] * len(test_imgs))
        
#         # Convert lists to numpy arrays for consistency with other datasets
#         self.train_data = np.array(self.train_data)
#         self.train_targets = np.array(self.train_targets)
#         self.test_data = np.array(self.test_data)
#         self.test_targets = np.array(self.test_targets)
        
class iOfficeHome(iData):
    use_path = True
    train_trsf = [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
    ]
    test_trsf = [
        transforms.Resize(256),
        transforms.CenterCrop(224),
    ]
    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0]),
    ]

    class_order = np.arange(260).tolist()

    def __init__(self, args):
        self.args = args
        class_order = np.arange(260).tolist()
        self.class_order = class_order
        self.domain_names = ["Art", "Clipart", "Product", "Real_World"]

    def download_data(self):
        data_path = self.args['data_path']
        
        # Initialize empty lists for data and labels
        self.train_data, self.train_targets = [], []
        self.test_data, self.test_targets = [], []

        num_classes = 65  # Each domain has 65 classes

        for domain_idx, domain in enumerate(self.domain_names):
            domain_path = os.path.join(data_path, domain)
            label_offset = domain_idx * num_classes  # Offset for the domain's labels
            
            for label, category in enumerate(sorted(os.listdir(domain_path))):
                category_path = os.path.join(domain_path, category)
                images = [os.path.join(category_path, img) for img in os.listdir(category_path) if img.endswith('.jpg')]

                # Split images into train and test
                train_imgs, test_imgs = train_test_split(images, test_size=0.2, random_state=12)

                # Assign labels with the offset for this domain
                adjusted_label = label + label_offset
                self.train_data.extend(train_imgs)
                self.train_targets.extend([adjusted_label] * len(train_imgs))
                self.test_data.extend(test_imgs)
                self.test_targets.extend([adjusted_label] * len(test_imgs))
        
        # Convert lists to numpy arrays for consistency with other datasets
        self.train_data = np.array(self.train_data)
        self.train_targets = np.array(self.train_targets)
        self.test_data = np.array(self.test_data)
        self.test_targets = np.array(self.test_targets)

class iOfficeHome_90(iData):
    use_path = True
    train_trsf = [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
    ]
    test_trsf = [
        transforms.Resize(256),
        transforms.CenterCrop(224),
    ]
    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0]),
    ]

    class_order = np.arange(260).tolist()

    def __init__(self, args):
        self.args = args
        class_order = np.arange(260).tolist()
        self.class_order = class_order
        self.domain_names = ["Art", "Clipart", "Product", "Real_World"]

    def download_data(self):
        data_path = self.args['data_path']
        
        # Initialize empty lists for data and labels
        self.train_data, self.train_targets = [], []
        self.test_data, self.test_targets = [], []

        num_classes = 65  # Each domain has 65 classes

        for domain_idx, domain in enumerate(self.domain_names):
            domain_path = os.path.join(data_path, domain)
            label_offset = domain_idx * num_classes  # Offset for the domain's labels
            
            for label, category in enumerate(sorted(os.listdir(domain_path))):
                category_path = os.path.join(domain_path, category)
                images = [os.path.join(category_path, img) for img in os.listdir(category_path) if img.endswith('.jpg')]

                # Split images into train and test
                train_imgs, test_imgs = train_test_split(images, test_size=0.1, random_state=12)

                # Assign labels with the offset for this domain
                adjusted_label = label + label_offset
                self.train_data.extend(train_imgs)
                self.train_targets.extend([adjusted_label] * len(train_imgs))
                self.test_data.extend(test_imgs)
                self.test_targets.extend([adjusted_label] * len(test_imgs))
        
        # Convert lists to numpy arrays for consistency with other datasets
        self.train_data = np.array(self.train_data)
        self.train_targets = np.array(self.train_targets)
        self.test_data = np.array(self.test_data)
        self.test_targets = np.array(self.test_targets)

class iOfficeHome_80norm(iData):
    use_path = True
    train_trsf = [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
    ]
    test_trsf = [
        transforms.Resize(256),
        transforms.CenterCrop(224),
    ]
    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]

    class_order = np.arange(260).tolist()

    def __init__(self, args):
        self.args = args
        class_order = np.arange(260).tolist()
        self.class_order = class_order
        # self.domain_names = ["Art", "Clipart", "Product", "Real_World"]
        self.domain_names = ["Real_World", "Clipart", "Product", "Art"]

    def download_data(self):
        data_path = self.args['data_path']
        
        # Initialize empty lists for data and labels
        self.train_data, self.train_targets = [], []
        self.test_data, self.test_targets = [], []

        num_classes = 65  # Each domain has 65 classes

        for domain_idx, domain in enumerate(self.domain_names):
            domain_path = os.path.join(data_path, domain)
            label_offset = domain_idx * num_classes  # Offset for the domain's labels
            
            for label, category in enumerate(sorted(os.listdir(domain_path))):
                category_path = os.path.join(domain_path, category)
                images = [os.path.join(category_path, img) for img in os.listdir(category_path) if img.endswith('.jpg')]

                # Split images into train and test
                train_imgs, test_imgs = train_test_split(images, test_size=0.2, random_state=12)

                # Assign labels with the offset for this domain
                adjusted_label = label + label_offset
                self.train_data.extend(train_imgs)
                self.train_targets.extend([adjusted_label] * len(train_imgs))
                self.test_data.extend(test_imgs)
                self.test_targets.extend([adjusted_label] * len(test_imgs))
        
        # Convert lists to numpy arrays for consistency with other datasets
        self.train_data = np.array(self.train_data)
        self.train_targets = np.array(self.train_targets)
        self.test_data = np.array(self.test_data)
        self.test_targets = np.array(self.test_targets)

class iOfficeHome_90norm(iData):
    use_path = True
    train_trsf = [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
    ]
    test_trsf = [
        transforms.Resize(256),
        transforms.CenterCrop(224),
    ]
    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]

    class_order = np.arange(260).tolist()

    def __init__(self, args):
        self.args = args
        class_order = np.arange(260).tolist()
        self.class_order = class_order
        # self.domain_names = ["Art", "Clipart", "Product", "Real_World"]
        self.domain_names = ["Real_World", "Clipart", "Product", "Art"]

    def download_data(self):
        data_path = self.args['data_path']
        
        # Initialize empty lists for data and labels
        self.train_data, self.train_targets = [], []
        self.test_data, self.test_targets = [], []

        num_classes = 65  # Each domain has 65 classes

        for domain_idx, domain in enumerate(self.domain_names):
            domain_path = os.path.join(data_path, domain)
            label_offset = domain_idx * num_classes  # Offset for the domain's labels
            
            for label, category in enumerate(sorted(os.listdir(domain_path))):
                category_path = os.path.join(domain_path, category)
                images = [os.path.join(category_path, img) for img in os.listdir(category_path) if img.endswith('.jpg')]

                # Split images into train and test
                train_imgs, test_imgs = train_test_split(images, test_size=0.1, random_state=12)

                # Assign labels with the offset for this domain
                adjusted_label = label + label_offset
                self.train_data.extend(train_imgs)
                self.train_targets.extend([adjusted_label] * len(train_imgs))
                self.test_data.extend(test_imgs)
                self.test_targets.extend([adjusted_label] * len(test_imgs))
        
        # Convert lists to numpy arrays for consistency with other datasets
        self.train_data = np.array(self.train_data)
        self.train_targets = np.array(self.train_targets)
        self.test_data = np.array(self.test_data)
        self.test_targets = np.array(self.test_targets)

class iPACS_80(iData):
    use_path = True
    train_trsf = [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
    ]
    test_trsf = [
        transforms.Resize(256),
        transforms.CenterCrop(224),
    ]
    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0]),
    ]

    class_order = np.arange(28).tolist()

    def __init__(self, args):
        self.args = args
        class_order = np.arange(28).tolist()
        self.class_order = class_order
        self.domain_names = ["art_painting", "cartoon", "photo", "sketch"]

    def download_data(self):
        data_path = self.args['data_path']
        
        # Initialize empty lists for data and labels
        self.train_data, self.train_targets = [], []
        self.test_data, self.test_targets = [], []

        num_classes = 7  # Each domain has 65 classes

        for domain_idx, domain in enumerate(self.domain_names):
            domain_path = os.path.join(data_path, domain)
            label_offset = domain_idx * num_classes  # Offset for the domain's labels
            
            for label, category in enumerate(sorted(os.listdir(domain_path))):
                category_path = os.path.join(domain_path, category)
                images = [os.path.join(category_path, img) for img in os.listdir(category_path) if img.endswith('.jpg') or img.endswith('.png')]

                # Split images into train and test
                train_imgs, test_imgs = train_test_split(images, test_size=0.2, random_state=12)

                # Assign labels with the offset for this domain
                adjusted_label = label + label_offset
                self.train_data.extend(train_imgs)
                self.train_targets.extend([adjusted_label] * len(train_imgs))
                self.test_data.extend(test_imgs)
                self.test_targets.extend([adjusted_label] * len(test_imgs))
        
        # Convert lists to numpy arrays for consistency with other datasets
        self.train_data = np.array(self.train_data)
        self.train_targets = np.array(self.train_targets)
        self.test_data = np.array(self.test_data)
        self.test_targets = np.array(self.test_targets)

class iPACS_80norm(iData):
    use_path = True
    train_trsf = [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
    ]
    test_trsf = [
        transforms.Resize(256),
        transforms.CenterCrop(224),
    ]
    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]

    class_order = np.arange(28).tolist()

    def __init__(self, args):
        self.args = args
        class_order = np.arange(28).tolist()
        self.class_order = class_order
        # self.domain_names = ["art_painting", "cartoon", "photo", "sketch"]
        self.domain_names = ["photo", "art_painting", "cartoon", "sketch"]

    def download_data(self):
        data_path = self.args['data_path']
        
        # Initialize empty lists for data and labels
        self.train_data, self.train_targets = [], []
        self.test_data, self.test_targets = [], []

        num_classes = 7  # Each domain has 65 classes

        for domain_idx, domain in enumerate(self.domain_names):
            domain_path = os.path.join(data_path, domain)
            label_offset = domain_idx * num_classes  # Offset for the domain's labels
            
            for label, category in enumerate(sorted(os.listdir(domain_path))):
                category_path = os.path.join(domain_path, category)
                images = [os.path.join(category_path, img) for img in os.listdir(category_path) if img.endswith('.jpg') or img.endswith('.png')]
                
                # Split images into train and test
                try:
                    train_imgs, test_imgs = train_test_split(images, test_size=0.2, random_state=12)
                except Exception as e:
                    import ipdb; ipdb.set_trace()

                # Assign labels with the offset for this domain
                adjusted_label = label + label_offset
                self.train_data.extend(train_imgs)
                self.train_targets.extend([adjusted_label] * len(train_imgs))
                self.test_data.extend(test_imgs)
                self.test_targets.extend([adjusted_label] * len(test_imgs))
        
        # Convert lists to numpy arrays for consistency with other datasets
        self.train_data = np.array(self.train_data)
        self.train_targets = np.array(self.train_targets)
        self.test_data = np.array(self.test_data)
        self.test_targets = np.array(self.test_targets)

class iVLCS_80(iData):
    use_path = True
    train_trsf = [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
    ]
    test_trsf = [
        transforms.Resize(256),
        transforms.CenterCrop(224),
    ]
    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0]),
    ]

    class_order = np.arange(20).tolist()

    def __init__(self, args):
        self.args = args
        class_order = np.arange(20).tolist()
        self.class_order = class_order
        self.domain_names = ["Caltech101", "LabelMe", "SUN09", "VOC2007"]

    def download_data(self):
        data_path = self.args['data_path']
        
        # Initialize empty lists for data and labels
        self.train_data, self.train_targets = [], []
        self.test_data, self.test_targets = [], []

        num_classes = 5  # Each domain has 65 classes

        for domain_idx, domain in enumerate(self.domain_names):
            domain_path = os.path.join(data_path, domain)
            label_offset = domain_idx * num_classes  # Offset for the domain's labels
            
            for label, category in enumerate(sorted(os.listdir(domain_path))):
                category_path = os.path.join(domain_path, category)
                images = [os.path.join(category_path, img) for img in os.listdir(category_path) if img.endswith('.jpg') or img.endswith('.png')]

                # Split images into train and test
                train_imgs, test_imgs = train_test_split(images, test_size=0.2, random_state=12)

                # Assign labels with the offset for this domain
                adjusted_label = label + label_offset
                self.train_data.extend(train_imgs)
                self.train_targets.extend([adjusted_label] * len(train_imgs))
                self.test_data.extend(test_imgs)
                self.test_targets.extend([adjusted_label] * len(test_imgs))
        
        # Convert lists to numpy arrays for consistency with other datasets
        self.train_data = np.array(self.train_data)
        self.train_targets = np.array(self.train_targets)
        self.test_data = np.array(self.test_data)
        self.test_targets = np.array(self.test_targets)

class iVLCS_80norm(iData):
    use_path = True
    train_trsf = [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
    ]
    test_trsf = [
        transforms.Resize(256),
        transforms.CenterCrop(224),
    ]
    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]

    class_order = np.arange(20).tolist()

    def __init__(self, args):
        self.args = args
        class_order = np.arange(20).tolist()
        self.class_order = class_order
        self.domain_names = ["Caltech101", "LabelMe", "SUN09", "VOC2007"]

    def download_data(self):
        data_path = self.args['data_path']
        
        # Initialize empty lists for data and labels
        self.train_data, self.train_targets = [], []
        self.test_data, self.test_targets = [], []

        num_classes = 5  # Each domain has 65 classes

        for domain_idx, domain in enumerate(self.domain_names):
            domain_path = os.path.join(data_path, domain)
            label_offset = domain_idx * num_classes  # Offset for the domain's labels
            
            for label, category in enumerate(sorted(os.listdir(domain_path))):
                category_path = os.path.join(domain_path, category)
                images = [os.path.join(category_path, img) for img in os.listdir(category_path) if img.endswith('.jpg') or img.endswith('.png')]
                
                # Split images into train and test
                try:
                    train_imgs, test_imgs = train_test_split(images, test_size=0.2, random_state=12)
                except Exception as e:
                    import ipdb; ipdb.set_trace()

                # Assign labels with the offset for this domain
                adjusted_label = label + label_offset
                self.train_data.extend(train_imgs)
                self.train_targets.extend([adjusted_label] * len(train_imgs))
                self.test_data.extend(test_imgs)
                self.test_targets.extend([adjusted_label] * len(test_imgs))
        
        # Convert lists to numpy arrays for consistency with other datasets
        self.train_data = np.array(self.train_data)
        self.train_targets = np.array(self.train_targets)
        self.test_data = np.array(self.test_data)
        self.test_targets = np.array(self.test_targets)

class iDomain_80norm(iData):
    use_path = True
    train_trsf = [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
    ]
    test_trsf = [
        transforms.Resize(256),
        transforms.CenterCrop(224),
    ]
    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]

    class_order = np.arange(345).tolist()

    def __init__(self, args):
        self.args = args
        class_order = np.arange(345).tolist()
        self.class_order = class_order
        self.domain_names = ["real", "clipart", "painting", "quickdraw", "sketch", "infograph"]

    def download_data(self):
        data_path = self.args['data_path']
        
        # Initialize empty lists for data and labels
        self.train_data, self.train_targets = [], []
        self.test_data, self.test_targets = [], []

        num_classes = 69  # Each domain has 65 classes

        for domain_idx, domain in enumerate(self.domain_names):
            domain_path = os.path.join(data_path, domain)
            label_offset = domain_idx * num_classes  # Offset for the domain's labels
            
            for label, category in enumerate(sorted(os.listdir(domain_path))):
                category_path = os.path.join(domain_path, category)
                images = [os.path.join(category_path, img) for img in os.listdir(category_path) if img.endswith('.jpg') or img.endswith('.png')]
                
                # Split images into train and test
                try:
                    train_imgs, test_imgs = train_test_split(images, test_size=0.2, random_state=12)
                except Exception as e:
                    import ipdb; ipdb.set_trace()

                # Assign labels with the offset for this domain
                adjusted_label = label + label_offset
                self.train_data.extend(train_imgs)
                self.train_targets.extend([adjusted_label] * len(train_imgs))
                self.test_data.extend(test_imgs)
                self.test_targets.extend([adjusted_label] * len(test_imgs))
        
        # Convert lists to numpy arrays for consistency with other datasets
        self.train_data = np.array(self.train_data)
        self.train_targets = np.array(self.train_targets)
        self.test_data = np.array(self.test_data)
        self.test_targets = np.array(self.test_targets)

def jpg_image_to_array(image_path):
    """
    Loads JPEG image into 3D Numpy array of shape 
    (width, height, channels)
    """
    with Image.open(image_path) as image:      
        image = image.convert('RGB')
        im_arr = np.fromstring(image.tobytes(), dtype=np.uint8)
        im_arr = im_arr.reshape((image.size[1], image.size[0], 3))                                   
    return im_arr

