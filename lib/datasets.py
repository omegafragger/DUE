from torch.utils import data
from torchvision import datasets, transforms


import os

import torch
from torch.utils.data import Dataset

from torchvision import datasets, transforms

from scipy.io import loadmat
from PIL import Image


import os
import torch
import numpy as numpy

from torchvision import datasets
from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from torch.utils.data.sampler import SubsetRandomSampler

import os
import glob
from torch.utils.data import Dataset
from PIL import Image

import torch
import os
import numpy as np
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from torch.utils.data import Dataset


corruption_types = ['brightness',
                    'defocus_blur',
                    'fog',
                    'gaussian_blur',
                    'glass_blur',
                    'jpeg_compression',
                    'motion_blur',
                    'saturate',
                    'snow',
                    'speckle_noise',
                    'contrast',
                    'elastic_transform',
                    'frost',
                    'gaussian_noise',
                    'impulse_noise',
                    'pixelate',
                    'shot_noise',
                    'spatter',
                    'zoom_blur']


EXTENSION = 'JPEG'
NUM_IMAGES_PER_CLASS = 500
CLASS_LIST_FILE = 'wnids.txt'
VAL_ANNOTATION_FILE = 'val_annotations.txt'


def get_SVHN(root):
    input_size = 32
    num_classes = 10

    # NOTE: these are not correct mean and std for SVHN, but are commonly used
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    train_dataset = datasets.SVHN(
        root + "/SVHN", split="train", transform=transform, download=True
    )
    test_dataset = datasets.SVHN(
        root + "/SVHN", split="test", transform=transform, download=True
    )
    return input_size, num_classes, train_dataset, test_dataset


def get_CIFAR10(root):
    input_size = 32
    num_classes = 10

    normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

    # Alternative
    # normalize = transforms.Normalize(
    #     (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
    # )

    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )

    train_dataset = datasets.CIFAR10(
        root + "/CIFAR10", train=True, transform=train_transform, download=True
    )

    test_transform = transforms.Compose([transforms.ToTensor(), normalize])
    test_dataset = datasets.CIFAR10(
        root + "/CIFAR10", train=False, transform=test_transform, download=False
    )

    return input_size, num_classes, train_dataset, test_dataset


def get_CIFAR100(root):
    input_size = 32
    num_classes = 100
    normalize = transforms.Normalize((0.5071, 0.4866, 0.4409), (0.2673, 0.2564, 0.2762))

    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )
    train_dataset = datasets.CIFAR100(
        root + "/CIFAR100", train=True, transform=train_transform, download=True
    )

    test_transform = transforms.Compose([transforms.ToTensor(), normalize])
    test_dataset = datasets.CIFAR100(
        root + "/CIFAR100", train=False, transform=test_transform, download=False
    )

    return input_size, num_classes, train_dataset, test_dataset


def get_TinyImageNet(root='./'):
    input_size = 32
    num_classes = 200

    normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])

    val_test_transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            normalize
    ])

    train_transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.RandomCrop(64, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
    ])

    # load the dataset
    data_dir = root

    test_dataset = TinyImageNet(data_dir,
                               split='val',
                               transform=val_test_transform,
                               in_memory=True)
    return input_size, num_classes, None, test_dataset


def get_CIFAR10_C(root='./', corruption_type='brightness', corruption_intensity=1):
    input_size = 32
    num_classes = 10

    normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010],)

    # define transform
    transform = transforms.Compose([transforms.ToTensor(), normalize,])

    test_dataset = CIFAR10_C(corruption_type=corruption_type, intensity=corruption_intensity, path=root, transform=transform,)

    return input_size, num_classes, None, test_dataset



all_datasets = {
    "SVHN": get_SVHN,
    "CIFAR10": get_CIFAR10,
    "CIFAR100": get_CIFAR100,
    "TinyImageNet": get_TinyImageNet,
    "CIFAR10_C": get_CIFAR10_C,
}


def get_dataset(dataset, root="./", **kwargs):
    return all_datasets[dataset](root, **kwargs)


def get_dataloaders(dataset, train_batch_size=128, root="./"):
    ds = all_datasets[dataset](root)
    input_size, num_classes, train_dataset, test_dataset = ds

    kwargs = {"num_workers": 4, "pin_memory": True}

    train_loader = data.DataLoader(
        train_dataset, batch_size=train_batch_size, shuffle=True, **kwargs
    )

    test_loader = data.DataLoader(
        test_dataset, batch_size=1000, shuffle=False, **kwargs
    )

    return train_loader, test_loader, input_size, num_classes



class TinyImageNet(Dataset):
    """Tiny ImageNet data set available from `http://cs231n.stanford.edu/tiny-imagenet-200.zip`.
    Parameters
    ----------
    root: string
        Root directory including `train`, `test` and `val` subdirectories.
    split: string
        Indicating which split to return as a data set.
        Valid option: [`train`, `test`, `val`]
    transform: torchvision.transforms
        A (series) of valid transformation(s).
    in_memory: bool
        Set to True if there is enough memory (about 5G) and want to minimize disk IO overhead.
    """
    def __init__(self, root, split='train', transform=None, target_transform=None, in_memory=False):
        self.root = os.path.expanduser(root)
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        self.in_memory = in_memory
        self.split_dir = os.path.join(root, self.split)
        self.image_paths = sorted(glob.iglob(os.path.join(self.split_dir, '**', '*.%s' % EXTENSION), recursive=True))
        self.labels = {}  # fname - label number mapping
        self.images = []  # used for in-memory processing

        # build class label - number mapping
        with open(os.path.join(self.root, CLASS_LIST_FILE), 'r') as fp:
            self.label_texts = sorted([text.strip() for text in fp.readlines()])
        self.label_text_to_number = {text: i for i, text in enumerate(self.label_texts)}

        if self.split == 'train':
            for label_text, i in self.label_text_to_number.items():
                for cnt in range(NUM_IMAGES_PER_CLASS):
                    self.labels['%s_%d.%s' % (label_text, cnt, EXTENSION)] = i
        elif self.split == 'val':
            with open(os.path.join(self.split_dir, VAL_ANNOTATION_FILE), 'r') as fp:
                for line in fp.readlines():
                    terms = line.split('\t')
                    file_name, label_text = terms[0], terms[1]
                    self.labels[file_name] = self.label_text_to_number[label_text]

        # read all images into torch tensor in memory to minimize disk IO overhead
        if self.in_memory:
            self.images = [self.read_image(path) for path in self.image_paths]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        file_path = self.image_paths[index]

        if self.in_memory:
            img = self.images[index]
        else:
            img = self.read_image(file_path)

        if self.split == 'test':
            return img
        else:
            # file_name = file_path.split('/')[-1]
            return img, self.labels[os.path.basename(file_path)]

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        tmp = self.split
        fmt_str += '    Split: {}\n'.format(tmp)
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str

    def read_image(self, path):
        img = Image.open(path)
        if (img.mode == 'L'):
            img = img.convert('RGB')
        return self.transform(img) if self.transform else img


class CIFAR10_C(Dataset):
    '''
    Defining a pytorch dataset for CIFAR-10-C images.
    '''
    def __init__(self, corruption_type, intensity, path='./', transform=None):
        '''
        corruption_type: indicates the type of corruption to construct the dataset on
        intensity: intensity of the corruption type for the dataset
        path: path for the dataset
        transform: transform to be applied on the samples
        '''
        
        # Read the dataset file
        data = torch.from_numpy(np.load(os.path.join(path, corruption_type + '.npy')))#.permute(0,3,1,2)
        label = torch.from_numpy(np.load(os.path.join(path, 'labels.npy')))

        # Morph the data to change based on the intensity
        self.data = data[((intensity-1)*10000) : (intensity*10000)].repeat_interleave(5, dim=0)
        self.label = label[((intensity-1)*10000) : (intensity*10000)].repeat_interleave(5, dim=0)
        self.transform = transform
        
    
    def __len__(self):
        return self.data.shape[0]
    
    
    def __getitem__(self, idx):
        sample = self.data[idx].numpy()
        label = self.label[idx].numpy()

        if self.transform:
            sample = self.transform(sample)
        
        return sample, label