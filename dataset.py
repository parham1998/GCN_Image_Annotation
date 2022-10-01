# =============================================================================
# Import required libraries
# =============================================================================
import os
import json
import numpy as np
from PIL import Image

import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from sklearn.utils import shuffle


# =============================================================================
# Create annotation dataset
# =============================================================================
class AnnotationDataset(torch.utils.data.Dataset):
    def __init__(self, root, annotation_path, aug_path=None, transforms=None):
        self.root = root
        self.transforms = transforms
        self.aug_path = aug_path
        #
        with open(annotation_path) as fp:
            json_data = json.load(fp)
        samples = json_data['samples']
        if aug_path is not None:
            with open(aug_path) as fp:
                json_aug_data = json.load(fp)
                samples = json_data['samples'] + json_aug_data['samples']
        samples = shuffle(samples, random_state=0)
        self.classes = json_data['labels']
        #
        self.imgs = []
        self.annotations = []
        for sample in samples:
            self.imgs.append(sample['image_name'])
            self.annotations.append(sample['image_labels'])
        # converting all labels of each image into a binary array
        # of the class length
        for idx in range(len(self.annotations)):
            item = self.annotations[idx]
            vector = [cls in item for cls in self.classes]
            self.annotations[idx] = np.array(vector, dtype=float)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, self.imgs[idx])
        image = Image.open(img_path).convert("RGB")
        annotations = torch.tensor(self.annotations[idx])
        if self.transforms is not None:
            image = self.transforms(image)
        return image, annotations

    def __len__(self):
        return len(self.imgs)


# =============================================================================
# Make data loader
# =============================================================================
def get_mean_std(args):
    if not args.augmentation:  # 4500 images
        mean = [0.3928, 0.4079, 0.3531]
        std = [0.2559, 0.2436, 0.2544]
    else:  # 18000 images
        mean = [0.4249, 0.4365, 0.3926]
        std = [0.1955, 0.1830, 0.1987]
    return mean, std


def get_transforms(args):
    mean, std = get_mean_std(args)
    #
    transform_train = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.RandomVerticalFlip(),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=mean,
            std=std,
        )
    ])
    transform_validation = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=mean,
            std=std,
        )
    ])
    return transform_train, transform_validation


def make_data_loader(args):
    root_dir = args.data_root_dir

    transform_train, transform_validation = get_transforms(args)
    #
    if not args.augmentation:
        train_set = AnnotationDataset(root=os.path.join(root_dir, 'images'),
                                      annotation_path=os.path.join(
                                      root_dir, 'train.json'),
                                      transforms=transform_train)
    else:
        train_set = AnnotationDataset(root=os.path.join(root_dir, 'images'),
                                      annotation_path=os.path.join(
                                      root_dir, 'train.json'),
                                      aug_path=os.path.join(
                                          root_dir, 'train_aug.json'),
                                      transforms=transform_train)
    train_loader = DataLoader(train_set,
                              batch_size=args.batch_size,
                              num_workers=args.num_workers,
                              shuffle=True)
    #
    validation_set = AnnotationDataset(root=os.path.join(root_dir, 'images'),
                                       annotation_path=os.path.join(
                                           root_dir, 'test.json'),
                                       transforms=transform_validation)
    validation_loader = DataLoader(validation_set,
                                   batch_size=args.batch_size,
                                   num_workers=args.num_workers,
                                   shuffle=False)
    #
    classes = validation_set.classes
    return train_loader, validation_loader, train_set.annotations, classes


# =============================================================================
# Word embedding
# =============================================================================
def word_embedding(glove_path, classes):
    with open(glove_path, 'r', encoding='UTF-8') as f:
        words = set()
        word_to_vec_map = {}
        for line in f:
            line = line.strip().split()
            curr_word = line[0]
            words.add(curr_word)
            word_to_vec_map[curr_word] = np.array(line[1:], dtype=np.float32)
    #
    emb = []
    for word in classes:
        emb.append(word_to_vec_map[word])
    return torch.from_numpy(np.array(emb))


# =============================================================================
# Define adjacency matrix
# =============================================================================
def adjacency_matrix(annotations, num_classes, th=0.1, p=0.2):
    adj = np.zeros((num_classes, num_classes))
    anno = np.array(annotations)
    sum_anno = np.sum(anno, axis=0)
    for i in range(0, num_classes):
        N = sum_anno[i]
        for j in range(0, num_classes):
            if i != j:
                M = np.sum(anno[:, i] * anno[:, j])
                adj[i, j] = M/N
    # binary
    adj[adj < th] = 0
    adj[adj >= th] = 1
    #
    adj = adj * p / (adj.sum(0, keepdims=True) + 1e-07)
    adj = adj + (1-p) * np.identity(num_classes, np.int32)
    return torch.Tensor(adj)
