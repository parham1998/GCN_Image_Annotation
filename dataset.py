# =============================================================================
# Import required libraries
# =============================================================================
import os
import json
import numpy as np
from PIL import Image

import torch
import torchvision.transforms as transforms
from sklearn.utils import shuffle


# =============================================================================
# Pytorch dataset format
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
# Corel-5k settings
# =============================================================================
def corel_5k(input_size=(448, 448)):
    batch_size = 32
    worker = 2
    input_size = input_size
    root = './Corel-5k/'

    # 4500 images 1
    mean = [0.3928, 0.4079, 0.3531]
    std = [0.2559, 0.2436, 0.2544]

    # 18000 images
    # mean = [0.4249, 0.4365, 0.3926]
    # std = [0.1955, 0.1830, 0.1987]

    transform_train = transforms.Compose([
        transforms.Resize(input_size),
        transforms.RandomVerticalFlip(),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=mean,
            std=std,
        ),
    ])
    transform_validation = transforms.Compose([
        transforms.Resize(input_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=mean,
            std=std,
        ),
    ])
    return (batch_size, worker, input_size, root, mean, std,
            transform_train, transform_validation)


# =============================================================================
# Define mean & std
# =============================================================================
'''def get_mean_and_std(trainloader):
    mean, std, n_b = 0, 0, 0
    for images, _ in trainloader:
        mean += images.mean([0, 2, 3])
        std += torch.mean(images**2, [0, 2, 3])
        n_b += 1
    mean = mean / n_b
    std = (std / n_b - mean**2)**0.5
    return mean, std
root = './Corel-5k/'
trainset = AnnotationDataset(
    root=os.path.join(root, 'images'),
    annotation_path=os.path.join(root, 'train.json'),
    aug_path=os.path.join(root, 'train_aug.json'),
    transforms=transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=32,
                                          shuffle=False)
mean, std = get_mean_and_std(trainloader)'''
