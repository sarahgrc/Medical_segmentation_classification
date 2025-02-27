import torch
from torchvision import transforms
from models3D import Clf3D
from models2D import *
from dataset.dataset import Dataset2D
#model3D = Clf3D('r2plus1d_18', num_class=2)

data_root = 'C:/Users/julie/OneDrive/Bureau/Sarah/cours 5A/projet M2/propre/data_clf_2d/archive'
train_dataset = Dataset2D(data_root=data_root,
                          mode= 'train')
test_dataset = Dataset2D(data_root=data_root,
                         mode= 'test')

train_dataset.get_info()
test_dataset.get_info()

# Dealing with medical images we have to have a special care with resizing and prefer cropping to certain size rather than resizing
train_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),

    # data augmentation
    transforms.RandomHorizontalFlip(p=0.3),
    transforms.RandomRotation(degrees=(-0.2,+0.2))
])
