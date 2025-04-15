import os
import pandas as pd
from torchvision import transforms, datasets
from collections import Counter
from torch.utils.data.dataset import Dataset
from .preprocessing import *

#labels = pd.read_csv('C:/Users/julie/OneDrive/Bureau/Sarah/Projets_Python/Computer_vision_MED/data/train_labels.csv', sep=',')
#print(labels.head(15))


class Dataset2D(Dataset):
    def __init__(self, data_root:str, mode:str,transforms = None):
        super().__init__()
        self.transform = transforms
        if mode not in ['train', 'val', 'test']:
            raise ValueError('Chose mode between : "train" or "test" ')
        self.mode = mode
        self.dataset = datasets.ImageFolder(str(data_root + '/' + mode), transform = self.transform)
        self.class_id = self.dataset.class_to_idx

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx] # -> img, label

    def get_info(self)->None:
        """ Display dataset information """
        class_count = Counter(self.dataset.targets)
        print(f' Classes identification : {self.class_id}', '\n',
              f' Classes distribution - {self.mode} : ','\n')

        total_len = len(self.dataset)
        for class_idx, count in class_count.items():
            print(f'class : {self.dataset.classes[class_idx]} - : {count/total_len *100:.2f}%')
