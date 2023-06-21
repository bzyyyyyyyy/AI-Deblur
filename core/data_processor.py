from typing import List
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import os
import torch


class DataLoad:

    def __init__(self, dirpath, split: float):
        self.dirpath = dirpath
        self.data_paths = self.__data_paths()
        i_split = int(len(self.data_paths) * split)
        self.data_train = self.data_paths[:i_split]
        self.data_test = self.data_paths[i_split:]
        self.in_trans = transforms.Compose([
            transforms.TenCrop(256),
            transforms.Resize(128),
            transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
            transforms.ToTensor()
        ])
        self.out_trans = transforms.Compose([
            transforms.CenterCrop(256),
            transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
            transforms.ToTensor()
        ])

    def get_train_dataloader(self):
        train_dataset = MyData(self.dirpath, self.data_train, self.in_trans, self.out_trans)
        return DataLoader(train_dataset, batch_size=10, num_workers=2)

    def get_test_dataloader(self):
        test_dataset = MyData(self.dirpath, self.data_test, self.in_trans, self.out_trans)
        return DataLoader(test_dataset, batch_size=10, num_workers=2)

    def __data_paths(self) -> list:
        rst = []
        files = os.listdir(self.dirpath)
        for file in files:
            try:
                img = Image.open(os.path.join(self.dirpath, file))
            except:
                pass
            else:
                if img.width >= 256 and img.height >= 256:
                    rst.append(file)
        return rst


class MyData(Dataset):

    def __init__(self, dirpath, datas: list, in_trans, out_trans):
        self.dirpath = dirpath
        self.datas = datas
        self.in_trans = in_trans
        self.out_trans = out_trans

    def __getitem__(self, item):
        img_path = os.path.join(self.dirpath, self.datas[item])
        img = Image.open(img_path).convert('RGB')
        imgs = self.in_trans(img)
        targets = self.out_trans(img)
        return [imgs, targets]

    def __len__(self):
        return len(self.datas)


if __name__ == '__main__':
    pass
