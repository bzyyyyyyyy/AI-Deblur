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
            # transforms.Resize(128),
            transforms.GaussianBlur(21),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
        ])
        self.out_trans = transforms.Compose([
            transforms.ToTensor()
        ])

    def get_train_dataloader(self):
        train_dataset = MyData(self.dirpath, self.data_train, self.in_trans, self.out_trans)
        for i in range(7):
            train_dataset += MyData(self.dirpath, self.data_train, self.in_trans, self.out_trans, trans=transforms.RandomCrop(128))
        return DataLoader(train_dataset, batch_size=16, num_workers=2)

    def get_test_dataloader(self):
        test_dataset = MyData(self.dirpath, self.data_test, self.in_trans, self.out_trans)
        return DataLoader(test_dataset, batch_size=16, num_workers=2)

    def __data_paths(self) -> list:
        rst = []
        files = os.listdir(self.dirpath)
        for file in files:
            try:
                img = Image.open(os.path.join(self.dirpath, file))
            except:
                pass
            else:
                if img.width >= 128 and img.height >= 128:
                    rst.append(file)
        return rst


class MyData(Dataset):

    def __init__(self, dirpath, datas: list, in_trans, out_trans, trans=transforms.CenterCrop(128)):
        self.dirpath = dirpath
        self.datas = datas
        self.in_trans = in_trans
        self.out_trans = out_trans
        self.trans = trans

    def __getitem__(self, item):
        img_path = os.path.join(self.dirpath, self.datas[item])
        img = Image.open(img_path).convert('RGB')
        img = self.trans(img)
        imgs = self.in_trans(img)
        targets = self.out_trans(img)
        return [imgs, targets]

    def __len__(self):
        return len(self.datas)


if __name__ == '__main__':
    pass
