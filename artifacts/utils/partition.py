import numpy as np
from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader
from PIL import Image

class Partition(Dataset):
    def __init__(self, data, targets, transform = None, data_type='img'):
        self.data = data
        self.targets = targets
        self.transform = transform
        self.target_transform = None
        self.data_type = data_type

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        if self.data_type == 'img':
            img = Image.fromarray(np.array(img))

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


class FolderPartition(Dataset):
    def __init__(self, data, targets, transform = None):
        self.loader = default_loader
        self.data = data
        # count_img = 0
        # count_path = 0
        # for path in data:
        #     # print(path)
        #     if type(path) == str:
        #         self.data.append(self.loader(path))
        #         count_path += 1
        #     if type(path) == PIL.Image.Image:
        #         count_img += 1
        # if count_path > 0:
        #     print('path: ', count_path)
        # if count_img > 0:
        #     print('img: ', count_img)
        self.targets = targets
        self.transform = transform
        self.target_transform = None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        sample = self.loader(data[index])
        target = self.targets[index]
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

