import os
import torch.utils.data as data
from torch.utils.data import Dataset
from PIL import Image, ImageOps
import numpy as np
import torch
from torch.utils.data.dataset import T_co
from torchvision import transforms

def read_image(path):
    img = Image.open(path)
    return np.array(img)

class resize:
    """resize the pic, and remain the ratio,use 0 padding """

    def __init__(self, reshape_size=224):
        self.reshape_size = reshape_size
        pass

    def __call__(self, img):
        w, h = img.size
        long = max(w, h)
        w, h = int(w / long * self.reshape_size), int(h / long * self.reshape_size)
        img = img.resize((w, h))
        delta_w, delta_h = self.reshape_size - w, self.reshape_size - h
        padding = (delta_w // 2, delta_h // 2, delta_w - (delta_w // 2), delta_h - (delta_h // 2))
        img = ImageOps.expand(img, padding)
        return img


class BAATrainDataset(Dataset):
    def __init__(self, df, file_path, boneage_mean, boneage_div, transforms):
        def preprocess_df(df):
            # nomalize boneage distribution
            df['zscore'] = df['boneage'].map(lambda x: (x - boneage_mean) / boneage_div)
            # change the type of gender, change bool variable to float32
            df['male'] = df['male'].astype('float32')
            df['bonage'] = df['boneage'].astype('float32')
            return df

        self.mean = boneage_mean
        self.div = boneage_div
        self.df = preprocess_df(df)
        self.file_path = file_path
        self.transforms = transforms

    def __getitem__(self, index):
        row = self.df.iloc[index]
        num = int(row['id'])
        return (self.transforms(image=read_image(f"{self.file_path}/{num}.png"))['image'],
                torch.Tensor([row['male']])), row['zscore']

    def __len__(self):
        return len(self.df)


class BAAValDataset(Dataset):
    def __init__(self, df, file_path, transforms):
        def preprocess_df(df):
            # change the type of gender, change bool variable to float32
            df['male'] = df['male'].astype('float32')
            df['bonage'] = df['boneage'].astype('float32')
            return df

        self.df = preprocess_df(df)
        self.file_path = file_path
        self.transforms = transforms

    def __getitem__(self, index):
        row = self.df.iloc[index]
        return (self.transforms(image=read_image(f"{self.file_path}/{int(row['id'])}.png"))['image'],
                torch.Tensor([row['male']])), row['boneage']

    def __len__(self):
        return len(self.df)