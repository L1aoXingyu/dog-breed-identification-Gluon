import mxnet as mx
import numpy as np
from mxnet import gluon as gl
from mxnet import nd
import json
import os


class DogDataSet(gl.data.Dataset):
    def __init__(self, label_file, img_path, transform):
        self._img_path = img_path
        self._transform = transform
        with open(label_file, 'r') as f:
            self.label_list = json.load(f)

    def __getitem__(self, idx):
        img_name = self.label_list[idx][0] + '.jpg'
        label = np.float32(self.label_list[idx][1])
        img = mx.image.imread(os.path.join(self._img_path, img_name))
        img = self._transform(img)
        return img, label

    def __len__(self):
        return len(self.label_list)


class TestDataSet(gl.data.Dataset):
    def __init__(self, img_path, transform):
        self._img_path = img_path
        self._img_list = os.listdir(img_path)
        self._transform = transform

    def __getitem__(self, idx):
        im_name = self._img_list[idx]
        im = mx.image.imread(os.path.join(self._img_path, im_name))
        im0, im1, im2, im3, im4, im5, im6, im7, im8, im9 = self._transform(im)
        return im_name, im0, im1, im2, im3, im4, im5, im6, im7, im8, im9

    def __len__(self):
        return len(self._img_list)