import numpy as np
#from utils import DepthNorm

from zipfile import ZipFile
from tensorflow.keras.utils import Sequence
#from augment import BasicPolicy
import pandas as pd
import os
#import matplotlib.pyplot as plt

def extract_zip(input_zip):
    input_zip=ZipFile(input_zip)
    return {name: pd.read_csv(input_zip.open(name), header=None) for name in input_zip.namelist() if '.csv' in name }

def uwdb_resize(img, resolution=480, padding=6):
    from skimage.transform import resize
    return resize(img, (resolution, int(resolution*4/3)), preserve_range=True, mode='reflect', anti_aliasing=True )

def get_uwdb_data(batch_size, uwdb_data_zipfile=r'shared/data/D5.zip'):
    data = extract_zip(uwdb_data_zipfile)
    # data = []
    uwdb2_train = pd.read_csv(r'./train_set.csv')
    uwdb2_test = pd.read_csv(r'./test_set.csv')

    shape_rgb = (batch_size, 480, 640, 3)
    shape_depth = (batch_size, 240, 320, 1)

    # Helpful for testing...
    if False:
        uwdb2_train = uwdb2_train[:10]
        uwdb2_test = uwdb2_test[:10]

    return data, uwdb2_train, uwdb2_test, shape_rgb, shape_depth

def get_uwdb_train_test_data(batch_size):
    data, UWDB_train, UWDB_test, shape_rgb, shape_depth = get_uwdb_data(batch_size)

    train_generator = UWDB_BasicRGBSequence(data, UWDB_train, batch_size=batch_size, shape_rgb=shape_rgb, shape_depth=shape_depth)
    test_generator = UWDB_BasicRGBSequence(data, UWDB_test, batch_size=batch_size, shape_rgb=shape_rgb, shape_depth=shape_depth)

    return train_generator, test_generator

def DepthNorm(x, maxDepth):
    return maxDepth / x
class UWDB_BasicRGBSequence(Sequence):
    def __init__(self, data, dataset, batch_size,shape_rgb, shape_depth):
        self.data = data
        self.dataset = dataset.sample(frac = 1,random_state=0)
        #self.dataset = dataset
        self.batch_size = batch_size
        self.N = len(self.dataset)
        self.shape_rgb = shape_rgb
        self.shape_depth = shape_depth
        self.maxDepth = 30.0

    def __len__(self):
        return int(np.ceil(self.N / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x, batch_y = np.zeros( self.shape_rgb ), np.zeros( self.shape_depth )

        # Augmentation of RGB images
        for i in range(batch_x.shape[0]):
            index = min((idx * self.batch_size) + i, self.N-1)

            sample = self.dataset.iloc[index]      
            x = np.asarray(self.data[sample['rgb']]).reshape(480,640,3,order='F')/255
            y = np.asarray(self.data[sample['D']]).reshape(480,640,1)
            y = DepthNorm(y, maxDepth=self.maxDepth)

            batch_x[i] = uwdb_resize(x, 480)
            batch_y[i] = uwdb_resize(y, 240)

        return batch_x, batch_y

# class NYU_BasicRGBSequence(Sequence):
#     def __init__(self, data, dataset, batch_size,shape_rgb, shape_depth):
#         self.data = data
#         self.dataset = dataset
#         self.batch_size = batch_size
#         self.N = len(self.dataset)
#         self.shape_rgb = shape_rgb
#         self.shape_depth = shape_depth
#         self.maxDepth = 1000.0

#     def __len__(self):
#         return int(np.ceil(self.N / float(self.batch_size)))

#     def __getitem__(self, idx):
#         batch_x, batch_y = np.zeros( self.shape_rgb ), np.zeros( self.shape_depth )
#         for i in range(self.batch_size):            
#             index = min((idx * self.batch_size) + i, self.N-1)

#             sample = self.dataset[index]

#             x = np.clip(np.asarray(Image.open( BytesIO(self.data[sample[0]]))).reshape(480,640,3)/255,0,1)
#             y = np.asarray(Image.open(BytesIO(self.data[sample[1]])), dtype=np.float32).reshape(480,640,1).copy().astype(float) / 10.0
#             y = DepthNorm(y, maxDepth=self.maxDepth)

#             batch_x[i] = nyu_resize(x, 480)
#             batch_y[i] = nyu_resize(y, 240)

#             # DEBUG:
#             #self.policy.debug_img(batch_x[i], np.clip(DepthNorm(batch_y[i])/maxDepth,0,1), idx, i)
#         #exit()

#         return batch_x, batch_y


if __name__ == '__main__':
    train_generator, test_generator = get_uwdb_train_test_data(4)
    print(train_generator[0])
    print(test_generator[0])
