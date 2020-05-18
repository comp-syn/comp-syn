# data helper code

# DataCollector

import os
import PIL
from PIL import Image
import numpy as np
from collections import defaultdict
from numba import jit

test_jzazbz_array = np.load('jzazbz_array.npy')

@jit
def rgb_array_to_jzazbz_array(rgb_array):
    jzazbz_array = np.zeros(rgb_array.shape)
    for i in range(rgb_array.shape[0]):
        for j in range(rgb_array.shape[1]):
            jzazbz_array[i][j] = test_jzazbz_array[rgb_array[i][j][0]][rgb_array[i][j][1]][rgb_array[i][j][2]]
    return jzazbz_array

class ImageData():
    def __init__(self, **kwargs):
        self.rgb_dict = defaultdict(lambda : None)
        self.jzazbz_dict = defaultdict(lambda : None)
        self.labels_list = []
        self.dims = None

    def load_image_dict_from_subfolders(self, path, label=None, compress_dims=(300,300)):
        assert os.path.isdir(path)
        compress_dims = self.dims if self.dims else compress_dims
        self.dims = compress_dims
        path = os.path.realpath(path)
        folders = os.listdir(path)
        for folder in folders:
            fp = os.path.join(path, folder)
            print(fp)
            assert os.path.isdir(fp)
            self.load_image_dict_from_folder(fp, label=label, compress_dims=compress_dims)
            self.store_jzazbz_from_rgb(label)
        self.labels_list = list(self.rgb_dict.keys())


    def load_image_dict_from_folder(self, path, label=None, compress_dims=(300,300), compute_jzazbz=True):
        assert os.path.isdir(path)
        compress_dims = self.dims if self.dims else compress_dims
        self.dims = compress_dims
        path = os.path.realpath(path)
        label = label or path.split('/')[-1]
        files = os.listdir(path)
        imglist = []
        arraylist = []
        for file in files:
            fp = os.path.join(path, file)
            img = None
            try:
                img = self.load_rgb_image(fp, compress_dims=compress_dims)
            except ValueError:
                print("error")
                continue
            if img is not None:
                imglist.append(img)
        
        if compute_jzazbz:
            self.store_jzazbz_from_rgb(label)
        self.rgb_dict[label] = imglist
        self.labels_list = list(self.rgb_dict.keys())
        
    def load_image_continuum_from_folder(self, path, continuum_files, idx=0, window=100, label=None, compress_dims=(300,300), compute_jzazbz=True):
        assert os.path.isdir(path)
        compress_dims = self.dims if self.dims else compress_dims
        self.dims = compress_dims
        path = os.path.realpath(path)
        label = label or path.split('/')[-1]
        #files = os.listdir(path)
        imglist = []
        arraylist = []
        
        files_in_window = continuum_files[idx:idx + window]
        
        for file in files_in_window:
            fp = os.path.join(path, file)
            img = None
            try:
                img = self.load_rgb_image(fp, compress_dims=compress_dims)
            except ValueError:
                print("error")
                continue
            if img is not None:
                imglist.append(img)
        
        if compute_jzazbz:
            self.store_jzazbz_from_rgb(label)
        self.rgb_dict[label] = imglist
        self.labels_list = list(self.rgb_dict.keys())


    def load_rgb_image(self, path, compress_dims=None):
        fmts = ['.jpg', '.jpeg', '.png', '.bmp']
        if os.path.isfile(path) and any([fmt in path.lower() for fmt in fmts]):
            try:
                img_raw = PIL.Image.open(path)
                if compress_dims:
                    assert len(compress_dims)==2
                    img_raw = img_raw.resize((compress_dims[0],compress_dims[1]),
                                                PIL.Image.ANTIALIAS)
                img_array = np.array(img_raw)[:,:,:3]

                assert len(img_array.shape)==3 and img_array.shape[-1]==3
                return img_array
            except:
                return None
                pass


    def store_jzazbz_from_rgb(self, labels=None):
        if labels:
            labels = labels if isinstance(labels, list) else [labels]
        else:
            labels = list(self.rgb_dict.keys())
        for label in labels:
            if label and label in self.rgb_dict.keys():
                try:
                    self.jzazbz_dict[label] = [rgb_array_to_jzazbz_array(rgb) for rgb in self.rgb_dict[label]]
                except:
                    
                    pass
                    

    def print_labels(self):
        self.labels_list = list(self.rgb_dict.keys())
        print(self.labels_list)

