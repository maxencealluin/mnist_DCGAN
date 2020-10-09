import struct
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

def show_img(image):
    plt.imshow((image + 1) / 2)
    plt.show()

def plot_grid(gen, sample, RGB=False):
    out = gen(sample).squeeze().detach().cpu().numpy()
    print(out.shape)
    if RGB:
        out = np.moveaxis(out, 1, 3)
    for i in range(16):
        plt.subplot(4,4,i+1)
        plt.tick_params(
        axis='both',  
        which='both', 
        bottom=False,  
        labelleft=False,
        left=False, 
        labelbottom=False) 
        if RGB:
            plt.imshow((out[i, :, :, :] + 1) / 2, vmin=0, vmax=1)
        else:
            plt.imshow(out[i, :, :], cmap='gray')

def read_images(file):
    with open(file,'rb') as f:
        magic, size = struct.unpack(">II", f.read(8))
        nrows, ncols = struct.unpack(">II", f.read(8))
        data = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))
        data = data.reshape((size, nrows, ncols))
    return data

def read_labels(file):
    with open(file, 'rb') as f:
        bytes = f.read(8)
        magic, size = struct.unpack(">II", bytes)
        # nrows, ncols = struct.unpack('>II', f.read(8))
        data_1 = np.fromfile(f,  dtype=np.dtype(np.uint8)).newbyteorder(">")    
    return np.array(data_1)

def rescale_data(data, resize = False):
    data = data.astype(np.int32)
    data = (data - 128) / 128
    if resize:
        data = data[:,:,2:-2, 2:-2]
    return data

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def load_mnist():
    x = read_images('data/train-images-idx3-ubyte')
    y = read_labels('data/train-labels-idx1-ubyte')
    return x, y