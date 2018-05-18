# script for reading each .cmp file from directories
# and then create images

import glob as g
import numpy as np
import matplotlib.pyplot as plt

#read files' name of directory and create an array of them
def read_dir(dir_name):
    files=g.glob(dir_name+"/*.cmp")
    cmp_list=[]
    for i in range(files.__len__()):
        cmp_list.append(read_cmp_file(files[i]))
    return cmp_list


#read .cmp files and converted to numpy array (transposed)
def read_cmp_file(cmp_filename):
    nfilt=64
    with open(cmp_filename, 'rb') as fid:
        cmp_data = np.fromfile(fid, dtype=np.float32, count=-1)

    cmp_data = cmp_data.reshape((-1, nfilt))  # where nfilt = 64
    # check if read is done right
    fbank = 10  # put a number between 0 and 63
    plt.plot(cmp_data[:, fbank])
    plt.show()
    return (cmp_data)




# ------------------------------------------------------------------------------
# convert .cmp file to image
#what is params ?? cmp_data

def convert_to_images(self, params):
    context_width = 17
    context_width = context_width
    n_frames, param_dim = params.shape
    border = context_width // 2

    params_with_borders = np.zeros(
        (2 * border + n_frames, param_dim), dtype=np.float32)
    params_with_borders[border:-border, :] = params
    params_with_borders[:border, :] = params[0, :]  # broadcast
    params_with_borders[-border:, :] = params[-1, :]  # broadcast

    params_with_borders_transposed = np.transpose(params_with_borders)

    params_as_images = np.zeros(
        (n_frames, param_dim, context_width, 1), dtype=np.float32)

    for i in range(n_frames):
        params_as_images[i, :, :,
                         0] = params_with_borders_transposed[:, i:i + context_width]

    return params_as_images
