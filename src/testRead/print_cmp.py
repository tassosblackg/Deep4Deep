import model_func as mf
import numpy as np
import matplotlib.pyplot as plt

def read_cmp_file(cmp_filename):
    nfilt = 64
    with open(cmp_filename, 'rb') as fid:
        cmp_data1 = np.fromfile(fid, dtype=np.float32, count=-1)

    cmp_data = cmp_data1.reshape((-1, nfilt))  # where nfilt = 64
    # @ check if read is done right
    fbank = 63  # put a number between 0 and 63
    plt.plot(cmp_data[:, fbank])
    plt.show()
    print(cmp_data.shape)
    return (cmp_data)



read_cmp_file('/home/tassos/Desktop/DATA_ASR/ASVspoof2017_V2_train_eval/E_1000030.cmp')
