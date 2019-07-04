import sys
sys.path.append('../')
import numpy as np
import read_img as ri


# labels= ri.read_label('train_info.txt')
# print(labels)

print('\n')

sub_labels = ri.read_subset_labels('train_info.txt',2600)
print(sub_labels)
print('\n')
ohe = ri.one_hot_encode(sub_labels)
print(ohe)
a = np.empty(0)
a=[[0. ,1.]]
# b= np.empty(0)
# b.append([1., 0.])
print('Tuple\n')
print(ohe[2,:])

if (np.array_equal(ohe[0,:],[0.,1.])):
    print('1')
else:
    print('0')
