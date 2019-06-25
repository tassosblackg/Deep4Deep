# from .. import read_img as ri
import sys
sys.path.append('../')
import read_img as ri

data,labels,n = ri.read_pred_data('ASVspoof2017_V2_train_eval','eval_info.txt')

print(type(data))
print('\n')
print(data.shape)
print(labels.shape)
