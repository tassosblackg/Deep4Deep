import read_img as ri


# labels= ri.read_label('train_info.txt')
# print(labels)

print('\n')

sub_labels = ri.read_subset_labels('train_info.txt',2600)
print(sub_labels)
print('\n')
ohe = ri.one_hot_encode(sub_labels)
print(ohe)
