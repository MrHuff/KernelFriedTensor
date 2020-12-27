import torch
import shutil
import os
import pickle

old_datasets = ['../public_data/' ,'../public_movielens_data/' ,'../tensor_data/']
new_datasets = ['../public_data_t_fixed/' ,'../public_movielens_data_t_fixed/' ,'../tensor_data_t_fixed/']
t_dim = [2,2,2]
side_info_pos = [2,1,2]

def time_fix(old_ds,new_ds_name,t_dim,pos):
    if not os.path.exists(new_ds_name):
        shutil.copytree(old_ds,new_ds_name)
    else:
        shutil.rmtree(new_ds_name)
        shutil.copytree(old_ds, new_ds_name)
    old_side_info = list(torch.load(old_ds+'side_info.pt'))
    original_shape = list(pickle.load(open(old_ds + 'full_tensor_shape.pickle', 'rb')))
    print(original_shape)
    print(old_side_info)
    old_side_info[pos] = torch.arange(0,original_shape[t_dim]).float().unsqueeze(-1)
    torch.save(old_side_info,new_ds_name+'side_info.pt')
    test = torch.load(new_ds_name+'side_info.pt')
    print(test[pos])

if __name__ == '__main__':
    for i,el in enumerate(old_datasets):
        time_fix(el,new_datasets[i],t_dim[i],side_info_pos[i])






