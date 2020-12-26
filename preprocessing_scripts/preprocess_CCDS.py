import torch
import numpy as np
from scipy.sparse import coo_matrix
import os
import pickle
import shutil
from numpy import genfromtxt

if __name__ == '__main__':
    save_dir = "../CCDS_data/"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    else:
        shutil.rmtree(save_dir)
        os.makedirs(save_dir)
    X  = torch.from_numpy(genfromtxt('../KFT_revision_benchmark_data/CCDS/X.csv', delimiter=',')-1).int()
    y  = torch.from_numpy(genfromtxt('../KFT_revision_benchmark_data/CCDS/Y.csv', delimiter=',')).float()
    loc_side_info = torch.from_numpy(genfromtxt('../KFT_revision_benchmark_data/CCDS/side_info.csv', delimiter=',')).float()
    time_side_info = torch.unique(X[:,-1]).int()
    print(time_side_info)
    print(loc_side_info)
    tensor_shape = torch.Size([17,125,156])
    torch.save((X,y),save_dir+'all_data.pt')
    torch.save((loc_side_info,time_side_info), save_dir + 'side_info.pt')
    with open(save_dir + 'full_tensor_shape.pickle', 'wb') as handle:
        pickle.dump(tensor_shape, handle, protocol=pickle.HIGHEST_PROTOCOL)
