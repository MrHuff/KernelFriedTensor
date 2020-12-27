import torch
import numpy as np
from scipy.sparse import coo_matrix
import os
import pickle
import shutil
if __name__ == '__main__':
    save_dir = "../traffic_data/"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    else:
        shutil.rmtree(save_dir)
        os.makedirs(save_dir)
    data = np.load('../KFT_revision_benchmark_data/traffic.npy')
    coo_mat = coo_matrix(data)
    X = np.stack([coo_mat.row,coo_mat.col]).transpose()
    y = coo_mat.data[:,np.newaxis]

    tmp_dat = torch.from_numpy(data)

    tensor_shape = tmp_dat.shape
    print(tensor_shape)
    X = torch.from_numpy(X).int()
    y = torch.from_numpy(y).float()
    time_side_info = torch.unique(X[:,0]).float().unsqueeze(-1)
    torch.save((X,y),save_dir+'all_data.pt')
    torch.save([time_side_info], save_dir + 'side_info.pt')

    with open(save_dir + 'full_tensor_shape.pickle', 'wb') as handle:
        pickle.dump(tensor_shape, handle, protocol=pickle.HIGHEST_PROTOCOL)








