from KFT.util import core_data_extract
import pandas as pd
import torch
import os
import pickle

if __name__ == '__main__':

    df = pd.read_csv('./ml-20m/ratings.csv')
    df=df.head(1000)
    indices = ['userId','movieId','timestamp']
    signal = ['rating']
    side_info_indices = []
    for el in indices:
        side_info_indices.append(df[el].sort_values())
    X,y,tensor_shape = core_data_extract(df,indices_list=indices,target_name=signal)
    save_dir = './public_movielens_data/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    torch.save((X,y),save_dir+'all_data.pt')
    with open(save_dir + 'full_tensor_shape.pickle', 'wb') as handle:
        pickle.dump(tensor_shape, handle, protocol=pickle.HIGHEST_PROTOCOL)
