from KFT.util import core_data_extract,generate_timestamp_side_info
import pandas as pd
import torch
import os
import pickle
from sklearn.preprocessing import StandardScaler

if __name__ == '__main__':
    scaler_movie = StandardScaler()

    df = pd.read_csv('./ml-20m/ratings.csv')
    df['timestamp'] = pd.to_datetime(df['timestamp'].values,infer_datetime_format=True,unit='s').hour
    movie_side_info = pd.read_csv('./ml-20m/genome-scores.csv')
    df = df[df['movieId'].isin(movie_side_info['movieId'].unique())]
    movie_side_info = movie_side_info[movie_side_info['movieId'].isin(df['movieId'].unique())]
    movie_side_info = movie_side_info.pivot(index='movieId', columns='tagId', values='relevance').sort_index()
    movie_side_info =  torch.from_numpy(scaler_movie.fit_transform(movie_side_info.fillna(0).to_numpy())).float()
    print(movie_side_info.shape)
    indices = ['userId','movieId','timestamp']
    signal = ['rating']
    side_info_indices = []
    for el in indices:
        side_info_indices.append(df[el].sort_values().values)
    time_side_info = generate_timestamp_side_info(side_info_indices[-1])
    X,y,tensor_shape = core_data_extract(df,indices_list=indices,target_name=signal)
    print(tensor_shape)
    save_dir = './public_movielens_data/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    torch.save((X,y),save_dir+'all_data.pt')
    torch.save((movie_side_info,time_side_info),save_dir+'side_info.pt')

    with open(save_dir + 'full_tensor_shape.pickle', 'wb') as handle:
        pickle.dump(tensor_shape, handle, protocol=pickle.HIGHEST_PROTOCOL)

