from KFT.util import core_data_extract,generate_timestamp_side_info
import pandas as pd
import torch
import os
import pickle
from sklearn.preprocessing import StandardScaler
def index_join(df,join,id):
    df = df[df[id].isin(join[id].unique())]
    return df
if __name__ == '__main__':
    dataset = 'ml-1m'
    scaler_movie = StandardScaler()
    if dataset=='ml-20m':
        df = pd.read_csv(f'./{dataset}/ratings.csv')
    else:
        df = pd.read_csv(f'./{dataset}/ratings.csv', sep="::", names=['userId', 'movieId', 'rating', 'timestamp'])
    if dataset=='ml-20m':
        df['timestamp'] = pd.to_datetime(df['timestamp'].values,infer_datetime_format=True,unit='s').hour
    else:
        df['timestamp'] = pd.to_datetime(df['timestamp'].values,infer_datetime_format=True,unit='s').year*12+pd.to_datetime(df['timestamp'].values,infer_datetime_format=True,unit='s').month
    movie_side_info = pd.read_csv(f'./{dataset}/genome-scores.csv')
    df = df[df['movieId'].isin(movie_side_info['movieId'].unique())]
    movie_side_info = movie_side_info[movie_side_info['movieId'].isin(df['movieId'].unique())]
    movie_side_info = movie_side_info.pivot(index='movieId', columns='tagId', values='relevance').sort_index()
    movie_side_info =  torch.from_numpy(scaler_movie.fit_transform(movie_side_info.fillna(0).to_numpy())).float()
    if dataset == 'ml-1m':
        # oh = OneHotEncoder()
        user_side_info = pd.read_csv(f'./{dataset}/users.csv', sep="::",
                                     names=['userId', 'Gender', 'Age', 'Occupation', 'Zip-code'])
        df = index_join(df, user_side_info, 'userId')
        user_side_info = index_join(user_side_info, df, 'userId')
        user_side_info = user_side_info.set_index('userId').sort_index()
        user_side_info[['Age', 'Occupation']] = user_side_info[['Age', 'Occupation']].astype('category')
        user_side_info = pd.get_dummies(user_side_info[['Gender', 'Age', 'Occupation', 'Zip-code']])
        user_side_info = torch.from_numpy(scaler_movie.fit_transform(user_side_info.fillna(0).to_numpy())).float()

    indices = ['userId','movieId','timestamp']
    signal = ['rating']
    side_info_indices = []
    for el in indices:
        side_info_indices.append(df[el].sort_values().values)
    time_side_info = generate_timestamp_side_info(side_info_indices[-1])
    X,y,tensor_shape = core_data_extract(df,indices_list=indices,target_name=signal)
    print(tensor_shape)
    save_dir = f'./public_movielens_data_{dataset}/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    torch.save((X,y),save_dir+'all_data.pt')

    if dataset == 'ml-1m':
        torch.save((user_side_info,movie_side_info,time_side_info), save_dir + 'side_info.pt')
    else:
        torch.save((movie_side_info,time_side_info),save_dir+'side_info.pt')

    with open(save_dir + 'full_tensor_shape.pickle', 'wb') as handle:
        pickle.dump(tensor_shape, handle, protocol=pickle.HIGHEST_PROTOCOL)

