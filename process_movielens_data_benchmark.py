import pandas as pd
import torch
from dask_ml.preprocessing import DummyEncoder,StandardScaler
import dask.dataframe as dd
from dask.distributed import Client,LocalCluster
import os
if __name__ == '__main__':
    if not os.path.exists('./movielens_parquet/'):

        cluster = LocalCluster(n_workers=63,threads_per_worker=1)
        client = Client(cluster)
        df = pd.read_csv('./ml-20m/ratings.csv')
        df['timestamp'] = pd.to_datetime(df['timestamp'].values,infer_datetime_format=True,unit='s').hour
        movie_side_info = pd.read_csv('./ml-20m/genome-scores.csv')
        df = df[df['movieId'].isin(movie_side_info['movieId'].unique())]
        movie_side_info = movie_side_info[movie_side_info['movieId'].isin(df['movieId'].unique())]
        movie_side_info = movie_side_info.pivot(index='movieId', columns='tagId', values='relevance').sort_index()
        movie_side_info.columns = movie_side_info.columns.map(str)
        movie_side_info = dd.from_pandas(movie_side_info,npartitions=100)
        df = dd.from_pandas(df,npartitions=1000)
        df = dd.merge(df, movie_side_info, on='movieId', suffixes=('', '_repeat'))
        print(df)
        categoricals = ['userId','movieId','timestamp']
        df = df.categorize(categoricals)
        print(df)
        df.to_parquet('./movielens_parquet/')
    scaler_movie = StandardScaler()
    de = DummyEncoder()
    categoricals = ['userId', 'movieId', 'timestamp']
    df = dd.read_parquet('./movielens_parquet/')
    df = df.categorize(categoricals)
    sd_ohe = de.fit_transform(df)
    sd_ohe = scaler_movie.fit_transform(sd_ohe)
    sd_ohe.to_parquet('./movielens_parquet_ohe/')









