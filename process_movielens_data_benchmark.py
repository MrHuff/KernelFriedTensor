import pandas as pd
import numpy as np
from dask_ml.preprocessing import DummyEncoder,StandardScaler
from sklearn.decomposition import PCA
import dask.dataframe as dd
from dask.distributed import Client,LocalCluster
from dask.diagnostics import ProgressBar
import os
if __name__ == '__main__':
    ProgressBar().register()
    if not os.path.exists('./ml-20m/genome-scores.csv'):
        pca = PCA(n_components=100)
        movie_side_info = pd.read_csv('./ml-20m/genome-scores.csv')
        movie_side_info = movie_side_info.pivot_table(index='movieId', columns='tagId', values='relevance')
        indices = np.int64(movie_side_info.index.values[:, np.newaxis])
        movie_side_info = pca.fit_transform(movie_side_info)
        print(np.cumsum(pca.explained_variance_ratio_))
        total = pd.DataFrame(np.concatenate([indices, movie_side_info], axis=1),
                             columns=['movieId'] + [f'PCA_{i}' for i in range(100)])
        total['movieId'] = total['movieId'].astype('int64')
        total.to_csv('./ml-20m/genome-scores_PCA.csv',index=False)
        del total


    if not os.path.exists('./movielens_parquet/'):
        # cluster = LocalCluster(n_workers=63, threads_per_worker=1)
        # client = Client(cluster)
        df = dd.read_csv('./ml-20m/ratings.csv',blocksize=8000000)
        print(df)
        df['timestamp'] = dd.to_datetime(df['timestamp'],unit='s').dt.hour
        print(df)
            # /= dd.to_datetime(df['timestamp'].values,infer_datetime_format=True,unit='s').hour
        movie_side_info = dd.read_csv('./ml-20m/genome-scores_PCA.csv',blocksize=8000000)
        movie_side_info.columns = movie_side_info.columns.map(str)
        df = dd.merge(df, movie_side_info, on='movieId', suffixes=('', '_repeat'))
        df.columns = df.columns.map(str)
        categoricals = ['userId','movieId','timestamp']
        df = df.categorize(categoricals)
        print(df)
        df.to_parquet('./movielens_parquet/',object_encoding='utf8')
    #TODO: figure out what to do with indices...
    # scaler_movie = StandardScaler()
    # de = DummyEncoder()
    # categoricals = ['userId', 'movieId', 'timestamp']
    # df = dd.read_parquet('./movielens_parquet/')
    # df = df.categorize(categoricals)
    # sd_ohe = de.fit_transform(df)
    # sd_ohe = scaler_movie.fit_transform(sd_ohe)
    # sd_ohe.to_parquet('./movielens_parquet_ohe/')









