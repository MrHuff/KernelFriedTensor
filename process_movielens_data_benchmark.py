import pandas as pd
import numpy as np
from dask_ml.preprocessing import DummyEncoder,StandardScaler
from sklearn.decomposition import PCA
import dask.dataframe as dd
from dask.distributed import Client,LocalCluster
from sklearn.feature_extraction import FeatureHasher
from dask.diagnostics import ProgressBar
from sklearn.compose import ColumnTransformer
import os
if __name__ == '__main__':
    ProgressBar().register()
    if not os.path.exists('./ml-20m/genome-scores_PCA.csv'):
        pca = PCA(n_components=25)
        movie_side_info = pd.read_csv('./ml-20m/genome-scores.csv')
        movie_side_info = movie_side_info.pivot_table(index='movieId', columns='tagId', values='relevance')
        indices = np.int64(movie_side_info.index.values[:, np.newaxis])
        movie_side_info = pca.fit_transform(movie_side_info)
        print(np.cumsum(pca.explained_variance_ratio_))
        total = pd.DataFrame(np.concatenate([indices, movie_side_info], axis=1),
                             columns=['movieId'] + [f'PCA_{i}' for i in range(25)])
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
    if not os.path.exists('./movielens_parquet_hashed'):
        categorical_columns = ['userId','movieId','timestamp']
        df = dd.read_parquet('./movielens_parquet/', index=False)
        hash_vector_size = 10
        ct = ColumnTransformer([(f't_{i}', FeatureHasher(n_features=hash_vector_size,
                                                         input_type='string'), i) for i in
                                range(len(categorical_columns))], verbose=True)
        to_be_hashed = df[categorical_columns].astype(str).compute()
        print(to_be_hashed)
        df = df.compute()
        df = df.drop(categorical_columns, axis=1)
        df = df.drop('index', axis=1)
        # to_be_hashed = to_be_hashed.map_partitions(ct.fit_transform).compute()
        to_be_hashed = ct.fit_transform(to_be_hashed).todense()
        print(to_be_hashed)
        to_be_hashed = pd.DataFrame(to_be_hashed, columns=[f'hash_{i}' for i in range(
            int(round(hash_vector_size * len(categorical_columns))))])
        df = df.reset_index(drop=True)
        to_be_hashed = to_be_hashed.reset_index(drop=True)
        df = pd.concat([df, to_be_hashed], axis=1)
        df = dd.from_pandas(df, npartitions=64)
        print(df.head())
        df.to_parquet('./movielens_parquet_hashed/')

    df = dd.read_parquet('./movielens_parquet_hashed/')
    Y = df['rating'].compute()
    X = df.drop('rating', axis=1)
    print(X)
    s = StandardScaler()
    X = s.fit_transform(X).compute()
    df = pd.concat([X, Y], axis=1)
    df = dd.from_pandas(df, npartitions=64)
    df.to_parquet('./movielens_parquet_hashed_scaled/')









