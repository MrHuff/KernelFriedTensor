import numpy as np
import time
import dask.dataframe as dd
import dask.array as da
from dask_ml.preprocessing import StandardScaler
import os
import pandas as pd
from dask.diagnostics import ProgressBar
from dask.distributed import Client,LocalCluster
from sklearn.feature_extraction import FeatureHasher
from sklearn.compose import ColumnTransformer
PATH = './raw_data_hm/'
np.random.seed(1337)

if __name__ == '__main__':
    ProgressBar().register()
    # cluster = LocalCluster(n_workers=63,threads_per_worker=1)
    # client = Client(cluster)
    if not os.path.exists('./hm_sales_parquet/'):
        try:
            df = pd.read_hdf(PATH + 'sales.h5')
        except:
            df = pd.read_csv(PATH + 'sales.csv')
            df.to_hdf(PATH + 'sales.h5', key='sales')

        articles = pd.read_csv(PATH + 'articles.csv')
        articles['unique_id'] = articles['article_id'].astype(str) + articles['enterprise_size_id'].astype(str)
        df['unique_id'] = df['article_id'].astype(str) + df['enterprise_size_id'].astype(str)
        category_list_article = [
            'graphical_appearance_id',
            'colour_id',
            'enterprise_size_id',
            'department_id',
            'product_season_id',
            'product_type_id',
            'product_group_no',
            'product_id',
            'article_id'
        ]

        for el in category_list_article:
            articles[el] = articles[el].astype(str)
        articles.to_csv(PATH + 'article_id_processed.csv', index=False)
        sd = dd.from_pandas(df, npartitions=64)
        sd.to_parquet('./hm_sales_parquet/')
    if not os.path.exists('./benchmark_data_lgbm/'):
        n_articles = 400000
        start = time.time()
        df = dd.read_parquet('./hm_sales_parquet/')#.compute()
        df = df.drop(['article_id','enterprise_size_id'],axis=1)
        end = time.time()
        print('read parquet file took {}'.format(end-start))
        print(df)
        start = time.time()

        df['unique_id'] = df['unique_id'].apply(str)
        df['location_id'] = df['location_id'].apply(str)
        sampled_list = np.random.choice(df['unique_id'].unique(),n_articles,replace=False)
        end = time.time()
        print('get samples took {}'.format(end-start))
        articles = dd.read_csv(PATH+'article_id_processed.csv')#.compute()#scheduler='processes'
        location = dd.read_csv(PATH+'location.csv')#.compute()
        articles['unique_id'] = articles['unique_id'].apply(str)
        location['location_id'] = location['location_id'].apply(str)

        articles = articles.categorize(['unique_id'])
        location = location.categorize(['location_id'])
        start = time.time()
        df = dd.merge(df, location, on='location_id',suffixes=('','_repeat'))
        end = time.time()
        print('join locations took {}'.format(end-start))
        df = df[df['unique_id'].isin(sampled_list)]
        articles = articles[articles['unique_id'].isin(sampled_list)]
        start = time.time()
        df = dd.merge(df, articles, on='unique_id',suffixes=('','_repeat'))
        end = time.time()
        print('join articles took {}'.format(end-start))
        df = df.drop('unique_id',axis=1)
        print(df)
        start = time.time()
        categorical_columns = [
                               'location_id',
                               'city_id',
                               'corporate_brand_id',
                               'graphical_appearance_id',
                               'colour_id',
                               'enterprise_size_id',
                               'department_id',
                               'product_season_id',
                               'product_type_id',
                               'product_group_no',
                               'product_id',
                                'article_id',
                              ]
        df[categorical_columns] = df[categorical_columns].astype(int)
        df = df.categorize(categorical_columns)
        df.to_parquet('./benchmark_data_lgbm/')
    if not os.path.exists('./benchmark_data_private_hashed/'):
        df = dd.read_parquet('./benchmark_data_lgbm/',index=False)
        categorical_columns = [
            'location_id',
            'city_id',
            'corporate_brand_id',
            'graphical_appearance_id',
            'colour_id',
            'enterprise_size_id',
            'department_id',
            'product_season_id',
            'product_type_id',
            'product_group_no',
            'product_id',
            'article_id',
        ]
        hash_vector_size = 5
        ct = ColumnTransformer([(f't_{i}', FeatureHasher(n_features=hash_vector_size,
                                                         input_type='string'), i) for i in range(len(categorical_columns))],verbose=True)
        to_be_hashed = df[categorical_columns].astype(str).compute()
        df = df.compute()
        df = df.drop(categorical_columns,axis=1)
        df = df.drop('index',axis=1)
        # to_be_hashed = to_be_hashed.map_partitions(ct.fit_transform).compute()
        to_be_hashed = ct.fit_transform(to_be_hashed)
        to_be_hashed = pd.DataFrame(to_be_hashed,columns=[f'hash_{i}' for i in range(60)])
        df = df.reset_index(drop=True)
        to_be_hashed=to_be_hashed.reset_index(drop=True)
        df= pd.concat([df,to_be_hashed],axis=1)
        df = dd.from_pandas(df,npartitions=64)
        df.to_parquet('./benchmark_data_private_hashed/')
    df = dd.read_parquet('./benchmark_data_private_hashed/')
    Y = df['total_sales'].compute()
    X = df.drop('total_sales',axis=1)
    print(X)
    s = StandardScaler()
    X = s.fit_transform(X).compute()
    df = pd.concat([X, Y], axis=1)
    df = dd.from_pandas(df,npartitions=64)
    df.to_parquet('./benchmark_data_private_hashed_scaled/')


