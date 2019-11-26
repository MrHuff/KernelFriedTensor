import numpy as np
import time
import dask.dataframe as dd
from dask_ml.preprocessing import DummyEncoder,StandardScaler
import os
import pandas as pd

PATH = './raw_data_hm/'
de = DummyEncoder()
s = StandardScaler()
np.random.seed(1337)

if __name__ == '__main__':
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
            'product_id'
        ]

        for el in category_list_article:
            articles[el] = articles[el].astype(str)
        articles.to_csv(PATH + 'article_id_processed.csv', index=False)
        sd = dd.from_pandas(df, npartitions=10)
        sd.to_parquet('./hm_sales_parquet/')

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

    #print(location)
    #print(articles)
    #df = df.set_index('location_id')
    #location = location.set_index('location_id')

    start = time.time()
    df = dd.merge(df, location, on='location_id',suffixes=('','_repeat'))
    end = time.time()
    print('join locations took {}'.format(end-start))
    #df = df.reset_index()



    #print(len(sampled_list))
    df = df[df['unique_id'].isin(sampled_list)]
    articles = articles[articles['unique_id'].isin(sampled_list)]
    #df = df.set_index('article_id')
    #articles = articles.set_index('article_id')
    start = time.time()
    df = dd.merge(df, articles, on='unique_id',suffixes=('','_repeat'))
    end = time.time()
    print('join articles took {}'.format(end-start))
    #df = df.reset_index()
    print(df)
    #df['article_id']=df['article_id'].apply(str)
    #df['location_id']=df['location_id'].apply(str)
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
                           'product_id'
                          ]
    df[categorical_columns] = df[categorical_columns].astype(int)
    df = df.categorize(categorical_columns)
    num_cols = df.columns[df.dtypes.apply(lambda c: np.issubdtype(c, np.number))]
    print(num_cols)
    print(df)
    df.to_parquet('./benchmark_data_lgbm/')