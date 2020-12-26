import pandas as pd
import numpy as np
import torch
from sklearn import preprocessing
from dask.diagnostics import ProgressBar
import dask
from sklearn.feature_extraction import FeatureHasher

scaler = preprocessing.MinMaxScaler()
np.random.seed(1337)
do_PCA = False
do_scale = False

"""
read data
"""
PATH = './raw_data_hm/'
save_path = '../tensor_data/'
n_article_list = [400000]
scale_list = [False]
ProgressBar().register()
for scale_ in scale_list:
    for n in n_article_list:
        do_scale = scale_
        n_articles = n
        do_PCA = False
        df = pd.read_parquet(PATH + 'sales_processed')
        articles = pd.read_csv(PATH + 'article_id_processed.csv')
        location = pd.read_csv(PATH + 'location.csv')
        articles = articles.drop(['article_id'], axis=1)
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
        category_list_location = [
            'corporate_brand_id',
            'city_id',
        ]

        h = FeatureHasher(n_features=10, input_type='string')
        articles[category_list_article] = articles[category_list_article].astype(str)
        for el in category_list_article:
            f = h.transform(articles[el])
            cols = [el + '_{}'.format(i) for i in range(10)]
            articles = pd.concat([articles, pd.DataFrame(f.toarray(), columns=cols)], axis=1)
            articles = articles.drop(el, axis=1)

        article_join = np.array(articles['unique_id'].unique())
        location_join = np.array(location['location_id'].unique())
        df = df[df['unique_id'].isin(article_join)]
        df = df[df['location_id'].isin(location_join)]

        """
        Match articles
        """
        print(len(df['unique_id'].unique()))
        article_list = np.random.choice(df['unique_id'].unique(), n_articles, False)
        articles = articles[articles['unique_id'].isin(article_list)]
        articles = articles.set_index('unique_id')  # .sort_index(level=[0])
        df = df[df['unique_id'].isin(article_list)]

        """
        Match locations
        """

        real_location_list = np.array(df['location_id'].unique())
        location = location[location['location_id'].isin(real_location_list)]
        location = location.set_index('location_id')  # .sort_index(level=[0])

        """
        Tensorize data
        """

        # df = df.head(100000)
        df['timestamp'] = df['year'].astype(str) + '-' + df['month'].astype(str) + '-1'
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.drop(['year', 'month'], axis=1)
        tensor = np.empty(tuple(df.nunique()[['location_id', 'unique_id', 'timestamp']].tolist()))
        tensor.fill(np.nan)
        df = df.set_index(
            ['location_id', 'unique_id', 'timestamp']
        ).sort_index(level=[0, 1, 2])
        index_list = [df.index.labels[i] for i in range(3)]
        tensor[tuple(index_list)] = df['total_sales'].values
        torch_tensor = torch.from_numpy(tensor)
        torch.save(torch_tensor, save_path + 'data_tensor_{}.pt'.format(n_articles))
        time = torch.transpose(torch.tensor([df.index.levels[2].year.values, df.index.levels[2].month.values]), 0, 1)

        torch.save(time, save_path + 'time_tensor_{}.pt'.format(n_articles))

        """
        Save side data
        """

        location[category_list_location] = location[category_list_location].astype(str)
        location = pd.get_dummies(location)
        print(location.isnull().values.any())
        torch_location = torch.from_numpy(location.values)
        torch.save(torch_location, save_path + 'location_tensor_{}.pt'.format(n_articles))

        print(articles.head())
        articles = articles.fillna(0)
        print(articles.isnull().values.any())
        torch_article = torch.from_numpy(articles.values)
        torch.save(torch_article, save_path + 'article_tensor_{}.pt'.format(n_articles))