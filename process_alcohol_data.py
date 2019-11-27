import pandas as pd
import torch
import numpy as np
import os
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

if __name__ == '__main__':
    PATH = './public_data/'

    if not os.path.isfile(PATH+'liqour_sales_processed.h5'):
        '''
        Generall preprocessing - this goes into lightgbm
        '''
        if not os.path.isfile(PATH + 'liqour_sales.h5'):

            df = pd.read_csv(PATH + 'iowa-liquor-sales.zip')
            df.to_hdf(PATH + 'liqour_sales.h5', key='liqour_sales')
        else:
            df = pd.read_hdf(PATH + 'liqour_sales.h5')

        print(len(df))
        df['geo_data'] = df['Store Location'].str.extract('\(([^)]+)\)')
        df['year'] = df['Date'].apply(lambda x: int(x[-4:]))
        df['day'] = df['Date'].apply(lambda x: int(x[-7:-5]))
        df['month'] = df['Date'].apply(lambda x: int(x[0:2]))
        df = df.drop(['Date', 'Invoice/Item Number', 'Vendor Number', 'Store Name', 'Address', 'County', 'Category Name',
                      'Vendor Name', 'Item Description', 'Sale (Dollars)', 'Volume Sold (Liters)', 'Volume Sold (Gallons)'],
                     axis=1)
        df = df.dropna()
        df['latitude'] = df['geo_data'].str.split(',').apply(lambda x: x[0]).astype('float32')
        df['longitude'] = df['geo_data'].str.split(',').apply(lambda x: x[1]).astype('float32')
        df = df.drop('geo_data', axis=1)
        df['State Bottle Cost'] = df['State Bottle Cost'].str.strip('$').astype('float32')
        df['State Bottle Retail'] = df['State Bottle Retail'].str.strip('$').astype('float32')
        categoricals = ['Store Location', 'Store Number', 'City', 'Zip Code', 'County Number', 'Category', 'Item Number']
        for el in categoricals:
            df[el] = df[el].astype('category')
            df[el] = df[el].cat.codes

        aggregate_on = [el for el in df.columns.tolist() if el not in ['Bottles Sold', 'day']]
        df = df.groupby(aggregate_on, as_index=False)['Bottles Sold'].sum()
        print(df.head())
        df = df[df['year'].isin([2016, 2015])]
        df.to_hdf(PATH + 'liqour_sales_processed.h5', key='liqour_sales_processed')

    """
    Tensor Processing
    """
    if not os.path.isfile(PATH+'public_data_tensor.pt'):
        df = pd.read_hdf(PATH + 'liqour_sales_processed.h5')
        time = df.drop_duplicates(['year','month'])
        time = time[['year','month']].sort_values(['year','month'])
        location = df.drop_duplicates('Store Location').set_index('Store Location')
        location = location[['City','Zip Code','Store Number','County Number','latitude','longitude']].sort_index()
        location[['City','Zip Code','Store Number','County Number']] = location[['City','Zip Code','Store Number','County Number']].astype('category')
        articles = df.drop_duplicates('Item Number').set_index('Item Number')
        articles = articles[['Category','Pack','Bottle Volume (ml)','State Bottle Cost','State Bottle Retail']].sort_index()
        articles['Category'] = articles['Category'].astype('category')

        articles = pd.get_dummies(articles)
        location = pd.get_dummies(location)
        df['timestamp'] = df['year'].astype(str)+'-'+ df['month'].astype(str) +'-1'
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        tensor = np.empty(tuple(df.nunique()[['Store Location','Item Number','timestamp']].tolist()))
        tensor.fill(np.nan)
        df = df.set_index(
            ['Store Location','Item Number','timestamp']
        ).sort_index(level=[0,1,2])
        index_list = [df.index.labels[i] for i in range(3)]
        tensor[tuple(index_list)] = df['Bottles Sold'].values
        torch_tensor = torch.from_numpy(tensor)
        torch.save(torch_tensor,PATH+'public_data_tensor.pt')
        time = torch.transpose(torch.tensor([df.index.levels[2].year.values,df.index.levels[2].month.values]),0,1)
        torch.save(time,PATH+'public_time_tensor.pt')
        torch_location = torch.from_numpy(location.values)
        torch.save(torch_location,PATH+'public_location_tensor.pt')
        torch_article = torch.from_numpy(articles.values)
        torch.save(torch_article,PATH+'public_article_tensor.pt')
    """
    PCA
    """
    if not os.path.isfile(PATH+'public_location_tensor_PCA.pt'):

        pca = PCA(n_components=10)
        location = torch.load(PATH + 'public_location_tensor.pt').numpy()
        location = pca.fit_transform(location)
        print(pca.explained_variance_ratio_)
        torch_location = torch.from_numpy(location)
        torch.save(torch_location, PATH + 'public_location_tensor_PCA.pt')

        articles = torch.load(PATH + 'public_article_tensor.pt').numpy()
        articles = pca.fit_transform(articles)
        print(pca.explained_variance_ratio_)
        torch_article = torch.from_numpy(articles)
        torch.save(torch_article, PATH + 'public_article_tensor_PCA.pt')

    """
    Scaling
    """
    if not os.path.isfile(PATH+'public_time_tensor_scaled.pt'):

        for name in ['article','location','time']:
            location = torch.load(f'{PATH}/public_{name}_tensor.pt').numpy()
            scaler_location =StandardScaler()
            location_scaled = torch.tensor(scaler_location.fit_transform(location))
            torch.save(location_scaled,f'{PATH}/public_{name}_tensor_scaled.pt')