import pandas as pd
import os
import dask.dataframe as dd
from dask_ml.preprocessing import DummyEncoder,StandardScaler
from dask.distributed import Client,LocalCluster

if __name__ == '__main__':
    PATH = './public_data/'
    cluster = LocalCluster(n_workers=63,threads_per_worker=1)
    client = Client(cluster)
    if not os.path.exists('./alcohol_sales_parquet/'):
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
        sd = dd.from_pandas(df, npartitions=1000)
        sd.to_parquet('./alcohol_sales_parquet/')
    categoricals = ['Store Location', 'Store Number', 'City', 'Zip Code', 'County Number', 'Category', 'Item Number']
    sd = dd.read_parquet('./alcohol_sales_parquet/')
    sd = sd.categorize(categoricals)
    de = DummyEncoder()
    scaler = StandardScaler()
    sd_ohe = de.fit_transform(sd)
    sd_ohe = scaler.fit_transform(sd_ohe)
    sd_ohe.to_parquet('./alcohol_sales_parquet_ohe/')



