import pandas as pd
import os
import dask.dataframe as dd
from dask_ml.preprocessing import StandardScaler
from dask.distributed import Client,LocalCluster
from dask.diagnostics import ProgressBar
from sklearn.feature_extraction import FeatureHasher
from sklearn.compose import ColumnTransformer
from KFT.benchmarks.utils import core_data_extract_df
import numpy as np
if __name__ == '__main__':
    PATH = './public_data/'
    ProgressBar().register()
    categorical_columns = ['Store Location', 'Store Number', 'City', 'Zip Code', 'County Number', 'Category',
                           'Item Number']

    # cluster = LocalCluster(n_workers=63,threads_per_worker=1)
    # client = Client(cluster)
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
        sd = dd.from_pandas(df, npartitions=64)
        sd.to_parquet('./alcohol_sales_parquet/')

    if not os.path.exists('./alcohol_sales_parquet_hashed'):
        df = dd.read_parquet('./alcohol_sales_parquet/',index=False)
        hash_vector_size = 25
        ct = ColumnTransformer([(f't_{i}', FeatureHasher(n_features=hash_vector_size,
                                                         input_type='string'), i) for i in range(len(categorical_columns))],verbose=True)
        to_be_hashed = df[categorical_columns].astype(str).compute()
        print(to_be_hashed)
        df = df.compute()
        df = df.drop(categorical_columns,axis=1)
        df = df.drop('index',axis=1)
        # to_be_hashed = to_be_hashed.map_partitions(ct.fit_transform).compute()
        to_be_hashed = ct.fit_transform(to_be_hashed).todense()
        print(to_be_hashed)
        to_be_hashed = pd.DataFrame(to_be_hashed,columns=[f'hash_{i}' for i in range(int(round(hash_vector_size*len(categorical_columns))))])
        df = df.reset_index(drop=True)
        to_be_hashed=to_be_hashed.reset_index(drop=True)
        df= pd.concat([df,to_be_hashed],axis=1)
        df = dd.from_pandas(df,npartitions=64)
        print(df.head())
        df.to_parquet('./alcohol_sales_parquet_hashed/')

    if not os.path.exists('./alcohol_sales_parquet_hashed'):

        df = dd.read_parquet('./alcohol_sales_parquet_hashed/')
        Y = df['Bottles Sold'].compute()
        X = df.drop('Bottles Sold',axis=1)
        print(X)
        s = StandardScaler()
        X = s.fit_transform(X).compute()
        df = pd.concat([X, Y], axis=1)
        df = dd.from_pandas(df,npartitions=64)
        df.to_parquet('./alcohol_sales_parquet_hashed_scaled/')

    if not os.path.exists('alcohol_sales_parquet_FFM'):
        df = dd.read_parquet('./alcohol_sales_parquet/')  # .compute()

        n_cat = 100
        df = df.reset_index(drop=True)
        Y = df['Bottles Sold'].compute()
        X = df.drop('Bottles Sold',axis=1).compute()
        df_num = X.select_dtypes(include=[np.float]).columns
        print(df_num)
        # for col in X.columns:
        #     print(X[col].unique())
        #     X[col] = X[col].fillna(X[col].mean())
        for el in df_num:
            X[el] =pd.cut(X[el],n_cat,labels=False)
        print(X.columns[-1])
        X = core_data_extract_df(X)
        X = X.reset_index(drop=True)
        Y=Y.reset_index(drop=True)
        df = pd.concat([X,Y],axis=1)
        df = dd.from_pandas(df, npartitions=64)
        df.to_parquet('./alcohol_sales_parquet_FFM/')
    df = dd.read_parquet('./alcohol_sales_parquet_FFM/').compute()
    print(df.min())
    print(df.max())
    print(df.nunique())


