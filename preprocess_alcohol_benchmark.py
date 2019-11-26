import pandas as pd
import torch
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
import re
import scipy.sparse as ssp
from sklearn.model_selection import train_test_split

def sparse_ohe(df, col, vals):
    """One-hot encoder using a sparse ndarray."""
    colaray = df[col].values
    # construct a sparse matrix of the appropriate size and an appropriate,
    # memory-efficient dtype
    spmtx = ssp.dok_matrix((df.shape[0], vals.shape[0]), dtype=np.uint8)
    # do the encoding
    spmtx[np.where(colaray.reshape(-1, 1) == vals.reshape(1, -1))] = 1
    print('done with the important')
    # Construct a SparseDataFrame from the sparse matrix
    dfnew = pd.SparseDataFrame(spmtx, dtype=np.uint8, index=df.index,
                               columns=[col + '_' + str(el) for el in vals])
    dfnew.fillna(0, inplace=True)
    return dfnew

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
        num_cols = df.columns[df.dtypes.apply(lambda c: np.issubdtype(c, np.number))]
        print(num_cols)
        df.to_hdf(PATH + 'liqour_sales_processed.h5', key='liqour_sales_processed')