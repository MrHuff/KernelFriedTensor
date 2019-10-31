import numpy as np
import pandas as pd
import torch
import re
from torch.utils.data import DataLoader,Dataset
# import dask.dataframe as dd
# from dask.diagnostics import ProgressBar
from sklearn.model_selection import train_test_split
import pickle


def print_model_parameters(model):
    for n,p in model.named_parameters():
        print(n)
        print(p.shape)
        print(p.requires_grad)

def get_int_dates(x):
    y = x.split('-')
    return list(map(lambda z: int(z), y))

def chunkify(lst, n):
    return [lst[i::n] for i in range(n)]

class read_benchmark_data():
    def __init__(self, y_name, path='benchmark_data.h5', seed=1337, backend='dask'):

        if backend == 'pandas':
            if '.h5' in path:
                self.df = pd.read_hdf(path)
            else:
                self.df = pd.read_parquet(path, engine='fastparquet')
            regex = re.compile(r"\[|\]|<", re.IGNORECASE)
            self.df.columns = [regex.sub("_", col) if any(x in str(col) for x in set(('[', ']', '<'))) else col for col
                               in
                               self.df.columns.values]

            self.Y = self.df[y_name]
            self.X = self.df.drop(y_name, axis=1)

        else:

            ProgressBar().register()
            self.df = dd.read_parquet(path)
            regex = re.compile(r"\[|\]|<", re.IGNORECASE)
            self.df.columns = [regex.sub("_", col) if any(x in str(col) for x in set(('[', ']', '<'))) else col for col
                               in
                               self.df.columns.values]

            self.Y = self.df[y_name]
            self.X = self.df.drop(y_name, axis=1)
            self.X = self.X.compute()
            self.Y = self.Y.compute()

        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(self.X,
                                                                                self.Y,
                                                                                test_size=0.2,
                                                                                random_state=seed
                                                                                )
        self.X_train, self.X_val, self.Y_train, self.Y_val = train_test_split(self.X_train,
                                                                              self.Y_train,
                                                                              test_size=0.25,
                                                                              random_state=seed
                                                                              )
        self.n = len(self.X_train)
        print("Succesfully loaded data")

    def get_batch(self, ratio):
        msk = np.random.rand(self.n) < ratio
        return self.X_train[msk], self.Y_train[msk]

    def get_test_batch(self, chunks):
        return chunkify(self.X_test, chunks), chunkify(self.Y_test, chunks)

    def get_validation_batch(self, chunks):
        return chunkify(self.X_val, chunks), chunkify(self.Y_val, chunks)

class kernel_adding_tensor():
    def __init__(self,tensor_path):
        self.data = torch.load(tensor_path + 'tensor_data.pt')
        try:
            self.m_side, self.n_side, self.t_side = torch.load(tensor_path + 'side_info.pt')
        except Exception as e:
            print(e)

        self.indices = (torch.isnan(self.data) == 0).nonzero()
        self.Y = self.data[torch.unbind(self.indices, dim=1)]
        self.indices = self.indices.numpy()
        self.Y = self.Y.numpy()
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(self.indices,
                                                                                self.Y,
                                                                                test_size=0.2,
                                                                                random_state=10
                                                                                )
        self.X_train, self.X_val, self.Y_train, self.Y_val = train_test_split(self.X_train,
                                                                              self.Y_train,
                                                                              test_size=0.25,
                                                                              random_state=10)
        self.X_train = torch.from_numpy(self.X_train)
        self.X_val = torch.from_numpy(self.X_val)
        self.X_test = torch.from_numpy(self.X_test)
        self.Y_train = torch.from_numpy(self.Y_train)
        self.Y_val = torch.from_numpy(self.Y_val)
        self.Y_test = torch.from_numpy(self.Y_test)

    def get_batch(self, rate):
        batch_msk = np.random.choice(self.X_train.shape[0], int(self.X_train.shape[0] * rate),
                                     replace=False)
        return self.X_train[batch_msk, :], self.data[torch.unbind(self.X_train[batch_msk, :], dim=1)]

    def get_test(self):
        return self.X_test, self.data[torch.unbind(self.X_test, dim=1)]

    def get_validation(self):
        return self.X_val, self.data[torch.unbind(self.X_val, dim=1)]

def process_old_setup(folder,tensor_name):
    data = torch.load(folder+tensor_name)
    shape = data.shape
    indices = (torch.isnan(data) == 0).nonzero() #All X:s
    Y = data[torch.unbind(indices, dim=1)] #All Y:s
    torch.save((indices,Y.float()),folder+'all_data.pt')
    with open(folder+'full_tensor_shape.pickle', 'wb') as handle:
        pickle.dump(shape, handle, protocol=pickle.HIGHEST_PROTOCOL)

def concat_old_side_info(PATH,paths):
    concated = []
    for f in paths:
        s = torch.load(PATH+f)
        concated.append(s.float())
    torch.save(concated,PATH+'side_info.pt')

def load_side_info(side_info_path,indices):
    container = {}
    side_info = torch.load(side_info_path + 'side_info.pt')
    for i,info in zip(indices,side_info):
        container[i] = {'data':info,'temporal':False}
    return container

class tensor_dataset(Dataset):
    def __init__(self, tensor_path,seed,mode):
        self.indices,self.Y = torch.load(tensor_path)
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(self.indices.numpy(),
                                                                                self.Y.numpy(),
                                                                                test_size=0.2,
                                                                                random_state=seed
                                                                                )
        self.X_train, self.X_val, self.Y_train, self.Y_val = train_test_split(self.X_train,
                                                                              self.Y_train,
                                                                              test_size=0.25,
                                                                              random_state=seed)
        if mode=='train':
            self.X = torch.from_numpy(self.X_train)
            self.Y = torch.from_numpy(self.Y_train)
        elif mode=='val':
            self.X = torch.from_numpy(self.X_val)
            self.Y = torch.from_numpy(self.Y_val)
        elif mode=='test':
            self.X = torch.from_numpy(self.X_test)
            self.Y = torch.from_numpy(self.Y_test)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx,:],self.Y[idx]

def get_dataloader_tensor(tensor_path,seed,mode,bs_ratio,cuda):
    ds = tensor_dataset(tensor_path,seed,mode)
    n = len(ds)
    bs = int(round(n*bs_ratio))
    return DataLoader(dataset=ds,batch_size=bs,pin_memory=cuda)

