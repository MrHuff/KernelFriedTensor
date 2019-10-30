import numpy as np
import pandas as pd
import torch
import re
from torch.utils.data import DataLoader,Dataset
import dask.dataframe as dd
from dask.diagnostics import ProgressBar
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
# class read_real_data():
#
#     def __init__(self, tensor_path, u_path, v_path, t_path, seed):
#         np.random.seed(seed)
#         self.seed = str(seed)
#         self.data = torch.load(tensor_path)
#         self.u_kernel_data = torch.load(u_path)
#         self.v_kernel_data = torch.load(v_path)
#         self.t_kernel_data = torch.load(t_path)
#
#         self.indices = (torch.isnan(self.data) == 0).nonzero()
#         self.Y = self.data[torch.unbind(self.indices, dim=1)]
#
#         self.indices = self.indices.numpy()
#         self.Y = self.Y.numpy()
#
#         self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(self.indices,
#                                                                                 self.Y,
#                                                                                 test_size=0.2,
#                                                                                 random_state=seed
#                                                                                 )
#         self.X_train, self.X_val, self.Y_train, self.Y_val = train_test_split(self.X_train,
#                                                                               self.Y_train,
#                                                                               test_size=0.25,
#                                                                               random_state=seed)
#         self.X_train = torch.from_numpy(self.X_train)
#         self.X_val = torch.from_numpy(self.X_val)
#         self.X_test = torch.from_numpy(self.X_test)
#         self.Y_train = torch.from_numpy(self.Y_train)
#         self.Y_val = torch.from_numpy(self.Y_val)
#         self.Y_test = torch.from_numpy(self.Y_test)
#
#     def get_batch(self, rate):
#         batch_msk = np.random.choice(self.X_train.shape[0], int(self.X_train.shape[0] * rate),
#                                      replace=False)
#         return self.X_train[batch_msk, :], self.data[torch.unbind(self.X_train[batch_msk, :], dim=1)]
#
#     def get_test(self):
#         return self.X_test, self.data[torch.unbind(self.X_test, dim=1)]
#
#     def get_validation(self):
#         return self.X_val, self.data[torch.unbind(self.X_val, dim=1)]
#
#
# class toy_data_tensor():
#     def __init__(self, path='toy_data.p'):
#         self.data_tensor = torch.load(path)
#         self.indices = np.array(list(itertools.product(*[range(el) for el in self.data_tensor.shape])))
#         self.shape = self.data_tensor.shape
#
#     def get_non_nan(self, sample_size=0.1):
#         np_indices = self.indices[np.random.choice(self.indices.shape[0], replace=False,
#                                                    size=int(sample_size * self.indices.shape[0])), :].T
#         pytorch_indices = [torch.LongTensor(el) for el in np_indices.tolist()]
#         return np_indices, pytorch_indices
#
#     def get_all_indices(self):
#         np_indices = self.indices.T
#         pytorch_indices = [torch.LongTensor(el) for el in np_indices.tolist()]
#         return np_indices, pytorch_indices
#
#
# class abstract_data_class(ABC):
#     def __init__(self, path='generated_data.csv'):
#         self.df = pd.read_csv(path, index_col=0)
#         np.random.seed(1337)
#
#
# class benchmark_data(abstract_data_class):
#     def __init__(self, path='generated_data.csv'):
#         super().__init__(path)
#         self.df = self.df.reset_index()
#         self.time_df = pd.DataFrame(list(map(get_int_dates, self.df['timestamp'].tolist())),
#                                     columns=['year', 'month', 'day'])
#         self.df = pd.concat([self.df, self.time_df], axis=1)
#         self.df = self.df.drop(['timestamp'], axis=1)
#         self.dummies = pd.get_dummies(self.df[['cities', 'article']])
#         self.df = pd.concat([self.df, self.dummies], axis=1)
#         self.df = self.df.drop(['cities', 'article'], axis=1)
#         self.df = self.df.dropna()
#         self.Y = self.df['sales_val']
#         self.X = self.df.loc[:, self.df.columns != 'sales_val']
#         msk = np.random.rand(len(self.X)) < 0.8
#         self.X_test = self.X[~msk]
#         self.X = self.X[msk]
#         self.Y_test = self.Y[~msk]
#         self.Y = self.Y[msk]
#
#
# class data_tensor(abstract_data_class):
#     def __init__(self, path='generated_data.csv'):
#         super().__init__(path)
#         self.t_kernel_data = torch.Tensor(list(map(get_int_dates, self.df['timestamp'].unique().tolist())))  # Get dates in int form to tensor
#         tensor_dim = tuple(self.df.nunique().tolist()[0:-1])
#         self.numpy_tensor = self.df['sales_val'].values.reshape(tensor_dim)
#         self.indices = np.argwhere(~np.isnan(self.numpy_tensor))
#         msk = np.random.rand(self.indices.shape[0]) < 0.8
#         self.test_indices = self.indices[~msk, :]
#         self.indices = self.indices[msk, :]
#         self.data = torch.Tensor(self.numpy_tensor)
#         self.shape = self.data_tensor.shape
#         self.number_of_dims = len(self.shape)
#
#     def get_batch(self, sample_size=0.1):
#         np_indices = self.indices[np.random.choice(self.indices.shape[0], replace=False,
#                                                    size=int(sample_size * self.indices.shape[0])), :].T
#         pytorch_indices = [torch.LongTensor(el) for el in np_indices.tolist()]
#         return np_indices, pytorch_indices
#
#     def get_all_indices(self):
#         np_indices = self.indices.T
#         pytorch_indices = [torch.LongTensor(el) for el in np_indices.tolist()]
#         return np_indices, pytorch_indices
#
#     def get_test_indices(self):
#         np_indices = self.test_indices.T
#         pytorch_indices = [torch.LongTensor(el) for el in np_indices.tolist()]
#         return np_indices, pytorch_indices

def process_old_setup(folder,tensor_name):
    data = torch.load(folder+tensor_name)
    shape = data.shape
    indices = (torch.isnan(data) == 0).nonzero() #All X:s
    Y = data[torch.unbind(indices, dim=1)] #All Y:s
    torch.save((indices,Y),folder+'all_data.pt')
    with open(folder+'full_tensor_shape.pickle', 'wb') as handle:
        pickle.dump(shape, handle, protocol=pickle.HIGHEST_PROTOCOL)

def load_side_info(side_info_path,indices):
    container = {}
    side_info = torch.load(side_info_path + 'side_info.pt')
    for i in indices:
        container[i] = side_info[i]
    return container

class tensor_dataset(Dataset):
    def __init__(self, tensor_path,seed,mode):

        self.indices,self.Y = torch.load(tensor_path)

        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(self.indices,
                                                                                self.Y,
                                                                                test_size=0.2,
                                                                                random_state=seed
                                                                                )
        self.X_train, self.X_val, self.Y_train, self.Y_val = train_test_split(self.X_train,
                                                                              self.Y_train,
                                                                              test_size=0.25,
                                                                              random_state=seed)
        if mode=='train':
            self.X = self.X_train
            self.Y = self.Y_train
        elif mode=='val':
            self.X = self.X_val
            self.Y = self.Y_val
        elif mode=='test':
            self.X = self.X_test
            self.Y = self.Y_test

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx,:],self.Y[idx]

def get_dataloader_tensor(tensor_path,seed,mode,bs_ratio):
    ds = tensor_dataset(tensor_path,seed,mode)
    n = len(ds)
    bs = int(round(n*bs_ratio))
    return DataLoader(dataset=ds,batch_size=bs)

