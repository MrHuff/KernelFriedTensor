import numpy as np
import pandas as pd
import torch
import re
from torch.utils.data import DataLoader,Dataset
from sklearn.model_selection import train_test_split
import pickle
import argparse
from io import BytesIO
import subprocess
from sklearn.preprocessing import StandardScaler

def generate_timestamp_side_info(sorted_timestamp_data):
    t = np.unique(sorted_timestamp_data)
    scaler_location = StandardScaler()
    t = scaler_location.fit_transform(t.reshape(-1, 1))
    return torch.from_numpy(t).float()
def core_data_extract(df,indices_list,target_name):
    tensor_shape = df.nunique()[indices_list].tolist()
    df = df.set_index(
        indices_list
    ).sort_index(level=[i for i in range(len(indices_list))])
    X = [df.index.codes[i] for i in range(len(indices_list))]
    y = df[target_name].values
    X = torch.tensor(X).int().t()
    y = torch.tensor(y).float()
    return X,y,tensor_shape

def get_free_gpu(n=3):
    gpu_stats = subprocess.check_output(["nvidia-smi", "--format=csv", "--query-gpu=memory.used,memory.free"])
    gpu_df = pd.read_csv(BytesIO(gpu_stats),
                         names=['memory.used', 'memory.free'],
                         skiprows=1)
    print('GPU usage:\n{}'.format(gpu_df))
    gpu_df['memory.free'] = gpu_df['memory.free'].map(lambda x: int(x.rstrip(' [MiB]')))
    idx = gpu_df.nlargest(n,['memory.free']).index.values
    for i in idx:
        print('Returning GPU{} with {} free MiB'.format(i, gpu_df.iloc[i]['memory.free']))
    return idx

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def job_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--PATH', type=str, nargs='?')
    parser.add_argument('--max_R', type=int, nargs='?', default=10, help='max_R')
    parser.add_argument('--reg_para_a', type=float, nargs='?', default=0., help='reg_para_a')
    parser.add_argument('--reg_para_b', type=float, nargs='?', default=1., help='reg_para_b')
    parser.add_argument('--batch_size_a', type=float, nargs='?', default=0., help='batch_size_a')
    parser.add_argument('--batch_size_b', type=float, nargs='?', default=1., help='batch_size_b')
    parser.add_argument('--max_lr', type=float, nargs='?', default=1., help='max_lr')
    parser.add_argument('--fp_16', default=False, help='fp_16',type=str2bool, nargs='?')
    parser.add_argument('--fused', default=False, help='fused',type=str2bool, nargs='?')
    parser.add_argument('--hyperits', type=int, nargs='?', default=20, help='hyperits')
    parser.add_argument('--save_path', type=str, nargs='?')
    parser.add_argument('--task', type=str, nargs='?')
    parser.add_argument('--epochs', type=int, nargs='?', default=10, help='epochs')
    parser.add_argument('--bayesian', default=False, help='fp_16',type=str2bool, nargs='?')
    parser.add_argument('--cuda', default=True, help='cuda',type=str2bool, nargs='?')
    parser.add_argument('--full_grad', default=True, help='full_grad',type=str2bool, nargs='?')
    parser.add_argument('--sub_epoch_V', type=int, nargs='?', default=100, help='sub_epoch_V')
    parser.add_argument('--sub_epoch_ls', type=int, nargs='?', default=100, help='sub_epoch_ls')
    parser.add_argument('--sub_epoch_prime', type=int, nargs='?', default=100, help='sub_epoch_prime')
    parser.add_argument('--seed', type=int, nargs='?', help='seed')
    parser.add_argument('--side_info_order', nargs='+', type=int)
    parser.add_argument('--temporal_tag', nargs='+', type=int)
    parser.add_argument('--architecture', type=int, nargs='?', default=0, help='architecture')
    parser.add_argument('--tensor_name', type=str,default='', nargs='?')
    parser.add_argument('--side_info_name', type=str,nargs='+')
    return parser

def print_ls_gradients(model):
    for n,p in model.named_parameters():
        if 'lengthscale' in n:
            print(n)
            print(p)
            print(p.grad)
        if 'period' in n:
            print(n)
            print(p)
            print(p.grad)

def print_model_parameters(model):
    for n,p in model.named_parameters():
        print(n)
        print(p.shape)
        print(p.requires_grad)
        print(p.device)

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
    def __init__(self, tensor_path,seed,mode,bs_ratio=1.):
        self.ratio = bs_ratio
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
            self.bs = int(round(self.X.shape[0]*bs_ratio))
        elif mode=='val':
            self.ratio=1.
            self.X = torch.from_numpy(self.X_val)
            self.Y = torch.from_numpy(self.Y_val)
            self.bs = int(round(self.X.shape[0]))
        elif mode=='test':
            self.ratio=1.
            self.X = torch.from_numpy(self.X_test)
            self.Y = torch.from_numpy(self.Y_test)
            self.bs = int(round(self.X.shape[0]))

    def get_batch(self):
        if self.ratio==1.:
            return self.X,self.Y
        else:
            batch_msk = np.random.choice(self.X.shape[0],self.bs,
                                         replace=False)
            return self.X[batch_msk, :], self.Y[batch_msk]

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx,:],self.Y[idx]

def get_dataloader_tensor(tensor_path,seed,mode,bs_ratio):
    ds = tensor_dataset(tensor_path,seed,mode,bs_ratio=bs_ratio)
    return ds

