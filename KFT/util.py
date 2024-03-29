import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader,Dataset
from sklearn.model_selection import train_test_split
import pickle
import argparse
from io import BytesIO
import subprocess
from sklearn.preprocessing import StandardScaler
import os
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import GPUtil

def plot_VI(save_path,idx_list,seed,predictions):
    df = predictions.groupby(idx_list).agg(['sum','count'])
    cal_list = [5,15,25,35,45]
    fig, ax = plt.subplots(1, len(cal_list)+1,figsize=(30,20),gridspec_kw={'width_ratios':[1,1,1,1,1,0.05]})
    ax[0].get_shared_y_axes().join(*ax[1:5])

    for i in range(len(cal_list)):
        rate = cal_list[i]
        calibration_rate = 'calibrated_{}'.format(rate)
        plot_df = df[calibration_rate]
        plot_df['ratio'] = plot_df['sum']/plot_df['count']
        plot_df = plot_df.reset_index()
        result = plot_df.pivot(index=idx_list[0], columns=idx_list[1], values='ratio')
        if i==len(cal_list)-1:
            sns.heatmap(result,cmap="RdYlGn",ax=ax[i],cbar=True,cbar_ax = ax[i+1],vmin=0, vmax=1)
            ax[i].collections[0].colorbar.set_label('Calibration rate')
        else:
            sns.heatmap(result,cmap="RdYlGn",ax=ax[i],cbar=False)
        ax[i].set_xlabel('')
        ax[i].set_ylabel('')
        ax[i].set_yticks([])
        ax[i].set_xticks([])
        ax[i].set_title(r'$1-2\alpha$ = {}%'.format(100-2*rate))
        ax[i].title.set_fontsize(40)
    for item in ([ax[-1].title, ax[-1].xaxis.label, ax[-1].yaxis.label]+ax[-1].get_yticklabels()):
        item.set_fontsize(40)
    plt.subplots_adjust(wspace=0.05, hspace=0)

    if seed is None:
        plt.savefig(save_path + f'VI_plot.png', bbox_inches='tight',
                    pad_inches=0)
    else:
        plt.savefig(save_path + f'VI_plot_{seed}.png', bbox_inches = 'tight',
            pad_inches = 0)

def get_test_errors(folder_path, sort_metric_name, data_path, split_mode, reverse=False, bayes=False, arch=0,metric_name='test_loss'):
    trial_files = os.listdir(folder_path)
    metrics = []
    for i in range(1,6):
        compare  = f'bayesian_{i}.p' if bayes else f'frequentist_{i}_architecture_{arch}.p'
        dataloader = get_dataloader_tensor(data_path, seed=i, bs_ratio=1.0, split_mode=split_mode)
        var_Y_test = dataloader.Y_te.var().numpy()
        for el in trial_files:
            if el == compare:
                trials = pickle.load(open(folder_path + el, "rb"))
                filtered = [ ]
                for el in trials.results:
                    if el['status'] == 'ok':
                        filtered.append(el)
                r_2 = abs(sorted(filtered, key=lambda x: x[sort_metric_name], reverse=reverse)[0][metric_name])
                print(r_2)
                test_error = ((1-r_2)*var_Y_test)**0.5
                print(test_error)
                metrics.append([test_error,var_Y_test])
    df = pd.DataFrame(metrics)
    df = df.describe()
    df = df.round(3)
    df.to_csv(folder_path+'test_error_ref.csv')


def post_process(folder_path,metric_name,reverse=False,bayesian = False):
    trial_files = os.listdir(folder_path)
    print(trial_files)
    metrics = []
    best_config = []
    for el in trial_files:
        if '.p'==el[-2:]:
            print(el)
            trials = pickle.load(open(folder_path + el, "rb"))
            best_res = sorted(trials.trials, key=lambda x: x['result'][metric_name], reverse=reverse)[0]['misc']['vals']
            best_trial = sorted(trials.results, key=lambda x: x[metric_name], reverse=reverse)[0]
            metrics.append(best_trial)
            best_config.append(best_res)
    df = pd.DataFrame(metrics)
    print(df.columns)
    if bayesian:
        for f in ['val_cal_dict', 'test_cal_dict']:
            dict_list = df[f].tolist()
            dict_list = [x for x in dict_list if x == x]
            print(dict_list)
            dict_df = pd.DataFrame(dict_list)
            df = df.drop([f],axis=1)
            df = pd.concat([df,dict_df],axis=1)
            df[f'{f}_tot_error'] = dict_df.sum(axis=1)
        df['val_loss_final'] = df['val_loss_final'].astype(float)
        df['test_loss_final'] = df['test_loss_final'].astype(float)

    print(df)
    df.to_csv(folder_path+'results.csv')
    df_config = pd.DataFrame(best_config)
    print(df_config)
    df = df.describe()
    df = df.round(3)
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
        print(df)
    df.to_csv(folder_path+'summary.csv')
    df_config.to_csv(folder_path+'config.csv')
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
    parser.add_argument('--split_mode', type=int, nargs='?', default=0, help='split_mode')
    parser.add_argument('--reg_para_a', type=float, nargs='?', default=0., help='reg_para_a')
    parser.add_argument('--reg_para_b', type=float, nargs='?', default=1., help='reg_para_b')
    parser.add_argument('--pos_weight', type=float, nargs='?', default=1., help='pos_weight')
    parser.add_argument('--mu_a', type=float, nargs='?', default=0., help='mu_a')
    parser.add_argument('--mu_b', type=float, nargs='?', default=1., help='mu_b')
    parser.add_argument('--sigma_a', type=float, nargs='?', default=1e-3, help='sigma_a')
    parser.add_argument('--sigma_b', type=float, nargs='?', default=1., help='sigma_b')
    parser.add_argument('--batch_size_a', type=float, nargs='?', default=0., help='batch_size_a')
    parser.add_argument('--batch_size_b', type=float, nargs='?', default=1., help='batch_size_b')
    parser.add_argument('--init_max', type=float, nargs='?', default=1e-1, help='batch_size_b')
    parser.add_argument('--max_lr', type=float, nargs='?', default=1., help='max_lr')
    parser.add_argument('--old_setup', default=False, help='old_setup',type=str2bool, nargs='?')
    parser.add_argument('--latent_scale', default=False, help='old_setup',type=str2bool, nargs='?')
    parser.add_argument('--factorize_latent', default=False, help='fused',type=str2bool, nargs='?')
    parser.add_argument('--hyperits', type=int, nargs='?', default=20, help='hyperits')
    parser.add_argument('--save_path', type=str, nargs='?')
    parser.add_argument('--task', type=str, nargs='?')
    parser.add_argument('--epochs', type=int, nargs='?', default=10, help='epochs')
    parser.add_argument('--bayesian', default=False, help='bayesian_VI',type=str2bool, nargs='?')
    parser.add_argument('--cuda', default=True, help='cuda',type=str2bool, nargs='?')
    parser.add_argument('--full_grad', default=False, help='full_grad',type=str2bool, nargs='?')
    parser.add_argument('--dual', default=False, help='dual',type=str2bool, nargs='?')
    parser.add_argument('--multivariate', default=False, help='dual',type=str2bool, nargs='?')
    parser.add_argument('--sub_epoch_V', type=int, nargs='?', default=100, help='sub_epoch_V')
    parser.add_argument('--seed', type=int, nargs='?', help='seed')
    parser.add_argument('--chunks', type=int, nargs='?', help='chunks',default=1)
    parser.add_argument('--side_info_order', nargs='+', type=int)
    parser.add_argument('--temporal_tag',default=None, nargs='+', type=int)
    parser.add_argument('--architecture', type=int, nargs='?', default=0, help='architecture')
    parser.add_argument('--tensor_name', type=str,default='', nargs='?')
    parser.add_argument('--special_mode', type=int,default=0, nargs='?')
    parser.add_argument('--delete_side_info', type=int,nargs='+')
    return parser

def job_parser_preloaded():
    parser = argparse.ArgumentParser()
    parser.add_argument('--idx', type=int, nargs='?', default=0, help='idx')
    parser.add_argument('--job_path', type=str,nargs='?')
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

def load_side_info(side_info_path):
    side_info = torch.load(side_info_path)
    return side_info

class forecast_dataset(Dataset):
    def __init__(self, tensor_path,T_dim,seed, bs_ratio=1.,periods=7,period_size=24,normalize=False):
        np.random.seed(seed)
        n_last_test = periods*period_size
        self.chunks = 1
        self.ratio = bs_ratio
        self.indices, self.Y_base = torch.load(tensor_path)
        self.indices = self.indices.numpy()
        self.Y_base = self.Y_base.numpy()
        if self.Y_base.ndim==1:
            self.Y_base=self.Y_base.reshape(-1,1)
        self.time_indices = self.indices[:,T_dim]
        max_time = self.time_indices.max()+1
        self.test_begin = max_time-n_last_test
        self.test_times = np.arange(self.test_begin,max_time)
        self.test_periods=np.array_split(self.test_times,periods)
        self.normalize = normalize
        self.true_test_Y = []
        self.pred_test_Y = []
        self.pred_X = []
        self.set_data(0)

    def set_data(self,i):
        test_indices = self.test_periods[i]
        max_train_ind = test_indices.min()
        self.train_indices = np.isin(self.time_indices,np.arange(0,max_train_ind))
        self.val_indices = np.isin(self.time_indices,test_indices)
        self.test_indices = self.val_indices
        self.transformer = StandardScaler()
        self.transformer.fit(self.Y_base[self.train_indices, :])

        for el,name in zip([self.train_indices,self.val_indices,self.test_indices],['tr','v','te']):
            setattr(self,f'X_{name}', torch.from_numpy(self.indices[el,:]).long())
            if self.normalize:
                setattr(self,f'Y_{name}', torch.from_numpy(self.transformer.transform(self.Y_base[el,:])).float())
            else:
                setattr(self,f'Y_{name}', torch.from_numpy(self.Y_base[el,:]).float())
        self.set_mode('train')

    def append_pred_Y(self,pred_Y,true_Y,X_s):
        if self.normalize:
            new_pred_Y = self.transformer.inverse_transform(pred_Y.squeeze().cpu().numpy())
            new_Y = self.transformer.inverse_transform(true_Y.squeeze().cpu().numpy())
        else:
            new_pred_Y = pred_Y.squeeze().cpu().numpy()
            new_Y= true_Y.squeeze().cpu().numpy()
        self.pred_test_Y.append(new_pred_Y)
        self.true_test_Y.append(new_Y)
        self.pred_X.append(X_s.numpy())
    def set_mode(self, mode):
        self.mode = mode
        if mode == 'train':
            self.X = self.X_tr
            self.Y = self.Y_tr
            self.bs = int(round(self.X.shape[0] * self.ratio))
        elif mode == 'val':
            self.ratio = 1.
            self.X = self.X_v
            self.Y = self.Y_v
            self.X_chunks = torch.chunk(self.X, self.chunks)
            self.Y_chunks = torch.chunk(self.Y, self.chunks)

        elif mode == 'test':
            self.ratio = 1.
            self.X = self.X_te
            self.Y = self.Y_te
            self.X_chunks = torch.chunk(self.X, self.chunks)
            self.Y_chunks = torch.chunk(self.Y, self.chunks)

    def get_batch(self):
        if self.ratio == 1.:
            return self.X, self.Y
        else:
            i_s = np.random.randint(0, self.X.shape[0] - 1 - self.bs)
            return self.X[i_s:i_s + self.bs, :], self.Y[i_s:i_s + self.bs]


    def get_chunk(self, i):
        return self.X_chunks[i], self.Y_chunks[i]


    def __len__(self):
        return self.X.shape[0]


    def __getitem__(self, idx):
        return self.X[idx, :], self.Y[idx]

class tensor_dataset(Dataset):
    def __init__(self, tensor_path, seed, bs_ratio=1., split_mode=0,normalize=False):
        if split_mode==0:
            test_size = 0.2
            val_size = 0.25
        if split_mode == 1:
            test_size = 0.2
            val_size = 0.2
        if split_mode ==2:
            test_size = 0.1
            val_size = 0.05
        if split_mode ==3:
            test_size = 0.1
            val_size = 0.1
        self.normalize = normalize
        self.chunks = 1
        self.ratio = bs_ratio
        self.indices,self.Y = torch.load(tensor_path)
        #Make a comment on some subleties, that this is akin to forecasting issue...
        # if tensor_path=='CCDS_data/all_data.pt':
        #     self.location_indices= self.indices[:, 1]
        #     max_ind = self.location_indices.max().item()
        #     unique_indices = np.array(list(range(max_ind)))
        #     val_indices = np.random.choice(unique_indices,round(max_ind*0.1)+1,replace=False)
        #     train_indices = np.setdiff1d(unique_indices,val_indices)
        #     self.train_indices = np.isin(self.location_indices,train_indices )
        #     self.val_indices = np.isin(self.location_indices, val_indices)
        #     self.X_train = self.indices[self.train_indices,:].numpy()
        #     self.Y_train = self.Y[self.train_indices].numpy()
        #     self.X_val = self.indices[self.val_indices,:].numpy()
        #     self.Y_val = self.Y[self.val_indices].numpy()
        #     self.Y_test = self.Y_val
        #     self.X_test = self.X_val
        # else:
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(self.indices.numpy(),
                                                                                self.Y.numpy(),
                                                                                test_size=test_size,
                                                                                random_state=seed
                                                                                )
        self.X_train, self.X_val, self.Y_train, self.Y_val = train_test_split(self.X_train,
                                                                              self.Y_train,
                                                                              test_size=val_size,
                                                                              random_state=seed)

        if self.normalize:
            self.transformer = StandardScaler()
            self.Y_train = self.transformer.fit_transform(self.Y_train.reshape(-1,1))
            self.Y_val = self.transformer.transform(self.Y_val.reshape(-1,1))
            self.Y_test = self.transformer.transform(self.Y_test.reshape(-1,1))

        self.X_tr = torch.from_numpy(self.X_train).long()
        self.Y_tr = torch.from_numpy(self.Y_train.squeeze()).float()
        self.X_v = torch.from_numpy(self.X_val).long()
        self.Y_v = torch.from_numpy(self.Y_val.squeeze()).float()
        self.X_te = torch.from_numpy(self.X_test).long()
        self.Y_te = torch.from_numpy(self.Y_test.squeeze()).float()
        self.set_mode('train')

    def set_mode(self,mode):
        self.mode = mode
        if mode == 'train':
            self.X = self.X_tr
            self.Y = self.Y_tr
            self.bs = int(round(self.X.shape[0] * self.ratio))
        elif mode == 'val':
            self.ratio = 1.
            self.X = self.X_v
            self.Y = self.Y_v
            self.X_chunks = torch.chunk(self.X,self.chunks)
            self.Y_chunks = torch.chunk(self.Y,self.chunks)

        elif mode == 'test':
            self.ratio = 1.
            self.X = self.X_te
            self.Y = self.Y_te
            self.X_chunks = torch.chunk(self.X,self.chunks)
            self.Y_chunks = torch.chunk(self.Y,self.chunks)

    def get_batch(self):
        if self.ratio==1.:
            return self.X,self.Y
        else:
            i_s = np.random.randint(0,self.X.shape[0]-1-self.bs)
            return self.X[i_s:i_s+self.bs, :], self.Y[i_s:i_s+self.bs]

    def get_chunk(self,i):
        return self.X_chunks[i],self.Y_chunks[i]
    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx,:],self.Y[idx]

class chunk_iterator():
    def __init__(self,X,y,shuffle,batch_size):
        self.X = X
        self.y = y
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.n = self.X.shape[0]
        self.chunks=self.n//batch_size+1
        self.perm = torch.randperm(self.n)
        if self.shuffle:
            self.X = self.X[self.perm,:]
            self.y = self.y[self.perm]
        self._index = 0
        self.it_X = torch.chunk(self.X,self.chunks)
        self.it_y = torch.chunk(self.y,self.chunks)
        self.true_chunks = len(self.it_X)

    def __next__(self):
        ''''Returns the next value from team object's lists '''
        if self._index < self.true_chunks:
            result = (self.it_X[self._index],self.it_y[self._index],)
            self._index += 1
            return result
        # End of Iteration
        raise StopIteration

    def __len__(self):
        return len(self.it_X)

class super_fast_iterator():
    def __init__(self,X,y,batch_size):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.n = self.X.shape[0]
        self.chunks=100#self.n//batch_size+1
        self._index = 0
        self.rand_range = self.n - self.batch_size - 1
        if self.rand_range<0:
            self.rand_range=1

    def __next__(self):
        ''''Returns the next value from team object's lists '''
        if self._index < self.chunks:
            i = np.random.randint(0, self.rand_range)
            i_end = i+self.batch_size
            result = (self.X[i:i_end,:],self.y[i:i_end])
            self._index += 1
            return result
        # End of Iteration
        raise StopIteration

class custom_dataloader():
    def __init__(self,dataset,bs_ratio,shuffle=False):
        self.dataset = dataset
        self.bs_ratio = bs_ratio
        self.batch_size = int(round(self.dataset.X.shape[0] * bs_ratio))
        self.shuffle = shuffle
        self.n = self.dataset.X.shape[0]
        self.len=self.n//self.batch_size+1
    def __iter__(self):
        if self.dataset.mode=='train':
            self.batch_size = int(round(self.dataset.X.shape[0] * self.bs_ratio))
        else:
            self.batch_size = self.dataset.X.shape[0]//5
        return chunk_iterator(X =self.dataset.X,
                              y = self.dataset.Y,
                              shuffle = self.shuffle,
                              batch_size=self.batch_size)
    def __len__(self):
        if self.dataset.mode=='train':
            self.batch_size = int(round(self.dataset.X.shape[0] * self.bs_ratio))
        else:
            self.batch_size = self.dataset.X.shape[0]//5

        return chunk_iterator(X =self.dataset.X,
                              y = self.dataset.Y,
                              shuffle = self.shuffle,
                              batch_size=self.batch_size).chunks



def get_dataloader_tensor(tensor_path, seed, bs_ratio, split_mode, forecast=False, T_dim=0, normalize=False, periods=7,
                          period_size=24):
    if forecast:
        ds = forecast_dataset(tensor_path=tensor_path, seed=seed, bs_ratio=bs_ratio, periods=periods,period_size=period_size,T_dim=T_dim,normalize=normalize)
    else:
        ds = tensor_dataset(tensor_path, seed, bs_ratio=bs_ratio, split_mode=split_mode,normalize=normalize)
    dat = custom_dataloader(dataset=ds,bs_ratio=bs_ratio,shuffle=True)
    return dat

