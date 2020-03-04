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

def plot_VI(save_path,idx_list,seed=None):
    if seed is None:
        try:
            predictions = pd.read_hdf(save_path+f'VI_predictions.h5')
        except:
            predictions = pd.read_parquet(save_path+f'VI_predictions')
    else:
        try:
            predictions = pd.read_hdf(save_path+f'VI_predictions_{seed}.h5')
        except:
            predictions = pd.read_parquet(save_path+f'VI_predictions_{seed}')
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
        ax[i].set_title(r'$\alpha$ = {}%'.format(rate))
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

def get_test_errors(folder_path,metric_name,data_path,reverse=False,):
    trial_files = os.listdir(folder_path)
    print(trial_files)
    metrics = []
    for i in range(1,6):
        dataloader = get_dataloader_tensor(data_path, seed=i, mode='test',
                                           bs_ratio=1.0)
        var_Y_test = dataloader.Y_te.var().numpy()
        for el in trial_files:
            if '.p' == el[-2:] and (f'_{i}.p' in el) :
                trials = pickle.load(open(folder_path + el, "rb"))
                r_2 = abs(sorted(trials.results, key=lambda x: x[metric_name], reverse=reverse)[0][metric_name])
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
    parser.add_argument('--temporal_tag', nargs='+', type=int)
    parser.add_argument('--architecture', type=int, nargs='?', default=0, help='architecture')
    parser.add_argument('--L', type=int, nargs='?', default=2, help='L')
    parser.add_argument('--tensor_name', type=str,default='', nargs='?')
    parser.add_argument('--side_info_name', type=str,nargs='+')
    parser.add_argument('--special_mode', type=int,default=0, nargs='?')
    parser.add_argument('--delete_side_info', type=int,nargs='+')
    parser.add_argument('--kernels', type=str,nargs='+')
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
        self.chunks = 1
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
        self.X_tr = torch.from_numpy(self.X_train).long()
        self.Y_tr = torch.from_numpy(self.Y_train).float()
        self.X_v = torch.from_numpy(self.X_val).long()
        self.Y_v = torch.from_numpy(self.Y_val).float()
        self.X_te = torch.from_numpy(self.X_test).long()
        self.Y_te = torch.from_numpy(self.Y_test).float()
        self.set_mode(mode)

    def set_mode(self,mode):
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

def get_dataloader_tensor(tensor_path,seed,mode,bs_ratio):
    ds = tensor_dataset(tensor_path,seed,mode,bs_ratio=bs_ratio)
    return ds

