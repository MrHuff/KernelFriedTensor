import pandas as pd
from generate_parameters import load_obj
from KFT.job_utils import *
from matplotlib import pyplot as plt
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 25)
# folder_2 = "jobs_CCDS_side_info_WLR_2"
folder_2 = "jobs_traffic_3"
folder = f"{folder_2}_results"

def get_best(df,fold_idx,sort_on):
    subdf = df[df['fold']==fold_idx].sort_values(sort_on,ascending=True)
    best = subdf.iloc[0,:]
    return best['job_ind'],best['file_name']

def get_errors_np(preds_cat,Y_cat):
    diff = preds_cat-Y_cat
    yabs_mean = np.abs(Y_cat).mean()
    mean_abs_err = np.abs(diff).mean()
    mse = np.mean(diff**2)
    NRSME = mse**0.5/yabs_mean
    ND = mean_abs_err/yabs_mean
    print('RMSE',mse**0.5)
    print('NRMSE',NRSME) #THIS IS OK ALREADY
    print('ND',ND)
    return mse**0.5,NRSME,ND

def get_exact_preds(j_tmp,job_ind,sort_on,i):
    d = pd.read_csv(f'{folder}/{job_ind}/test_df.csv', index_col=0)
    print(d)
    d = d.sort_values(sort_on)
    best_ind = d.index.values.astype(int)[0]
    model_info = torch.load(f'{folder}/{job_ind}/frequentist_0_model_hyperit={best_ind + 1}.pt')
    j_tmp.init(model_info['parameters'])
    j_tmp.load_dumped_model(best_ind + 1)
    j_tmp.model.turn_on_all()
    j_tmp.model.to(j_tmp.device)
    j_tmp.init_dataloader(0.01)
    j_tmp.dataloader.dataset.set_data(i)
    j_tmp.dataloader.dataset.set_mode('test')
    with torch.no_grad():
        total_loss, Y, y_preds, Xs = j_tmp.get_preds()
        y_preds, Y = j_tmp.inverse_transform(y_preds, Y)
    rmse,nrsme,ND =  get_errors_np(y_preds, Y)
    return y_preds,Y,rmse,nrsme,ND,Xs

def forecast_plot_traffic(X, Y, preds, fold_idx):
    X_cpu = X.cpu().numpy()
    Y_cpu = Y
    preds = preds
    slice_indices= [0,1,2,3,4]
    mask = np.isin(X_cpu[:,1],[0,1,2,3,4])
    y_true = Y[mask]
    preds_ = preds[mask]
    x_subset = X_cpu[mask,:]
    df = pd.DataFrame(np.concatenate([x_subset,y_true[:,np.newaxis],preds_[:,np.newaxis]],axis=1))
    df = df.sort_values(by=[1, 0])
    for i in slice_indices:
        subset = df[df[1]==i]
        plt.plot(subset[0],subset[2],'-.',label='True values',color='b')
        plt.plot(subset[0],subset[3],'-',label='Forecasts',color='b')
        plt.xlabel('Time index')
        plt.ylabel('Value')
        plt.legend()
        plt.savefig(f'traffic_forecast_{fold_idx}.png')
        plt.clf()










if __name__ == '__main__':
    sort_on = 'RMSE'
    folds_nr = 5
    df = pd.read_csv(f"analysis_{folder_2}.csv",index_col=0)
    Y_cat = []
    preds_cat = []

    # Get the job dict
    # init the job_object
    # load val_df.csv
    # sort on appropriate thing to get index ND or NRSME
    # move to cuda and no grad.
    # Also get the dataloader
    # Add forecast plots roflmao
    key_init = load_obj('job_0.pkl', f"{folder_2}/")
    j_tmp = job_object(key_init)
    df_metrics = []
    X = []
    for i in range(folds_nr):
        job_ind,file_name = get_best(df,i,sort_on)
        key_load = load_obj(file_name,f"{folder_2}/")
        j_tmp.save_path =  f'{folder}/{job_ind}'
        y_preds,Y,rmse,nrsme,ND,Xs = get_exact_preds(j_tmp,job_ind,sort_on,i)

        forecast_plot_traffic(Xs, Y, y_preds,i)

        Y_cat.append(Y)
        preds_cat.append(y_preds)
        df_metrics.append([rmse,nrsme,ND])
    preds_cat=np.concatenate(preds_cat)
    Y_cat = np.concatenate(Y_cat)
    get_errors_np(preds_cat, Y_cat)
    df_metrics = pd.DataFrame(df_metrics,columns=['RSME','NRSME','ND'])
    print(df_metrics.describe())



