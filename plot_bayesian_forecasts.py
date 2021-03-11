import pandas as pd
import torch
import pickle
from calculate_overall_score_forecast import *
folder_2 = 'jobs_traffic_baysian_WLR_2'
folder = f'{folder_2}_results'
i = 6
job_ind = f'job_{i}'


def get_exact_preds(j_tmp,job_ind,best_ind,i):
    model_info = torch.load(f'{folder}/{job_ind}/bayesian_0_model_hyperit={best_ind + 1}.pt')
    j_tmp.load_dumped_model(best_ind + 1)
    j_tmp.model.turn_on_all()
    j_tmp.model.to(j_tmp.device)
    j_tmp.init_dataloader(0.01)
    j_tmp.dataloader.dataset.set_data(i)
    j_tmp.dataloader.dataset.set_mode('test')
    with torch.no_grad():
        all_samples = []
        Y = []
        all_X = []
        # mean_preds = []
        for i, (X, y) in enumerate(j_tmp.dataloader):
            if j_tmp.train_config['cuda']:
                X = X.to(j_tmp.device)
                y = y.to(j_tmp.device)
            _y_preds = []
            for j in range(100):
                _y_pred_sample, KL = j_tmp.model.sample(X)
                if j_tmp.normalize_Y:
                    _y_pred_sample, y_copy = j_tmp.inverse_transform(_y_pred_sample, y)
                else:
                    _y_pred_sample = _y_pred_sample.cpu().numpy()
                    y_copy = y.cpu().numpy()
                if not j_tmp.task == 'regression':
                    _y_pred_sample = torch.sigmoid(_y_pred_sample)
                _y_preds.append(_y_pred_sample)
            all_X.append(X.cpu().numpy())
            Y.append(y_copy)
            y_sample = np.stack(_y_preds, axis=1)
            all_samples.append(y_sample)
            # mean_preds.append(j_tmp.dataloader.dataset.transformer.inverse_transform(j_tmp.model.mean_forward(X).cpu().numpy()))
    Y_preds = np.concatenate(all_samples, axis=0)
    Y_true =  np.concatenate(Y,axis=0)
    X_all = np.concatenate(all_X,axis=0)
    # mean = np.concatenate(mean_preds,axis=0)
    return X_all,Y_preds,Y_true#,mean


if __name__ == '__main__':
    trials = pickle.load(open(folder + f'/{job_ind}/bayesian_0.p', "rb"))
    best_trial = sorted(trials.trials, key=lambda x: x['result']['test_loss'], reverse=True)[0]
    best_tid = best_trial['tid']
    key_init = load_obj(f'{job_ind}.pkl', f"{folder_2}/")
    j_tmp = job_object(key_init)
    j_tmp.save_path = f'{folder}/{job_ind}'
    X,preds,y_t = get_exact_preds(j_tmp,job_ind,best_tid,i)
    mean  = np.mean(preds,axis=1)
    std = preds.std(axis=1)
    slice_indices = [100]
    mask = np.isin(X[:,1],slice_indices)
    y_true = y_t[mask]
    mean = mean[mask]
    std = std[mask]
    x_subset = X[mask,:]
    df = pd.DataFrame(np.concatenate([x_subset,y_true[:,np.newaxis],mean[:,np.newaxis],std[:,np.newaxis]],axis=1))
    df = df.sort_values(by=[1, 0])

    plt.plot(df[0], df[2], label='Ground truth')
    plt.plot(df[0], df[3], label='Predictions')
    plt.fill_between(df[0], df[3]-df[4], df[3]+df[4], color='b', alpha=0.1)
    plt.savefig("bayesian_plot_test.png")
    plt.clf()