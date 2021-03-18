from plot_bayesian_forecasts import *
import torch
import pandas as pd
pd.set_option('display.max_columns', None)  # or 1000
pd.set_option('display.max_rows', None)  # or 1000
pd.set_option('display.max_colwidth', None)  # or 199
def get_likelihood(j_tmp,best_ind):
    j_tmp.load_dumped_model(best_ind + 1)
    j_tmp.model.turn_on_all()
    j_tmp.model.to(j_tmp.device)
    j_tmp.init_dataloader(0.01)
    total_cal_error_test, test_cal_dict, predictions, test_likelihood, test_ELBO = j_tmp.calculate_calibration(
        mode='test', task=j_tmp.train_config['task'])
    return test_ELBO,test_likelihood

def find_p_file(path):
    print(path)
    files = os.listdir(path)
    for el in files:
        if el[-2:]=='.p':
            return el,int(el.split('_')[-1].split('.')[0])

def save_likelihood_metrics(folder,folder_2,job_indices,dataset_name,model_name):
    columns = ['dataset', 'model', '$\Xi_{0.05}$', '$\Xi_{0.15}$', '$\Xi_{0.25}$', '$\Xi_{0.35}$', '$\Xi_{0.45}$',
               '$\Xi$', '$R^2$', '$\eta_{\text{criteria}}$', 'ELBO', 'log-likelihood']
    trial_list = []
    for job_ind in job_indices:
        path_a = folder + f'/job_{job_ind}'
        fname,j_seed = find_p_file(path_a)
        trials = pickle.load(open(folder + f'/job_{job_ind}/{fname}', "rb"))
        best_trial = sorted(trials.trials, key=lambda x: x['result']['test_loss_final']['R2'], reverse=True)[0]
        best_tid = best_trial['tid']
        print(best_trial)
        key_init = load_obj(f'job_{j_seed}.pkl', f"{folder_2}/")
        # if ('test_ELBO' in best_trial['result'].keys()) and ('test_likelihood' in best_trial['result'].keys()):
        #     ELBO = best_trial['result']['test_ELBO']
        #     likelihood = best_trial['result']['test_likelihood']
        # else:
        j_tmp = job_object(key_init)
        j_tmp.save_path = f'{folder}/job_{job_ind}'
        ELBO,likelihood = get_likelihood(j_tmp,best_tid)
        best_trial['test_ELBO'] = ELBO
        best_trial['test_likelihood'] = likelihood
        trial_list.append([dataset_name,model_name,
                           best_trial['result']['test_cal_dict'][5],
                           best_trial['result']['test_cal_dict'][15],
                           best_trial['result']['test_cal_dict'][25],
                           best_trial['result']['test_cal_dict'][35],
                           best_trial['result']['test_cal_dict'][45],
                           best_trial['result']['test_loss']+best_trial['result']['test_loss_final']['R2'],
                           best_trial['result']['test_loss_final']['R2'],
                           best_trial['result']['test_loss'],
                           best_trial['test_ELBO'],
                           best_trial['test_likelihood'],
                           ])
        df = pd.DataFrame(trial_list,columns=columns)
        df.to_csv(folder+'/all_results.csv')


if __name__ == '__main__':
    columns = ['dataset', 'model', '$\Xi_{0.05}$', '$\Xi_{0.15}$', '$\Xi_{0.25}$', '$\Xi_{0.35}$', '$\Xi_{0.45}$',
               '$\Xi$', '$R^2$', '$\eta_{\text{criteria}}$', 'ELBO', 'log-likelihood']
    # tex_final_name = 'retail_bayesian_results'
    # folder_2_list = [
    #                  'retail_20_bayesian_dual_univariate',
    #                  'retail_20_bayesian_dual_multivariate',
    #                  'retail_20_bayesian_dual_univariate_LS',
    #                  'retail_20_bayesian_dual_multivariate_LS',
    #                         'retail_benchmark_bayesian',
    #                  ]
    # model_name = [r'\makecell{2-way, WLR\\Dual, Univ.}',
    #               r'\makecell{2-way, WLR\\Dual, Multv.}',
    #               r'\makecell{2-way, LS\\Dual, Univ.}',
    #               r'\makecell{2-way, LS\\Dual, Multv.}',
    #               r'\makecell{2-way \\ No Side Info}',
    #               ]
    # dataset_name = ['Retail']*len(folder_2_list)

    # tex_final_name = 'alcohol_bayesian_results'
    # folder_2_list = [
    #                 'alcohol_bayesian_dual_univariate',
    #                 'alcohol_bayesian_dual_multivariate',
    #                  'alcohol_bayesian_dual_univariate_LS',
    #                  'alcohol_bayesian_dual_multivariate_LS',
    #                  'alchohol_bayesian_benchmark',
    # ]
    # model_name = [r'\makecell{P-way, WLR\\Dual, Univ.}',
    #               r'\makecell{P-way, WLR\\Dual, Multv.}',
    #               r'\makecell{P-way, LS\\Dual, Univ.}',
    #               r'\makecell{P-way, LS\\Dual, Multv.}',
    #               r'\makecell{P-way \\ No Side Info}',
    #               ]
    # dataset_name = ['Alcohol']*len(folder_2_list)

    tex_final_name = 'movielens_bayesian_results'
    folder_2_list = [
                    'movielens_20_bayesian_dual_univariate',
                    # 'movielens_20_bayesian_dual_multivariate',
                     'movielens_20_bayesian_dual_univariate_LS',
                     'movielens_20_bayesian_dual_multivariate_LS',
                     'movielens_20_benchmark_bayesian',
    ]
    model_name = [
                r'\makecell{P-way, WLR\\Dual, Univ.}',
                  # r'\makecell{P-way, WLR\\Dual, Multv.}',
                  r'\makecell{P-way, LS\\Dual, Univ.}',
                  r'\makecell{P-way, LS\\Dual, Multv.}',
                  r'\makecell{P-way \\ No Side Info}',
                  ]
    dataset_name = ['Movielens-20m']*len(folder_2_list)
    #

    # tex_final_name = 'traffic_bayesian'
    # folder_2_list = [
    #                  'jobs_traffic_baysian_WLR_3',
    #                  'jobs_traffic_baysian_LS_3',
    #                  'jobs_CCDS_baysian_WLR_3',
    #                  'jobs_CCDS_baysian_LS_3',
    #                  ]
    # model_name = [r'\makecell{P-way, WLR\\Dual, Univ.}',
    #               r'\makecell{P-way, LS\\Dual, Univ.}',
    #               r'\makecell{P-way, WLR\\Dual, Univ.}',
    #               r'\makecell{P-way, LS\\Dual, Univ.}',
    #               ]
    # dataset_name = ['Traffic','Traffic','CCDS','CCDS']

    job_indices = [0, 1, 2]
    summary_df_list = []
    for i, folder_2 in enumerate(folder_2_list):
        folder = f'{folder_2}_results'
        save_likelihood_metrics(folder,folder_2,job_indices,dataset_name[i],model_name[i])
        if os.path.exists(folder+'/all_results.csv'):
            df = pd.read_csv(folder+'/all_results.csv',index_col=0)
            summary = df.describe().transpose()
            tmp_df = summary[['mean','std']]
            tmp_df['latex_col'] = tmp_df['mean'].apply(lambda x: '\makecell{$'+str(round(x,3))+'$') + tmp_df['std'].apply(lambda x: r'\\ $\pm '+str(round(x,3))+'$}')
            tmp_df = tmp_df['latex_col'].tolist()
            tmp_df.insert(0,dataset_name[i])
            tmp_df.insert(0,model_name[i])
            summary_df_list.append(tmp_df)
    df = pd.DataFrame(summary_df_list,columns=columns)
    print(df)
    df.to_latex(f'{tex_final_name}.tex',escape=False,index=None)
