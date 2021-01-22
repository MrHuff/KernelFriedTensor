import warnings
from KFT.util import job_parser_preloaded
from KFT.job_utils import run_job_func
from generate_parameters import load_obj
import os

if __name__ == '__main__':
    warnings.simplefilter("ignore")
    args = vars(job_parser_preloaded().parse_args())
    job_path = args['job_path']
    idx = args['idx']
    job_list = os.listdir(job_path)
    print(job_list)
    job_name_args = job_list[idx]
    job_args = load_obj(job_name_args,job_path+'/')
    job_args['save_path'] = job_path+'_results/'+f'job_{idx}'
    run_job_func(job_args)

