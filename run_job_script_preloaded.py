import warnings
from KFT.util import job_parser_preloaded
from KFT.job_utils import run_job_func
import os

if __name__ == '__main__':
    warnings.simplefilter("ignore")
    args = vars(job_parser_preloaded().parse_args())
    job_path = args['job_path']
    idx = args['idx']
    job_list = os.listdir(job_path)
    job_args = job_list[idx]
    
    run_job_func(job_args)

