import warnings
from KFT.util import job_parser
from KFT.job_utils import run_job_func

if __name__ == '__main__':
    warnings.simplefilter("ignore")
    args = vars(job_parser().parse_args())
    run_job_func(args)

