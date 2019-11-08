import warnings
from KFT.util import job_parser
from KFT.job_utils import run_job_func

if __name__ == '__main__':
    with warnings.catch_warnings(): #There are some autograd issues fyi, might wanna fix it sooner or later
        warnings.simplefilter("ignore")
        args = vars(job_parser().parse_args())
        run_job_func(args)

