from KFT.benchmarks.linear_regression import linear_job_class
from KFT.benchmarks.utils import job_parser_FMM_and_linear
if __name__ == '__main__':
    args = vars(job_parser_FMM_and_linear().parse_args())
    l = linear_job_class(seed=args['seed'],y_name=args['y_name'],data_path=args['data_path'],save_path=args['SAVE_PATH'],params=args)
    l.run()
