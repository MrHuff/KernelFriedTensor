from KFT.util import get_test_errors


if __name__ == '__main__':
    data_path = './public_movielens_data/all_data.pt'
    get_test_errors('./movielens_job_arch_0/','test_R2',data_path)