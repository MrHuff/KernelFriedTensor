from KFT.util import get_test_errors


if __name__ == '__main__':
    data_path = './report_movielens_data_ml-1m/all_data.pt'
    get_test_errors('./report_movielens_1m_job_arch_2_dual_split_1_variant/','test_loss',data_path,split_mode=0,reverse=False)