from KFT.util import post_process
import os

if __name__ == '__main__':
    dirs = os.listdir('./')
    for d in dirs:
        if 'report_movielens_1m_neural_beat_variant' in d:
            try:
                post_process(f'./{d}/','test_R2',reverse=False)
            except:
                try:
                    post_process(f'./{d}/','test_loss',reverse=False)
                except:
                    continue