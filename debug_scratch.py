import torch
import pandas as pd
pd.set_option("display.max_rows", 100)
pd.set_option('display.max_columns', 500)
from matplotlib import pyplot as plt
import pickle
import pandas as pd
if __name__ == '__main__':
    trials = pickle.load(open('job_dir_traffic/results_traffic_data/frequentist_0.p',
                              "rb"))
    results = trials.results
    val_df=[]
    test_df=[]
    for el in results:
        val_df.append(el['other_test'])
        test_df.append(el['other_val'])
    val_df = pd.DataFrame(val_df)
    test_df = pd.DataFrame(test_df)
    val_df =val_df.sort_values(by=['R2'],ascending=False)
    test_df = test_df.sort_values(by=['R2'],ascending=False)


