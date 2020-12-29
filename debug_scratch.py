import torch
import pandas as pd
pd.set_option("display.max_rows", 100)
pd.set_option('display.max_columns', 500)
from matplotlib import pyplot as plt
if __name__ == '__main__':
    test  = torch.load('test_run/bayesian_1_model_hyperit=1.pt')
    # print(test)
    predictions = pd.read_parquet('test_run/VI_predictions_1')
    print(predictions)

    plt.hist(predictions['mean'],1000)
    plt.show()
    plt.hist(predictions['y_true'],1000)
    plt.show()

