import re
from sklearn.model_selection import train_test_split
import dask.dataframe as dd
from dask.diagnostics import ProgressBar
import torch
import numpy as np
from sklearn.metrics.regression import mean_squared_error
import  sklearn.metrics as metrics

def get_auc(Y,y_pred):
    fpr, tpr, thresholds = metrics.roc_curve(Y, y_pred, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    return auc
def get_R_square(Y,y_pred):
    mse = mean_squared_error(Y,y_pred)
    var = np.var(Y)
    return -(1-mse/var)


class read_benchmark_data():
    def __init__(self, y_name, path, seed):
        ProgressBar().register()
        self.df = dd.read_parquet(path)
        regex = re.compile(r"\[|\]|<", re.IGNORECASE)
        self.df.columns = [regex.sub("_", col) if any(x in str(col) for x in set(('[', ']', '<'))) else col for col
                           in
                           self.df.columns.values]

        self.Y = self.df[y_name]
        self.X = self.df.drop(y_name, axis=1)
        self.X = self.X.compute()
        self.Y = self.Y.compute()

        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(self.X,
                                                                                self.Y,
                                                                                test_size=0.2,
                                                                                random_state=seed
                                                                                )
        self.X_train, self.X_val, self.Y_train, self.Y_val = train_test_split(self.X_train,
                                                                              self.Y_train,
                                                                              test_size=0.25,
                                                                              random_state=seed
                                                                              )
        self.n = len(self.X_train)
        print("Succesfully loaded data")

    def set_mode(self, mode):
        if mode == 'train':
            self.X = torch.from_numpy(self.X_train).float()
            self.Y = torch.from_numpy(self.Y_train).float()
            self.bs = int(round(self.X.shape[0] * self.ratio))
        elif mode == 'val':
            self.ratio = 1.
            self.X = torch.from_numpy(self.X_val).float()
            self.Y = torch.from_numpy(self.Y_val).float()
            self.X_chunks = torch.chunk(self.X, self.chunks)
            self.Y_chunks = torch.chunk(self.Y, self.chunks)

        elif mode == 'test':
            self.ratio = 1.
            self.X = torch.from_numpy(self.X_test).float()
            self.Y = torch.from_numpy(self.Y_test).float()
            self.X_chunks = torch.chunk(self.X, self.chunks)
            self.Y_chunks = torch.chunk(self.Y, self.chunks)

    def get_batch(self):
        if self.ratio == 1.:
            return self.X, self.Y
        else:
            batch_msk = np.random.choice(self.X.shape[0], self.bs,
                                         replace=False)
            return self.X[batch_msk, :], self.Y[batch_msk]

    def get_chunk(self, i):
        return self.X_chunks[i], self.Y_chunks[i]

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
            return self.X[idx, :], self.Y[idx]

