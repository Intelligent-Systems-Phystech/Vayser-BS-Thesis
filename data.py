'''
module for data preparation
'''


import pandas as pd
import torch

from sklearn.model_selection import train_test_split
from sklearn import preprocessing

def easy_data_preparation(filename, sep=',', test_size=0.1, cat = None):
    '''
        Preprocessing for standard datasets
        Including:
            1. Filling null values
            2. Labelling
            3. Min-Max scaling
    '''
    data = pd.read_csv(filename, sep = sep)
    data.loc[:,  data.dtypes == 'object'] = data.select_dtypes(['object']).fillna('No information')
    data.loc[:,  data.dtypes == 'int'] = data.select_dtypes(['int']).apply(lambda x: x.fillna(x.mean()))
    data.loc[:,  data.dtypes == 'float'] = data.select_dtypes(['float']).apply(lambda x: x.fillna(x.mean()))
    data.loc[:,  data.dtypes == 'object'] = data.select_dtypes(['object']).apply(lambda x: x.astype('category'))
    data.loc[:,  data.dtypes == 'category'] = data.select_dtypes(['category']).apply(lambda x: x.cat.codes)
    data_values = data.values #returns a numpy array
    min_max_scaler = preprocessing.MinMaxScaler()
    data_scaled = min_max_scaler.fit_transform(data_values)
#     data_scaled = pd.DataFrame(data_scaled, columns = data.columns)
    y = data_scaled[:, -1]
    X = data_scaled[:, :-1]
    if cat:
        y = y[:cat]
        X = X[:cat, :]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    y_train, y_test = y_train.reshape(-1, 1), y_test.reshape(-1, 1)
    return X_train, X_test, y_train, y_test

def get_ctg_data(test_size):
    '''specific dataset'''
    
    ctg = pd.read_excel("CTG.xls", sheet_name = 1, skiprows = [0])
    ctg = ctg.drop(["Unnamed: 9", "Unnamed: 31", "Unnamed: 42", "Unnamed: 44",
                                   "b", "e", "AC", "FM", "UC", "DL", "DS", "DP", "DR", "Nmax", 
                                    "Nzeros", "A", "B", "C", "D", "E", "AD", "DE", "LD", "FS", 
                                    "SUSP", "CLASS", "Tendency", "DS.1", "DP.1"], axis = 1)
    ctg.dropna(inplace = True)
    ctg[ctg["NSP"] == 3] = 2
    ctg = pd.concat((ctg[ctg["NSP"] == 1].sample(500, random_state=1), ctg[ctg["NSP"] == 2]))

    X = ctg.values
    ndim = X.shape[1] - 1
    y = X[:, -1] - 1
    X = X[:, :ndim]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    y_train, y_test = y_train.reshape(-1, 1), y_test.reshape(-1, 1)
    
    return X_train, X_test, y_train, y_test, ndim

def tensor(a, b, c, d, device):
    return torch.tensor(a, dtype = torch.float).to(device), torch.tensor(b, dtype = torch.float).to(device), \
           torch.tensor(c, dtype = torch.float).to(device), torch.tensor(d, dtype = torch.float).to(device)
