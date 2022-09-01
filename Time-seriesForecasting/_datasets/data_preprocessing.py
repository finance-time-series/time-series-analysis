import os
import pathlib
import sys

import _datasets.utils as utils
import pandas as pd
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

def preprocessBank(train_seq, y_seq, type, NCDE=True, train_split=0.7, valid_split=0.15, test_split=0.15):
    here = pathlib.Path(__file__).resolve().parent

    if type == 'elec': csv = 'ECOS_전자금융공동망_0104_2206'
    elif type == 'suburb': csv = 'ECOS_지방은행공동망_9706_2206'
    elif type == 'other': csv = 'ECOS_타행환공동망_9001_2206'
    elif type == 'cd': csv = 'ECOS_CD공동망_9001_2206'
    elif type == 'cms': csv = 'ECOS_CMS공동망_9606_2206'

    df = pd.read_csv(here / f'{csv}.csv')
    df.rename(columns = {df.columns[0]:"date", df.columns[1]:"number", df.columns[2]:"amount"}, inplace = True)
    df = df[['number','amount']]
    df = df.replace(',', '', regex=True)
    df = df.apply(pd.to_numeric)
    df['amount'] /= 1000
    df = df.reset_index()
    df.rename(columns = {'index':'date'}, inplace = True)

    data_X = df[['number', 'amount']]
    dtime = np.expand_dims(utils.normalize(df['date'].values), axis=1)

    data_y = np.expand_dims(df['amount'].values, axis=1)
    data_X = np.concatenate([data_X, data_y], axis=1)

    full_seq = train_seq + y_seq
    seq_data = None
    for i in range(len(data_X) - full_seq + 1):
        if NCDE:
            new_seq = np.concatenate([dtime[i:i + full_seq], data_X[i:i + full_seq]], axis=1)
        else:
            new_seq = data_X[i:i + full_seq]
        new_seq = np.expand_dims(new_seq, axis=0)

        if seq_data is None:
            seq_data = new_seq
        else:
            seq_data = np.concatenate([seq_data, new_seq], axis=0)

    total_num = seq_data.shape[0] 
    train_len, valid_len = int(total_num * train_split), int(total_num * valid_split) + int(total_num * train_split)
    trainX, trainy = seq_data[:train_len, :train_seq, :-1], seq_data[:train_len, train_seq:, -1]
    validX, validy = seq_data[train_len:valid_len, :train_seq, :-1], seq_data[train_len:valid_len, train_seq:, -1]
    testX, testy = seq_data[valid_len:, :train_seq, :-1], seq_data[valid_len:, train_seq:, -1]

    return trainX, trainy, validX, validy, testX, testy

def preprocessTransfer(train_seq, y_seq, type, NCDE=True, train_split=0.7, valid_split=0.15, test_split=0.15):
    here = pathlib.Path(__file__).resolve().parent

    folder_loc = '_processed_data/Transfer'
    checkFile_loc = f"Transfer_trainX_{train_seq}_{y_seq}_{NCDE}_{train_split}_{valid_split}_{test_split}.npy"

    if os.path.exists(here / folder_loc / checkFile_loc):
        trainX, trainy, validX, validy, testX, testy = utils.load_processed_data(folder_loc, train_seq, y_seq, NCDE, \
                                                                                 train_split, valid_split, test_split)

    else:
        df = pd.read_csv(here / f'transfer.csv')
        normTime = utils.timeDF_normalize(df)
        df['seconds'] = normTime
        df['balance']/=1000

        full_seq = train_seq + y_seq
        timeLst = df.groupby('account_id')['seconds'].apply(list)
        balanceLst = df.groupby('account_id')['balance'].apply(list)

        timeLst = timeLst[balanceLst.map(len)>full_seq].reset_index(drop = True)
        balanceLst = balanceLst[balanceLst.map(len)>full_seq].reset_index(drop = True)

        timeLsts, balanceLsts = timeLst.values, balanceLst.values
        seq_data = None
        for i in range(len(timeLsts)):
            ti, am = timeLsts[i], balanceLsts[i]
            j = 0
            while True:
                if (j+1)*full_seq > len(am): break
                else:
                    if NCDE:
                        am_npy = np.expand_dims(np.expand_dims(np.array(am[j*full_seq:(j+1)*full_seq]), axis=0), axis=2)
                        ti_npy = np.expand_dims(np.expand_dims(np.array(ti[j*full_seq:(j+1)*full_seq]), axis=0), axis=2)
                        new_seq = np.concatenate([ti_npy, am_npy], axis=2)
                    else:
                        new_seq = np.expand_dims(np.expand_dims(np.array(am[j*full_seq:(j+1)*full_seq]), axis=0), axis=2)

                    if seq_data is None:
                        seq_data = new_seq
                    else:
                        seq_data = np.concatenate([seq_data, new_seq], axis=0)
                    
                    j+=1

        total_num = seq_data.shape[0] 
        train_len, valid_len = int(total_num * train_split), int(total_num * valid_split) + int(total_num * train_split)
        trainX, trainy = seq_data[:train_len, :train_seq, :], seq_data[:train_len, train_seq:, -1]
        validX, validy = seq_data[train_len:valid_len, :train_seq, :], seq_data[train_len:valid_len, train_seq:, -1]
        testX, testy = seq_data[valid_len:, :train_seq, :], seq_data[valid_len:, train_seq:, -1]

        utils.save_processed_data(folder_loc, train_seq, y_seq, NCDE, train_split, valid_split, test_split, \
                            trainX, trainy, validX, validy, testX, testy)

    return trainX, trainy, validX, validy, testX, testy