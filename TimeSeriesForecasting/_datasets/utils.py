import pathlib
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')


def normalize(data):
    """
    Execute min-max normalization to data

    Args:
        data: target data

    Returns:
        norm_data: normalized data

    """
    numerator = data - np.min(data, 0)
    denominator = np.max(data, 0) - np.min(data, 0)
    norm_data = numerator / (denominator + 1e-7)
    return norm_data

def timeDF_normalize(df):

    df['date'] = df['date']+19000000
    df['date'] = pd.to_datetime(df['date'].astype('str'))
    df['seconds'] = df['date'].astype(np.int64)// 10 ** 9

    min_sec = df['seconds'].min()
    df['seconds'] -= min_sec

    min_sec, max_sec = df['seconds'].min(), df['seconds'].max()

    df['seconds'] = (df['seconds'] - min_sec) / (max_sec - min_sec)

    return df['seconds']


def load_processed_data(folder_loc, train_seq, y_seq, NCDE, train_split, valid_split, test_split):
    here = pathlib.Path(__file__).resolve().parent

    dataset_name = folder_loc.split('/')[1]

    trainX_loc = f"{dataset_name}_trainX_{train_seq}_{y_seq}_{NCDE}_{train_split}_{valid_split}_{test_split}.npy"
    trainy_loc = f"{dataset_name}_trainy_{train_seq}_{y_seq}_{NCDE}_{train_split}_{valid_split}_{test_split}.npy"
    validX_loc = f"{dataset_name}_validX_{train_seq}_{y_seq}_{NCDE}_{train_split}_{valid_split}_{test_split}.npy"
    validy_loc = f"{dataset_name}_validy_{train_seq}_{y_seq}_{NCDE}_{train_split}_{valid_split}_{test_split}.npy"
    testX_loc = f"{dataset_name}_testX_{train_seq}_{y_seq}_{NCDE}_{train_split}_{valid_split}_{test_split}.npy"
    testy_loc = f"{dataset_name}_testy_{train_seq}_{y_seq}_{NCDE}_{train_split}_{valid_split}_{test_split}.npy"

    trainX, trainy = np.load(here / folder_loc / trainX_loc), np.load(here / folder_loc / trainy_loc)
    validX, validy = np.load(here / folder_loc / validX_loc), np.load(here / folder_loc / validy_loc)
    testX, testy = np.load(here / folder_loc / testX_loc), np.load(here / folder_loc / testy_loc)

    return trainX, trainy, validX, validy, testX, testy


def save_processed_data(folder_loc, train_seq, y_seq, NCDE, train_split, valid_split, test_split, \
                        trainX, trainy, validX, validy, testX, testy):
    here = pathlib.Path(__file__).resolve().parent

    dataset_name = folder_loc.split('/')[1]

    trainX_loc = f"{dataset_name}_trainX_{train_seq}_{y_seq}_{NCDE}_{train_split}_{valid_split}_{test_split}.npy"
    trainy_loc = f"{dataset_name}_trainy_{train_seq}_{y_seq}_{NCDE}_{train_split}_{valid_split}_{test_split}.npy"
    validX_loc = f"{dataset_name}_validX_{train_seq}_{y_seq}_{NCDE}_{train_split}_{valid_split}_{test_split}.npy"
    validy_loc = f"{dataset_name}_validy_{train_seq}_{y_seq}_{NCDE}_{train_split}_{valid_split}_{test_split}.npy"
    testX_loc = f"{dataset_name}_testX_{train_seq}_{y_seq}_{NCDE}_{train_split}_{valid_split}_{test_split}.npy"
    testy_loc = f"{dataset_name}_testy_{train_seq}_{y_seq}_{NCDE}_{train_split}_{valid_split}_{test_split}.npy"

    np.save(f"{here}/{folder_loc}/{trainX_loc}", trainX)
    np.save(f"{here}/{folder_loc}/{trainy_loc}", trainy)
    np.save(f"{here}/{folder_loc}/{validX_loc}", validX)
    np.save(f"{here}/{folder_loc}/{validy_loc}", validy)
    np.save(f"{here}/{folder_loc}/{testX_loc}", testX)
    np.save(f"{here}/{folder_loc}/{testy_loc}", testy)

    return None
