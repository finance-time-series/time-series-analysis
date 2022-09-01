import warnings
from enum import Enum

import torch
import torchcde
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

warnings.filterwarnings('ignore')


class NCDE_USE_TYPE(Enum):
    NONE = 1  # not use
    ONLY = 2  # only use (coeffs)
    WITH = 3  # use with


def getDataloaders(trainX, trainy, validX, validy, testX, testy, ncde_use_type, device, batch_size, trainF=None, validF=None, testF=None):
    if ncde_use_type == NCDE_USE_TYPE.ONLY:
        # to tensor and device
        trainX, trainy, validX, validy, testX, testy = to_tensor_device(device, trainX, trainy, validX, validy, testX,
                                                                        testy)

        train_coeff = torchcde.hermite_cubic_coefficients_with_backward_differences(trainX)
        valid_coeff = torchcde.hermite_cubic_coefficients_with_backward_differences(validX)
        test_coeff = torchcde.hermite_cubic_coefficients_with_backward_differences(testX)

        train_dataset = TensorDataset(train_coeff, trainy)
        valid_dataset = TensorDataset(valid_coeff, validy)
        test_dataset = TensorDataset(test_coeff, testy)

    elif ncde_use_type == NCDE_USE_TYPE.WITH:

        # to tensor and device
        trainX, trainF, trainy, validX, validF, validy, testX, testF, testy \
            = to_tensor_device(device, trainX, trainF, trainy, validX, validF, validy, testX, testF, testy)

        train_coeff = torchcde.hermite_cubic_coefficients_with_backward_differences(trainX)
        valid_coeff = torchcde.hermite_cubic_coefficients_with_backward_differences(validX)
        test_coeff = torchcde.hermite_cubic_coefficients_with_backward_differences(testX)

        train_dataset = TensorDataset(trainX, train_coeff, trainF, trainy)
        valid_dataset = TensorDataset(validX, valid_coeff, validF, validy)
        test_dataset = TensorDataset(testX, test_coeff, testF, testy)

    elif ncde_use_type == NCDE_USE_TYPE.NONE:
        # to tensor and device
        trainX, trainy, validX, validy, testX, testy = to_tensor_device(device, trainX, trainy, validX, validy, testX,
                                                                        testy)

        train_dataset = TensorDataset(trainX, trainy)
        valid_dataset = TensorDataset(validX, validy)
        test_dataset = TensorDataset(testX, testy)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

    return train_dataloader, valid_dataloader, test_dataloader


def to_tensor_device(device, *dataset):
    return tuple(torch.Tensor(data).to(device) for data in dataset)
