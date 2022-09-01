import os
import sys

parent_path = (os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
sys.path.append(parent_path)

import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from _datasets.data_preprocessing import preprocessBank, preprocessTransfer
from config import get_config
import models as models

from utils import getDataloaders, NCDE_USE_TYPE
from torch.utils.tensorboard import SummaryWriter
import math
from sklearn.metrics import mean_absolute_percentage_error

def getModelandDataProcessor(args):
    model_name, data = args.model, args.data

    if data == "bank": dataProcessor = preprocessBank
    elif data == "tran": dataProcessor = preprocessTransfer
    else: print("Wrong dataset input!")

    if model_name == 'ncde':
        isNCDE, typeNCDE = True, NCDE_USE_TYPE.ONLY
        model = models.NeuralCDE
        
    else:
        isNCDE, typeNCDE = False, NCDE_USE_TYPE.NONE
        if model_name == 'rnn':
            model = models.RNNNet
        elif model_name == 'lstm':
            model = models.LSTMNet
        elif model_name == 'gru':
            model = models.GRUNet

    return model, dataProcessor, isNCDE, typeNCDE
        

def train_forecasting(model, dataloader, optimizer, mse_loss, device):
    true_ys, pred_ys = torch.Tensor().to(device), torch.Tensor().to(device)
    for batch in dataloader:
        batch_coeffs, batch_y = batch

        optimizer.zero_grad()
        pred_y = model(batch_coeffs)
        true_ys = torch.concat([true_ys, batch_y])
        pred_ys = torch.concat([pred_ys, pred_y])
        loss = mse_loss(pred_y, batch_y)

        loss.backward()
        optimizer.step()

    mseLoss = mse_loss(pred_ys, true_ys)
    mapeLoss = mean_absolute_percentage_error(true_ys.detach().cpu().numpy(), pred_ys.detach().cpu().numpy()) * 100

    return mseLoss, mapeLoss


def eval_forecasting(model, dataloader, mse_loss, device):
    with torch.no_grad():
        true_ys, pred_ys = torch.Tensor().to(device), torch.Tensor().to(device)

        for batch in dataloader:
            batch_coeffs, batch_y = batch
            pred_y = model(batch_coeffs)
            true_ys = torch.concat([true_ys, batch_y])
            pred_ys = torch.concat([pred_ys, pred_y])

        mseLoss = mse_loss(pred_ys, true_ys)
        mapeLoss = mean_absolute_percentage_error(true_ys.detach().cpu().numpy(), pred_ys.detach().cpu().numpy()) * 100

    return mseLoss, mapeLoss

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":

    # set configuration
    args = get_config()
    print(args)

    # fix randomness
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(args.seed)

    # GPU setting
    gpu = 'cuda:' + args.device
    device = torch.device(gpu)

    tb_train_writer = SummaryWriter(log_dir=f'../_tensorboard/{args.model}/train')
    tb_valid_writer = SummaryWriter(log_dir=f'../_tensorboard/{args.model}/valid')
    tb_test_writer = SummaryWriter(log_dir=f'../_tensorboard/{args.model}/test')

    # get the model and the pre-processing ftn
    trainModel, preprocessData, isNCDE, typeNCDE = getModelandDataProcessor(args)

    trainX, trainy, validX, validy, testX, testy = preprocessData(train_seq = args.train_seq, y_seq = args.y_seq, type = args.type, NCDE = isNCDE)

    train_dataloader, valid_dataloader, test_dataloader = getDataloaders(
        trainX, trainy, validX, validy, testX, testy, ncde_use_type=typeNCDE, device=device, batch_size=args.batch)
    input_size, output_size = trainX.shape[2], 1
    model = trainModel(input_size, args.hidden_size, output_size, args.n_layers, args.hh_size)
    model.to(device)

    print(model)
    print(f"# PARAMS : {count_parameters(model)}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = torch.nn.MSELoss(reduction='mean')

    bestEpoch, bestTrainMSE, bestValidMSE, bestTestMSE = 0, math.inf, math.inf, math.inf

    # Train
    for epoch in range(1, args.epoch + 1):
        if epoch==1:
            torch.cuda.reset_max_memory_allocated(device)
            baseline_memory = torch.cuda.memory_allocated(device)

        model.train()
        train_mse_loss, train_mape_loss = train_forecasting(model, train_dataloader, optimizer, criterion, device)

        model.eval()
        valid_mse_loss, valid_mape_loss = eval_forecasting(model, valid_dataloader, criterion, device)
        test_mse_loss, test_mape_loss = eval_forecasting(model, test_dataloader, criterion, device)

        if epoch==1:
            memory_usage = torch.cuda.max_memory_allocated(device) - baseline_memory
            print(f"Memory_usage:{memory_usage}")

        print('[Epoch: {:3} (MSE)] Train: {:.4f} | Valid : {:.4f} | Test : {:.4f}'.format(epoch, train_mse_loss.item(),
                                                                                          valid_mse_loss.item(),
                                                                                          test_mse_loss.item()))
        print('[Epoch: {:3} (MAPE)] Train: {:.4f} | Valid : {:.4f} | Test : {:.4f}'.format(epoch, train_mape_loss.item(),
                                                                                        valid_mape_loss.item(),
                                                                                        test_mape_loss.item()))

        if bestValidMSE > valid_mse_loss:
            bestEpoch, bestTrainMSE, bestValidMSE, bestTestMSE, bestTrainMAPE, bestValidMAPE, bestTestMAPE =\
                 epoch, train_mse_loss.item(), valid_mse_loss.item(), test_mse_loss.item(), train_mape_loss, valid_mape_loss, test_mape_loss

            print(
                "- Best loss (MSE) update!! Train: {:.4f} | Valid : {:.4f} | Test : {:.4f} at Epoch {:3}".format(bestTrainMSE,
                                                                                                           bestValidMSE,
                                                                                                           bestTestMSE,
                                                                                                           bestEpoch))
            print(
                "- Best loss (MAPE) update!! Train: {:.4f} | Valid : {:.4f} | Test : {:.4f} at Epoch {:3}".format(bestTrainMAPE,
                                                                                                           bestValidMAPE,
                                                                                                           bestTestMAPE,
                                                                                                           bestEpoch))

        tb_train_writer.add_scalar("{} / {} / Loss: mse".format(args.data, args.model), train_mse_loss / len(train_dataloader), epoch)
        tb_valid_writer.add_scalar("{} / {} / Loss: mse".format(args.data, args.model), valid_mse_loss / len(valid_dataloader), epoch)
        tb_test_writer.add_scalar("{} / {} / Loss: mse".format(args.data, args.model), test_mse_loss / len(test_dataloader), epoch)

    print(
        "[Final (BEST MSE)] Train: {:.4f} | Valid : {:.4f} | Test : {:.4f} at Epoch {:3}".format(bestTrainMSE, bestValidMSE,
                                                                                             bestTestMSE, bestEpoch))

    print(
        "[Final (BEST MAPE)] Train: {:.4f} | Valid : {:.4f} | Test : {:.4f} at Epoch {:3}".format(bestTrainMAPE, bestValidMAPE,
                                                                                             bestTestMAPE, bestEpoch))

    tb_train_writer.flush()
    tb_train_writer.close()
    tb_valid_writer.flush()
    tb_valid_writer.close()
    tb_test_writer.flush()
    tb_test_writer.close()
