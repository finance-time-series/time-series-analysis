import matplotlib.pyplot as plt
import numpy as np

types = ["cd","cms","elec","other","suburb"]
for ty in types:
    rnn_pred = np.load(f'./vis/bank_{ty}/rnn/best_test_predy.npy')
    rnn_true = np.load(f'./vis/bank_{ty}/rnn/best_test_truey.npy')
    lstm_pred = np.load(f'./vis/bank_{ty}/lstm/best_test_predy.npy')
    lstm_true = np.load(f'./vis/bank_{ty}/lstm/best_test_truey.npy')
    gru_pred = np.load(f'./vis/bank_{ty}/gru/best_test_predy.npy')
    gru_true = np.load(f'./vis/bank_{ty}/gru/best_test_truey.npy')
    ncde_pred = np.load(f'./vis/bank_{ty}/ncde/best_test_predy.npy')
    ncde_true = np.load(f'./vis/bank_{ty}/ncde/best_test_truey.npy')

    rnn_pred, rnn_true = rnn_pred.squeeze(), rnn_true.squeeze()
    lstm_pred, lstm_true = lstm_pred.squeeze(), lstm_true.squeeze()
    gru_pred, gru_true = gru_pred.squeeze(), gru_true.squeeze()
    ncde_pred, ncde_true = ncde_pred.squeeze(), ncde_true.squeeze()

    plt.figure(figsize=[20, 15])
    plt.plot(range(gru_pred.shape[0]),gru_true, label = "TRUE", linestyle="--", linewidth=6)
    plt.plot(range(gru_pred.shape[0]),rnn_pred, label = "RNN", linewidth=6)
    plt.plot(range(gru_pred.shape[0]),lstm_pred, label = "LSTM", linewidth=6)
    plt.plot(range(gru_pred.shape[0]),gru_pred, label = "GRU", linewidth=6)
    plt.plot(range(gru_pred.shape[0]),ncde_pred, label = "NCDE", linewidth=6)
    plt.legend(fontsize=48)
    plt.ylabel("amount (unit:trillion)", fontsize = 55)
    plt.xlabel("day", fontsize = 55)
    plt.yticks(fontsize=55)
    plt.xticks(fontsize=55)
    plt.tight_layout()
    plt.savefig(f'./vis/results/bank_{ty}.png')