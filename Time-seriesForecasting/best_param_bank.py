import os

types = ["cd","cms","elec","other","suburb"]

rnn_lrs = [1e-2, 1e-3, 1e-2, 1e-2, 1e-4]
rnn_hiddens = [2048, 2048, 2048, 2048, 128]
rnn_layers = [1, 1, 1, 1, 1]

lstm_lrs = [1e-4, 1e-3, 1e-2, 1e-2, 1e-4]
lstm_hiddens = [2048, 2048, 2048, 2048, 1024]
lstm_layers = [1, 1, 1, 1, 1]

gru_lrs = [1e-4, 1e-3, 1e-2, 1e-2, 1e-3]
gru_hiddens = [2048, 1024, 2048, 256, 2048]
gru_layers = [1, 1, 1, 1, 1]

ncde_lrs = [1e-2, 1e-2, 1e-3, 1e-2, 1e-3]
ncde_hiddens = [256, 256, 64, 128, 512]
ncde_hhs = [256, 512, 512, 256, 64]
ncde_layers = [2, 3, 3, 2, 4]

## common params ##
data = "bank"
seed = 2022
train_seq = 30
y_seq = 1
epoch = 1000
weight_decay = 1e-5
batch = 128
###################

for i in range(len(types)):
    # RNN
    os.system(f"python main.py --model rnn --data {data} --type {types[i]} --seed {seed} --train_seq {train_seq} --y_seq {y_seq} --epoch {epoch} \
        --lr {rnn_lrs[i]} --batch {batch} --weight_decay {weight_decay} --n_layer {rnn_layers[i]} --hidden_size {rnn_hiddens[i]} \
            > ./results/{data}_{types[i]}_rnn_final.csv")

    # LSTM
    os.system(f"python main.py --model lstm --data {data} --type {types[i]} --seed {seed} --train_seq {train_seq} --y_seq {y_seq} --epoch {epoch} \
        --lr {lstm_lrs[i]} --batch {batch} --weight_decay {weight_decay} --n_layer {lstm_layers[i]} --hidden_size {lstm_hiddens[i]} \
            > ./results/{data}_{types[i]}_lstm_final.csv")

    # GRU
    os.system(f"python main.py --model gru --data {data} --type {types[i]} --seed {seed} --train_seq {train_seq} --y_seq {y_seq} --epoch {epoch} \
        --lr {gru_lrs[i]} --batch {batch} --weight_decay {weight_decay} --n_layer {gru_layers[i]} --hidden_size {gru_hiddens[i]} \
            > ./results/{data}_{types[i]}_gru_final.csv")

    # NCDE
    os.system(f"python main.py --model ncde --data {data} --type {types[i]} --seed {seed} --train_seq {train_seq} --y_seq {y_seq} --epoch {epoch} \
        --lr {ncde_lrs[i]} --batch {batch} --weight_decay {weight_decay} --n_layer {ncde_layers[i]} --hidden_size {ncde_hiddens[i]} --hh_size {ncde_hhs[i]} \
            > ./results/{data}_{types[i]}_ncde_final.csv")
