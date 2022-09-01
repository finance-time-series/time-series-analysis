import os

rnn_lrs = [1e-3]
rnn_hiddens = [1024]
rnn_layers = [1]

lstm_lrs = [1e-2]
lstm_hiddens = [256]
lstm_layers = [1]

gru_lrs = [1e-3]
gru_hiddens = [1024]
gru_layers = [1]

ncde_lrs = [1e-3]
ncde_hiddens = [512]
ncde_hhs = [128]
ncde_layers = [3]

## common params ##
data = "tran"
seed = 2022
train_seq = 30
y_seq = 1
epoch = 300
weight_decay = 1e-5
batch = 1024
###################

i=0
# RNN
os.system(f"python main.py --model rnn --data {data} --seed {seed} --train_seq {train_seq} --y_seq {y_seq} --epoch {epoch} \
    --lr {rnn_lrs[i]} --batch {batch} --weight_decay {weight_decay} --n_layer {rnn_layers[i]} --hidden_size {rnn_hiddens[i]} \
        > ./results/{data}_rnn_final.csv")

# LSTM
os.system(f"python main.py --model lstm --data {data} --seed {seed} --train_seq {train_seq} --y_seq {y_seq} --epoch {epoch} \
    --lr {lstm_lrs[i]} --batch {batch} --weight_decay {weight_decay} --n_layer {lstm_layers[i]} --hidden_size {lstm_hiddens[i]} \
        > ./results/{data}_lstm_final.csv")

# GRU
os.system(f"python main.py --model gru --data {data} --seed {seed} --train_seq {train_seq} --y_seq {y_seq} --epoch {epoch} \
    --lr {gru_lrs[i]} --batch {batch} --weight_decay {weight_decay} --n_layer {gru_layers[i]} --hidden_size {gru_hiddens[i]} \
        > ./results/{data}_gru_final.csv")

# NCDE
os.system(f"python main.py --model ncde --data {data} --seed {seed} --train_seq {train_seq} --y_seq {y_seq} --epoch {epoch} \
    --lr {ncde_lrs[i]} --batch {batch} --weight_decay {weight_decay} --n_layer {ncde_layers[i]} --hidden_size {ncde_hiddens[i]} --hh_size {ncde_hhs[i]} \
        > ./results/{data}_ncde_final.csv")
