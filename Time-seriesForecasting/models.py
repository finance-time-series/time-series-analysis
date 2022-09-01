import torch
import torch.nn as nn
import torch.nn.functional as F
import torchcde

### RNN ###
class RNNNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, hidden_hidden_channels=128):
        super(RNNNet, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size

        self.rnn = nn.RNN(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(in_features=hidden_size, out_features=output_size)

    def forward(self, x):
        _, h = self.rnn(x)

        # predict t+1
        y = self.fc(F.relu(h))

        # (num_layers, batch_size, output_size) → (batch_size, output_size)
        y = y.squeeze(dim=0)

        return y

### LSTM ###
class LSTMNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, hidden_hidden_channels=128):
        super(LSTMNet, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(in_features=hidden_size, out_features=output_size)

    def forward(self, x):
        _, (h, _) = self.lstm(x)
        # predict t+1
        y = self.fc(F.relu(h))

        # (num_layers, batch_size, output_size) → (batch_size, output_size)
        y = y.squeeze(dim=0)

        return y

### GRU ###
class GRUNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, hidden_hidden_channels=128):
        super(GRUNet, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size

        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(in_features=hidden_size, out_features=output_size)

    def forward(self, x):
        h_init = self.init_hidden(batch_size=x.size(0), device=x.get_device())
        _, h = self.gru(x, h_init)

        # predict t+1
        y = self.fc(h.relu())

        # (num_layers, batch_size, output_size) → (batch_size, output_size)
        y = y.squeeze(dim=0)

        return y

    def init_hidden(self, batch_size, device):
        weight = next(self.parameters()).data
        h_init = weight.new(self.num_layers, batch_size, self.hidden_size).zero_().to(device)
        return h_init

### Neural CDE ###
class CDEFunc(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels, hidden_hidden_channels, n_layers):
        super(CDEFunc, self).__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.n_layers = n_layers
        self.hidden_hidden_channels = hidden_hidden_channels

        self.linear_in = torch.nn.Linear(hidden_channels, hidden_hidden_channels)
        self.linears = torch.nn.ModuleList(torch.nn.Linear(hidden_hidden_channels, hidden_hidden_channels)
                        for _ in range(n_layers - 1))
        self.linear_out = torch.nn.Linear(hidden_hidden_channels, input_channels * hidden_channels)

    def forward(self, t, z):
        z = self.linear_in(z)
        z = z.relu()
        for linear in self.linears:
            z = linear(z)
            z = z.relu()
        z = self.linear_out(z)
        z = z.tanh()
        z = z.view(z.size(0), self.hidden_channels, self.input_channels)
        return z


class NeuralCDE(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels, output_channels, n_layers, hidden_hidden_channels, interpolation="cubic"):
        super(NeuralCDE, self).__init__()

        self.func = CDEFunc(input_channels, hidden_channels, hidden_hidden_channels, n_layers)
        self.initial = torch.nn.Linear(input_channels, hidden_channels)
        self.readout = torch.nn.Linear(hidden_channels, output_channels)
        self.interpolation = interpolation

    def forward(self, coeffs):
        if self.interpolation == 'cubic':
            X = torchcde.CubicSpline(coeffs)
        elif self.interpolation == 'linear':
            X = torchcde.LinearInterpolation(coeffs)
        else:
            raise ValueError("Only 'linear' and 'cubic' interpolation methods are implemented.")

        X0 = X.evaluate(X.interval[0])
        z0 = self.initial(X0)

        z_T = torchcde.cdeint(X=X,
                              z0=z0,
                              func=self.func,
                              t=X.grid_points,
                              method='rk4')

        z_T = z_T[:,-1,:]
        pred_y = self.readout(z_T)
        return pred_y