import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import argparse
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import StepLR


parser = argparse.ArgumentParser()

parser.add_argument('--epochs', type=int, default=50, help='input dimension')
parser.add_argument('--input_size', type=int, default=1, help='input dimension')
parser.add_argument('--output_size', type=int, default=1, help='output dimension')
parser.add_argument('--hidden_size', type=int, default=128, help='hidden size')
parser.add_argument('--num_layers', type=int, default=1, help='num layers')
parser.add_argument('--lr', type=float, default=0.05, help='learning rate')
parser.add_argument('--batch_size', type=int, default=30, help='batch size')
parser.add_argument('--optimizer', type=str, default='adam', help='type of optimizer')
parser.add_argument('--device', default=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
parser.add_argument('--step_size', type=int, default=5, help='step size')
parser.add_argument('--pred_step_size', type=int, default=30, help='pred step size')
parser.add_argument('--gamma', type=float, default=0.5, help='gamma')
parser.add_argument('--input_window', type=int, default=180)

args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)


def get_mape(v, v_):
    return np.mean(np.abs((v_ - v) / v_))


def get_rmse(v, v_):
    return np.sqrt(np.mean((v_ - v) ** 2))


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, batch_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.num_directions = 1
        self.batch_size = batch_size
        self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=True)
        self.linear = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input_seq):
        batch_size, seq_len = input_seq.shape[0], input_seq.shape[1]
        h_0 = torch.randn(self.num_directions * self.num_layers, batch_size, self.hidden_size).to(device)
        c_0 = torch.randn(self.num_directions * self.num_layers, batch_size, self.hidden_size).to(device)
        output, _ = self.lstm(input_seq, (h_0, c_0))
        pred = self.linear(output)
        pred = pred[:, -1, :]

        return pred


def generateDataset(B):
    data = pd.read_csv('PigPrice.csv')

    train = data[:int(len(data) * 0.67)]
    test = data[int(len(data) * 0.67):len(data)]

    def process(dataset, batch_size):
        load = dataset[dataset.columns[0]]
        load = load.tolist()
        m, n = np.max(load), np.min(load)
        load = (load - n) / (m - n)
        seq = []
        for i in range(len(load) - args.input_window):
            train_seq = []
            train_label = []
            for j in range(i, i + args.input_window):
                x = [load[j]]
                train_seq.append(x)
            train_label.append(load[i + args.input_window])
            train_seq = torch.FloatTensor(train_seq)
            train_label = torch.FloatTensor(train_label).view(-1)
            seq.append((train_seq, train_label))

        seq = MyDataset(seq)
        seq = DataLoader(dataset=seq, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=True)

        return seq, [m, n]

    trainSet, lis1 = process(train, B)
    evalSet, lis2 = process(test, B)

    predSet = data[len(data)-args.input_window:len(data)]
    predSet = predSet.values.tolist()
    m, n = np.max(predSet), np.min(predSet)
    predSet = (predSet - n) / (m - n)

    return trainSet, evalSet, predSet, lis1, lis2, [m, n]


def train(args, model, Dtr):
    print("=" * 45)
    print("Training...")
    loss_function = nn.MSELoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 weight_decay=args.weight_decay)
    scheduler = StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    loss = 0
    for i in range(args.epochs):
        for (seq, label) in Dtr:
            seq = seq.to(device)
            label = label.to(device)
            y_pred = model(seq)
            loss = loss_function(y_pred, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print('Epoch', i+1, ':', loss.item())

        scheduler.step()


def evaluate(model, Dte, lis):
    print("=" * 45)
    print("Evaluating...")
    loss_function = nn.MSELoss().to(device)
    loss, mape, rmse, out, target = [], [], [], [], []
    for (seq, label) in Dte:
        seq = seq.to(device)
        label = label.to(device)
        pred = model(seq)
        loss.append(loss_function(pred, label).detach().cpu().numpy())
        pred = pred.detach().cpu().numpy()
        label = label.detach().cpu().numpy()
        out_unnormalized = (lis[0] - lis[1])*pred + lis[1]
        label_unnormalized = (lis[0] - lis[1])*label + lis[1]
        mape.append(get_mape(out_unnormalized, label_unnormalized))
        rmse.append(get_rmse(out_unnormalized, label_unnormalized))
        out.append(out_unnormalized[0][0])
        target.append(label_unnormalized[0][0])
    print('Validation loss:', np.mean(loss))
    print('Validation MAPE:', np.mean(mape))
    print('Validation RMSE:', np.mean(rmse))
    x = range(1, len(target) + 1)
    plt.plot(x, target, c='green', marker='*', ms=1, alpha=0.75, label='Ground True')
    plt.plot(x, out, c='red', marker='o', ms=1, alpha=0.75, label='Prediction')
    plt.grid(axis='y')
    plt.legend()
    plt.show()


def predict(model, predSet, lisPred, input_window, pred_step):
    print("=" * 45)
    print("Predicting...")
    predSet = predSet.tolist()
    prediction = []
    for i in range(pred_step):
        inputSet = torch.FloatTensor(predSet[-input_window:]).to(device)
        inputSet = torch.unsqueeze(inputSet, dim=0)
        output = model(inputSet)
        output = (lisPred[0] - lisPred[1]) * output.detach().cpu().numpy()[0][0] + lisPred[1]
        prediction.append(output)
        predSet.append([output])
    orgdata = pd.read_csv('PigPrice.csv').values
    plt.figure(figsize=(8, 4))
    x = range(1, len(orgdata) + 1)
    plt.plot(x, orgdata, c='black', marker='*', ms=1, alpha=0.75, label='Ground True')
    x = range(len(orgdata) + 1, len(orgdata) + 1 + len(prediction))
    plt.plot(x, prediction, c='red', marker='*', ms=1, alpha=0.75, label='Prediction')
    plt.grid(axis='y')
    plt.legend()
    plt.show()
    print(prediction)



if __name__ == '__main__':
    torch.manual_seed(7)
    trainSet, EvalSet, predSet, lisTrain, lisEval, lisPred = generateDataset(B=1)
    model = LSTM(args.input_size, args.hidden_size, args.num_layers, args.output_size, batch_size=args.batch_size).to(device)

    train(args, model, trainSet)
    evaluate(model, EvalSet, lisEval)
    predict(model, predSet, lisPred, args.input_window, args.pred_step_size)


