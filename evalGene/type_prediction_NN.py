import torch as th
import torch.nn as nn

from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
import argparse
import copy
import csv
    
class Net(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Net, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(in_dim, int(in_dim/2)),
            nn.BatchNorm1d(int(in_dim/2)),
            nn.Dropout(),
            nn.ReLU(),
            nn.Linear(int(in_dim/2), out_dim),
            nn.Sigmoid()
        )
    def forward(self, x):
        out = self.main(x)
        return out

class Scheduler():
    def __init__(self, optimizer, init_lr, n_warmup, epochs):
        self.optim = optimizer
        self.n_warmup = n_warmup
        self.lr_multiplier = 0.1
        self.init_lr = init_lr
        self.total_epochs = epochs

    def zero_grad(self):
        self.optim.zero_grad()

    def step_and_update_lr(self, curr_eph):
        self.update_lr(curr_eph)
        self.optim.step()
        
    def update_lr(self, curr_eph):
        if curr_eph < self.n_warmup:
            lr = self.init_lr * self.lr_multiplier 
        else:
            lr = self.init_lr * max(0.0,float(self.total_epochs-curr_eph))/float(max(1.0,self.total_epochs-self.n_warmup))
        for param_group in self.optim.param_groups:
            param_group['lr'] = lr

def load_data(samples):
    x_ls = []
    y_ls = []
    for i in range(len(samples)):
        g1 = samples.iloc[i,0]
        g2 = samples.iloc[i,1]
        x_ls.append([g1, g2])
        y_ls.append(samples.iloc[i,2:])
    return np.array(x_ls), np.array(y_ls)

def map_to_vec(samples, embeddings):
    x_ls = []
    for i in range(len(samples)):
        x_ls.append(np.concatenate((embeddings[int(samples[i,0].item())], embeddings[int(samples[i,1].item())])).tolist())
    return th.FloatTensor(x_ls)    

def main():
    parser = argparse.ArgumentParser(description='Predict protein interaction type')
    parser.add_argument('-model', help='Embedding model', type=str)
    parser.add_argument('-dset', help='protein-protein interactions', type=str)
    parser.add_argument('-dim', help='Embedding dimension', type=int)
    parser.add_argument('-lr', help='Learning rate', type=float, default=0.01)
    parser.add_argument('-gpu', help='GPU id', type=int, default=0)
    parser.add_argument('-burnin', help='Epochs of burn in', type=int, default=20)
    parser.add_argument('-epochs', help='Number of epochs', type=int, default=300)
    parser.add_argument('-batchsize', help='Batchsize', type=int, default=1000)
    parser.add_argument('-path', help='model path', type=str)
    parser.add_argument('-print_each', help='Print loss each n-th epoch', type=int, default=10)
    parser.add_argument('-patience', help='early stop patience', type=int, default=20)
    parser.add_argument('-fout', help='Prediction output', type=str)
    opt = parser.parse_args()

    # load embeddings
    embeddings = np.load(opt.path)

    # dataset processing
    print("... load data ...")
    if opt.dset[-3:] == "tsv":
        data = pd.read_csv(opt.dset, sep="\t")
    else:
        data = pd.read_csv(opt.dset)
        
    device = th.device('cuda:'+str(opt.gpu) if th.cuda.is_available() else 'cpu')
    X, y = load_data(data) #

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42, stratify=y_train)
    out_dim = y_train.shape[1]

    net = Net(2*opt.dim, out_dim).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = th.optim.Adam(net.parameters(), lr=opt.lr)
    scheduler = Scheduler(optimizer, opt.lr, opt.burnin, opt.epochs)

    # Dataloader
    train_dataset = TensorDataset(th.FloatTensor(X_train), th.FloatTensor(y_train.astype('float64')))
    val_dataset = TensorDataset(th.FloatTensor(X_val), th.FloatTensor(y_val.astype('float64')))
    test_dataset = TensorDataset(th.FloatTensor(X_test), th.FloatTensor(y_test.astype('float64')))
    train_loader = DataLoader(
        train_dataset,
        batch_size=opt.batchsize,
        shuffle=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=opt.batchsize,
        shuffle=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
    )

    # Train the model
    print("... Train Network ...")
    opt_eph = 0
    opt_loss = np.inf
    opt_model_state_dict = net.state_dict()
    counter = 0
    
    for epoch in range(opt.epochs):
        epoch_loss = []
        net.train()
        for samples, labels in train_loader:
            samples = map_to_vec(samples, embeddings)
            samples = samples.to(device)
            labels = labels.to(device)
            preds = net(samples)
            loss = criterion(preds, labels)
            scheduler.zero_grad()
            loss.backward()
            scheduler.step_and_update_lr(epoch)
            epoch_loss.append(loss.item())

        with th.no_grad():
            net.eval()
            val_loss = []
            for samples, labels in val_loader:
                samples = map_to_vec(samples, embeddings)
                samples = samples.to(device)
                labels = labels.to(device)
                preds = net(samples)
                loss = criterion(preds, labels)
                val_loss.append(loss.item())
            if np.mean(val_loss) < opt_loss:
                opt_loss = np.mean(val_loss)
                opt_eph = epoch
                opt_model_state_dict = copy.deepcopy(net.state_dict())
            else:
                counter += 1
                if counter == opt.patience:
                    print("Early stopping")
                    break

        if (epoch+1) % opt.print_each == 0:
            print("Epoch [{}/{}] Train Loss: {:.3f}  Val Loss: {:.3f}".format(epoch+1, opt.epochs, np.mean(epoch_loss), np.mean(val_loss)))

    # Calculate the test result
    net.load_state_dict(opt_model_state_dict)
    print("Optimal tuning: Epoch {}, Val Loss: {:.3f}".format(opt_eph+1, opt_loss))
    y = []
    yhat = []
    with th.no_grad():
        net.eval()
        for samples, labels in test_loader:
            samples = map_to_vec(samples, embeddings)
            samples = samples.to(device)
            preds = net(samples)
            label = labels.tolist()[0]
            pred = preds.cpu().tolist()[0]
            pred = [1 if d>=0.5 else 0 for d in pred]
            y.append(label)
            yhat.append(pred)

    y = np.vstack(y)
    yhat = np.vstack(yhat)
    acc = accuracy_score(y, yhat)
    macro_f1 = f1_score(y, yhat, average='macro')
    micro_f1 = f1_score(y, yhat, average='micro')
    print(opt.model)
    print("Acc: "+str(acc))
    print("macro-F1: "+str(macro_f1))
    print("micro-F1: "+str(micro_f1))
    
    eval_list = [opt.model, "Acc: {:.4f}".format(acc), "Mac-F1: {:.4f}".format(macro_f1), "Mic-F1: {:.4f}".format(micro_f1)]
    
    with open('type_evaluation.tsv', mode='a', newline='') as f:
        wr = csv.writer(f, delimiter='\t')
        wr.writerow(eval_list)
        
if __name__ == '__main__':
    main()