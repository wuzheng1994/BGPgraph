import torch
import pytorch_lightning as pl
from torch_geometric.loader import DataLoader
import sys
sys.path.append("/home/wuzheng/Packet/Graph_network/")
from Mydataset import MyOwnDataset
from collections import Counter
from autoencoder import Autoencoder
from torch import nn
from torch_geometric.utils import degree
import time
import numpy as np
from numpy import percentile
from scipy.stats import rankdata, iqr

class Tracer(pl.LightningModule):
    def __init__(self):
        super(Tracer, self).__init__()
        self.model = Autoencoder.load_from_checkpoint(checkpoint_path=model_path, hidden_channels= 64, num_features = 5)
    
    def forward(self, batch, focus_as):
        x, edge_index, batch_index, edge_weight = batch.x, batch.edge_index, batch.batch, batch.edge_weight
        deg_sqrt = torch.sqrt(degree(edge_index[1], num_nodes=x.size(0)))
        z, x_nor = self.model.encoder(x, edge_index, batch_index, edge_weight)
        x_hat, _ = self.model.decoder(z, edge_index, batch_index, edge_weight)
        
        loss_x = torch.mean(nn.functional.mse_loss(x_hat, x_nor, reduction="none"), dim=1)
        loss_foc = loss_x[focus_as]
        
        
        # loss_x = nn.functional.mse_loss(x_hat, x_nor, reduction="mean")
        # loss_str = nn.functional.mse_loss(x_hat_str, z_str, reduction="mean")
        # loss = torch.tensor([loss_x, loss_str])
        return loss_foc
        
if __name__ == "__main__":
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_path = "/home/wuzheng/Packet/Graph_network/GCN/lightning_logs/version_load9/checkpoints/last-v4.ckpt"
    
    dataset = MyOwnDataset('/data/data/graph_data6/Outage1/')
    clf_model = Tracer()
    # print('The label of the first data:', dataset[0].y, dataset[0].ind)

    print(f'Number of class:', dataset.num_classes)
    print('The distribution is {}'.format(Counter([int(d.y[0]) for d in dataset])))
    # anomaly_list = [i for i, d in enumerate(dataset) if int(d.y[0]) != 0]
    anomaly_dataset = dataset[:]

    num = len(anomaly_dataset)
    print("num:", num)
    print("Complete loading model!")
    
    # focus_as = torch.tensor([32934, 6939,  20940,   3573,   6762, 267613,  21433,  11537, 136907,   6453]).to(device)
    focus_as = torch.tensor([32934]).to(device)
    
    # trainer = pl.Trainer(max_epochs = num_epochs, val_check_interval=1.0, log_every_n_steps=3, accelerator="gpu", enable_progress_bar=1)
    model = clf_model.eval().to(device)
    foc_loss_list = []
    ind_list = []
    label_list = []
    with torch.no_grad():
        for i in range(len(anomaly_dataset)):
            ind = anomaly_dataset[i].ind
            label_list.append(anomaly_dataset[i].y)
            ind_list.append(ind)
            node_seq = anomaly_dataset[i].mapping.to(device)
            focus_as_tran = []
            for as_ in focus_as:
                index_ = torch.argwhere(node_seq == as_).squeeze(0)
                focus_as_tran.append(index_)
            focus_as_tran = torch.cat(focus_as_tran, dim = 0).to(device)
            foc = model(anomaly_dataset[i].to(device), focus_as_tran)
            foc_loss_list.append(foc)
    
    label_list = torch.cat(label_list, dim=0).numpy()
    ind_list = torch.cat(ind_list, dim=0).numpy()
    print("ind_list:", ind_list)
    # exit()   
    z = np.array([f.tolist() for f in foc_loss_list]).T
    # print(z.shape, label_list.shape, ind_list.shape)
    com = np.vstack((ind_list, label_list, z)).T
    
    # z = z.T
    # print("z:",z)
    
    # epsilon = 1e-4
    # med = np.median(z, axis=1)
    # print('med:',med)
    # err_iqr = iqr(z, axis=1)
    # print('err_iqr:', err_iqr)
    # q = np.abs((z - med.reshape(focus_as.size(0),1)) / (err_iqr.reshape(focus_as.size(0),1) + epsilon))
    # print("q:", q)
    
    filew = "/home/wuzheng/Packet/Graph_network/plot/foc_performance1.dat"
    with open(filew, 'w') as fw:
        for i in com:
            fw.write(','.join([str(p) for p in i]))
            fw.write('\n')

    
    
        
