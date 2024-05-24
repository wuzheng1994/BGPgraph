import torch
import pytorch_lightning as pl
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool, global_max_pool
from torch.nn import Linear
from torch import nn
from torch_geometric.loader import DataLoader
import sys
sys.path.append("/home/wuzheng/Packet/Graph_network/")
from Mydataset import MyOwnDataset
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from collections import Counter
from torch.nn import BatchNorm1d
from Edge_conv import Edge_GCNConv
import logging
from pyg_test import GCN
from sklearn.metrics import accuracy_score

logger_ = logging.getLogger("pytorch_lightning")
logger = pl.loggers.TensorBoardLogger(save_dir='/home/wuzheng/Packet/Graph_network/GCN/', version='version_load9', name='lightning_logs')
Batch_num = 10

class encoder(nn.Module):
    def __init__(self, hidden_channels, num_features):
        super(encoder, self).__init__()
        torch.manual_seed(12345)
        self.norm1 = nn.BatchNorm1d(num_features)
        self.conv1 = GCNConv(num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)

    def forward(self, x, edge_index, batchsize, edge_weight):
        x_nor = self.norm1(x)
        # x_structure = torch.cat([global_mean_pool(x_nor, batchsize), global_max_pool(x_nor, batchsize)], dim =1)
        x = self.conv1(x, edge_index, edge_weight)
        x = x.relu()
        x = self.conv2(x, edge_index, edge_weight)
        x = x.relu()
        x = self.conv3(x, edge_index, edge_weight)
        return x, x_nor

class decoder(nn.Module):
    def __init__(self, hidden_channels, num_features):
        super(decoder, self).__init__()
        self.norm2 = nn.BatchNorm1d(hidden_channels)
        self.conv4 = GCNConv(hidden_channels, hidden_channels)
        self.conv5 = GCNConv(hidden_channels, hidden_channels)
        self.conv6 = GCNConv(hidden_channels, num_features, add_self_loops=True)
        self.lin1 = Linear((num_features + hidden_channels*2) * 2, hidden_channels)
        self.lin2 = Linear(hidden_channels, 1)
        
    def forward(self, x, edge_index, batchsize, edge_weight):
        x = self.norm2(x)
        x = self.conv4(x, edge_index, edge_weight)
        x = x.relu()
        x_mean = global_mean_pool(x, batchsize)
        x_max = global_max_pool(x, batchsize)
        x_structure_1 = torch.cat([x_mean, x_max], dim =1)
        x = self.conv5(x, edge_index, edge_weight)
        x = x.relu()
        x_mean = global_mean_pool(x, batchsize)
        x_max = global_max_pool(x, batchsize)
        x_structure_2 = torch.cat([x_mean, x_max], dim =1)
        x = self.conv6(x, edge_index, edge_weight)
        x = x.relu()
        x_mean = global_mean_pool(x, batchsize)
        x_max = global_max_pool(x, batchsize)
        x_structure_3 = torch.cat([x_mean, x_max], dim =1)

        x_structure = torch.cat([x_structure_1, x_structure_2, x_structure_3], dim =1)
        
        lin1_out = self.lin1(x_structure)
        y_hat = self.lin2(lin1_out)
        return x, y_hat

class Autoencoder(pl.LightningModule):
    def __init__(self, hidden_channels, num_features) -> None:
        super(Autoencoder, self).__init__()
        self.encoder = encoder(hidden_channels = hidden_channels, num_features=num_features)
        self.decoder = decoder(hidden_channels = hidden_channels, num_features=num_features)
        self.sig = nn.Sigmoid()
        self.loss = nn.BCELoss()
        # self.decoder2 = decoder2(hidden_channels = hidden_channels, num_features=num_features)
    
    def training_step(self, batch):
        x, edge_index, batchsize, edge_weight = batch.x, batch.edge_index, batch.batch, batch.edge_weight        
        z, x_nor = self.encoder(x, edge_index, batchsize, edge_weight)
        x_hat, y_hat = self.decoder(z, edge_index, batchsize, edge_weight)
        # adj_hat, adj = self.decoder2(x, edge_index, batchsize, edge_weight)
        target = torch.where(batch.y != 0.0, 1.0, 0.0)
        # nor_label = target == 0.0
        nor_label = batch.y == 0.0
        batch_label = torch.arange(Batch_num).to(device)
        batch_label = batch_label[nor_label]
        batch_isin = torch.isin(batchsize, batch_label)
        loss_attr = nn.functional.mse_loss(x_hat[batch_isin], x_nor[batch_isin], reduction='sum')
        
        loss_cross = self.loss(self.sig(y_hat.squeeze()), target.squeeze())
        
        loss = loss_attr + loss_cross
        self.log("loss_cross_tra", loss_cross, batch_size=Batch_num)
        self.log("training_loss", loss, batch_size=Batch_num)
        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

    def training_epoch_end(self, outputs) -> None:
        train_loss = 0.0
        idx = 0
        for i in outputs:
            idx += 1
            train_loss += i['loss']
        train_loss = train_loss / idx
        logger_.info(f"Epoch: {self.current_epoch}: training_epoch_end--loss/train: loss {train_loss}")

    def validation_epoch_end(self, loss):
        val_loss = 0.0
        idx = 0
        for output in loss:
            idx += 1
            val_loss += output
        val_loss = output / idx
        logger_.info(f"Epoch: {self.current_epoch}: val_epoch_end--loss/train: loss {val_loss}") 
        self.log("loss/val", val_loss, batch_size = Batch_num)

    def validation_step(self, batch, batch_idx):
        x, edge_index, batch_index, edge_weight = batch.x, batch.edge_index, batch.batch, batch.edge_weight
        z, x_nor = self.encoder(x, edge_index, batch_index, edge_weight)
        x_hat, y_hat = self.decoder(z, edge_index, batch_index, edge_weight)

        nor_label = batch.y == 0.0

        batch_label = torch.arange(Batch_num).to(device)
        batch_label = batch_label[nor_label]

        batch_isin = torch.isin(batch_index, batch_label)

        loss_x = nn.functional.mse_loss(x_hat[batch_isin], x_nor[batch_isin], reduction="sum")
        target = torch.where(batch.y != 0.0, 1.0, 0.0)
        
        loss_cross = self.loss(self.sig(y_hat.squeeze()), target.squeeze())
        # print("loss_cross: ",loss_cross)
        self.log("loss_cross_val", loss_cross, batch_size=Batch_num)
        # loss = loss_x + loss_str
        loss = loss_x + loss_cross
        # loss = loss_cross
        self.log("validation loss,", loss, batch_size=Batch_num)
        return loss       

# features = []
# def hook(module, input, output):
#     features.append(input)
#     return None

checkpoint_callback = ModelCheckpoint(
    monitor='loss/val',
    filename='sample-mnist-{epoch:02d}-{val_loss:.2f}',
    save_top_k=1,
    mode='min',
    save_last=True
)

if __name__ == '__main__':
    # dataset = MyOwnDataset('/home/wuzheng/Packet/Graph_network/graph_data2/')
    # dataset = MyOwnDataset('/data/data/graph_data4/Data/')
    dataset = MyOwnDataset('/data/data/graph_data5/')
    torch.manual_seed(12345)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    print(f'Number of class:', dataset.num_classes)
    print('The distribution is {}'.format(Counter([int(d.y[0]) for d in dataset])))
    # nor_list = [i for i, d in enumerate(dataset) if int(d.y[0]) == 0]
    # print("nor list:",len(nor_list))
    
    nor_dataset = dataset[:]
    nor_dataset = nor_dataset.shuffle()
    num = len(nor_dataset)
    train_num = num // 5 * 4

    train_dataset = nor_dataset[:train_num]
    test_dataset = nor_dataset[train_num:]

    print(f'Number of training graphs: {len(train_dataset)}')
    print(f'Number of test graphs: {len(test_dataset)}')

    train_loader = DataLoader(train_dataset, batch_size=Batch_num, shuffle=True, num_workers=10)
    test_loader = DataLoader(test_dataset, batch_size=Batch_num, shuffle=True)

    # '''
    # load the model.
    # '''
    # path = "/home/wuzheng/Packet/Graph_network/lightning_logs/mymy_name5/checkpoints/last.ckpt"
    # clf_model = GCN.load_from_checkpoint(checkpoint_path=path, hidden_channels= 64, num_features = 4, num_classes = 4)
    
    lighning_model = Autoencoder(hidden_channels=64, num_features=5)
    
    # net_new_parameters = lighning_model.state_dict().copy()
    # pre_name = list(clf_model.state_dict().keys())[:11]
    # cur_name = list(lighning_model.state_dict().keys())[:11]
    # map_name = dict(zip(pre_name, cur_name))
    # print("map_name:",map_name)
    
    # for n in pre_name:
    #     net_new_parameters[map_name[n]] = clf_model.state_dict()[n]
    #     if n == "norm.weight":
    #         print(net_new_parameters[map_name[n]])
    # lighning_model.load_state_dict(net_new_parameters)
    # print("Complete loading model!")
    
    # lighning_model.linear.register_forward_hook(hook=hook)
    num_epochs=30
    val_check_interval = 0.25
    
    trainer = pl.Trainer(max_epochs = num_epochs, val_check_interval=1.0, log_every_n_steps=3, accelerator="gpu", logger=logger, enable_progress_bar=1, callbacks=[checkpoint_callback])
    trainer.fit(lighning_model, train_loader, test_loader)