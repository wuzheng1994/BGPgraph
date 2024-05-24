import torch
import pytorch_lightning as pl
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool, global_max_pool
import torch_geometric.transforms as T

from torch.nn import Linear
from torch_geometric.loader import DataLoader
import sys
sys.path.append('/home/wuzheng/Packet/99code/')
from Mydataset import MyOwnDataset
from pytorch_lightning.callbacks import ModelCheckpoint
from collections import Counter
from torch.nn import BatchNorm1d
import logging

use_gdc = True

logger_ = logging.getLogger("pytorch_lightning")
logger = pl.loggers.TensorBoardLogger(save_dir='/home/wuzheng/Packet/99code/', version='ph', name='lightning_logs')


class GCN(pl.LightningModule):
    def __init__(self, hidden_channels, num_features, num_classes) -> None:
        super(GCN, self).__init__()
        torch.manual_seed(12345)
        self.norm = BatchNorm1d(num_features)
        self.conv1 = GCNConv(num_features, hidden_channels, add_self_loops=True)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.linear1 = Linear(hidden_channels * 2 * 3, hidden_channels)
        self.linear2 = Linear(hidden_channels, num_classes)
        self.starter, self.ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        self.validation_step_outputs = []
        self.train_step_outputs = []

    def forward(self, x, edge_index, batchsize, edge_weight):
        x = self.norm(x)
        x = self.conv1(x, edge_index, edge_weight=edge_weight)
        x = x.relu()
        x1 = torch.cat([global_mean_pool(x, batch=batchsize), global_max_pool(x, batch=batchsize)], dim=1)
        x = self.conv2(x, edge_index, edge_weight=edge_weight)
        x = x.relu()
        x2 = torch.cat([global_mean_pool(x, batch=batchsize), global_max_pool(x, batch=batchsize)], dim=1)
        x = self.conv3(x, edge_index, edge_weight=edge_weight)
        x_mean = global_mean_pool(x, batch=batchsize)
        x_max = global_max_pool(x, batch=batchsize)
        x = torch.cat([x_mean, x_max], dim =1)
        x_com = torch.cat([x,x1,x2], dim=1)
        x_com = F.dropout(x_com, p=0.5, training=self.training)
        x_com = self.linear1(x_com)
        x_com = self.linear2(x_com)
        return x_com

    def training_step(self, batch, batch_idx):
        
        x, edge_index, edge_weight = batch.x, batch.edge_index, batch.edge_attr
        batch_idx = batch.batch
        x_out = self.forward(x, edge_index, batch_idx, edge_weight)
        loss = F.cross_entropy(x_out, batch.y)
        pred = x_out.argmax(-1)
        label = batch.y
        # print('batch label:', label)
        accuracy = (pred == label).sum() / pred.shape[0]
        self.log("loss/train", loss, batch_size=10, on_step=True)
        self.log("accuracy/train", accuracy, batch_size=10, on_step=True)
        self.train_step_outputs.append({"loss":loss, "accuracy": accuracy})
        return {"loss":loss, "accuracy": accuracy}

    def on_train_epoch_end(self) -> None:
        train_loss = 0.0
        train_acc = 0.0
        idx = 0
        for i in self.train_step_outputs:
            idx += 1
            train_acc += i['accuracy']
            train_loss += i['loss']
        train_loss = train_loss / idx
        train_acc = train_acc / idx
        logger_.info(f"Epoch: {self.current_epoch}: training_epoch_end--loss/train: loss {train_loss}, accuracy {train_acc}") 

    def on_validation_epoch_end(self):

        val_loss = 0.0
        num_correct = 0
        num_total = 0

        for output, pred, labels in self.validation_step_outputs:

            val_loss += F.cross_entropy(output, labels, reduction="sum")
            num_correct += (pred == labels).sum()
            num_total += pred.shape[0]
            val_accuracy = num_correct / num_total
            val_loss = val_loss / num_total
        logger_.info(f"Epoch: {self.current_epoch}: val_epoch_end--loss/train: loss {val_loss}, accuracy {val_accuracy}") 

        self.log("accuracy/val", val_accuracy, batch_size=10)
        self.log("loss/val", val_loss, batch_size=10)

    def validation_step(self, batch, batch_index):
        x, edge_index, edge_weight = batch.x, batch.edge_index, batch.edge_attr
        batch_index = batch.batch
        x_out = self.forward(x, edge_index, batch_index, edge_weight)
        loss = F.cross_entropy(x_out, batch.y)
        pred = x_out.argmax(-1)
        self.validation_step_outputs.append([x_out, pred, batch.y])
        return x_out, pred, batch.y       

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        return optimizer

    def predict_step(self, batch, batch_idx):
        
        x, edge_index, edge_weight = batch.x, batch.edge_index, batch.edge_attr
        batch_index = batch.batch
        
        self.starter.record()
        self.forward(x, edge_index, batch_index, edge_weight)
        self.ender.record()
        torch.cuda.synchronize()
        inference_time = self.starter.elapsed_time(self.ender)
        return inference_time*1e-3

features = []
def hook(module, input, output):
    features.append(input)
    return None

checkpoint_callback = ModelCheckpoint(
    monitor='loss/val',
    filename='sample-mnist-{epoch:02d}-{val_loss:.2f}',
    save_top_k=1,
    mode='min',
    save_last=True
)

if __name__ == '__main__':
    # dataset = MyOwnDataset('/data/data/graph_data6/Prefix_hijack/')
    dataset = MyOwnDataset("/data/data/graph_data5/")
    torch.manual_seed(12345)
    
    dataset = dataset.shuffle()
    train_dataset = dataset[:1152]
    test_dataset = dataset[1152:]    
    
    print(f'Number of training graphs: {len(train_dataset)}')
    print(f'Number of test graphs: {len(test_dataset)}')
    print(f'Number of class:', dataset.num_classes)
    print('The distribution is {}'.format(Counter([int(d.y[0]) for d in dataset])))
    # print(dataset.num_classes, dataset.num_features)

    train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True, num_workers=10)
    test_loader = DataLoader(test_dataset, batch_size=10, shuffle=True)

    lighning_model = GCN(hidden_channels=64, num_features=5, num_classes=4) 
    
    # lighning_model.linear.register_forward_hook(hook=hook)
    num_epochs=40
    val_check_interval = 0.25
    
    trainer = pl.Trainer(max_epochs = num_epochs, val_check_interval=1.0, log_every_n_steps=3, accelerator="gpu", callbacks=[checkpoint_callback], logger=logger, enable_progress_bar=1)
    trainer.fit(lighning_model, train_loader, test_loader)
    # inference_time = trainer.predict(lighning_model, dataloaders=test_loader)





