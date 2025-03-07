import torch
import pytorch_lightning as pl
from torch_geometric.loader import DataLoader
import sys
sys.path.append("/home/wuzheng/Packet/99code/")
from Mydataset3 import MyOwnDataset
from collections import Counter
from pyg_test_Dec import GCN
from torch import nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix, precision_score
import time

def compute_missing_rate(y_predict, y_label):
    matrix = confusion_matrix(y_pred=y_predict, y_true=y_label, )
    tn, fp, fn, tp =  matrix.ravel()
    missing_rate = fn / (tp + fn)
    false_rate = fp / (fp + tn)
    return (missing_rate, false_rate)


class MyDetector(pl.LightningModule):
    def __init__(self, hidden_channels, num_features, num_classes):
        super(MyDetector, self).__init__()
        self.hidden_channels = hidden_channels
        self.num_features = num_features
        self.num_classes = num_classes
        self.model = GCN.load_from_checkpoint(checkpoint_path=model_path, hidden_channels=self.hidden_channels, num_features=self.num_features, num_classes=self.num_classes)
    
    def forward(self, batch):
        x, edge_index, batch_index, edge_weight = batch.x, batch.edge_index, batch.batch, batch.edge_weight
        x_com = self.model(x, edge_index, batch_index, edge_weight)
        pred = x_com.argmax(-1)
        x_com_soft = F.softmax(x_com)
        return pred, x_com_soft


def hook(module, input, output):
    # print("model's hook:", output)
    features.append(output.detach())
    return None

if __name__ == "__main__":
    
    features = []
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_path = "/home/wuzheng/Packet/99code/lightning_logs/ph/checkpoints/sample-mnist-epoch=26-val_loss=0.00.ckpt"
    
    # dataset = MyOwnDataset('/data/data/graph_data5/')
    dataset = MyOwnDataset('/data/data/graph_data6/Prefix_hijack/')
    
    torch.manual_seed(12345)
    clf_model = MyDetector(hidden_channels=64, num_features=5, num_classes=2)
    print(f'Number of class:', dataset.num_classes)
    print('The distribution is {}'.format(Counter([int(d.y[0]) for d in dataset])))

    # anomaly_list = [i for i, d in enumerate(dataset) if int(d.y[0]) != 0]
    # anomaly_dataset = dataset[anomaly_list]

    # num = len(anomaly_list)
    # train_num = num // 3 *2
    # print("test name:", anomaly_dataset[0].ind)
    print("Complete loading model!")

    y_true_list = []
    y_pred_list = []
    ind_list = []
    
    model = clf_model.eval().to(device)
    
    for child in model.modules():
        if isinstance(child, nn.Linear) and child.in_features == 384:
            child.register_forward_hook(hook=hook)
    x_com_list = []
    with torch.no_grad():       
        starter = time.time()        
        for i in range(len(dataset)):
            y_true = dataset[i].y
            ind = dataset[i].ind
            
            pred, x_com = model(dataset[i].to(device))
            x_com_list.append(list(x_com[0])[1])
            # print('x_com:', list(x_com[0])[1])
            ind_list.append(int(ind))
            y_pred_list.append(int(pred))
            y_true_list.append(int(y_true))
    
    # exit()
    
    
    ender = time.time()
    running_time= ender - starter
    print("running time:",running_time/1440)
    
    # print(y_pred_list)
    # print(Counter(y_pred_list))
    # exit()
    # print(y_true_list)
    # for i in range(len(y_true_list)):
    #     if y_pred_list[i] == 0 and y_true_list[i] != 0:
    #         print(ind_list[i])
    # print(ind_list)
    
    a = sorted(zip(ind_list, range(len(ind_list))), key=lambda x: x[0])
    ind_sorted = [i[1] for i in a]
    
    '''
    提取输出层表征
    '''
    features = [i.squeeze().tolist() for i in features]
    print('x_com_list:', x_com_list)
    with open('/home/wuzheng/Packet/99code/GCN/linear_output_ph.txt', 'w') as fw:
        # for f in features:
        #     fw.write(','.join([str(i) for i in f]))
        #     fw.write('\n')
    
        fw.write(','.join([str(float(i)) for i in x_com_list]))
        fw.write('\n')
        fw.write(','.join([str(i) for i in y_true_list]))
        fw.write('\n')
        fw.write(','.join([str(i) for i in y_pred_list]))
        fw.write('\n')
        fw.write(','.join([str(i) for i in ind_sorted]))

    # print("names: ", model.named_modules)
    # print(list(model.modules()))
    exit()
    
    # print("Feature:", features)

    # print(sorted(zip(ind_list, y_pred_list), key=lambda x: x[0]))
    
    # accuracy = accuracy_score(y_true=y_true_list, y_pred=y_pred_list)
    # recall = recall_score(y_true=y_true_list, y_pred=y_pred_list, average='micro')
    # f1 = f1_score(y_true=y_true_list, y_pred=y_pred_list, average='micro')
    # miss_rate, false_rate = compute_missing_rate(y_pred_list, y_true_list)
    
    # print('Overall accuracy', accuracy)
    # print("Recall score:", recall)
    # print("f1_score:", f1)
    
    # # print("miss rate:", miss_rate, "false rate:", false_rate)
    # # print("precision:", precision_score(y_true=y_true_list, y_pred=y_pred_list, average='micro'))
    # print('running time:', running_time / len(y_pred_list))
    
