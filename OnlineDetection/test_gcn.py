import torch
import pytorch_lightning as pl
from torch_geometric.loader import DataLoader
# sys.path.append("/home/wuzheng/Packet/99code/")
from Mydataset import MyOwnDataset
from collections import Counter
from pyg_test_418 import GCN
from torch import nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix, roc_auc_score
import time
import numpy as np
from sklearn.preprocessing import label_binarize
import matplotlib as mpl
import os

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
        # x_com_soft = F.softmax(x_com, dim=0)
        return pred

def hook(module, input, output):
    features.append(output.detach())
    return None

if __name__ == "__main__":
    num_worker=5
    features = []
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_path = "/home/wuzheng/Packet/99code/lightning_logs/my_name_online2/checkpoints/last-v7.ckpt"
    pt_path = "/data/data/onlineDetect/onlineTest_pt/"
    name_list = os.listdir(pt_path)
    name_list = [i.rstrip('.pt') for i in sorted(name_list)[:-2]]
    
    for name in name_list:
        print("name:", name)
        dataset = MyOwnDataset(root = pt_path, version= f"{name}", pk_path= f"/data/data/onlineDetect/pkFile/{name}/")
        torch.manual_seed(12345)
        clf_model = MyDetector(hidden_channels=64, num_features=5, num_classes=4)

        print("Complete loading model!")

        y_pred_list = []
        ind_list = []
        
        model = clf_model.eval().to(device)

        x_com_list = []
        with torch.no_grad():       
            starter = time.time()        
            for i in range(len(dataset)):
                ind = dataset[i].ind            
                pred = model(dataset[i].to(device))
                # x_com_list.append(list(x_com[0])[1])
                ind_list.append(int(ind))
                y_pred_list.append(int(pred))
            ender = time.time()
        running_time= ender - starter
        # accuracy = accuracy_score(y_true=y_true_list, y_pred=y_pred_list)
        
        print("running time:",running_time/1440)
        # print("accuracy:", accuracy)
        # miss_rate, false_rate = compute_missing_rate(y_pred_list, y_true_list)
        # print(miss_rate, false_rate)
        # exit()
        # print(y_pred_list)
        print(Counter(y_pred_list))
        # print(x_com_list)
        # exit()
        # exit()
        # print(y_true_list)
        # for i in range(len(y_true_list)):
        #     if y_pred_list[i] == 0 and y_true_list[i] != 0:
        #         print(ind_list[i])
        # print(ind_list)
        
        # a = sorted(zip(range(len(ind_list), ind_list)))
        # ind_sorted = [i[1] for i in a]
        # print("ind_sorted:", ind_sorted)
        label_path = f"/home/wuzheng/Packet/99code/onlineAD/results/{name}.dat"
        with open(label_path, 'w') as fw:
            # fw.write(','.join([str(i) for i in y_true_list]))
            # fw.write('\n')
            fw.write(','.join([str(i) for i in ind_list]))
            fw.write('\n')
            fw.write(','.join([str(i) for i in y_pred_list]))
        print(f"{name} complete.")
        # '''
        # 提取输出层表征
        # '''
        # features = [i.squeeze().tolist() for i in features]
        # fw_path = "/home/wuzheng/Packet/99code/plot/feature_rl1.dat"
        # with open(fw_path, 'w') as fw:
        #     for f in features:
        #         fw.write(','.join([str(i) for i in f]))
        #         fw.write('\n')
        
        # print(features)
        # print('x_com_list:', x_com_list)
        # with open('/home/wuzheng/Packet/99code/GCN/linear_output_ph.txt', 'w') as fw:    
        #     fw.write(','.join([str(float(i)) for i in x_com_list]))
        #     fw.write('\n')
        #     fw.write(','.join([str(i) for i in y_true_list]))
        #     fw.write('\n')
        #     fw.write(','.join([str(i) for i in y_pred_list]))
        #     fw.write('\n')
        #     fw.write(','.join([str(i) for i in ind_list]))
        
        # accuracy = accuracy_score(y_true=y_true_list, y_pred=y_pred_list)
        # print("accuracy:", accuracy)
        # # miss_rate, false_rate = compute_missing_rate(y_pred_list, y_true_list)
        # y_pred_list = np.array(y_pred_list)
        # y_true_list = np.array(y_true_list)
        
        # y_true_binarized = label_binarize(y_true_list, classes=[0, 1, 2, 3])
        # y_pred_binarized = label_binarize(y_pred_list, classes=[0, 1, 2, 3])
        
        # print("AUC: ", roc_auc_score(y_true=y_true_binarized, y_score=y_pred_binarized, average='macro', multi_class='ovr'))
        # print(f"False Negative Rate: {miss_rate}, False Positive Rate: {false_rate}")
        # print('running time:', running_time / len(y_pred_list))