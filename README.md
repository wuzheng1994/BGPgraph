# BGPgraph

## Abstract
Detecting anomalous BGP (Border Gateway Protocol) traffic is crucial for ensuring the security of autonomous system (AS)-level networks. Due to the dynamic nature of BGP routes, the large volume of routes involved, and the limited visibility of the whole topology, it is difficult to identify BGP anomalies and even harder to trace anomalous ASes. Although many works have been proposed recently, no existing methods have effectively addressed the problem of detecting BGP anomalies and locating the anomaly sources due to the above challenges. This paper proposes a novel BGP anomaly detection method named GraphBGP. GraphBGP can obtain the updated AS-level graph timely, accurately detect BGP anomalies, and identify the anomaly type while being able to trace the problematic ASes precisely. Specifically, considering the evolving nature of BGP status, GraphBGP models the AS-level network as a graph, embeds it with node and edge attributes, and updates the graph timely by effectively and intelligently tracking BGP updates. Based on the up-to-date and enriched AS graph, GraphBGP customizes the detection and tracing models based on graph convolutional networks (GCN), achieving accurate and seamless detection and location of anomalies. Comprehensive experiments on the real-world and synthetic datasets demonstrate that GraphBGP achieves the highest accuracy in anomaly detection while requiring much less inference time, even under incomplete visibility of the BGP networks. Moreover, GraphBGP can accurately trace the problematic ASes within a time interval of 0.007s after detecting the anomaly.

## Organization
In this paper, the framework of BGPGraph includes data collection, method design, instance labeling, and online anomlay detections.

### 1. Topo_construction 
---
<div align="center">
  <img src="figure 1.png" alt="示例图片" width="300">
</div>   

``Data_generator.py`` 数据生成器，按照固定窗口生成Updates消息  
``Routes.py`` 增量更新topo所用信息  
``Pyg_data_mul.py`` 增量更新拓扑并提取相关特征

<div align="center">
<img src="figure 2.png" alt="本地图片">
</div>

### 2. GCN_model & Autoencoder   
---   
``python3 pyg_tesy_418.py `` 训练GCN模型   
 
``python3 autoencoder.py `` 加载GCN的预训练模型，并且训练auto-encoder模型       
 
``python3 Mydataset.py`` 加载所用GCN数据，包括节点属性、边属性和邻接矩阵

``test_autoencoder.py`` 测试auto-encoder模型使用  

``test_GCN.py`` 测试GCN模型使用   

## Requirements
networkx --2.8.8  
torch -- 1.13.1  
torch_geometric -- 2.4.0  
torch-scatter -- 2.1.2
pytorch_lightning -- 2.1.3