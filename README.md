# BGPgraph


### Organization of code
<div style="text-align: center;",align="center">
<img src="figure 1.png" alt="本地图片"， width=300>
</div>  

(1) ./Topo_construction    
``Data_generator.py`` 数据生成器，按照固定窗口生成Updates消息  
``Routes.py`` 增量更新topo所用信息  
``Pyg_data_mul.py`` 增量更新拓扑并提取相关特征

<div style="text-align: center;",align="center">
<img src="figure 2.png" alt="本地图片">
</div>

(2) ./GCN_model  
``pyg_test_418.py`` 训练GCN模型  
``python3 pyg_tesy_418.py ``  
``autoencoder.py`` 加载GCN的预训练模型，并且训练auto-encoder模型
``python3 autoencoder.py ``  
``Mydataset.py`` 加载所用GCN数据，包括节点属性、边属性和邻接矩阵
``test_autoencoder.py`` 测试auto-encoder模型使用  
``test_GCN.py`` 测试GCN模型使用  

### Several toolkits may be needed to run the code
networkx --2.8.8  
torch -- 1.13.1  
torch_geometric -- 2.4.0  
torch-scatter -- 2.1.2
pytorch_lightning -- 2.1.3