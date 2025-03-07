import torch
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.utils import from_networkx
from torch_geometric.loader import DataLoader
import networkx as nx
import numpy as np
import os
import pickle
from concurrent.futures import ThreadPoolExecutor, as_completed
import tqdm
import argparse

'''
multiprocessing version
'''

class graph_data:
    def __init__(self, graph, label, ind):
        self.data = graph.edges()
        self.x = nx.get_node_attributes(graph, 'feature')
        self.label = label
        self.ind = ind
        self.edge_attribute = nx.get_edge_attributes(graph, 'prefix_count')

def convert_node_label_v2(edges, node_sequence):
    nodes = node_sequence
    node_ind = range(len(node_sequence))
    mapping = dict(zip(nodes, node_ind))
    edges_transformed = [(mapping[e[0]], mapping[e[1]]) for e in edges]
    return edges_transformed, mapping

def edge_attribute_transform(new_edge, mapping, edge_attribute_o):
    
    '''
    Parameter:
    Input:
        new_edge: list # 映射过后的边
        mapping: dict
    Output:
        edge_attribute
    
    1. 将 edge_attribute_o 转化为mapping后的mapping_attribute{}；
    2. 按照edge_index,对attribute进行排序；
    '''

    mapping_attribute = {}
    for edge in edge_attribute_o:
        mapping_attribute[(mapping[edge[0]], mapping[edge[1]])] = edge_attribute_o[edge]
    
    edge_attribute = []
    
    for n in range(len(new_edge)):
        edges = (new_edge[n][0], new_edge[n][1])
        # 如果找不到怎么办
        if mapping_attribute.get(edges):
            edge_attribute.append(mapping_attribute[edges])
        elif mapping_attribute.get((edges[1], edges[0])):
            edge_attribute.append(mapping_attribute[(edges[1], edges[0])])
        else:
            edge_attribute.append(0)
    return edge_attribute

def worker(pl_path):
    # print('in worker!')
    try:
        with open(pl_path,'rb') as p:
            graph_data = pickle.load(p)
    except:
        return None

    edges = list(graph_data.data)
    label_b = int(graph_data.label)
    label_ = torch.Tensor([label_b]).long()
    
    edge_attribute_o = graph_data.edge_attribute
    feature = graph_data.x
    node_sequence = feature.keys()
    new_edges, mapping = convert_node_label_v2(edges, node_sequence)
    edge_index = torch.tensor(new_edges, dtype=torch.long).t().contiguous()
    edge_index = edge_index.view(2, -1)
    feature_mapped = []
    
    # feature mapping
    for k_id in feature:
        if mapping.get(k_id) != None:
            feature_mapped.append((mapping[k_id], feature[k_id]))
        else:
            print('Error!', k_id)
    feature_mapped = feature.values()
    # feature_mapped = sorted(feature_mapped, key=lambda x: x[0])

    nor_x = torch.FloatTensor([f[:2] + [int(f[2])]+f[3:] for f in feature_mapped])
    node_sequence = torch.tensor([int(n) for n in node_sequence], dtype=torch.int32)
    
    # edge_attribute_o = data.edge_attribute
    edge_attribute = edge_attribute_transform(new_edges, mapping, edge_attribute_o)  # 1111111
    edge_attribute = torch.Tensor(edge_attribute).unsqueeze(1)
    data = Data(edge_index= edge_index, y= label_, x= nor_x, edge_attr= edge_attribute, mapping=node_sequence)
    data.ind = torch.Tensor([graph_data.ind])
    return data

class MyOwnDataset(InMemoryDataset):
    def __init__(
        self, 
        root,
        version,
        pk_path,
        transform=None, 
        pre_transform=None, 
        pre_filter=None):
        self.version = version
        self.pk_path = pk_path
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])
        
        # self.raw_dir = pk_path
        # self.num_worker = num_worker
        # print("processed:", self.processed_paths[0])
        # print("workers:",self.num_worker)
    
    @property
    def raw_dir(self):
        # 自定义 raw_dir 路径
        return self.pk_path
    
    @property
    def raw_file_names(self):
        return []

    @property
    def processed_dir(self):
        # 自定义 processed_dir 路径
        return self.root
    
    @property
    def processed_file_names(self):
        return [f'{self.version}.pt']

    def process(self):
        # Read data into huge `Data` list.
        # print(self.raw_dir)
        files = sorted(os.listdir(self.raw_dir), key=lambda x: int(x.rstrip('.pkl')))
        # print("raw_files:", files)
        # exit()
        files_path = [os.path.join(self.raw_dir, i) for i in files]
        print("The number of files:",len(files_path))

        data_list = []

        from tqdm.contrib.concurrent import process_map
        # from multiprocessing import Pool
        data_list = process_map(worker, files_path, max_workers=num_worker, chunksize=1)
        data_list = [data for data in data_list if data is not None]
        # with Pool(processes=5) as pool:        
        #     data_list = pool.imap(worker, files_path[-5:], chunksize=1)
        #     data_list = [data for data in data_list if data is not None]
        # with ThreadPoolExecutor(max_workers=num_worker) as executor:
        #     result = list(tqdm(executor.map(worker, files_path), total=len(files_path)))
        
        # with tqdm(total=len(data), desc="Processing") as pbar:
        #     for _ in as_completed(result):
        #         pbar.update(1)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        # print('data_list:', data_list)
        data, slices = self.collate(data_list)
        
        torch.save((data, slices), self.processed_paths[0])


def main():
    parser = argparse.ArgumentParser(description='程序功能描述')
    parser.add_argument('-n', '--name', type=str, help='name')
    args = parser.parse_args()
    name = args.name
    print(f"Processing {name}...")
    pk_path = f"/data/data/onlineDetect/pkFile/{name}/"
    pt_path = "/data/data/onlineDetect/onlineTest_pt/"
    dataset = MyOwnDataset(root=pt_path, version=name, pk_path=pk_path)
    print('The number of samples',len(dataset))


if __name__ == "__main__":
    num_worker = 4
    main()
    # name_list = os.listdir("/data/data/onlineDetect/pkFile/")
    # name = "20240724"
    # # print(sorted(name_list))
    # # exit()
    # # for name in sorted(name_list)[6:]:
    # print(f"Processing {name}...")
    # pk_path = f"/data/data/onlineDetect/pkFile/{name}/"
    pt_path = "/data/data/onlineDetect/onlineTest_pt/"
    # # file_dir = os.listdir(pk_path)
    # # for file in file_dir:
    #     # class_ = {'Rl': 1,'Prefix':2, 'Outage': 3, 'normal':0}
    # dataset = MyOwnDataset(root=pt_path, version=name)
    # # dataset = MyOwnDataset(root ='/data/data/graph_data6/Outage1/', num_worker=5)
    # # loader = DataLoader(dataset, batch_size=10)

    # print('The number of samples',len(dataset))