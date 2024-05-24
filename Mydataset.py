import torch
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.utils import from_networkx
from torch_geometric.loader import DataLoader
import networkx as nx
import numpy as np
import os
import pickle

class graph_data:
    def __init__(self, graph, label, ind):
        self.data = graph.edges()
        self.x = nx.get_node_attributes(graph, 'feature')
        self.label = label
        self.ind = ind
        self.edge_attribute = nx.get_edge_attributes(graph, 'prefix_count')

# def convert_node_label(edge, node_sequence, first_label=0):
#     '''
#     relabel the node name by sorted method for the correspoding relationship between node and adj.
#     Parameter:
#         edge: list
#         x: torch.tensor

#     output:
#         new_edge: the edges after mapping
#         mapping: the mapping relationship
#     '''
#     nodes = sorted(list(node_sequence))
#     N = len(nodes)
#     mapping = dict(zip(nodes, range(first_label, first_label + N)))
    
#     new_edge = []
#     for i, j in edge:
#         new_edge.append([mapping[i], mapping[j]])
#     return new_edge, mapping

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
    
    # print('add___:',mapping_attribute)
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

class MyOwnDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None, num_worker=4):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])
        # self.y_ = []
        self.num_worker = num_worker
        print(self.processed_paths[0])
        # self.root = '/home/wuzheng/Packet/Graph_network/graph_data4/'
    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ['data.pt']

    # @property
    # def get_label(self):
    #     return self.y_
    
    def process(self):
        # Read data into huge `Data` list.
        files = sorted(os.listdir(self.raw_dir))
        files_path = [os.path.join(self.raw_dir, i) for i in files ]
        print(files_path)
        class_ = {'Rl': 1,'Prefix':2, 'Outage': 3, 'normal':0}
        token = None
        data_list = []
        count_ = 0
        for file in files_path:
            print(file)
            token = None
            for k_id in class_:
                if k_id in file:
                    token = class_[k_id]
            pls = os.listdir(file)
            for pl in pls:
                pl_path = os.path.join(file, pl)
                try:
                    with open(pl_path,'rb') as p:
                        graph_data = pickle.load(p)
                except:
                    print('the file is {}'.format(pl_path))
                    continue
                edges = list(graph_data.data)
                # self.y_.append(graph_data.label)
                if graph_data.label == '1':
                    label = torch.Tensor([token]).long()
                else:
                    label = torch.Tensor([0]).long()
                
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

                nor_x = torch.FloatTensor([f[1][:2] + [int(f[1][2])]+f[1][3:] for f in feature_mapped])

                # edge_attribute_o = data.edge_attribute
                edge_attribute = edge_attribute_transform(new_edges, mapping, edge_attribute_o)  # 1111111
                edge_attribute = torch.Tensor(edge_attribute).unsqueeze(1)
                
                data = Data(edge_index= edge_index, y= label, x= nor_x, edge_attr= edge_attribute, mapping=list(node_sequence))
                data.ind = torch.Tensor([graph_data.ind])
                data_list.append(data)
                count_ += 1
                if count_ % 100 == 0:
                    print('Haved completed the {} hundred: {}'.format(count_ / 100, count_))

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        # print("data_list:", data_list)
        data, slices = self.collate(data_list)
        
        torch.save((data, slices), self.processed_paths[0])

if __name__ == "__main__":
    
    dataset = MyOwnDataset('/data/data/graph_data5/')
    loader = DataLoader(dataset, batch_size=10)

    print('The number of samples',len(dataset))
    print(dataset[0].edge_attr.size())

