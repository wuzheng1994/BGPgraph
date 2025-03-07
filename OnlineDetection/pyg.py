#encoding: utf-8
import os
import time
import networkx as nx
import pickle
import torch
import numpy as np
from mysql_connect import Data_from_mysql
import sys
# sys.path.append('/home/wuzheng/Feature_extraction/')
from Routes_vp import Routes
from Data_generator import data_generator_wlabel
from multiprocessing import Manager
from itertools import repeat
from collections import OrderedDict

# second time transfer to '%Y-%m-%d %H:%M:%S'.
def s2t(seconds:int) -> str:
    utcTime = time.gmtime(seconds)
    strTime = time.strftime("%Y-%m-%d %H:%M:%S",utcTime)
    return strTime

# str time transfer to second time.
def t2s(str_time:str) -> int:
    time_format = '%Y-%m-%d %H:%M:%S'
    time_int = int(time.mktime(time.strptime(str_time, time_format)))
    return time_int

'''
build the network by the routes.
'''

def buildGraph(routes):
    # build the graph from priming data.
    graph = nx.Graph()
    edges = set()
    for peer_as in routes.keys():
        for prefix in routes[peer_as]:
            if routes[peer_as][prefix] != None:
                as_path = routes[peer_as][prefix]
                if not ('{' in as_path):
                    as_list = as_path.split(' ')
                    for i in range(len(as_list)-1):
                        if as_list[i] != as_list[i+1]:
                            edges.add((as_list[i], as_list[i+1]))                 
    graph.add_edges_from(edges)
    return graph

def nx_to_pyg(G):
    adj = nx.to_scipy_sparse_array(G).tocoo()
    row = torch.from_numpy(adj.row.astype(np.int64)).to(torch.long)
    col = torch.from_numpy(adj.col.astype(np.int64)).to(torch.long)
    edge_index = torch.stack([row, col], dim=0)
    print('edge:\n',edge_index)
    return edge_index

def updateGraph(G, addedge, removeedge):
    G.add_edges_from(addedge)
    G.remove_edges_from(removeedge)
    return G

class graph_data:
    def __init__(self, graph, label, ind):
        self.data = graph.edges()
        self.x = nx.get_node_attributes(graph, 'feature')
        self.label = label
        self.ind = ind
        self.edge_attribute = nx.get_edge_attributes(graph, 'prefix_count')

def node_degree(graph):
    return nx.degree(graph)

def file2second(filelist):
    '''
    transform for filelist
    '''
    file_dir = {}
    for l in filelist:
        timestamp = l.split('.')[1:3]
        timestring = timestamp[0][:4] + '-' + timestamp[0][4:6] + '-' +  timestamp[0][6:8] + " " + timestamp[1][:2] + ":" + timestamp[1][2:4] + ":" + "00"
        int_num = t2s(timestring)
        file_dir[int_num] = l
    return file_dir

def file2second_(file):
    '''
    param: file the priming_file
    function: transform for single file
    
    return: 
    start timestring
    end timestring
    '''
    timestamp = file.split('.')[1:3]
    startstring = timestamp[0][:4] + '-' + timestamp[0][4:6] + '-' +  timestamp[0][6:8] + " " + timestamp[1][:2] + ":" + timestamp[1][2:4] + ":" + "00"
    num_int = t2s(startstring)
    endstring = s2t(num_int + refresh_time)
    return startstring, endstring

def worker(input_):

    (priming_file, updates_files), d = input_ 
    # date_time = priming_file.split('.')[1]
    updates_files_path = updates_files
    start_time, end_time = file2second_(priming_file)

    filedir = priming_file.split('.')[1]
    # print("priming_file:",priming_file)
    # print("updates_file:", updates_files_path)
    save_path_day = os.path.join(save_path, filedir)
    if not os.path.exists(save_path_day):
        os.mkdir(save_path_day)
    
    r = Routes(priming_file)
    r.collect_routes() # collect route
    
    r1 = r.get_route 
    r.link_compute() 
    # print('the length of key:', len(list(r1.keys())))
    r.get_interval()
    print('Complete collection!')
    Period = 1
    # updates_files = sorted([os.path.join(data_path, i) for i in os.listdir(data_path)])
    init_graph = buildGraph(r1)
    G = init_graph

    degree = nx.degree(G)

    pagerank = nx.pagerank(G)
    feature = {} 
    for k_id in nx.nodes(G):
        if d.get(k_id):
            feature[k_id] = [*d[k_id], pagerank[k_id], degree[k_id]]
        else: 
            feature[k_id] = [0,0,"75725", pagerank[k_id], degree[k_id]]
        
    # feature = {k_id: [*d[k_id], pagerank[k_id], degree[k_id]] if d.get(k_id) else: [0,0,"75725", pagerank[k_id], degree[k_id]] for k_id in nx.nodes(G)} 

    link_dict = r.get_link_prefix_dict

    nx.set_edge_attributes(G, link_dict, 'prefix_count')
    nx.set_node_attributes(G, feature, 'feature')

    with open(os.path.join(save_path_day, '0.pkl'), 'wb') as f:
        pickle.dump(graph_data(G, "0", 0), f)

    a = data_generator_wlabel(updates_files_path, Period, start_time= start_time, end_time= end_time, anomaly_start_time= start_time, anomaly_end_time=end_time)
    # print("have construct the generator!")
    count = 1
    graph = G
    for update in a:
        # print('Enter loop!')
        starter = time.time()
        label = 0
        # print('label:',label)
        # update = update
        # print(update)
        
        addedge, removeedge = r.compute_edge(update, directed=True)
        
        graph = updateGraph(graph, addedge, removeedge)
        # print("Have updated!")
        ###########################################################################################

        degree = nx.degree(graph)
        # 根据图计算pagerank值
        pagerank = nx.pagerank(graph)
        # 从数据库中提取数据
        # asns = set(G.nodes)  
        feature = {} 
        for k_id in nx.nodes(G):
            if d.get(k_id):
                feature[k_id] = [*d[k_id], pagerank[k_id], degree[k_id]]
            else: 
                feature[k_id] = [0, 0, "75725", pagerank[k_id], degree[k_id]]
        # feature = {k_id: [*d[k_id], pagerank[k_id], degree[k_id]] for k_id in nx.nodes(G)}
        # 将特征融合在图中
        nx.set_node_attributes(G, feature,'feature')
        link_dict = r.get_link_prefix_dict # 可以采用两种策略：根据updates数量：1.  整体计算（已完成）；2. 计算变化的部分, 现在采取第二部分

        nx.set_edge_attributes(G ,link_dict, 'prefix_count')
        ender = time.time()
        print("elpase:", ender - starter)
        
        with open(os.path.join(save_path_day,'{}.pkl').format(count), 'wb') as f:
            pickle.dump(graph_data(graph, label, count), f)
        print(f"{filedir}: the {count} has completed!")
        count += 1
    print("The day {} has completed".format(filedir))    


if __name__ == "__main__":
    
    # initialized info.
    database = Data_from_mysql("192.168.1.79", username="root", database = "Stream")
    database.fetdata_from_mysql('All')
    database_info = database.get_asn_info
    database.finsh_fetch()
    
    path = "/data/data/onlineDetect/onlineTest_txtgb/"
    # priming_path = '/data/data/onlineDetect/onlineTest_txtgb/rrc00_bview.20240701.0000.txt'

    files = os.listdir(path=path)
    files = sorted(files, key=lambda x: x[4:])
    save_path = "/data/data/onlineDetect/pkFile/"
    
    rib_files = os.listdir('/data/data/onlineDetect/onlineTest_ribtxt/')
    rib_name_dict = {}
    for rib in rib_files:
        d = rib.split('.')[1]
        rib_name_dict[d] = rib
    # print("rib_name_dict:", rib_name_dict)
    task_list = []
    # print("The length of files:", len(files[1:]))
    # print(list(zip(files,range(len(files)))))
    # exit()
    for file in files[21:60]:
        update_path = os.path.join(path , file)
        event_time = (f"{file[0:4]}-{file[4:6]}-{file[6:8]} 00:00:00", f"{file[0:4]}-{file[4:6]}-{file[6:8]} 23:59:59") 
        # print(event_time[0])
        # event_time_start = t2s(event_time[0])
        # event_time_end = t2s(event_time[1])
        # priming_dir = '/data/data/onlineDetect/onlineTest_txtgb/'
        # print("Name:", rib_name_dict['20240809'])
        if rib_name_dict.get(file):
            priming_path = f'/data/data/onlineDetect/onlineTest_ribtxt/{rib_name_dict[file]}'
        else:
            continue
        
        refresh_time = 24 * 3600
        updates_files = os.listdir(update_path)
        
        updates_sorted = file2second(updates_files)

        updates_sorted = [os.path.join(update_path, i[1]) for i in sorted(updates_sorted.items(), key=lambda x: x[0])]
        # print(updates_sorted)
        # exit()

        task_list.append((priming_path, updates_sorted))
        # shared_d = {}
        # for k in database_info:
        #     shared_d[k] = database_info[k]
        # worker(((priming_path, updates_sorted_path),shared_d))
        # exit()
    # print(task_list)
    # exit()
    print("Begining work!")
    from tqdm.contrib.concurrent import process_map
    num_worker = 5
    with Manager() as manager:
        shared_d = manager.dict()
        for k in database_info:
            shared_d[k] = database_info[k]
        task_list_share = zip(task_list, list(repeat(shared_d, len(task_list))))
        res = process_map(worker, task_list_share, max_workers=num_worker)
        print('Over!')

