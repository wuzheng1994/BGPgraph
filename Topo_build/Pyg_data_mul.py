import os
import json
import time
from Routes_vp.Routes_vp import Routes
# from Data_generator import data_generator_wlabel
from Data_generator_random_sampling import data_generator_wlabel
import networkx as nx
import pickle
import torch
import numpy as np
from mysql_connect import Data_from_mysql
from multiprocessing import Queue, Process, JoinableQueue, Manager, Lock

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


def get_link_prefix_dict(Interval, link_prefix_dict):
    '''
    calculate the weight of prefix from self.link_prefix_dict.
    '''
    start = time.time()
    results_ = {k_id: compute_con_value(p_l) for k_id, p_l in link_prefix_dict.items()}
    print('end:', time.time() - start)
    weight_value = edge_weight_computing(Interval, results_.values())
    link_weight = dict(zip(results_.keys(), weight_value))
    return link_weight

def compute_con_value(p_l):
    weight = 0
    for p in p_l:
        if '.' in p:
            weight += (32 - int(p.split('/')[1])) / 32
        elif ":" in p:
            weight += (128 - int(p.split('/')[1])) / 128
    return weight

def edge_weight_computing(interval, edge_value):
    if not isinstance(edge_value, list):
        edge_value = list(edge_value)
    edge_weight = np.digitize(edge_value, interval)
    return edge_weight

def Producer(path, save_path, data_path, anomaly_time, event_time, q, Interval, database_info, sampling_rate):
    print("Have started Producer!")
    priming_path = path + 'priming_data/txt/'
    r = Routes(priming_path)
    r.collect_routes() # 收集路由
    r.select_vps # 选择采集点
    r1 = r.get_route # 对vp进行清理，并进行输出
    r.link_compute() # 收集self.link_prefix_list
    print('the length of key:', len(list(r1.keys())))
    r.get_interval() # 获取间隔
    
    for i in range(6):
        Interval[i] = r.interval_[i]
    # 需要修改属性值
    print('Complete collection!')
    Period = 1
    updates_files = sorted([os.path.join(data_path, i) for i in os.listdir(data_path)])
    init_graph = buildGraph(r1)
    G = init_graph

    degree = nx.degree(G)
    asns = list(G.nodes)
    
    # get data from database
    database.fetch_asnset(asns)
    database_info = database.get_asn_info
    pagerank = nx.pagerank(G)
    feature = {k_id: [*database_info[k_id], pagerank[k_id], degree[k_id]] for k_id in nx.nodes(G)} # 添加边属性等

    # 修改内部属性，同时返回边属性
    link_dict = r.get_link_prefix_dict

    # 设置节点和边属性
    nx.set_edge_attributes(G, link_dict, 'prefix_count')
    nx.set_node_attributes(G, feature, 'feature')

    with open(os.path.join(save_path, '0.pkl'), 'wb') as f:
        pickle.dump(graph_data(G, 0, 0), f)
    
    a = data_generator_wlabel(updates_files, Period, start_time= event_time[0], end_time= event_time[1], anomaly_start_time= anomaly_time[0], anomaly_end_time=anomaly_time[1], sampling_rate=sampling_rate)
    count = 1
    graph = G
    for update in a:
        label = update[1]
        print('label:',label)
        update = update[0]
        # 构建图结构：根据窗口内的Updates计算需要添加的边和去掉的边
        addedge, removeedge = r.compute_edge(update, directed=True)
        # 根据添加和去掉的边，更新图结构
        graph = updateGraph(graph, addedge, removeedge)
        link_prefix = dict(r.link_prefix_dict)
        q.put((graph, label, count, link_prefix))
        G = graph
        count += 1
    q.join()    

def Customer(save_path, q, Interval):
    pid = os.getpid()
    print('Have started Consumer!', pid)
    # GF = GraphFeature()
    while True:
        graph, label, idx, link_prefix_dict = q.get()
        degree = nx.degree(graph)
        # 根据图计算pagerank值
        pagerank = nx.pagerank(graph)
        # 从数据库中提取数据
        asns = set(graph.nodes)
        asndiff = asns - database.asn_info_dict.keys() # the difference of asn set between asn and existing asn.
        print("the length of asndiff:", len(asndiff))
        if asndiff:
            database.fetch_asnset(asndiff)
        database_info = database.get_asn_info
        print("Interval:", Interval)
        link_dict = get_link_prefix_dict(Interval, link_prefix_dict)
        feature = {k_id: [*database_info[k_id], pagerank[k_id], degree[k_id]] for k_id in nx.nodes(graph)}
        # 将特征融合在图中
        nx.set_node_attributes(graph, feature,'feature')

        nx.set_edge_attributes(graph ,link_dict, 'prefix_count')
        with open(os.path.join(save_path,'{}.pkl').format(idx), 'wb') as f:
            pickle.dump(graph_data(graph, label, idx), f)
        print(f"Have complete {idx} sample!")
        q.task_done()

if __name__ == "__main__":
    
    database = Data_from_mysql("192.168.1.79", username="root", database = "Stream")
    lock = Lock()
    # collector = 'route-views.chicago'
    # path = "/data/wuzheng_data/anomaly_event/route_leak/event2/"
    # data_path = path + 'txt/' + collector
    # event_time = ("2019-06-24 00:00:00", "2019-06-25 00:00:00")
    # anomaly_time = ("2019-06-24 10:34:25", "2019-06-24 12:38:54")
    # # save_path = '/home/wuzheng/Feature_extraction/data/Rl2/'
    # save_path = '/data/wuzheng_data/anomaly_event/Data/Rl2_sampling/Rl2_sampling_0.4/'
    
    # collector = 'route-views3'
    # path = "/data/wuzheng_data/anomaly_event/prefix_hijack/event2/"
    # data_path = path + 'txt/' + collector
    # event_time = ("2020-7-30 00:00:00", "2020-7-31 00:00:00")
    # anomaly_time = ("2020-7-30 00:55:00", "2020-7-30 02:35:00")
    # save_path = '/data/wuzheng_data/anomaly_event/Data/Prefix_hijack1_sampling/Prefix_hijack1_sampling_0.4/'

    # collector = 'route-views.amsix'
    # path = "/data/wuzheng_data/anomaly_event/outage/event1/"
    # data_path = path + 'txt/' + collector
    # event_time = ("2021-10-04 00:00:00", "2021-10-05 00:00:00")
    # anomaly_time = ("2021-10-04 15:07:00", "2021-10-04 21:49:00")
    # save_path = '/data/wuzheng_data/anomaly_event/Data/Outage1_sampling_0.4'
    
    collector = 'rrc01'
    path = "/data/wuzheng_data/anomaly_event/route_leak/event1/"
    data_path = path + 'txt/' + collector
    event_time = ("2018-11-12 00:00:00", "2018-11-13 00:00:00")
    anomaly_time = ("2018-11-12 21:12:00", "2018-11-12 22:32:00")
    save_path = '/data/wuzheng_data/anomaly_event/Data/Rl1_sampling/Rl1_sampling_0.2/'
    
    # collector = 'route-views.chile'
    # path = "/data/wuzheng_data/anomaly_event/outage/event1/"
    # data_path = path + 'txt/' + collector
    # event_time = ("2022-06-21 00:00:00", "2022-06-22 00:00:00")
    # anomaly_time = ("2022-06-21 06:27:00", "2022-06-21 07:42:00")
    # save_path = '/data/wuzheng_data/anomaly_event/Data/Outage2/'
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    start_time = time.time()
    q = JoinableQueue(10)
    period = 1
    sampling_rate = 0.2
    manager = Manager()
    Interval = manager.Array('f', [0.0]*6, lock =False)
    database_info = manager.dict(lock=True)
    producer = Process(target=Producer, args=(path, save_path, data_path, anomaly_time, event_time, q, Interval, database_info, sampling_rate))
    
    num_consumer = 2
    consumer_list = []
    for i in range(num_consumer):
        consumer_list.append(Process(target=Customer, args=(save_path, q, Interval)))
    
    producer.start()

    for i in range(num_consumer):
        consumer_list[i].daemon = True

    for i in range(num_consumer):
        consumer_list[i].start()
    
    producer.join()
    print('producer Over!')
    
    
    database.finsh_fetch()
    duration = time.time() - start_time
    print('The elapse time is ',duration)