# encoding: utf-8
import os
import numpy as np
from collections import defaultdict
# import pandas as pd
import time

from numba import njit


'''
return the state of route.
'''

prefix_weight_dict_v4 = {0:0, 1: 0.96875, 2: 0.9375, 3: 0.90625, 4: 0.875, 5: 0.84375, 6: 0.8125, 7: 0.78125, 8: 0.75, 9: 0.71875, 10: 0.6875, 11: 0.65625, 12: 0.625, 13: 0.59375, 14: 0.5625, 15: 0.53125, 16: 0.5, 17: 0.46875, 18: 0.4375, 19: 0.40625, 20: 0.375, 21: 0.34375, 22: 0.3125, 23: 0.28125, 24: 0.25, 25: 0.21875, 26: 0.1875, 27: 0.15625, 28: 0.125, 29: 0.09375, 30: 0.0625, 31: 0.03125, 32: 0.0}
prefix_weight_dict_v6 = {0:0, 1: 0.9921875, 2: 0.984375, 3: 0.9765625, 4: 0.96875, 5: 0.9609375, 6: 0.953125, 7: 0.9453125, 8: 0.9375, 9: 0.9296875, 10: 0.921875, 11: 0.9140625, 12: 0.90625, 13: 0.8984375, 14: 0.890625, 15: 0.8828125, 16: 0.875, 17: 0.8671875, 18: 0.859375, 19: 0.8515625, 20: 0.84375, 21: 0.8359375, 22: 0.828125, 23: 0.8203125, 24: 0.8125, 25: 0.8046875, 26: 0.796875, 27: 0.7890625, 28: 0.78125, 29: 0.7734375, 30: 0.765625, 31: 0.7578125, 32: 0.75, 33: 0.7421875, 34: 0.734375, 35: 0.7265625, 36: 0.71875, 37: 0.7109375, 38: 0.703125, 39: 0.6953125, 40: 0.6875, 41: 0.6796875, 42: 0.671875, 43: 0.6640625, 44: 0.65625, 45: 0.6484375, 46: 0.640625, 47: 0.6328125, 48: 0.625, 49: 0.6171875, 50: 0.609375, 51: 0.6015625, 52: 0.59375, 53: 0.5859375, 54: 0.578125, 55: 0.5703125, 56: 0.5625, 57: 0.5546875, 58: 0.546875, 59: 0.5390625, 60: 0.53125, 61: 0.5234375, 62: 0.515625, 63: 0.5078125, 64: 0.5, 65: 0.4921875, 66: 0.484375, 67: 0.4765625, 68: 0.46875, 69: 0.4609375, 70: 0.453125, 71: 0.4453125, 72: 0.4375, 73: 0.4296875, 74: 0.421875, 75: 0.4140625, 76: 0.40625, 77: 0.3984375, 78: 0.390625, 79: 0.3828125, 80: 0.375, 81: 0.3671875, 82: 0.359375, 83: 0.3515625, 84: 0.34375, 85: 0.3359375, 86: 0.328125, 87: 0.3203125, 88: 0.3125, 89: 0.3046875, 90: 0.296875, 91: 0.2890625, 92: 0.28125, 93: 0.2734375, 94: 0.265625, 95: 0.2578125, 96: 0.25, 97: 0.2421875, 98: 0.234375, 99: 0.2265625, 100: 0.21875, 101: 0.2109375, 102: 0.203125, 103: 0.1953125, 104: 0.1875, 105: 0.1796875, 106: 0.171875, 107: 0.1640625, 108: 0.15625, 109: 0.1484375, 110: 0.140625, 111: 0.1328125, 112: 0.125, 113: 0.1171875, 114: 0.109375, 115: 0.1015625, 116: 0.09375, 117: 0.0859375, 118: 0.078125, 119: 0.0703125, 120: 0.0625, 121: 0.0546875, 122: 0.046875, 123: 0.0390625, 124: 0.03125, 125: 0.0234375, 126: 0.015625, 127: 0.0078125, 128: 0.0}



class Routes():
    '''
    routes:
    {
        peer_as: {prefix: [as_path]}
    }
    
    '''
    def __init__(self, path):  
        self.path = path
        self.routes_ = defaultdict(lambda: defaultdict(str))
        self.mode = "ribs"
        self.link_prefix_dict = defaultdict(lambda: set())
        self.selected_vp = None
        self.interval_ = None
        self.link_weight_counter = defaultdict(lambda:float())
        
    def collect_routes(self):
        routes_ = defaultdict(lambda: defaultdict(str))
        if self.mode == 'updates':
            files = os.listdir(self.path)
            for f in files:
                f_path = os.path.join(self.path, f)
                with open(f_path,'r') as file:
                    for l in file:
                        if l.strip() != '':
                            line = l.strip().split('|')
                            prefix_ = line[5]
                            peer_asn = line[4]
                            op_ = line[2]
                            path_ = line[6]
                            if op_ == 'A':
                                if '{' not in path_:
                                    if not routes_[peer_asn].get(prefix_) or len(path_.split(" ")) < len(routes_[peer_asn][prefix_]):
                                        routes_[peer_asn][prefix_] = path_
                            elif op_ == 'W':
                                routes_[peer_asn][prefix_] = None
                            else:
                                pass
            
        elif self.mode == 'ribs':
            # print('begining collecting the ribs info')
            if os.path.isdir(self.path):
                files = os.listdir(self.path)
            else:
                files= [self.path]
            for f in files:
                f_path = os.path.join(self.path,f)
                print("f_path:", f_path)
                with open(f_path, 'r') as file:
                    for l in file:
                        if l.strip() != '':
                            line = l.strip().split('|')
                            prefix_ = line[5]
                            peer_asn = line[4]
                            op_ = line[2]
                            path_ = line[6]
                            if '{' not in path_:
                                if not routes_[peer_asn].get(prefix_) or len(path_.split(" ")) < len(routes_[peer_asn][prefix_]): # 选出最短路径
                                    routes_[peer_asn][prefix_] = path_
                            else:
                                pass
        self.routes_ = routes_  

    @property
    def select_vps(self):
        vp_dict = {}
        for vp in self.routes_.keys():
            vp_dict[vp] = len(self.routes_[vp])
        self.selected_vp = [vp for vp in vp_dict if vp_dict[vp] > 1e3]  # 修改

    @property
    def get_route(self):
        '''
        get routes
        '''
        # 对VP进行清理，然后输出
        if self.select_vps != None:
            print(self.select_vps)
            rm_vp = set(self.routes_.keys()) - set(self.selected_vp)
            if len(rm_vp) != 0:
                for vp in rm_vp:
                    self.routes_.pop(vp, None)
        return self.routes_
    
    def get_interval(self):
        '''
        get interval bins for digitalization.
        in: self.link_prefix_dict
        out: self.interval interval bins for digitalization.
        '''
        
        s = time.time()
        v_ = [compute_con_value(p_l) for p_l in self.link_prefix_dict.values()]
        print('the time of get_interval:', time.time() - s)
        _, self.interval_ = np.histogram(v_, bins=5)
        print("get interval:", self.interval_)

    def get_link_prefix_weight_full(self):
        '''
        test the performance for full update.
        in: self.link_prefix_dict link_prefix_dict link: prefix list.
        out: link_dig link weight for each link.
        '''
        weight = []
        for link in self.link_prefix_dict:
            a = compute_con_value(self.link_prefix_dict[link])
            weight.append(a)
        weight_dig = edge_weight_computing(weight)
        link_dig = zip(self.link_prefix_dict.keys(), weight_dig)
        return dict(link_dig)
    
    @property
    def get_link_prefix_dict(self):
        
        '''
        calculate the weight of prefix from self.link_prefix_dict.
        in: link_weight_counter not digitalized edge weight
        out: link_dig dict digitalized edge weight
        '''
        
        weight_dig = edge_weight_computing(self.interval_, self.link_weight_counter.values())
        link_dig = zip(self.link_weight_counter.keys(), weight_dig)
        return dict(link_dig)
    

    def compute_edge(self, updates, directed=False):
        '''
        
        Incremental updating the link_prefix_dict and route_, output the added edges and removed edges.
        routes format: routes[peer_asn][prefix] = as_path
        updates format: [bgpdump_line,label]
        out: (add_edges, remove_edges) edges for incremental update
        '''


        edge_updates = []
        route_ = self.routes_
        for update in updates:
            op_ = update[2]
            peer_as = update[4]
            prefix_ = update[5]
            if peer_as in self.selected_vp:
                if op_ == 'A':
                    as_path = update[6]
                    if as_path != route_[peer_as][prefix_]:
                        if '{' not in as_path:
                            if route_[peer_as][prefix_] == None: 
                                route_[peer_as][prefix_] = as_path
                                as_path_list = as_path.split(' ')
                                for l in range(len(as_path_list)-1):
                                    if as_path_list[l] != as_path_list[l+1]:
                                        edge_updates.append(('A', prefix_, [as_path_list[l], as_path_list[l+1]]))
                            else:
                                as_path_o = route_[peer_as][prefix_]
                                route_[peer_as][prefix_] = as_path
                                as_path_o_list = as_path_o.split(' ')
                                for l in range(len(as_path_o_list)-1): # 隐式撤销
                                    if as_path_o_list[l] != as_path_o_list[l+1]:
                                        edge_updates.append(('W', prefix_, [as_path_o_list[l], as_path_o_list[l+1]]))
                                as_path_list = as_path.split(' ')
                                for l in range(len(as_path_list)-1):
                                    if as_path_list[l] != as_path_list[l+1]:
                                        edge_updates.append(('A',prefix_, [as_path_list[l], as_path_list[l+1]]))
                elif op_ == 'W':
                    if route_[peer_as][prefix_] != None:
                        as_path_list = route_[peer_as][prefix_].split(' ')
                        for p in range(len(as_path_list)-1):
                            if as_path_list[p] != as_path_list[p+1]:
                                edge_updates.append(('W', prefix_, [as_path_list[p], as_path_list[p+1]]))
                        route_[peer_as][prefix_] = None
                else:
                    pass

        self.routes_ = route_
        # undirected graph or direccted
        link_prefix_dict_ = self.link_prefix_dict
        link_weight_counter = self.link_weight_counter
        
        edge_combine = {}
        if not directed:
            for e in edge_updates:
                edge = tuple(sorted(e[2]))
                pf = e[1]
                if edge not in edge_combine:
                    edge_combine[edge] = 0
                if e[0] == 'W':
                    edge_combine[edge] -= 1
                    if pf in link_prefix_dict_[edge]:
                        link_prefix_dict_[edge].remove(pf)
                        weight_ = compute_prefix_weight(pf)
                        link_weight_counter[edge] -= weight_

                elif e[0] == 'A':
                    edge_combine[edge] += 1
                    if pf not in link_prefix_dict_[edge]:
                        link_prefix_dict_[edge].add(pf)
                        weight_ = compute_prefix_weight(pf)
                        link_weight_counter[edge] += weight_
        
        else:
            for e in edge_updates:
                edge = tuple(e[2])
                if edge not in edge_combine:
                    edge_combine[edge] = 0
                if e[0] == 'W':
                    edge_combine[edge] -= 1
                elif e[0] == 'A':
                    edge_combine[edge] += 1

        add_edges = []
        remove_edges = []
        self.link_prefix_dict = link_prefix_dict_
        for idx in edge_combine:
            if edge_combine[idx] > 0:
                add_edges.append(idx)
            elif edge_combine[idx] < 0:
                remove_edges.append(idx)

        del edge_updates
        del edge_combine
        return (add_edges, remove_edges)

    def compute_edge_withroute(self, updates, directed=False):
        '''
        collect all the added edges from "route table" for full update.
        in: self.route_ route table
        out: route_edge edge list
        '''
        routes_ = self.routes_
        for update in updates:
            op_ = update[2]
            peer_as = update[4]
            prefix_ = update[5]
            if peer_as in self.selected_vp:
                if op_ == "A":
                    as_path = update[6]
                    if as_path != routes_[peer_as][prefix_]:
                        if '{' not in as_path: 
                            routes_[peer_as][prefix_] = as_path                
                elif op_ == 'W':
                    if routes_[peer_as][prefix_] != None:
                        routes_[peer_as][prefix_] = None
                else:
                    pass

        route_edge = set()
        for vp in routes_:
            for p in routes_[vp]:
                if routes_[vp][p] != None:
                    link_list = routes_[vp][p].split(' ')
                    for p in range(len(link_list)-1):
                            if link_list[p] != link_list[p+1]:
                                route_edge.add(tuple(sorted([link_list[p], link_list[p+1]])))
        self.routes_ = routes_
        return tuple(route_edge)

    def link_compute(self):
        """
        collect self.link_prefix_dict from route table for full update.
        in: routes:
        {
        peer_as: {prefix: [as_path]}
        }
        """
        link_prefix_dict_ = defaultdict(lambda: set())
        link_weight_counter = defaultdict(lambda:float())
        routes_ = self.routes_
        for peer_ in routes_:
            for prefix_ in routes_[peer_]:
                as_path = routes_[peer_][prefix_]
                if as_path != None:
                    as_path_list = as_path.split(' ')
                    for i in range(len(as_path_list)-1):
                        if as_path_list[i] != as_path_list[i+1]:
                            link = sorted([as_path_list[i], as_path_list[i+1]])
                            if prefix_ not in link_prefix_dict_[tuple(link)]:
                                link_prefix_dict_[tuple(link)].add(prefix_)
                                weight_ = compute_prefix_weight(prefix_)
                                link_weight_counter[tuple(link)] += weight_
                
        self.link_prefix_dict = link_prefix_dict_
        self.link_weight_counter = link_weight_counter


def compute_con_value(p_l):
    '''
    caculate the weight for prefix set of one link.
    in: p_l list/set prefix set
    out: weight float The weight of the link
    '''
    weight = 0
    for p in p_l:
        if '.' in p:
            weight += (32 - int(p.split('/')[1])) / 32
        elif ":" in p:
            weight += (128 - int(p.split('/')[1])) / 128
    return weight

def compute_prefix_weight(pf):
    '''
    pick the split weight of prefix
    in: pf str prefix
    out: weight float the split weight
    '''
    length_ = int(pf.split('/')[1])

    if "." in pf:
        return prefix_weight_dict_v4[length_]
    elif ":" in pf:
        return prefix_weight_dict_v6[length_]
    else:
        raise ValueError

def route_construct(path):
    print("1:",path)
    priming_path = path + 'priming_data/txt/'
    r = Routes(priming_path) # construct the class Route
    print("2:")
    r.collect_routes() # collect the routes from ribs
    print("3:")
    r.select_vps # select the vps
    print("4:", r.select_vps)
    # r1 = r.get_route # get the selected routes filtering from vps
    return r

# @njit(nogil=True, cache=True)
def edge_weight_computing(interval, edge_value):
    '''
    caculate the weight for each link.
    in: interval list the initial interval for digitizing edges weight.
    in: edge_value list the edge weight before digitize.
    out: edge_weight after digition.
    '''
    if not isinstance(edge_value, list):
        edge_value = list(edge_value)
    edge_weight = np.digitize(edge_value, interval)
    return edge_weight

    
if __name__ == "__main__":
    print(prefix_weight_dict_v4)
    pass
