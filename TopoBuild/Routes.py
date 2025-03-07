from collections import defaultdict
import os


class Routes:
    
    '''
    From RIB or priming updates datas, extract routes info as truth-grouding info.
    routes:
    {
        prefix: {peer_as: [as_path]}
    }
    
    link_prefix_dict:
    {
        link: set(prefix)        
    }
    '''
     
    def __init__(self, path):
        self.path = path
        self.routes_ = defaultdict(lambda: defaultdict(str))
        self.mode = "ribs"
        self.link_prefix_dict = defaultdict(lambda: set())

    def collect_routes(self):
        if self.mode == 'updates':
            files = os.listdir(self.path)
            for f in files:
                f_path = os.path.join(self.path,f)
                with open(f_path,'r') as file:
                    for l in file:
                        if l.strip() != '':
                            line = l.strip().split('|')
                            prefix = line[5]
                            peer_asn = line[4]
                            op_ = line[2]
                            if op_ == 'A':
                                if '{' not in line[6]:
                                    self.routes_[prefix][peer_asn] = line[6]
                            elif op_ == 'W':
                                self.routes_[prefix][peer_asn] = None
                            else:
                                pass

        elif self.mode == 'ribs':
            files = os.listdir(self.path)
            for f in files:
                f_path = os.path.join(self.path,f)
                with open(f_path, 'r') as file:
                    for l in file:
                        if l.strip() != '':
                            line = l.strip().split('|')
                            prefix = line[5]
                            peer_asn = line[4]
                            op_ = line[2]
                            if '{' not in line[6]:
                                self.routes_[prefix][peer_asn] = line[6]
                            else:
                                pass

    @property
    def get_route(self):
        return self.routes_
    
    @property
    def get_link_prefix_dict(self):
        link_dict = {}
        for link in self.link_prefix_dict:
            link_dict[link] = len(self.link_prefix_dict[link])
        return link_dict
        # return self.link_prefix_dict
    
    def compute_edge(self, updates, directed=False):
        # routes format: routes[prefix][peer_asn] = as_path
        # updates format: [bgpdump_line,label]
        edge_updates = []
        for update in updates:
            time_ = update[1]
            op_ = update[2]
            peer_as = update[4]
            prefix_ = update[5]
            if op_ == 'A':
                as_path = update[6]
                if as_path != self.routes_[prefix_][peer_as]:
                    if '{' not in as_path:
                        if self.routes_[prefix_][peer_as] == None: 
                            self.routes_[prefix_][peer_as] = as_path
                            
                            as_path_list = as_path.split(' ')
                            for l in range(len(as_path_list)-1):
                                if as_path_list[l] != as_path_list[l+1]:
                                    edge_updates.append(('A', prefix_, [as_path_list[l], as_path_list[l+1]]))
                                    # self.link_prefix_dict[(as_path_list[l], as_path_list[l+1])].add(prefix_)

                        else:
                            as_path_o = self.routes_[prefix_][peer_as]
                            self.routes_[prefix_][peer_as] = as_path
                            as_path_o_list = as_path_o.split(' ')
                            for l in range(len(as_path_o_list)-1): # 隐式撤销
                                if as_path_o_list[l] != as_path_o_list[l+1]:
                                    edge_updates.append(('W', prefix_, [as_path_o_list[l], as_path_o_list[l+1]]))
                                    # if prefix_ in self.link_prefix_dict[(as_path_o_list[l], as_path_o_list[l+1])]:
                                    #     self.link_prefix_dict[(as_path_o_list[l], as_path_o_list[l+1])].remove(prefix_)
                            as_path_list = as_path.split(' ')
                            for l in range(len(as_path_list)-1):
                                if as_path_list[l] != as_path_list[l+1]:
                                    edge_updates.append(('A', prefix_, [as_path_list[l], as_path_list[l+1]]))
                                    # self.link_prefix_dict[(as_path_list[l], as_path_list[l+1])].add(prefix_)    
            
            elif op_ == 'W':
                if self.routes_[prefix_][peer_as] != None:
                    as_path_list = self.routes_[prefix_][peer_as].split(' ')
                    for p in range(len(as_path_list)-1):
                        if as_path_list[p] != as_path_list[p+1]:
                            edge_updates.append(('W', prefix_, [as_path_list[p], as_path_list[p+1]]))
                            # if prefix_ in self.link_prefix_dict[(as_path_o_list[l], as_path_o_list[l+1])]:
                            #     self.link_prefix_dict[(as_path_list[p], as_path_list[p+1])].remove(prefix_)
                    self.routes_[prefix_][peer_as] = None
            else:
                pass

        # undirected graph or direccted
        edge_combine = {}
        if not directed:
            for e in edge_updates:
                edge = tuple(sorted(e[2]))
                pf = e[1]
                if edge not in edge_combine:
                    edge_combine[edge] = 0
                    # self.link_prefix_dict[edge] = set()
                if e[0] == 'W':
                    edge_combine[edge] -= 1
                    if pf in self.link_prefix_dict[edge]:
                        self.link_prefix_dict[edge].remove(pf)
                elif e[0] == 'A':
                    edge_combine[edge] += 1
                    self.link_prefix_dict[edge].add(pf)
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
        
        for idx in edge_combine:
            if edge_combine[idx] > 0:
                add_edges.append(idx)
            elif edge_combine[idx] < 0:
                remove_edges.append(idx)
        
        del edge_updates
        del edge_combine
        return (add_edges, remove_edges)
    
    def link_compute(self):
        """
        routes:
        {
            prefix: {peer_as: [as_path]}
        }
        
        link_dict:
        {
            count
        }
        """
        # link_prefix = defaultdict(lambda:set())
        for prefix in self.routes_:
            for peer in self.routes_[prefix]:
                as_path = self.routes_[prefix][peer]
                if as_path != None:
                    as_path_list = as_path.split(' ')
                    for i in range(len(as_path_list)-1):
                        if as_path_list[i] != as_path_list[i+1]:
                            link = (as_path_list[i], as_path_list[i+1])
                            self.link_prefix_dict[link].add(prefix)