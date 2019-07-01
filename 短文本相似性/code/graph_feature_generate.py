from __future__ import print_function

import time
import numpy as np
import pandas as pd
from collections import deque
from itertools import combinations


train = pd.read_csv('../data/train.csv')
test = pd.read_csv('.../data/test.csv')


def gen_graph(train):
    """
    把输入数据转化为以字典表示的无向图
    """ 
    data = train[train['label']==1][['q1','q2']]
    graph = {}
    for i in range(len(data)):
        if data.iloc[i,0] not in graph.keys():
            graph[data.iloc[i,0]] = set([data.iloc[i,1]])
        else:
            graph[data.iloc[i,0]].add(data.iloc[i,1])
    
        if data.iloc[i,1] not in graph.keys():
            graph[data.iloc[i,1]] = set([data.iloc[i,0]])
        else:
            graph[data.iloc[i,1]].add(data.iloc[i,0])
    
    return graph


def bfs_visited(ugraph, start_node):
    """
    输入无向图ugraph和一个节点start_node
    返回从这个节点出发，通过广度优先搜索访问的所有节点的集合
    """
    # initialize Q to be an empty queue
    que = deque()
    # initialize visited
    visited = [start_node]
    # enqueue(que, start_node)
    que.append(start_node)
    while len(que) > 0:
        current_node = que.popleft()
        neighbours = ugraph[current_node]
        for nei in neighbours:
            if nei not in visited:
                visited.append(nei)
                que.append(nei) 
    return set(visited)


def cc_visited(ugraph):
    """
    输入无向图ugraph
    返回一个list，list的元素是每个连通分量的节点构成的集合
    """
    remaining_nodes = list(ugraph.keys())
    connected_components = []
    while len(remaining_nodes) > 0 :
        # choose the first element in remaining_nodes to be the start_node
        start_node = remaining_nodes[0]
        # use bfs_visited() to get the connected component containing start_node
        con_component = bfs_visited(ugraph, start_node)
        # update connected_components
        connected_components.append(con_component)
        # update remaining_nodes
        remaining_nodes = list(set(remaining_nodes) - con_component)
    return connected_components


 def Dijkstra(ugraph, connected_component, start_node):
    '''
    返回start_node到connected_component所有节点的最短距离
    '''
    # 初始化
    minv = start_node
    visited = set()
    
    # 源顶点到其余各顶点的初始路程
    dist = dict([(node,np.float('inf')) for node in connected_component])
    dist[minv] = 0
    
    # 遍历集合V中与A直接相邻的顶点，找出当前与A距离最短的顶点
    while len(visited) < len(connected_component):
        visited.add(minv)
        # 确定当期顶点的距离
        for v in ugraph[minv]:
            if dist[minv] + 1 < dist[v]:   # 如果从当前点扩展到某一点的距离小与已知最短距离 
                dist[v] = dist[minv] + 1   # 对已知距离进行更新
        
        # 从剩下的未确定点中选择最小距离点作为新的扩散点
        new = np.float('inf')                                      
        for w in connected_component - visited:   
            if dist[w] < new: 
                new = dist[w]
                minv = w  
    return dist


def get_graph_distance(data, train_graph, connected_components, training_data=True):
    '''
    1. 如果q1,q2在一个连通图上：返回q1,q2的距离d
    2. 如果q1,q2不在一个连通图上: 令d(q1, q2) = 1000
    '''
    n = data.shape[0]
    
    # 初始化
    record_distance = {}  #用来记录已经计算过的距离
    result_distance = [1000 for i in range(n)]
    
    for i in range(n):
        q1 = data.loc[i,'q1']
        q2 = data.loc[i,'q2']

        # 如果是训练数据的相似问题，则dist=1
        if training_data and data.loc[i,'label'] == 1:
            result_distance[i] = 1
        
        # 如果已经计算过，直接取出计算过的值
        elif (q1,q2) in record_distance.keys():
            result_distance[i] = record_distance[(q1,q2)]

        elif (q2,q1) in record_distance.keys():
            result_distance[i] = record_distance[(q2,q1)]

        else:       
            # check whether q1,q2 are in one connected_componets
            for cc in connected_components:
                if (q1 in cc) and (q2 in cc):
                    # 连通图cc,q1到其它节点的距离
                    q1_dist = Dijkstra(train_graph, cc, q1)
                    # 把计算过的距离保存起来
                    new_dict = dict([((q1,node),q1_dist[node]) for node in q1_dist.keys()])
                    record_distance.update(new_dict)          
                    result_distance[i] = q1_dist[q2]            
                    break

    result_distance = pd.DataFrame(np.array(result_distance), index=data.index)
    result_distance.columns = ['graph_distance']
    
    return result_distance


def get_independent_groups(train, train_graph_distance, connected_components):
    
    # 找出不相似的问题对
    data = train[train.label == 0]
    
    independent_groups = []
       
    for i in data.index:
        q1 = data.loc[i,'q1']
        q2 = data.loc[i,'q2']
        
        if train_graph_distance.loc[i, 'graph_distance'] == 1000:
            # 查看它们是否有连通图
            cc1 = set([])
            cc2 = set([])
            for cc in connected_components:
                if q1 in cc:
                    cc1 = cc
                if q2 in cc:
                    cc2 = cc
            if len(cc1) > 0 and len(cc2) > 0 and (cc1,cc2) not in independent_groups and (cc2,cc1) not in independent_groups:
                independent_groups.append((cc1,cc2))
                
    return independent_groups


def get_graph_features(test, test_graph_distance, independent_groups):
    
    n = test.shape[0]
    
    # 初始化, 0 表示从训练集的graph无法确定是否相似， 1表示确定相似，-1表示确定不相似
    graph_features = [0 for i in range(n)]
    
    for i in range(n):
        q1 = test.loc[i,'q1']
        q2 = test.loc[i,'q2']

        if test_graph_distance.loc[i,'graph_distance'] < 1000:
            graph_features[i] = 1
        else:
            # 看看q1和q2是否在independent group里面，如果在，则q1，q2确定不相似
            for ig in independent_groups:
                if (q1 in ig[0] and q2 in ig[1]) or (q1 in ig[1] and q2 in ig[0]):
                    graph_features[i] = -1
      
    graph_features = pd.DataFrame(np.array(graph_features), index=test.index)
    graph_features.columns = ['graph_features']
    
    return graph_features


def gen_similar_data(train_graph, connected_components, max_cc_size):
    '''
    对每个连通图，求排列的相似集合，不用考虑图距离

    如果连通图节点只有2个，直接break
    '''    
    similar_pairs = set()
    
    for cc in connected_components:
        if len(cc) > 2 and len(cc) <= max_cc_size:
            res = set(combinations(cc, 2))
            similar_pairs.update(res)
        else:
            continue
    
    return similar_pairs


def gen_dissimilar_data(independent_groups, max_group_size):
    '''
    如果q1, q2不相似，且存在连通图cc1包含q1，和cc2包含q2，则cc1和cc2的任意组合均不相似
    max_group_size用来控制返回的问题对数量，设为46，对应100万左右的问题对
    '''
    dissimilar_pairs = set()  
    for ig in independent_groups:
        cc1 = ig[0]
        cc2 = ig[1]
        # 限制连通图大小，不然太多了
        if len(cc1) < max_group_size and len(cc2) < max_group_size:
            for q1 in cc1:
                for q2 in cc2:
                    dissimilar_pairs.add((q1, q2))
    return dissimilar_pairs
 
    
    
def data_augmentation(train, similar_data, dissimilar_data):
    '''
    与train数据去重，生成平衡数据集
    similar_data: dict,{(q1,q2): d(q1,q2)}
    dissimilar_data: set, {(q1,q2)}
    '''
   
    #问题对转化为set格式
    train_data1 = set([(train.loc[i,'q1'], train.loc[i,'q2']) for i in train.index])
    train_data2 = set([(train.loc[i,'q2'], train.loc[i,'q1']) for i in train.index])
    
    # 查看（q1,q2）组合是否与train数据重复,如重复，则去掉
    similar_pairs = similar_data - train_data1
    similar_pairs = list(similar_pairs - train_data2)[:1000000]
    dissimilar_pairs = dissimilar_data - train_data1
    dissimilar_pairs = list(dissimilar_pairs - train_data2)[:1000000]
    
    
    # 生成新的训练数据并导出
    new_data = []
    new_data.extend(similar_pairs)
    new_data.extend(dissimilar_pairs)
    
    new_data = pd.DataFrame(np.array(new_data))
    new_data.columns = ['q1','q2']
    new_data['label'] = 0
    new_data.loc[0:len(similar_pairs)-1, 'label'] = 1
    
    return new_data


if __name__ == '__main__':
	## 先生成图
	print('Generating Graph...')
	start = time.time()
	train_graph = gen_graph(train)
	end = time.time()
	print('Graph generated. Time used {:0.1f} mins'.format((end-start)/60))

	## 寻找各连通分项（大概7分钟）
	print('Searching Connected Components...')
	start = time.time()
	connected_components = cc_visited(train_graph)
	end = time.time()
	print('Search finished. Time used {:0.1f} mins'.format((end-start)/60))

	start = time.time()
	train_graph_distance = get_graph_distance(train, train_graph, connected_components, training_data=True)
	end = time.time()
	print('Compute train_graph_distance finished. Time used {:0.1f} mins'.format((end-start)/60))

	start = time.time()
	independent_groups = get_independent_groups(train, train_graph_distance, connected_components)
	end = time.time()
	print('Compute independent_groups finished. Time used {:0.1f} mins'.format((end-start)/60))

	# 通过训练集的ugraph，计算test的图特征
	# start = time.time()
	# test_graph_distance = get_graph_distance(test, train_graph, connected_components, training_data=False)
	# graph_features = get_graph_features(test, test_graph_distance, independent_groups)
	# end = time.time()
	# print('Generate graph_features finished. Time used {:0.1f} mins'.format((end-start)/60))
	# 由于实际中，直接获取到test data不现实，所以这种方式获得的特征，对实际应用用处不大。
	# 本项目计算过程中，不使用这样生成的图特征。
	# 只适用图的关系，来做数据扩增

	start = time.time()
	similar_data = gen_similar_data(train_graph, connected_components, 1350)
	end = time.time()
	print('Generate similar_data finished. Time used {:0.1f} mins'.format((end-start)/60))

	start = time.time()
	dissimilar_data = gen_dissimilar_data(independent_groups, 50)
	end = time.time()
	print('Generate dissimilar_data finished. Time used {:0.1f} mins'.format((end-start)/60))

	start = time.time()
	new_data = data_augmentation(train, similar_data, dissimilar_data)
	end = time.time()
	print('Generate new data finished. Time used {:0.1f} mins'.format((end-start)/60))

	new_data = new_data[['label', 'q1', 'q2']]

	new_data.to_csv('../data/augmentation_results.csv')
	print('Have saved data to \'augmentation_results.csv\'')