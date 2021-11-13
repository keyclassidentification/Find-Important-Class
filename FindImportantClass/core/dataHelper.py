import numpy as np
import networkx as nx

def copy_directed_graph(graph):
    new_graph = nx.DiGraph()
    nodes = graph.nodes
    edges = graph.edges
    for node in nodes:
        new_graph.add_node(node)
    for edge in edges:
        weight = graph.get_edge_data(edge[0],edge[1])[0]['weight']
        new_graph.add_edge(edge[0],edge[1],weight=weight)
    return new_graph

def adj_to_bias(adj, sizes, nhood=1):
    nb_graphs = adj.shape[0]
    mt = np.empty(adj.shape)
    for g in range(nb_graphs):
        mt[g] = np.eye(adj.shape[1])
        for _ in range(nhood):
            test1 = adj[g]
            temp = adj[g] + np.eye(adj.shape[1])
            mt[g] = np.matmul(mt[g], temp)
        for i in range(sizes[g]):
            for j in range(sizes[g]):
                if mt[g][i][j] > 0.0:
                    mt[g][i][j] = 1.0
    return -1e9 * (1.0 - mt)

def get_neighbours(G,u,in_out='in'):
    """
    Output in degree neighbor node and out degree neighbor node
    :param G:
    :param u:
    :param in_out:
    :return:
    """
    neighbours = set()
    if in_out == 'in':
        edges = set(G.in_edges(u))
        neighbours.clear()
        for edge in edges:
            neighbours.add(edge[0])
    elif in_out == 'out':
        edges = set(G.out_edges(u))
        neighbours.clear()
        for edge in edges:
            neighbours.add(edge[1])
    return neighbours

# File Related
def read_node_label(filename, skip_head=False):
    fin = open(filename, 'r')
    X = []
    Y = []
    while 1:
        if skip_head:
            fin.readline()
        l = fin.readline()
        if l == '':
            break
        vec = l.strip().split(' ')
        X.append(vec[0])
        Y.append(vec[1])
        print(vec[0])
    fin.close()
    return X, Y

def save_node_label(filename, x,y,z):
    f_out = open(filename, 'w+')
    i = 0
    for idx in x:
        t=''
        t = str(x[i]) + ' '+str(y[i]) + ' '+str(z[i])
        f_out.write(t)
        f_out.write('\n')
        i = i + 1
    f_out.close()
    return x,y,z

def save_rank(filename, x):
    f_out = open(filename, 'w+')
    i = 0
    for idx in x:
        t=''
        t = str(x[i])
        f_out.write(t)
        f_out.write('\n')
        i = i + 1
    f_out.close()