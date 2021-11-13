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

