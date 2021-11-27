import networkx

from other.minClass import compute_H
from helper.fileHelper import save_node_label
from helper.graphHelper import copy_directed_graph, update_weight, generate_un


def otherBaseLine():
    # Input the source class information path
    name = 'weight/test'
    net_path = 'E:/实验/DATA/' + name + '.txt'
    graph = networkx.read_pajek(net_path)
    graph = copy_directed_graph(graph)

    graph = update_weight(graph)
    edges = graph.edges()

    remove_edges = []
    for edge in edges:
        if edge[0] == edge[1]:
            remove_edges.append(edge)
    for edge in remove_edges:
        graph.remove_edge(edge[0], edge[0])
    un_graph = generate_un(graph)  # 非有向图

    """
    计算HITS
    """
    try:
        data = networkx.hits(graph, max_iter=1000)  # (hubs,authorities)
        test = sorted(data[1].items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
        save_node_label('E:/实验/DATA/' + name + '/HITS.txt', test)
    except Exception as e:
        print(e)

    """
    计算degree
    """
    try:
        data = networkx.degree_centrality(graph, weight='weight')
        test = sorted(data.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
        save_node_label('E:/实验/DATA/' + name + '/Deg.txt', test)
    except Exception as e:
        print(e)

    """
    计算 betweenness
    """
    try:
        data = networkx.betweenness_centrality(un_graph)
        test = sorted(data.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
        save_node_label('E:/实验/DATA/' + name + '/Between.txt', test)
    except Exception as e:
        print(e)

    '''
    计算 closeness
    '''
    try:
        data = networkx.closeness_centrality(un_graph)
        test = sorted(data.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
        save_node_label('E:/实验/DATA/' + name + '/Closeness.txt', test)
    except Exception as e:
        print(e)

    '''
    计算 core number
    '''
    try:
        data = networkx.core_number(un_graph)
        test = sorted(data.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
        save_node_label('E:/实验/DATA/' + name + '/CoreNum.txt', test)
    except Exception as e:
        print(e)

    '''
    计算page rank
    '''
    try:
        data = networkx.pagerank(graph, weight='weight')
        test = sorted(data.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
        save_node_label('E:/实验/DATA/' + name + '/PageRank.txt', test)
    except Exception as e:
        print(e)

    '''
    计算min class
    '''
    try:
        data = compute_H(graph, weight='weight')
        test = sorted(data.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
        save_node_label('E:/实验/DATA/' + name + '/MinClass.txt', test)
    except Exception as e:
        print(e)