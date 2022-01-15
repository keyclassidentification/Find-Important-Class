import networkx
import degree
import datetime

from other.minClass import compute_H
from helper.fileHelper import save_other_baseline_rank
from helper.graphHelper import copy_directed_graph, update_weight, generate_un

def otherBaseLine(baseLine):
    name = 'weight/jgroups'
    # Input the source class information path
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
    if baseLine == 'hits':
        try:
            data = networkx.hits(graph, max_iter=1000)  # (hubs,authorities)
            test = sorted(data[1].items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
            save_other_baseline_rank('E:/实验/DATA/' + name + '/HITS.txt', test)
        except Exception as e:
            print(e)

    """
    计算degree
    """
    if baseLine == 'degree':
        try:
            data = degree.degree_centrality(graph, weight='weight')
            test = sorted(data.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
            save_other_baseline_rank('E:/实验/DATA/' + name + '/Deg.txt', test)
        except Exception as e:
            print(e)

    '''
    计算indegree
    '''
    if baseLine == 'indegree':
        try:
            data = degree.in_degree_centrality(graph, weight='weight')
            test = sorted(data.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
            save_other_baseline_rank('E:/实验/DATA/' + name + '/InDeg.txt', test)
        except Exception as e:
            print(e)

    '''
    计算out degree
    '''
    if baseLine == 'outdegree':
        try:
            data = degree.out_degree_centrality(graph, weight='weight')
            test = sorted(data.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
            save_other_baseline_rank('E:/实验/DATA/' + name + '/OutDeg.txt', test)
        except Exception as e:
            print(e)

    """
    计算 betweenness
    """
    if baseLine == 'betweenness':
        try:
            data = networkx.betweenness_centrality(un_graph)
            test = sorted(data.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
            save_other_baseline_rank('E:/实验/DATA/' + name + '/Between.txt', test)
        except Exception as e:
            print(e)

    '''
    计算 core number
    '''
    if baseLine == 'corenumber':
        try:
            data = networkx.core_number(un_graph)
            test = sorted(data.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
            save_other_baseline_rank('E:/实验/DATA/' + name + '/CoreNum.txt', test)
        except Exception as e:
            print(e)

    '''
    计算page rank
    '''
    if baseLine == 'pagerank':
        try:
            data = networkx.pagerank(graph, weight='weight')
            test = sorted(data.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
            save_other_baseline_rank('E:/实验/DATA/' + name + '/PageRank.txt', test)
        except Exception as e:
            print(e)

    '''
    计算min class
    '''
    if baseLine == 'minclass':
        try:
            data = compute_H(graph, weight='weight')
            test = sorted(data.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
            save_other_baseline_rank('E:/实验/DATA/' + name + '/MinClass.txt', test)
        except Exception as e:
            print(e)

if __name__ == '__main__':
    start1 = datetime.datetime.now()
    otherBaseLine('minclass');
    end1 = datetime.datetime.now()
    print('minclass run times:' + str(end1 - start1))

    start2 = datetime.datetime.now()
    otherBaseLine('pagerank');
    end2 = datetime.datetime.now()
    print('pagerank run times:' + str(end2 - start2))

    start3 = datetime.datetime.now()
    otherBaseLine('hits');
    end3 = datetime.datetime.now()
    print('hits run times:' + str(end3 - start3))

    start4 = datetime.datetime.now()
    otherBaseLine('corenumber');
    end4 = datetime.datetime.now()
    print('corenumber run times:' + str(end4 - start4))

    start5 = datetime.datetime.now()
    otherBaseLine('betweenness');
    end5 = datetime.datetime.now()
    print('betweenness run times:' + str(end5 - start5))

    start6 = datetime.datetime.now()
    otherBaseLine('indegree');
    end6 = datetime.datetime.now()
    print('indegree run times:' + str(end6 - start6))

    start7 = datetime.datetime.now()
    otherBaseLine('outdegree');
    end7 = datetime.datetime.now()
    print('outdegree run times:' + str(end7 - start7))

    start8 = datetime.datetime.now()
    otherBaseLine('degree');
    end8 = datetime.datetime.now()
    print('degree run times:' + str(end8 - start8))