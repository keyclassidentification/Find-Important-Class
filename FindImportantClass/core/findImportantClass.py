
import numpy as np
import networkx as nx
import tensorflow as tf
from helper.graphHelper import copy_directed_graph,get_neighbours
from helper.fileHelper import read_node_label,save_rank

className = 'tomcat'
def loadData():
    G = nx.read_pajek('E:/实验/DATA/weight/'+className+'.txt')
    G = copy_directed_graph(G)

    x, y = read_node_label('E:/实验/DATA/weight/class_'+className+'.txt')

    Y = []
    for tempname in y:
        Y.append(eval(tempname))

    # 获取图的邻接矩阵
    nodes = []
    tempnodes = G.nodes()
    for node in tempnodes:
        nodes.append(node)
    bias = nx.adjacency_matrix(G)
    count = bias.shape[0]
    bias = bias.todense()
    bias = bias[np.newaxis]
    weight_mat = bias

    return G,x,Y,weight_mat

Graph,IdNumber,name, wei= loadData()

tf.compat.v1.disable_eager_execution()

def calculateImportantScore():
    nodes = []
    tempnodes = Graph.nodes()
    for node in tempnodes:
        nodes.append(node)

    # Set the score = sum(nei) 当前节点的分数为其邻居节点分数和权重乘积之和
    weights = (wei.A)[0]
    edges = Graph.edges()
    weight_set = set()
    for edge in edges:
        weight = Graph.get_edge_data(edge[0], edge[1])['weight']
        weight_set.add(weight)
    max_weight = max(weight_set)

    tempScores = {}
    for h in IdNumber:
        index1 = IdNumber.index(h)
        tempName1 = name[index1].strip()
        matr_ind = nodes.index(tempName1)
        wei1 = 0
        inner = get_neighbours(Graph, tempName1, in_out='in')
        for m in inner:
            if m != '0':
                index2 = name.index(m)
                matr_ind1 = nodes.index(m)
                tempWei = weights[matr_ind1][matr_ind] / max_weight
                wei1 = wei1 + tempWei
        tempScores[tempName1] = wei1

    weiOut = {}
    for h in IdNumber:
        index1 = IdNumber.index(h)
        tempName1 = name[index1].strip()
        matr_ind = nodes.index(tempName1)
        wei1 = 0
        inner = get_neighbours(Graph, tempName1, in_out='out')
        for m in inner:
            if m != '0':
                matr_ind1 = nodes.index(m)
                wei1 = wei1 + weights[matr_ind][matr_ind1] / max_weight
        weiOut[tempName1] = wei1

    scores = {}
    for h in IdNumber:
        index1 = IdNumber.index(h)
        tempName1 = name[index1].strip()
        matr_ind = nodes.index(tempName1)
        tempscore = tempScores[tempName1]
        inner = get_neighbours(Graph, tempName1, in_out='in')
        for n in inner:
            if n != '0':
                index2 = name.index(n)
                matr_ind1 = nodes.index(n)
                auth = 1;
                if weights[matr_ind1][matr_ind] < 1:
                    auth = weights[matr_ind1][matr_ind] / max_weight
                currentWei = float(weights[matr_ind1][matr_ind] / max_weight)
                totalWeiOut = weiOut[n]
                wei1 = currentWei / totalWeiOut
                neiScore = tempScores[n]
                tempscore = tempscore + (wei1 * neiScore) * auth
        scores[tempName1] = tempscore

    return scores

if __name__ == "__main__":
    scores = {}
    scores = calculateImportantScore()
    sorted_id = sorted(scores.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)

    IdNumber1 = []
    name1 = []
    scores1 = []

    for id in sorted_id:
        gy = id[0]
        temInd = name.index(gy)
        IdNumber1.append(IdNumber[temInd])
        name1.append(gy)
        scores1.append(id[1])
    #save_node_label('../data/result.txt', IdNumber1, name1, scores1)
    save_rank('E:/实验/DATA/weight/'+className+'/KeyClass.txt', name1)
    test = 'end'
