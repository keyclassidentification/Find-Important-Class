
import numpy as np
from class2vec import Node2Vec
import networkx as nx
import tensorflow as tf
from helper.graphHelper import copy_directed_graph, adj_to_bias,get_neighbours
from helper.fileHelper import read_node_label,save_node_label,save_rank
from core.classAttention import getNodeAttentions

def loadData():
    G = nx.read_pajek('E:/实验/DATA/weight/pdf.txt')
    G = copy_directed_graph(G)

    #walk_length:10, num_walks:80
    model = Node2Vec(G, walk_length=10, num_walks=80,
                     p=2, q=0.1, workers=1, use_rejection_sampling=0)#p=0.25, q=4
    # 通过word2vec训练node2vec向量模型
    model.train(window_size=5, iter=3)
    # 获取经过word2vec训练之后的嵌入
    embeddings = model.get_embeddings()

    x, y = read_node_label('E:/实验/DATA/weight/class_pdf.txt')

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
    bias_mat = adj_to_bias(bias, [count], nhood=1)

    a = []
    for num in embeddings:
        a.append(embeddings[num])
    em = np.array(a)

    return G,x,Y,em,weight_mat, bias_mat

head = 50
Graph,IdNumber,name, features, wei,bias= loadData()
ft_size = features.shape[1]
nb_nodes = features.shape[0]

tf.compat.v1.disable_eager_execution()
# temp add
embedding_fn_size = 312
# end
def scoreNetwork():
    weights = tf.Variable(tf.compat.v1.truncated_normal([embedding_fn_size, 1], stddev=0.1))

    biases = tf.Variable(tf.compat.v1.truncated_normal([1], stddev=0.1))

    embedding = tf.compat.v1.placeholder(tf.float32, [nb_nodes, ft_size])
    ftr_in = tf.compat.v1.placeholder(dtype=tf.float32, shape=(1, nb_nodes, ft_size))

    weight_in = tf.compat.v1.placeholder(dtype=tf.float32, shape=(1, nb_nodes, nb_nodes))
    bias_in = tf.compat.v1.placeholder(dtype=tf.float32, shape=(1, nb_nodes, nb_nodes))
    e = features.astype(float)
    output = tf.compat.v1.layers.dense(embedding, embedding_fn_size, activation=tf.nn.relu,
                                       kernel_initializer=tf.compat.v1.truncated_normal_initializer(
                                           stddev=0.1))

    output1 = tf.nn.dropout(output, 0.9)
    output2 = tf.matmul(output1, weights) + biases
    sco = tf.nn.sigmoid(output2)

    attns = getNodeAttentions(ftr_in, weight_mat=weight_in, bias_mat=bias_in,out_sz=64, activation=tf.nn.elu)

    with tf.compat.v1.Session() as sess:
        sess.run(tf.group(tf.compat.v1.global_variables_initializer(), tf.compat.v1.local_variables_initializer()))

        finalemb = features.reshape(1, features.shape[0], features.shape[1])

        iniScores,attentions = sess.run([sco, attns], feed_dict={embedding: features, ftr_in: finalemb,weight_in:wei,bias_in: bias})

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
        scores = {}
        for h in IdNumber:
            index1 = IdNumber.index(h)
            tempName1 = name[index1].strip()
            matr_ind = nodes.index(tempName1)
            tempscore = 0
            inner = get_neighbours(Graph, tempName1, in_out='in')
            for n in inner:
                if n != '0':
                    index2 = name.index(n)
                    matr_ind1 = nodes.index(n)
                    wei1 = float(weights[matr_ind1][matr_ind])
                    weight = wei1/float(max_weight)
                    tempscore = tempscore + weight*iniScores[index2]
            scores[tempName1] = tempscore
        sess.close()

    return scores

if __name__ == "__main__":
    scores = {}
    for num in range(head):
        score = scoreNetwork()
        if num == 0:
            scores = score
        else:
            for (key,value) in score.items():
               scores[key] = value + score[key]

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
    save_node_label('../data/result.txt', IdNumber1, name1, scores1)
    save_rank('E:/实验/DATA/weight/pdf/KeyClass.txt', name1)
    test = 'end'
