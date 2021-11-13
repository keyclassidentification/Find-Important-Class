import tensorflow as tf

def getNodeAttentions(seq,weight_mat, bias_mat,out_sz, activation=tf.nn.elu):
    with tf.name_scope('my_attn'):
        #计算Wh得到seq_fts,变换后的特征矩阵
        seq_fts = tf.compat.v1.layers.conv1d(seq, out_sz, 1, use_bias=False)

        # simplest self-attention possible
        #节点投影
        f_1 = tf.compat.v1.layers.conv1d(seq_fts, 1, 1)
        #节点邻居投影
        f_2 = tf.compat.v1.layers.conv1d(seq_fts, 1, 1)
        #将 f_2 转置之后与 f_1 叠加，得到注意力矩阵
        logits = f_1 + tf.transpose(f_2, [0, 2, 1]) #+weight_mat
        #注意力权重
        coefs = tf.nn.softmax(tf.nn.leaky_relu(logits) + bias_mat)

        return coefs[0]