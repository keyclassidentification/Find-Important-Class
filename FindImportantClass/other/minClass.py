#coding:utf-8
import math
import networkx as nx

from helper.graphHelper import get_neighbours

PER_SUM_W_IN = {}
PER_SUM_W_OUT = {}
PER_h = {}

NODE_PATH = set()

def per_compute(G,weight=None):
    for node in G:
        PER_SUM_W_IN[node] = Wsum_w_in(G,node,weight)
        PER_SUM_W_OUT[node] = Wsum_v_out(G,node,weight)
    for node in G:
        PER_h[node] = h(G,node,weight=weight)
    return PER_h

def mutual_weight(G, u, v, weight=None):
    try:
        a_uv = G[u][v].get(weight, 1)
    except KeyError:
        a_uv = 0
    return a_uv

def H(G,w,weight=None):
    r = PER_h[w]
    if len(get_neighbours(G,w,in_out='in')) > 0 and w not in NODE_PATH:
        NODE_PATH.add(w)
        if w=='org.apache.tools.ant.Task':
            nei=get_neighbours(G,w,in_out='in')
        for v in get_neighbours(G,w,in_out='in'):
            wei = mutual_weight(G,v,w,weight=weight)
            out = PER_SUM_W_OUT[v]
            r = r + (mutual_weight(G,v,w,weight=weight)/PER_SUM_W_OUT[v]) * PER_h[v]
        return r
    else:
        return r

def h(G,w,weight=None):
    r = 0
    for x in get_neighbours(G,w,in_out='in'):
        Pxw = mutual_weight(G,x,w,weight=weight)
        r = r + Pxw * math.log(Pxw)
        # print('1-r',1-r)
    return (1 - r) * PER_SUM_W_IN[w]

def Wsum_w_in(G,w,weight=None):
    return sum(mutual_weight(G,y,w,weight=weight) for y in get_neighbours(G,w,in_out='in'))
def Wsum_v_out(G,v,weight=None):
    return sum(mutual_weight(G,v,u,weight=weight) for u in get_neighbours(G,v,in_out='out'))

def compute_H(G,weight=None):
    per_compute(G,weight=weight)
    all_H = {}
    for w in G:
        NODE_PATH.clear()
        all_H[w]= H(G,w,weight=weight)
    return all_H