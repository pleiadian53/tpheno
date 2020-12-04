# encoding: utf-8

from collections import defaultdict
import pandas as pd
from pandas import DataFrame, Series
import os, gc, sys
from os import getenv 
import time, re, string, collections

try:
    import cPickle as pickle
except:
    import pickle

import random
import scipy
import numpy as np

# local modules 
from batchpheno import icd9utils, utils, predicate, qrymed2
from batchpheno.utils import div, indent
from config import seq_maker_config, sys_config
from pattern import medcode as pmed
import seqparams

 
# This class represents a directed graph using adjacency
# list representation
class Graph(object):
 
    # Constructor
    def __init__(self):
 
        # default dictionary to store graph
        self.graph = defaultdict(list)
        # self.path = {}
 
    # function to add an edge to graph
    def addEdge(self,u,v):
        self.graph[u].append(v)
 
    # Function to print a BFS of graph
    def BFS(self, s):
 
        path = [] # keep track of reachable vertices

        # Mark all the vertices as not visited
        visited = {k: False for k in self.graph.keys()} # [False]*(len(self.graph))
 
        # Create a queue for BFS
        queue = []
 
        # Mark the source node as visited and enqueue it
        queue.append(s)
        visited[s] = True
 
        while queue:
 
            # Dequeue a vertex from queue and print it
            x = queue.pop(0)
            # print v,
            path.append(x)
 
            # Get all adjacent vertices of the dequeued
            # vertex s. If a adjacent has not been visited,
            # then mark it visited and enqueue it
            for v in self.graph[x]:
                if visited[v] == False:
                    queue.append(v)
                    visited[v] = True
        return path

def toDict(G):
    """
    G: adjacency list

    """ 
    adict = {}
    for v, vset in G.items(): 
        adict[v] = set(vset)
    return adict

def index(G): 
    """
    Build a map from G's vertices to integers. 
    """
    adict = {}  # map vertex into integers
    
    # n = 0    
    for u, vset in G.items(): 
        if not adict.has_key(u): adict[u] = len(adict) 
        for v in vset: 
            if not adict.has_key(v): 
                adict[v] = len(adict) 

    return adict 

def vMap(G):
    vtoi = index(G)  # build indices
    print('info> n_vertices: %d' % len(vtoi))

    G2 = {}
    for u, vset in G.items(): 
        for v in vset: 
            iu = vtoi[u]
            if not G2.has_key(u): G2[iu] = set()
            G2[iu].add(vtoi[v])

    # find inverse map 
    itov = {v:k for k, v in vtoi.items()}  

    return (G2, itov)

def vMapInv(G, itov):
    """
    vmap: vertices to integers
    """
    # find reverse map

    G2 = {}
    for u, vset in G.items(): 
        assert isinstance(u, int)
        for v in vset: 
            assert isinstance(v, int)
            up, vp = itov[u], itov[v]
            if not G2.has_key(up): G2[up] = set()
            G2[up].add(vp)
    return G2

def linearize(torder, remove_index=True, ignoreList=None, to_str=True, sep=''):
    # e.g. set(['I1']), set(['C1']), set(['P2', 'I3', 'P1', 'V4', 'V2']), set(['L2', 'I4', 'D1']), set(['F1', 'I2', 'I5', 'V1', 'V3', 'L1', 'N1'])]

    orderedList = []
    for vset in torder: 
        for v in vset: 
            orderedList.append(v)
    
    print('info> ordered list:\n%s\n' % orderedList)
    if remove_index: 
        p = re.compile('(?P<vertex>\w+)(?P<index>\d+)')
        if ignoreList is None: ignoreList = ['ep', 'epsilon', ]

        for i, v in enumerate(orderedList): 
            if v in ignoreList: continue   # dummy or null tokens

            m = p.match(v)
            assert m is not None, "%s does not match %s" % (v, p.pattern)
            orderedList[i] = m.group('vertex')

        if to_str: 
            print('  + %s' % orderedList)
            olist = []
            for v in orderedList: 
                if v in ignoreList: continue   # dummy or null tokens
                olist.append(v)
            orderedStr = sep.join(olist)
            return orderedStr

    return orderedList

def t_peptide():  # also seqAlgo
    from toposort import toposort, toposort_flatten
    import graph

    # construct DAG 

    pdag = {}
    pdag['I0'] = ['ep', ]
    pdag['I1'] = ['I0', ] # self transition 
    pdag['C1'] = ['I1', ]
    pdag['V1'] = ['I2', 'D1', 'I4', 'V2', 'I3', 'C1', 'I1', ]
    pdag['I2'] = ['D1', 'I4', 'V2', 'I3', 'C1', 'I1', ]
    pdag['I3'] = ['C1', 'I1', ]
    pdag['D1'] = ['I4', 'V2', 'I3', 'C1', 'I1', ]
    pdag['I4'] = ['V2', 'I3', 'C1', 'I1']
    pdag['L1'] = ['V1', 'I2', 'D1', 'I4', 'V2', 'I3', 'C1', 'I1']
    pdag['V2'] = ['I3', 'C1', 'I1'] 
    pdag['F1'] = ['L1', 'V1', 'I2', 'D1', 'I4', 'V2', 'I3', 'C1', 'I1']
    pdag['N1'] = ['I5','V1', 'I2', 'D1', 'I4', 'V2', 'I3', 'C1', 'I1',]
    pdag['P1'] = ['L2', 'I4', 'V2', 'I3', 'C1', 'I1']
    pdag['V3'] = ['I2', 'D1', 'I4', 'V2', 'I3', 'C1', 'I1', ]
    pdag['V4'] = ['P1', 'L2', 'I4', 'V2', 'I3', 'C1', 'I1', ]
    pdag['P2'] = ['L2', 'I4', 'V2', 'I3', 'C1', 'I1', ]
    pdag['L2'] = ['I4', 'V2', 'I3', 'C1', 'I1', ]
    pdag['I5'] = ['V1', 'I2', 'D1', 'I4', 'V2', 'I3', 'C1', 'I1', ]

    print('info> size(G): %d' % len(pdag))
    G = toDict(pdag)   # convert list to set
    GI, itov = vMap(G)  # map vertices to integers; GI: vertices in integers while preserving graph struct, itov: maps from integers to vertices
    torder = list(toposort(GI)) 
    print('info> ordering in integer indices ...')
    print torder

    # maps back to original rep 
    print('info> original ordering ...')
    # GT = vMapInv(GIS, itov)
    # print GT 
    # [set([14]), set([13]), set([1]), set([6]), set([0, 12, 2, 4, 18]), set([11, 3, 5]), set([7, 8, 9, 10, 15, 16, 17])]
    
    torder2 = []
    for vset in torder: 
        vset2 = set()
        for v in vset: 
            vset2.add(itov[v])
        torder2.append(vset2)
    print torder2

    print('info> linearize ...')
    print linearize(torder2, remove_index=True)

    return 

def t_traversal(**kargs): 
    # Driver code
    # Create a graph given in the above diagram
    g = Graph()
    g.addEdge(0, 1)
    g.addEdge(0, 2)
    g.addEdge(1, 2)
    g.addEdge(2, 0)
    g.addEdge(2, 3)
    g.addEdge(3, 3)
 
    print "Following is Breadth First Traversal (starting from vertex 2)"
    path = g.BFS(2)
    print path

    return

def t_traversal2(**kargs): 
    # build graph from file

    # e.g. 'tpheno/data-exp/grouped_labs-Pshallow-Tbmeasurement-PTSD.csv'
    basedir = sys_config.read('DataExpRoot')
    fname = 'grouped_labs-Pshallow-Tbmeasurement-PTSD.csv'
    fpath = os.path.join(basedir, fname)
   
    # header = ['group', 'individual_lab', 'group_description', 'individual_description']
    df = pd.read_csv(fpath, sep='|', header=0, index_col=False, error_bad_lines=True)

    g = Graph()
    tokens = set()
    tokens.update(df['group'].values) 
    tokens.update(df['individual_lab'].values)
    tokens = sorted(list(tokens))  # set to sorted list
    print('> found %d unique lab codes' % len(tokens))

    # # encode 
    # LI = {}  # lab to index
    # for i, token in enumerate(tokens): 
    #     LI[token] = i
    # IL = {i:l for l, i in LI.items()}

    concept0 = {ch: [] for ch in df['group'].values}
    for i, row in df.iterrows(): 
        leader, member = row['individual_lab'], row['group']
        concept0[leader].append(member)
        g.addEdge(member, leader)  # memeber -> leader 
    print('info> size of graph: %d' % len(g.graph))

    lab = 106547 # 2294 # 106547
    # print('info> lab %s => index: %d\n' % (lab, LI[lab]))
    print('info> start lab: %d' % lab)
    path = g.BFS(lab)
    # print('info> lab %s => %s\n' % (lab, [IL[v] for v in path]))
    print path   # [106547, 110937, 110978, 111175, 113610, 163200] these are mutually reachable => should be further clustered 

    cluster = set(path)
    cluster.add(lab)

    i0, ch0 = 0, -1
    for i, ch in enumerate(cluster): # each member must be one of the leaders in concept0 
        if i == i0: 
            members = concept0[ch] 
            ch0 = ch
        else: 
            members2 = concept0[ch]
            if members2 == members: 
                print(' + group %d ~ %d' % (i, i0))
            else: 
                if len(members2) > len(members): 
                    if len(set(members)-set(members2)) == 0:
                        print('+ group %d :- group %d' % (ch0, ch))
                        print(' + [%d] => %s is a subset of [%d] => %s' % (ch0, members, ch, members2))
                elif len(members) > len(members2): 
                    if len(set(members2)-set(members)) == 0:
                        print('+ group %d :- group %d' % (ch, ch0))
                        print(' + [%d] is a subset of %s VS [%d] => %s' % (ch, members2, ch0, members))
                else: 
                    print(' + group %d <> %d' % (i, i0))
                    print(' + [%d] => %s VS [%d] => %s' % (ch, members2, ch0, members))

    # concepts = {}
    # for ch in df['group'].values: 
    #     path = g.BFS(lab)  # contains all other leaders reachable from
        
    #     for chi in path: 

    #     cluster = set(path)
    #     cluster.add(lab)


    # print('info> among %d initial groups, %d need to be futher consolidated.' % (df['group'].values))
   
    return

def test(**kargs): 
    # t_traversal(**kargs)
    # t_traversal2(**kargs)

    # sequence problems represented in graphs 
    t_peptide()

    return

if __name__ == "__main__": 
    test()