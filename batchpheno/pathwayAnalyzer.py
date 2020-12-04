# encoding: utf-8

import matplotlib
matplotlib.use('Agg') # use a non-interactive backend such as Agg (for PNGs), PDF, SVG or PS.
from matplotlib import pyplot as plt
import seaborn as sns

from gensim.models import doc2vec
from collections import namedtuple
import collections

import csv
import re
import string

import sys, os, random 

from pandas import DataFrame, Series
import pandas as pd
import numpy as np

try:
    import cPickle as pickle
except:
    import pickle

# doc2vec experiments 
from gensim.models import Doc2Vec
# import gensim.models.doc2vec
from collections import OrderedDict

# local modules (assuming that PYTHONPATH has been set)
from batchpheno import icd9utils, sampling, qrymed2, utils, dfUtils
from config import seq_maker_config, sys_config
from batchpheno.utils import div
from pattern import medcode as pmed
import seqparams
import analyzer
import seqAnalyzer as sa 
import seqUtils, plotUtils
import evaluate
import algorithms  # count n-grams
from seqparams import TSet

# clustering algorithms 
from sklearn.cluster import KMeans, MiniBatchKMeans, AgglomerativeClustering, AffinityPropagation, SpectralClustering, DBSCAN
# from sklearn.cluster import KMeans

from sklearn.neighbors import kneighbors_graph

# evaluation 
from sklearn import metrics
from scipy.spatial import distance
from sklearn.neighbors import NearestNeighbors  # kNN


###############################################################################################################
#
#   Module 
#   ------ 
#   tpheno.seqmaker.pathwayAnalyzer
#
#   Related 
#   -------
#   tpheno.seqmaker.seqCluser (doc2vec + cluster analysis)
#      t_pathway()
#    
#   tpheno.seqmaker.seqAnalyzer (word2vec)
#
# 
#
###############################################################################################################

class Params(object): 
    """
    Params class lists commonly-used parameters as a reference and sets their default values


    """
    read_mode =  'doc' # documents/doc (one patient one sequence) or sequences/seq
    tset_type = 'binary'  # multiclass
    seq_ptype = 'regular'  # random, diag, med, 
    d2v_method = 'tfidf'   # PVDM: distributed memory, average, 
    std_method = 'minmax'  # standardizing feature values
    
    # [params] auxiliary 
    cluster_method = 'kmeans'  # see seqCluster.run_cluster_analysis(...) 
    n_clusters = 50
    topn_clusters_only = False # analyze topn clusters only
    topn_ngrams = 5  # analyze top N ngrams
    cluster_label_policy = 'max_vote'

    # [params] I/O
    identifier = 'T%s-C%s-S%s-D2V%s' % (tset_type, cluster_method, seq_ptype, d2v_method)
    # outputdir = os.path.join(os.getcwd(), 'data')
    
    # fname_cluster_stats = 'cluster_stats-%s-%s.csv' % (cluster_label_policy, identifier)
    # load_motif = kargs.get('load_motif', kargs.get('load_', True)) # load global motif

    # [params] ordering 
    partial_order = True  # document: partialy structured or strictly structured (i.e. sensitive to strict ordering)
### end class params

def cluster_purity(): 

    basedir = outputdir = os.path.join(os.getcwd(), 'data')
    fname = 'cluster_stats-max_vote-Tbinary-Ckmeans-Sregular-D2Vtfidfavg.csv'

def t_groupby(**kargs):
    """


    Reference
    ---------
    1. http://pandas.pydata.org/pandas-docs/stable/groupby.html

    """

    basedir = os.path.join(os.getcwd(), 'data') 
    
    # [template]
    # motifs-Cdiagnosis-L0-CID10-kmeans-Sregular-D2Vtfidf.csv

    # [params]
    seq_ptype = kargs.get('seq_ptype', 'regular')
    d2v_method = kargs.get('d2v_method', 'tfidf')
    tset_type = kargs.get('tset_type', kargs.get('ttype', 'binary'))
    ctype = kargs.get('cytpe', 'diagnosis')  # medication
    label = kargs.get('label', 0)  
    mtype = kargs.get('cid', kargs.get('mtype', 0)) # cid
    cluster_method = kargs.get('cluster_method', 'kmeans')  # minibatch, dbscan
    
    # [params] derived 
    fname = 'motifs-C%s-L%s-CID%s-%s-S%s-D2V%s.csv' % (ctype, label, mtype, cluster_method, seq_ptype, d2v_method)

    


    fpath = os.path.join(basedir, fname) 
    df = pd.read_csv(fpath, sep='|', header=0, index_col=False, error_bad_lines=True)
    nrow = df.shape[0]
    print('io> loaded df of dim: %s' % str(df.shape))
    print('verify> top:\n%s\n' % df.head(20))

    header = ['length', 'ngram', 'count', 'global_count', 'ratio']
    pivots = ['length', ] # groupby attributes
    sort_fields = ['ratio', ]
    n_clusters = 50

    # [note] By default the group keys are sorted during the groupby operation
    groups = df.groupby(pivots)  # as_index=False # set as_index to False => pivots becomes a separate column instead of an index
    
    dfx = []
    for n, dfg in groups: 
        # df.apply(lambda x: x.order(ascending=False)) # take top 3, .head(3)
        dfg_sorted = dfg.sort(sort_fields, ascending=False, inplace=False)

        # doc-length-adjusted tf 
        dfg_sorted['tf'] = dfg_sorted['count']/nrow

        # idf: poll all cluster files

        dfx.append(dfg_sorted)
        # print('> length: %d:\n%s\n' % (n, df_group))

    df = pd.concat(dfx, ignore_index=True)
    print('verify> top:\n%s\n' % df.head(20))
    
    return

def test(): 
    return

if __name__ == "__main__": 
    test()