# encoding: utf-8

import matplotlib
matplotlib.use('Agg') # use a non-interactive backend such as Agg (for PNGs), PDF, SVG or PS.
from matplotlib import pyplot as plt
import seaborn as sns

from gensim.models import doc2vec
from collections import namedtuple
import collections, itertools

import csv
import re
import string

import sys, os, random, math, gc

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
import analyzer, vector
# encoding: utf-8

"""
#
#   Module 
#   ------ 
#   tpheno.seqmaker.pathwayAnalyzer
#   
#   Input (and Operational Dependency)
#   ----------------------------------
#   Mainly outputs and byproducts generated from seqmaker.seqAnalyzer and seqmaker.seqCluster
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
"""
import seqAnalyzer as sa 
import seqReader as sr
# import seqUtils, plotUtils
import evaluate
import algorithms, seqAlgo  # count n-grams, sequence algorithms, etc.
from seqparams import TSet

# clustering algorithms 
from sklearn.cluster import KMeans, MiniBatchKMeans, AgglomerativeClustering, AffinityPropagation, SpectralClustering, DBSCAN
# from sklearn.cluster import KMeans

from sklearn.neighbors import kneighbors_graph

# evaluation 
from sklearn import metrics
from scipy.spatial import distance
from sklearn.neighbors import NearestNeighbors  # kNN

class Params(object): 
    """
    Params class lists commonly-used parameters as a reference and sets their default values


    """
    read_mode =  'doc' # documents/doc (one patient one sequence) or sequences/seq
    tset_type = 'binary'  # multiclass
    seq_ptype = 'regular'  # random, diag, med, 
    d2v_method = 'tfidfavg'   # PVDM: distributed memory, average, 
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


def analyze_pathway(X, y=None, D=None, clusters=None, **kargs): # seqmaker.seqCluster
    pass 

def analyze_pathway_batch(X, y=None, D=None, clusters=None, **kargs): 
    pass 

# [cluster]
def cluster_purity(**kargs): 
    """

    Memo
    ----
    1. example input file: 
       cluster_stats-max_vote-Tbinary-Ckmeans-Sregular-D2Vtfidfavg.csv

    """
    basedir = outputdir = os.path.join(os.getcwd(), 'data')

    # [params]
    seq_ptype = kargs.get('seq_ptype', 'regular')
    d2v_method = kargs.get('d2v_method', 'tfidfavg') # or 'tfidf'
    tset_type = kargs.get('tset_type', kargs.get('ttype', 'binary'))
    ctype = kargs.get('cytpe', 'diagnosis')  # medication
    label = kargs.get('label', 0)  
    mtype = kargs.get('cid', kargs.get('mtype', 0)) # cid
    cluster_method = kargs.get('cluster_method', 'kmeans')  # minibatch, dbscan
    identifier = 'T%s-C%s-S%s-D2V%s' % (tset_type, cluster_method, seq_ptype, d2v_method)
    sep = '|'
    save_df = kargs.get('save_df', True)

    cluster_label_policy = kargs.get('cluster_label_policy', 'max_vote')
    fname = 'cluster_stats-%s-%s.csv' % (cluster_label_policy, identifier)
    fpath = os.path.join(basedir, fname)

    assert os.path.exists(fpath), "input: %s does not exist" % fpath 
    df = pd.read_csv(fpath, sep=sep, header=0, index_col=False, error_bad_lines=True)
    print('io> loaded cluster purity file (dim: %s) from: %s' % (str(df.shape), fpath))

    # [params] query
    pivots = ['label', ] # groupby attributes
    sort_fields = ['ratio', ]
    groups = df.groupby(pivots)  # as_index=False # set as_index to False => pivots becomes a separate column instead of an index

    dfx = []
    for label, dfg in groups: 
        print('group> label: %s => dim: %s' % (label, str(dfg.shape)))
        dfx.append(dfg.sort(sort_fields, ascending=False, inplace=False))

    df = pd.concat(dfx, ignore_index=True)
    print('io> top:\n%s\n' % df.head(20))

    if save_df: 
        fpath_new = fpath
        print('io> saving new dataframe to %s' % fpath_new)
        df.to_csv(fpath_new, sep=sep, index=False, header=True)  

    return df

def eval_cluster(**kargs):

    # global motif file (e.g. motifs-CMOPmixed-global-total-prior-Sregular-D2Vtfidfavg.csv)
    ctype, scope, order_type, policy_type, d2v_method, = ('mixed', 'global', 'part', 'prior', 'tfidfavg')
    fname = 'motifs-CMOP%s-%s-%s-%s-Sregular-D2V%s.csv' % (ctype, scope, order_type, policy_type, d2v_method)

    # cluster motif files
    ctype, scope, order_type, policy_type, d2v_method, = ('mixed', 'global', 'part', 'prior', 'tfidfavg')
    fname = 'motifs-CIDL9-1-COPmixed-part-prior-Ckmeans-Sregular-D2Vtfidfavg.csv'

    return 

def get_idf(nc, ncf_ct, base=10): 
    """

    Input
    -----
    nc: total number of documents/clusters
    ncf_ct: number of documents/clusters that contain term t

    Reference
    ---------
    1. http://stackoverflow.com/questions/25169297/how-to-take-the-logarithm-with-base-n-in-numpy
    2. http://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction

    """
    # import math
    
    # n: total number of docs 
    # ndf: |d in D and t in d|, number of docs that contain word t
    # t: word

    return math.log((1.0+nc)/(1.0+ndf), base)+1

def eval_tfidf(**kargs): 
    """
    Subsummed by seqmaker.seqCluser 

    Example tfidf file: 
        tfidf-diagnosis-Tbinary-Ckmeans-Sregular-D2Vtfidfavg.csv
    """
    def to_tuple(ngr, delimit=' '):
        # 272.0 272.0 401.9  
        return tuple(ngr.strip().split(delimit))

    basedir = os.path.join(os.getcwd(), 'data') 
    
    # [template]
    # motifs-Cdiagnosis-L0-CID10-kmeans-Sregular-D2Vtfidf.csv

    # [params]
    seq_ptype = kargs.get('seq_ptype', 'regular')
    d2v_method = kargs.get('d2v_method', 'tfidfavg')
    tset_type = kargs.get('tset_type', kargs.get('ttype', 'binary'))
    ctype = kargs.get('cytpe', 'diagnosis')  # medication
    label = kargs.get('label', 0)  
    mtype = kargs.get('cid', kargs.get('mtype', 0)) # cid
    cluster_method = kargs.get('cluster_method', 'kmeans')  # minibatch, dbscan
    cluster_label_policy = kargs.get('cluster_label_policy', 'max_vote')
    identifier = 'T%s-C%s-S%s-D2V%s' % (tset_type, cluster_method, seq_ptype, d2v_method)

    n_clusters = 50
    ulabels = [0, 1, ]  # [todo]
    sep = '|'

    # cfreqx = {}
    header = ['ngram', 'cluster_freq', ] # ngram|cluster_freq
    fname_tfidf = 'tfidf-%s-%s.csv' % (ctype, identifier) # identifier: (tset_type, cluster_method, seq_ptype, d2v_method)
    fpath = os.path.join(basedir, fname_tfidf)
    df = pd.read_csv(fpath, sep=sep, header=0, index_col=False, error_bad_lines=True)
    print('io> loaded tfidf stats file of dim: %s from %s' % (str(df.shape), fpath))
    keys = df['ngram'].values
    values = df['cluster_freq'].values
    cfreqx = dict(zip(keys, values))
    
    print cfreqx
    # sys.exit(0)

    # populate cluster motif files to find top N motifs 
    header = ['length', 'ngram', 'count', 'global_count', 'ratio', ]
    cmotifs, cstats = {}, {}
    d2v_method = 'tfidfavg'
    for ulabel in ulabels: 
        if not cmotifs.has_key(ulabel): cmotifs[ulabel] = {}
        if not cstats.has_key(ulabel): cstats[ulabel] = 0

        for cid in range(n_clusters): 
        	# if not cmotifs[ulabel].has_key(cid): cmotifs[ulabel]
            fname_cmotifs = 'motifs-C%s-L%s-CID%s-%s-S%s-D2V%s.csv' % (ctype, label, cid, cluster_method, seq_ptype, d2v_method)
            fpath = os.path.join(basedir, fname_cmotifs) 
            if os.path.exists(fpath): 

                # cmotifs[ulabel][cid] = dfc = pd.read_csv(fpath, sep=sep, header=0, index_col=False, error_bad_lines=True)
                dfc = pd.read_csv(fpath, sep=sep, header=0, index_col=False, error_bad_lines=True)
                
                nrow = dfc.shape[0]
                print('io> loaded df for (l: %s, cid: %s) => dim: %s' % (ulabel, cid, str(dfc.shape)))

                scorex = [] # tfidf
                for r, row in dfc.iterrows(): 
                    ngr, cnt = row['ngram'], row['count']
                    # ngrt = to_tuple(ngr)
                    assert ngr in cfreqx, "n_gram: %s is not in tfidf table" % str(ngr)
                    ncf_ct = cfreqx.get(ngr, 0) # number of clusters containing this ngram
                    score_idf = get_idf(nc=n_clusters, ncf_ct=ncf_ct)
                    score_tf = cnt/(nrow+0.0)   # length-adjusted term frequency
                    score_tfidf = score_tf * score_idf
                    scorex.append(score_tfidf)
                
                dfc['tfidf'] = scorex 
                cstats[ulabel] += 1 

                # saving tfidf scores
                fpath_new = fpath 
                print('io> updating %s with tfidf score ...' % fname_cmotifs)
                df.to_csv(fpath_new, sep=sep, index=False, header=True) 
            else: 
            	print('warning> file: %s does not exist' % fname_cmotifs)

    for ulabel in ulabels: 
    	print("verify> label: %s, n_files: %d" % (ulabel, cstats[ulabel]))

    # [params]
    pivots = ['length', ]
    sort_fields = ['tfidf', ]

    # sort each n-gram according to tfidf scores
    for ulabel in ulabels: 
        if not cmotifs.has_key(ulabel): cmotifs[ulabel] = {}
        if not cstats.has_key(ulabel): cstats[ulabel] = 0

        for cid in range(n_clusters): 
        	# if not cmotifs[ulabel].has_key(cid): cmotifs[ulabel]
            fname_cmotifs = 'motifs-C%s-L%s-CID%s-%s-S%s-D2V%s.csv' % (ctype, label, mtype, cluster_method, seq_ptype, d2v_method)
            fpath = os.path.join(basedir, fname_cmotifs) 
            if os.path.exists(fpath): 
            	dfc = pd.read_csv(fpath, sep=sep, header=0, index_col=False, error_bad_lines=True)
            	groups = df.groupby(pivots)  # as_index=False # set as_index to False => pivots becomes a separate column instead of an index
    
                dfx = []
                for n, dfg in groups: 
                	dfx.append(dfg.sort(sort_fields, ascending=False, inplace=False))
                dfc = pd.concat(dfx, ignore_index=True)

                # saving sorted motifs
                fpath_new = fpath 
                print('io> updating %s after sorting ...' % fname_cmotifs)
                df.to_csv(fpath_new, sep=sep, index=False, header=True)                 

    return

def adjust_motifs(X, y=None, D=None, clusters=None):
    """

    Memo
    ----
    1. example global motif files 
       motifs-CMOPmixed-global-part-noop-Sregular-D2Vtfidfavg.csv

    """
    from seqCluster import build_data_matrix_ts, cluster_analysis, eval_motif, eval_cluster_motif, map_clusters
    import evaluate, seqparams
    from pattern import diabetes as diab

    cmetrics = {}
    if X is None or y is None or D is None: 
        D, ts = build_data_matrix_ts(read_mode=read_mode, seq_ptype=seq_ptype, d2v_method=d2v_method, tset_type=tset_type)
        X, y = evaluate.transform(ts, standardize_=std_method) # default: minmax
        # X, y, D = getXYD(read_mode=read_mode, seq_ptype=seq_ptype, d2v_method=d2v_method, tset_type=tset_type) 
    if clusters is None: 
        # run cluster analysis
        clusters, cmetrics = cluster_analysis(X=X, y=y, n_clusters=n_clusters, cluster_method=cluster_method,
                                    seq_ptype=seq_ptype, d2v_method=d2v_method, 
                                    save_=True, load_=True)

    assert clusters is not None and D is not None
    cluster_to_docs = map_clusters(clusters, D)  # cluster ID => a set of documents (d in D) in the same cluster 

    for ctype in ('diagnosis', 'medication', 'mixed',):  # ~ seq_ptype: (diag, med, regular)
        D0 = {cid:[] for cid in cluster_to_docs.keys()}
        seq_ptype = seqparams.normalize_ptype(ctype)  # overwrites global value
        
        for cid, docs in cluster_to_docs.items(): 
            for doc in docs: 
                D0[cid].append(diab.transform(doc, policy='noop', inclusive=True, seq_ptype=seq_ptype)) # before the first diagnosis, inclusive

        # seq_ptype (ctype) i.e. seq_ptype depends on ctype; assuming that d2v_method is 'global' to this routine
        # identifier = 'CM%s-%s-O%s-C%s-S%s-D2V%s' % (ctype, mtype, order_type, cluster_method, seq_ptype, d2v_method)
        # [memo] otype (partial_order)
        gfmotifs = eval_motif(D0, topn=None, ng_min=1, ng_max=8, 
                                partial_order=partial_order, 
                                mtype='global', ctype=ctype, ptype='noop',
                                cluster_method=cluster_method, seq_ptype=seq_ptype, d2v_method=d2v_method,
                                    save_=True, load_=True)  # n (as in n-gram) -> topn counts [(n_gram, count), ]

    return 

def filter_motifs(n_clusters=10, **kargs):
    """

    Dependency 
    ----------
    seqmaker.analyze_pathway

    """ 
    identifier = 'G%s-COP%s-%s-%s-nC%s-nL%s-D2V%s' % (cohort_name, ctype, order_type, policy_type, n_clusters, n_classes, d2v_method)

    cids = range(n_clusters)

    for cid in cids: 
        fname_tfidf_local = 'pathway_C%s-%s.csv' % (cid, identifier)  # [I/O] local
        assert fname_tfidf_local.find(cohort_name) > 0
        fpath_local = os.path.join(outputdir, fname_tfidf_local) 

    return


def t_groupby(**kargs):
    """


    Reference
    ---------
    1. http://pandas.pydata.org/pandas-docs/stable/groupby.html

    Memo
    ----
    1. Outputs from t_pathway()

        identifier = 'T%s-C%s-S%s-D2V%s' % (tset_type, cluster_method, seq_ptype, d2v_method)
        fname_tfidf = 'tfidf-%s-%s.csv' % (ctype, identifier) # identifier: (tset_type, cluster_method, seq_ptype, d2v_method)
        fname_ngram_cidx = 'ngram_cidx-%s-%s.csv' % (ctype, identifier) # label policy as part of the identifier? 

        fname_cluster_stats = 'cluster_stats-%s-%s.csv' % (cluster_label_policy, identifier)

    """

    basedir = os.path.join(os.getcwd(), 'data') 
    
    # [template]
    # motifs-Cdiagnosis-L0-CID10-kmeans-Sregular-D2Vtfidf.csv

    # [params]
    seq_ptype = kargs.get('seq_ptype', 'regular')
    d2v_method = kargs.get('d2v_method', 'tfidfavg')
    tset_type = kargs.get('tset_type', kargs.get('ttype', 'binary'))
    ctype = kargs.get('cytpe', 'diagnosis')  # medication
    label = kargs.get('label', 0)  
    mtype = kargs.get('cid', kargs.get('mtype', 0)) # cid
    cluster_method = kargs.get('cluster_method', 'kmeans')  # minibatch, dbscan
    cluster_label_policy = kargs.get('cluster_label_policy', 'max_vote')

    # [params] derived 
    
    # cmotifs: cluster motifs
    fname_cmotifs = 'motifs-C%s-L%s-CID%s-%s-S%s-D2V%s.csv' % (ctype, label, mtype, cluster_method, seq_ptype, d2v_method)
    fpath = os.path.join(basedir, fname_cmotifs) 

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

def pLookupTable(**kargs):   # p: process
    return processLookupTable(**kargs)
def processLookupTable(**kargs): 
    """
    Process medical coding look up table (e.g. ICD-9 lookup)
    """

    cohort_name = kargs.get('cohort', 'PTSD')
    basedir = seqparams.get_basedir(cohort=cohort_name) 

    div(message='Sort lookup table by code ...')    
    # e.g. token_lookup-diag.csv
    content_types = ['diag', 'med', ]
    sort_keys = ['code', 'med']  # [todo] unify the sort key? 

    stem = 'token_lookup'
    sep = '|'

    for i, ctype in enumerate(content_types): 
        fname = '%s-%s.csv' % (stem, ctype)
        fpath = os.path.join(basedir, fname)
        df = pd.read_csv(fpath, sep=sep, header=0, index_col=False, error_bad_lines=True)
        df = df.sort_values(sort_keys[i], ascending=True)

        print('io> saving sorted lookup table (~ %s) to %s' % (sort_keys[i], fpath))
        df.to_csv(fpath, sep=sep, index=False, header=True)  

    return 

def findRepresentativeNGramsByPresence(**kargs):

    # [params]
    cohort_name = kargs.get('cohort', 'PTSD')
    basedir = seqparams.get_basedir(cohort=cohort_name) 

    # [params] file ID
    content_type = ctype = kargs.get('ctype', 'diagnosis') # content type (what's in the sequence?): medication, mixed
    order_type = otype = kargs.get('otype', 'total')
    policy_type = ptype = kargs.get('ptype', 'prior')  # cut point: pre-diagnosis (prior), post-diagnosis (posterior), whole sequence (mixed)
    cluster_method = kargs.get('cluster_method', 'kmeans')
    seq_ptype = kargs.get('seq_ptype', seqparams.normalize_ptype(ctype))  # diag, regular, med
    d2v_method = kargs.get('d2v_method', 'tfidfavg')
    identifier = 'G%s-COP%s-%s-%s-nC%s-nL%s-D2V%s' % (cohort_name, ctype, order_type, policy_type, n_clusters, n_classes, d2v_method)

    # ['length', 'ngram', 'tf_global', 'cluster_freq', 'cluster_occurrence', ] 
    df = loadGlobalMotifs(**kargs); assert df is not None and not df.empty
    dim0 = df.shape
    print('io> loaded global motif file (dim: %s)' % str(dim0))

    documents, timestamps = loadDocuments(seq_ptype=seq_ptype, cohort=cohort_name)  # [params] cohort, ctype ('mixed'), include_timestamps (True)
    if len(timestamps) > 0: assert len(documents) == len(timestamps)

    # it seems time-consuming to loop through all temporal docs to see which subset has a given n-gram

    return 

def findRepresentativeNGrams(**kargs): # representativeness by a particular score such as TFIDF
    return pClusterPathway(**kargs)
def pClusterPathway(**kargs): # p: process or postprocessing 
    """
    Find popular n-grams on the global and cluster levels. 

    Parameters 
    ----------
    cluster_freq_max: an upperbound of # of clusters that an n-gram appears
    

    Input
    -----
    1a. global motif files 
        tpheno/seqmaker/data/<cohort>/pathway/pathway_global*
           example: pathway_global-GPTSD-COPdiagnosis-partial-noop-nC10-nL1-D2Vtfidfavg.csv

    2a. cluster-level motif 
        tpheno/seqmaker/data/<cohort>/pathway/pathway_C*
    
    Output   a -> b
    ------
    1b. global motifs that occur only in few clusters (subtype-indicating motifs)
        tpheno/seqmaker/data/<cohort>/pathway/pathway_global_derived-* 
    2b. cluster-level motif 
        tpheno/seqmaker/data/<cohort>/pathway/pathway_cluster_derived-* 


    * cluster level motifs
    topn_per_cluster = 20

    Memo
    ----
    1. global level 
       columns: ['length', 'ngram', 'tf_global', 'cluster_freq', 'cluster_occurrence', ] 
       where, 
           cluster_freq: number of clusters that contain a given n-gram 
           cluster_occurrence: the clusters (IDs) that contain a given n-gram

    2. cluster level 
       columns:  columns: ['cid', 'length', 'ngram', 'tfidf', 'tf_cluster', 'tf_global', 'cluster_freq', 'label', 'cluster_occurrence', ]  (+ label if n_classes > 1)

       note that 'tf_cluster' is the term frequency (of a given n-gram) within the cluster 
             but 'cluster_freq' is the number of clusters that contain a given n-gram 

    Questions 
    ---------
    Using global motifs 
       + find dominant n-gram motifs for each cluster 
           > n-grams that appear in at most 'm' clusters (e.g. m =1)
               > among all the above n-grams, list the top k n-grams 

    Using cluster motifs 
       + find dominant cluster-level n-grams
           > n-grams with high ratios between tf within cluster and tf at global scope

    """
    def count_n_clusters(s, sep=' '): 
        # if s in (None, float('nan')): return 0 
        if isinstance(s, int): 
            return 1 
        else: 
            try: 
                return len(s.split(sep))
            except: 
                pass
        raise ValueError, "Not valid cluser IDs: %s" % s

    cohort_name = kargs.get('cohort', 'PTSD')
    basedir = seqparams.get_basedir(cohort=cohort_name)   # tpheno/seqmaker/data/<cohort>
    sep = '|'  # column separator

    # each cluster has its own pathway file (which is derived from cluster motifs & which includes tf-idf scores, etc.)
    n_classes = kargs.get('n_classes', 1)
    n_clusters = kargs.get('n_clusters', 10) 
    cIDs = range(0, n_clusters)   # CID

    # label = kargs.get('label', 1)
    content_type = ctype = kargs.get('ctype', 'diagnosis') # content type (what's in the sequence?): medication, mixed
    order_type = otype = kargs.get('otype', 'total')
    policy_type = ptype = kargs.get('ptype', 'prior')  # cut point: pre-diagnosis (prior), post-diagnosis (posterior), whole sequence (mixed)
    cluster_method = kargs.get('cluster_method', 'kmeans')
    seq_ptype = kargs.get('seq_ptype', seqparams.normalize_ptype(ctype))  # diag, regular, med
    d2v_method = kargs.get('d2v_method', 'tfidfavg')

    identifier = 'G%s-COP%s-%s-%s-nC%s-nL%s-D2V%s' % (cohort_name, ctype, order_type, policy_type, n_clusters, n_classes, d2v_method)

    div(message='1. Analyzing global motifs ...')
    ### global level pathway file (sorted by ['length', 'tf_global', ] followed by ['cluster_freq'] in ascending order)
    # e.g. pathway_global-GPTSD-COPdiagnosis-total-prior-nC10-nL1-D2Vtfidfavg.csv

    # first groupby cluster IDs and find their unique (or close to unique) n-grams (ranked by cluster ngrams)
    # [params]
    cluster_freq_max = kargs.get('cluster_freq_max', 1)  # set an upperbound of # of clusters that an n-gram appears
    topn_per_cluster = 20   # parameter for global scope

    # [input]
    fname_global = 'pathway_global-%s.csv' % identifier # identifier: (tset_type, cluster_method, seq_ptype, d2v_method)
    
    # path after organing the files according to their properties
    basedir1 = os.path.join(basedir, 'pathway') 
    fpath = os.path.join(basedir1, fname_global)
    assert os.path.exists(fpath), 'invalid path? %s' % fpath 

    print('io> reading global pathway file (tfidf scores, etc.): %s' % fname_global)
    df = pd.read_csv(fpath, sep=sep, header=0, index_col=False, error_bad_lines=True)
    dim0 = df.shape

    # [filter] 
    print('filter> take only n-grams that occur in few clusters (cluster_freq <= %d)' % cluster_freq_max)
    df1 = df.loc[df['cluster_freq'] <= cluster_freq_max]
    
    if cluster_freq_max == 1: 
        df1['cluster_occurrence'] = df1['cluster_occurrence'].astype(int)

    print('transform> sort, groupby, filter ... ')
    # [params]
    dfx = []
    gcols = ['cluster_occurrence', ]
    n_groups = 0
    
    pivot = 'tf_global'
    apply_group_sort = True
    clouds = []

    # df[['col2','col3']] = df[['col2','col3']].apply(pd.to_numeric)
    # df1[gcols] = df1[gcols].apply(pd.to_numeric)
    for g, dfi in df1.groupby(gcols):  # group by cluster occurrences (cids)
        try: 
            dfi[gcols] = dfi[gcols].apply(pd.to_numeric) # single cluster

        except: 
            print('  + non-numerical cluster groups: %s' % dfi[gcols].values)

        dfi1 = dfi.nlargest(topn_per_cluster, pivot)  # 20, tf_global
        dfi2 = dfi1.sort_values(pivot, ascending=False)  # tf_global
        
        dfx.append(dfi2) # highest global counts (but if it's only in one cluster, then ~ cluster counts)
        n_groups += 1
        clouds.append(g)  # [0, 1, 2, 3, ... '0', '1', '2', '3', ...]  # need to consolidate

    df = pd.concat(dfx, ignore_index=True)
    print('verify> cluster groups (i.e. clouds): %s' % clouds)

    if apply_group_sort: 

        # list n-grams in the order of CIDs 
        p1, p2 = 'cluster_occurrence', 'tf_global',
        df = df.sort_values(p1, ascending=True, kind='mergesort') # sort_values(p2, ascending=False, kind='mergesort')
        # ['length', 'ngram', 'tf_global', 'cluster_freq', 'cluster_occurrence', ]

    # re-ordering attributes
    # cluster_occurrence: In which clusters does a given ngram appear? 
    df = df[['cluster_occurrence', 'ngram', 'length', 'tf_global', 'cluster_freq']]
 
    # derived 

    # [params][todo] add parameters? cluster_freq_max, topn_per_cluster
    identifier1 = 'G%s-COP%s-%s-%s' % (cohort_name, ctype, order_type, policy_type)
    fname_out = 'pathway_global_derived-%s.csv' % identifier1 # identifier: (tset_type, cluster_method, seq_ptype, d2v_method)
    fpath_out = os.path.join(basedir1, fname_out)
    df.to_csv(fpath_out, sep='|', index=False, header=True)
    print('output1> saved the filtered global pathway file (params: cluster_freq_max: %d, n_groups: %d) to:\n%s\n' % \
        (cluster_freq_max, n_groups, fpath_out))

    # [log] dim ((2216999, 5) -> (200, 5))   2M+ -> 200
    print('info> transformed data: dim (%s -> %s)' % (str(dim0), str(df.shape)))

    # save the selected cluster-level representative n-grams for each cluster ID for futher analysis? 

    df2 = df1 # df1.loc[df1['cluster_freq'] == 1]
    # cluster representatives: cid -> represntative n-grams (more unique and with high counts)
    cRepr = zip(df2['cluster_occurrence'].values, df2['ngram'].values)
    cNgramCounts = zip(df2['ngram'].values, df2['tf_global'].values)
    wsep = ' '

    # aggregate n-grams 
    cluster_ngram = {}  # cluster ID -> n-grams
    cloud_ngram = {} # cIDs -> n-grams  ... same n-gram occurring in multiple clusters
    for cidr, ngr in cRepr: 
        # cids = str(cidstr).split()

        if isinstance(cidr, str):  #  cids is of "object" type because of the mixture of integer and strings (multiple clusters)
            cids = cidr.split(wsep)
        else: 
            assert isinstance(cidr, int)
            cids = [cidr]
        
        if len(cids) == 1:   # single cluster 
            cid = cids[0]
            if not cluster_ngram.has_key(cid): cluster_ngram[cid] = []
            cluster_ngram[cid].append(ngr) 
        else: 
            # cluster map 
            for cid in cids: 
                if not cluster_ngram.has_key(cid): cluster_ngram[cid] = []
                cluster_ngram[cid].append(ngr) 

            # cloud map: keep track of the n-grams that occur in mutliple clusters separately
            cloud = tuple(cids)
            if not cloud_ngram.has_key(cloud): cloud_ngram[cloud] = []
            cloud_ngram[cloud].append(ngr)
    
    div(message='2. Analyze top representative n-grams for each cluster using CLUSTER MOTIFS files (n_clusters=%d >=? %d)' % \
        (len(cluster_ngram), n_clusters))
    
    # cluster level pathway files (1)
    #   + header = ['length', 'ngram', 'count', 'global_count', 'ratio']  # 'ratio': ccnt/gcnt
    #   + example file: 

    # cluster level pathway files (2) (originally sorted by ['length', 'tfidf', ])
    #   + header: ['length', 'ngram', 'tfidf', 'tf_cluster', 'tf_global', 'cluster_occurrence', ]   // 'label' if n_classes > 1
    #   + example file: pathway_C9-GPTSD-COPdiagnosis-total-prior-nC10-nL1-D2Vtfidfavg.csv
    #   + query 
    #       > find all n-grams that appear in at most 'm' clusters (m = 1)   => set X 
    #           > sort X ~ tf-idf scores > take top 'k' 
    #              
    qcol = ['cluster_occurrence', ]
    col_n_clusters = 'cluster_freq'
    col_topn = 'tfidf'

    sep = '|'
    cReprMap = dict(cRepr)

    focused_lengths = [1, 2, 4, 8]
    cluster_freq_max = 1
    # topn_per_cluster = 50
    topn_per_length = topn_per_n = 20
    apply_group_sort = True

    header = ['cid', 'ngram', 'tfidf', 'tf_cluster', 'tf_global', ] # last two attributes are for plotting
    adict = {h: [] for h in header}
    # identifier = 'G%s-COP%s-%s-%s-nC%s-nL%s-D2V%s' % (cohort_name, ctype, order_type, policy_type, n_clusters, n_classes, d2v_method)
    print('params> cluster_freq_max=%d + topn_per_length=%d' % (cluster_freq_max, topn_per_n))

    for cid in cIDs: # ngr: n-gram in tuple repr
        print('  + Cluster %s ' % cid)
        # input data header: length|ngram|tfidf|tf_cluster|tf_global|cluster_occurrence
        fname = 'pathway_C%s-%s.csv' % (cid, identifier)  # [I/O] local
        basedir1 = os.path.join(basedir, 'pathway')
        fpath = os.path.join(basedir1, fname)   # [input] tpheno/seqmaker/data/<cohort>/pathway/pathway_C*
        dfc = pd.read_csv(fpath, sep=sep, header=0, index_col=False, error_bad_lines=True)
        nrow0 = dfc.shape[0]

        # find number of occurrences for each cluster-level n-grams (obtained from global, cluster motif files)
        # cluster_ngrams = cRepr.get(cid, None)
        # if cluster_ngrams is not None: 
        #     pass
        # multiple cids (same n-gram occurred in multiple clusters, small set)

        if not col_n_clusters in dfc.columns: 
            dfc[col_n_clusters] = dfc['cluster_occurrence'].apply(count_n_clusters)  # apply count n_cluster function to each row

        # [filter] filter by number of clusters (that contain a given n-gram)
        print('  ++ [filter] by number of clusters that contain a given n-gram (max=%d)' % cluster_freq_max)
        cond_cf = dfc[col_n_clusters] <= cluster_freq_max   # chooose only cluster-specific ngrams (those which only appear in few clusters)
        dfc = dfc.loc[cond_cf]

        # [filter] filter by lengths
        print('  ++ [filter] retain only those with lengths=%s' % focused_lengths)
        cond_length = dfc['length'].isin(focused_lengths)
        dfc = dfc.loc[cond_length]
        nrow1 = dfc.shape[0]
        print('  ++++ [filter result] nrow: %d -> %d (cluster_freq_max=%d, focused_lengths=%s)' % \
            (nrow0, nrow1, cluster_freq_max, focused_lengths))

        # [sort/group] foreach length, list top 'k' n-grams ~ tfidf
        # also sort tf-idf (should have been sorted by 'tfidf')
        if apply_group_sort: 
            dfx = []
            for i, (g, dfi) in enumerate(dfc.groupby(['length', ])):  # group by cluster IDs
                # if i == 0: print('     + columns: %s' % dfi.columns.values)  # ['length' 'ngram' 'tf_global' 'cluster_freq' 'cluster_occurrence']
                dfi1 = dfi.nlargest(topn_per_n, col_topn)  # col_topn: 'tfidf'
                dfi1 = dfi1.sort_values(col_topn, ascending=False)
                dfx.append(dfi1)
            dfc = pd.concat(dfx, ignore_index=True)
            # dfc.sort_values([length])
            print('   + nrow: %d -> %d (topn_per_n=%d)' % (nrow1, dfc.shape[0], topn_per_n))

        # save? [todo]
        
        # extract data
        adict['cid'].extend([cid] * dfc.shape[0])
        adict['ngram'].extend(dfc['ngram'].values)
        adict['tfidf'].extend(dfc['tfidf'].values) 
        adict['tf_cluster'].extend(dfc['tf_cluster'].values)
        adict['tf_global'].extend(dfc['tf_global'].values)
    
    # save final aggregated data
    fname_out = 'pathway_cluster_derived-%s.csv' % identifier1 # identifier: (tset_type, cluster_method, seq_ptype, d2v_method)
    basedir1 = os.path.join(basedir, 'pathway')  # [output] tpheno/seqmaker/data/<cohort>/pathway/pathway_cluster_derived-* 
    fpath_out = os.path.join(basedir1, fname_out)

    df = DataFrame(adict, columns=header)
    df.to_csv(fpath_out, sep='|', index=False, header=True)
    print('output2> saved derived cluster-level pathway data (dim: %s) to %s' % (str(dfc.shape), fpath_out))

    ### longest common subsequence 
    # pClusterPathway2(ctype=content_type, otype=order_type, ptype=policy_type)

    return

def pClusterPathway1a(**kargs):
    return pClusterPathwayTFIDF(**kargs) 
def pClusterPathway1b(**kargs): 
    return pClusterPathwayLength(**kargs)

def pClusterPathwayLength(**kargs):  # length regulated
    kargs['cluster_freq_max'] = kargs.get('cluster_freq_max', 1)  # select only those n-grams that appear in few clusters (e.g. 1)
    focused_lengths = kargs['focused_lengths'] = [1, 2, 3, 4]  # small lengths
    kargs['topn_per_length'] = kargs.get('topn_per_length', 15)
    kargs['save_'] = True 
    kargs['suffix'] = 'length'
    div(message='Preparing length-regulated cluster n-grams (focused_length=%s) ...' % focused_lengths)
    return rankClusterMotifs(**kargs)
def pClusterPathwayTFIDF(**kargs): 
    return rankClusterMotifs(**kargs)

def rankClusterMotifs(**kargs): 
    """
    use this to select representative n-grams ranked by their tf-idf scores 
    output can be used furhter by pClusterPathwayLCS

    Input
    -----
    foreach cluster 
        tpheno/seqmaker/data/<cohort>/pathway_C* 
            e.g. pathway_C9-GPTSD-COPdiagnosis-total-posterior-nC10-nL1-D2Vtfidfavg.csv

    Output
    ------
        tpheno/seqmaker/data/<cohort>/pathway_cluster_ranked-*
            e.g. pathway_cluster_ranked-length-GPTSD-COPdiagnosis-total-prior.csv

    """
    def count_n_clusters(s, sep=' '): 
        # if s in (None, float('nan')): return 0 
        if isinstance(s, int): 
            return 1 
        else: 
            try: 
                return len(s.split(sep))
            except: 
                pass
        raise ValueError, "Not valid cluser IDs: %s" % s

    cohort_name = kargs.get('cohort', 'PTSD')
    basedir = seqparams.get_basedir(cohort=cohort_name) # [input]

    sep = '|'  # column separator

    # each cluster has its own pathway file (which is derived from cluster motifs & which includes tf-idf scores, etc.)
    n_classes = kargs.get('n_classes', 1)
    n_clusters = kargs.get('n_clusters', 10) 
    cIDs = range(0, n_clusters)   # CID

    # label = kargs.get('label', 1)

    # [params] free parameters
    content_type = ctype = kargs.get('ctype', 'diagnosis') # content type (what's in the sequence?): medication, mixed
    order_type = otype = kargs.get('otype', 'total')
    policy_type = ptype = kargs.get('ptype', 'prior')  # cut point: pre-diagnosis (prior), post-diagnosis (posterior), whole sequence (mixed)
    
    cluster_method = kargs.get('cluster_method', 'kmeans')
    seq_ptype = kargs.get('seq_ptype', seqparams.normalize_ptype(ctype))  # diag, regular, med
    d2v_method = kargs.get('d2v_method', 'tfidfavg')

    identifier = 'G%s-COP%s-%s-%s-nC%s-nL%s-D2V%s' % (cohort_name, ctype, order_type, policy_type, n_clusters, n_classes, d2v_method)
    identifier1 = 'G%s-COP%s-%s-%s' % (cohort_name, ctype, order_type, policy_type)
    ### global level pathway file (sorted by ['length', 'tf_global', ] followed by ['cluster_freq'] in ascending order)
    # e.g. pathway_global-GPTSD-COPdiagnosis-total-prior-nC10-nL1-D2Vtfidfavg.csv

    # first groupby cluster IDs and find their unique (or close to unique) n-grams (ranked by cluster ngrams)
    # [params]
    cluster_freq_max = kargs.get('cluster_freq_max', 1)  # set an upperbound of # of clusters that an n-gram appears
    # topn_per_cluster = 50

    # [params] which motif lengths to focus on (if not specified, then ALL)
    max_length = 10
    focused_lengths = kargs.get('focused_lengths', range(1, max_length+1)) # ALL or just a selected few: [1, 2, 4, 8]

    # [params] select only top-ranked n-grams 
    col_n_clusters = 'cluster_freq'
    col_topn = kargs.get('criteria', 'tfidf') # ranking criteria
    topn_per_length = topn_per_n = kargs.get('topn_per_length', 100) # retain 100 most 'popular' n-grams (from which to compute LCSs)
    apply_group_sort = True

    # [params] output file 
    header = ['cid', 'ngram', 'tfidf', 'tf_ratio', ] 
    adict = {h: [] for h in header}
    save_ = kargs.get('save_', False)

    div(message='Select cluster-level representative ngrams ~ %s' % col_topn)
    print('  + [params] cluster_freq_max: %d, topn_per_n: %d' % (cluster_freq_max, topn_per_n))
    cluster_ngram = {cid: [] for cid in cIDs}
    for cid in cIDs: # foreach cluster-level pathway file (with motifs n = 1 ~ max_length [10])
        print('  + cluster ID=%s ...' % cid)
        # input data header: length|ngram|tfidf|tf_cluster|tf_global|cluster_occurrence
        fname = 'pathway_C%s-%s.csv' % (cid, identifier)  # [I/O] local
        basedir1 = os.path.join(basedir, 'pathway')
        fpath = os.path.join(basedir1, fname)
        dfc = pd.read_csv(fpath, sep=sep, header=0, index_col=False, error_bad_lines=True)
        nrow0 = dfc.shape[0]

        if not col_n_clusters in dfc.columns: 
            dfc[col_n_clusters] = dfc['cluster_occurrence'].apply(count_n_clusters)  # apply count n_cluster function to each row

        # [filter] filter by number of clusters (that contain a given n-gram)
        print('  ++ [filter] by number of clusters that contain a given n-gram (max=%d)' % cluster_freq_max)
        cond_cf = dfc[col_n_clusters] <= cluster_freq_max 
        dfc = dfc.loc[cond_cf]

        # [filter] filter by lengths
        print('  ++ [filter] retain only those with lengths=%s' % focused_lengths)
        cond_length = dfc['length'].isin(focused_lengths)
        dfc = dfc.loc[cond_length]
        nrow1 = dfc.shape[0]
        print('  +++ nrow: %d -> %d | cluster_freq_max=%d, focused_lengths=%s' % (nrow0, nrow1, cluster_freq_max, focused_lengths))

        # [transform]
        dfc['tf_ratio'] = dfc['tf_cluster']/dfc['tf_global'].astype(float)

        # [filter] foreach length, list top 'k' n-grams ~ tfidf
        #          also sort tf-idf (should have been sorted by 'tfidf')
        if apply_group_sort: 
            dfx = []
            for i, (g, dfi) in enumerate(dfc.groupby(['length', ])):  # group by cluster IDs
                # if i == 0: print('     + columns: %s' % dfi.columns.values)  # ['length' 'ngram' 'tf_global' 'cluster_freq' 'cluster_occurrence']
                dfi1 = dfi.nlargest(topn_per_n, col_topn)  # col_topn: 'tfidf', 't_cluster', 'tf_global', 'tf_ratio'
                # dfi1 = dfi1.sort_values(col_topn, ascending=False)
                dfx.append(dfi1)
            dfc = pd.concat(dfx, ignore_index=True)
            # dfc.sort_values([length])
            print('   + nrow: %d -> %d (topn_per_n=%d)' % (nrow1, dfc.shape[0], topn_per_n))

        # save? [todo]
        
        # extract data
        adict['cid'].extend([cid] * dfc.shape[0])
        adict['ngram'].extend(dfc['ngram'].values)
        adict['tfidf'].extend(dfc['tfidf'].values)  # cluster to global term frequence ratio
        adict['tf_ratio'].extend(dfc['tf_ratio'].values)

        cluster_ngram[cid].extend(dfc['ngram'].values)
    ### end foreach cluster (cid)
    
    # save final aggregated data
    if save_: 
        suffix = kargs.get('suffix', 'tfidf')  

        # identifier: (tset_type, cluster_method, seq_ptype, d2v_method)
        fname_out = 'pathway_cluster_ranked-%s.csv' % identifier1 if suffix is None else 'pathway_cluster_ranked-%s-%s.csv' % (suffix, identifier1)
        basedir1 = os.path.join(basedir, 'pathway')
        fpath_out = os.path.join(basedir1, fname_out)

        df = DataFrame(adict, columns=header)
        df.to_csv(fpath_out, sep='|', index=False, header=True)
        print('output> saved derived cluster-level pathway data (dim: %s) to %s' % (str(df.shape), fpath_out))

    return cluster_ngram

def clusterDocuments(**kargs):
    """

    Related 
    -------
    seqmaker.seqCluster.analyze_pathway_batch

    """
    import itertools 
    import seqCluster as sc

    # [params] cohort (e.g. diabetes, PTSD)
    cohort_name = kargs.get('cohort', 'PTSD')

    # [params] I/O
    outputdir = basedir = seqparams.get_basedir(cohort=cohort_name)  # os.path.join(os.getcwd(), cohort_name)

    # [params]
    seq_compo = composition = kargs.get('composition', 'condition_drug') # sequence composition
    read_mode = kargs.get('read_mode', 'doc') # documents/doc (one patient one sequence) or sequences/seq
    tset_type = kargs.get('tset_type', 'binary')
    std_method = kargs.get('std_method', 'minmax') # feature standardizing/scaling/preprocessing method
    n_sample, n_sample_small = 1000, 100

    # [params] cluster 
    n_clusters = kargs.get('n_clusters', 10)
    optimize_k = kargs.get('optimize_k', False)
    range_n_clusters = kargs.get('range_n_clusters', None) # None => to be determined by gap statistics 
    min_n_clusters, max_n_clusters = kargs.get('min_n_clusters', 1), kargs.get('max_n_clusters', 200)

    # [params] classification 
    n_classes = kargs.get('n_classes', 1)

    # [params] pathway 
    min_freq = kargs.get('min_freq', 2) # a dictionary or a number; this is different from 'min_count' used for w2v and d2v models

    d2v_methods = ['tfidfavg', ]
    cluster_methods = ['kmeans', ]

    for d2v_method in d2v_methods: 
        # kargs['d2v_method'] = d2v_method

        for cluster_method in cluster_methods: 
            # kargs['cluster_method'] = cluster_method
            # [params] experiemntal settings for n-gram analysis: otype, ctype, ptype
            
            otypes = ['partial', 'total', ]  # n-gram with ordering considered? 
            ctypes = ['diagnosis', 'medication', 'mixed',]
            ptypes = ['prior', 'noop', 'posterior',]
            for params in itertools.product(otypes, ctypes, ptypes): 
                # cluster_method = params[0]
                order_type, ctype, policy_type = params[0], params[1], params[2]  

                # document cluster
                DCluster = sc.cluster_documents(order_type=order_type, policy_type=policy_type, ctype=ctype, 
                                        n_classes=n_classes,  
                                        d2v_method=d2v_method,
                                        cohort=cohort_name, 
                                            cluster_method=cluster_method,
                                            n_clusters=n_clusters, optimize_k=optimize_k, 
                                                 min_n_clusters=min_n_clusters, max_n_clusters=max_n_clusters, 
                                                 min_freq=min_freq)

                # save results? this is already done within seqCluster.cluster_analysis()
                # seq_ptype_prime = 'regular'
                # f_tdoc = '%s_%s-%s-%s.csv' % (tdoc_prefix, tdoc_stem, cohort_name, seq_ptype_prime)  # e.g. condition_drug_timed_seq-PTSD-regular.csv
                # fpath_tdoc = os.path.join(basedir_tdoc, f_tdoc)
                # ifile = f_tdoc
                # print('input> reading from processed temporal doc file (.csv): %s' % f_tdoc)
                # assert os.path.exists(fpath_tdoc), 'Invalid input path: %s' % fpath_tdoc
                # df_seq = pd.read_csv(fpath_tdoc, sep='|', header=0, index_col=False, error_bad_lines=True)

    

    return 

def transformByType(seq, seq_ptype='regular'): 
    """

    Related
    -------
    seqReader.transform_by_ptype(seq, **kargs)
    """
    from pattern import medcode as pmed
    seq_ptype = seqparams.normalize_ptype(seq_ptype)  

    if seq_ptype == 'regular': 
        return seq # no op

    if seq_ptype == 'random': 
        random.shuffle(seq) # inplace op 
    elif seq_ptype == 'diag': 
        seq = [token for token in seq if pmed.isCondition(token)]
    elif seq_ptype == 'med': 
        seq = [token for token in seq if not pmed.isCondition(token) and pmed.isMed(token)]
    else: 
        # return transform_by_ptype_diabetes(seq, **kargs)
        raise NotImplementedError, "unrecognized seq_ptype: %s" % seq_ptype
    return seq
def transformByType2(seq, tseq, seq_ptype='regular'): 
    """

    Related
    -------
    seqReader.transform_by_ptype(seq, **kargs)
    """
    from pattern import medcode as pmed
    seq_ptype = seqparams.normalize_ptype(seq_ptype)  
    assert len(seq) == len(tseq)

    if seq_ptype == 'regular': 
        return (seq, tseq) # no op

    seqIndex = range(len(seq))
    if seq_ptype == 'random': 
        random.shuffle(seqIndex)
    elif seq_ptype == 'diag': 
        seqIndex = [i for i, token in enumerate(seq) if pmed.isCondition(token)]
    elif seq_ptype == 'med': 
        seq = [i for i, token in enumerate(seq) if not pmed.isCondition(token) and pmed.isMed(token)]
    else: 
        # return transform_by_ptype_diabetes(seq, **kargs)
        raise NotImplementedError, "unrecognized seq_ptype: %s" % seq_ptype
    
    seq2 = [seq[i] for i in seqIndex]  # indices of different order or just a subset
    tseq2 = [tseq[i] for i in seqIndex]

    return (seq2, tseq2)


def loadGlobalMotifs(**kargs): # global motifs obtained from seqmaker.seqCluster 
    cohort_name = kargs.get('cohort', 'PTSD')
    basedir = seqparams.get_basedir(cohort=cohort_name) 
    sep = '|'  # column separator

    # each cluster has its own pathway file (which is derived from cluster motifs & which includes tf-idf scores, etc.)
    n_classes = kargs.get('n_classes', 1)
    n_clusters = kargs.get('n_clusters', 10) 
    cIDs = range(0, n_clusters)   # CID

    # label = kargs.get('label', 1)
    content_type = ctype = kargs.get('ctype', 'diagnosis') # content type (what's in the sequence?): medication, mixed
    order_type = otype = kargs.get('otype', 'total')
    policy_type = ptype = kargs.get('ptype', 'prior')  # cut point: pre-diagnosis (prior), post-diagnosis (posterior), whole sequence (mixed)
    cluster_method = kargs.get('cluster_method', 'kmeans')
    seq_ptype = kargs.get('seq_ptype', seqparams.normalize_ptype(ctype))  # diag, regular, med
    d2v_method = kargs.get('d2v_method', 'tfidfavg')

    identifier = 'G%s-COP%s-%s-%s-nC%s-nL%s-D2V%s' % (cohort_name, ctype, order_type, policy_type, n_clusters, n_classes, d2v_method)

    ### global level pathway file (sorted by ['length', 'tf_global', ] followed by ['cluster_freq'] in ascending order)
    # e.g. pathway_global-GPTSD-COPdiagnosis-total-prior-nC10-nL1-D2Vtfidfavg.csv

    # first groupby cluster IDs and find their unique (or close to unique) n-grams (ranked by cluster ngrams)

    # [input]
    fname_global = 'pathway_global-%s.csv' % identifier # identifier: (tset_type, cluster_method, seq_ptype, d2v_method)
    
    # path after organing the files according to their properties
    basedir1 = os.path.join(basedir, 'pathway') 
    fpath = os.path.join(basedir1, fname_global)
    assert os.path.exists(fpath), 'invalid path? %s' % fpath 

    print('io> reading global pathway file (tfidf scores, etc.): %s' % fname_global)
    df = pd.read_csv(fpath, sep=sep, header=0, index_col=False, error_bad_lines=True)

    return df

def loadClusterMotifs(**kargs):  
    """

    Use
    ---
    evalClusterDiff


    Memo
    ---- 
    1 raw cluster motifs vs processed? 

        pathway_C9-GPTSD-COPdiagnosis-total-prior-nC10-nL1-D2Vtfidfavg.csv 
           n = 11836
           
           - filtered by min_freq_local = 3


        cluster_motifs-CIDL9-1-COPdiagnosis-total-prior-Ckmeans-Sdiag-D2Vtfidfavg.csv
           n = 413359

    """ 
    cohort_name = kargs.get('cohort', 'PTSD')
    basedir = seqparams.get_basedir(cohort=cohort_name) # [input]
    sep = '|'  # column separator

    # each cluster has its own pathway file (which is derived from cluster motifs & which includes tf-idf scores, etc.)
    n_classes = kargs.get('n_classes', 1)
    n_clusters = kargs.get('n_clusters', 10) 

    targetCID = kargs.get('cID', None)
    if targetCID is None: 
        cIDs = range(0, n_clusters)   # CID
    else: 
        if hasattr(targetCID, '__iter__'): 
            cIDs = targetCID
        else: 
            cIDs = [targetCID]

    print('info> target cluster IDs: %s (nC=%d)' % (cIDs, len(cIDs)))

    # [params] free parameters
    content_type = ctype = kargs.get('ctype', 'diagnosis') # content type (what's in the sequence?): medication, mixed
    order_type = otype = kargs.get('otype', 'total')
    policy_type = ptype = kargs.get('ptype', 'prior')  # cut point: pre-diagnosis (prior), post-diagnosis (posterior), whole sequence (mixed)
    
    cluster_method = kargs.get('cluster_method', 'kmeans')
    seq_ptype = kargs.get('seq_ptype', seqparams.normalize_ptype(ctype))  # diag, regular, med
    d2v_method = kargs.get('d2v_method', 'tfidfavg')

    # first groupby cluster IDs and find their unique (or close to unique) n-grams (ranked by cluster ngrams)
    # [params]
    cluster_freq_max = kargs.get('cluster_freq_max', 1)  # set an upperbound of # of clusters that an n-gram appears

    loadClusterPathway = kargs.get('load_selected_motifs', True)

    # return the dataframe itself or an generator of dataframe? 
    # [note] cannot mix a return statement in a generator
    # if targetCID is not None and len(cIDs) == 1: pass 

    for cid in cIDs: # foreach cluster-level pathway file (with motifs n = 1 ~ max_length [10])
        print('  + cluster ID=%s ...' % cid)
        
        if loadClusterPathway: # i.e. processed cluster motifs             
            # input data header: length|ngram|tfidf|tf_cluster|tf_global|cluster_occurrence
            identifier = 'G%s-COP%s-%s-%s-nC%s-nL%s-D2V%s' % (cohort_name, content_type, order_type, policy_type, n_clusters, n_classes, d2v_method)
            fname = 'pathway_C%s-%s.csv' % (cid, identifier)  # [I/O] local
            basedir1 = os.path.join(basedir, 'pathway')
            fpath = os.path.join(basedir1, fname)
            if not os.path.exists(fpath): 
                fpath = os.path.join(basedir, fname)
                assert os.path.exists(fpath), "could not find %s" % fname

            dfc = pd.read_csv(fpath, sep=sep, header=0, index_col=False, error_bad_lines=True)
            # nrow0 = dfc.shape[0]

        else: # e.g. cluster_motifs-CIDL9-1-COPdiagnosis-total-prior-Ckmeans-Sdiag-D2Vtfidfavg.csv
            label = kargs.get('label', 1)
            identifier2 = 'CIDL%s-%s-COP%s-%s-%s-C%s-S%s-D2V%s' % (cid, label, content_type, order_type, policy_type, cluster_method, seq_ptype, d2v_method) 
            fname = 'cluster_motifs-%s.csv' % identifier2
            basedir2 = os.path.join(basedir, 'cluster_motifs')
            fpath = os.path.join(basedir2, fname)
            if not os.path.exists(fpath): 
                fpath = os.path.join(basedir, fname)
                assert os.path.exists(fpath), "could not find %s" % fname

            dfc = pd.read_csv(fpath, sep=sep, header=0, index_col=False, error_bad_lines=True)
            nrow0 = dfc.shape[0]

        yield dfc 

def loadClusters(**kargs): 

    # [params] cohort (e.g. diabetes, PTSD)
    cohort_name = kargs.get('cohort', 'PTSD')

    # [params] I/O
    basedir = seqparams.get_basedir(cohort=cohort_name)  # os.path.join(os.getcwd(), cohort_name)

    # [params]
    # n_classes = kargs.get('n_classes', 1)
    # n_clusters = kargs.get('n_clusters', 10) 
    # cIDs = range(0, n_clusters)   # CID

    # load clusters
    # [input] e.g. cluster_ids-Ckmeans-Pdiag-D2Vtfidfavg-GPTSD.csv
    ctype = content_type = kargs.get('ctype', 'mixed')  
    seq_ptype = seqparams.normalize_ptype(ctype)  # 'regular', 'mixed' all mapped to regular
    
    d2v_method = kargs.get('d2v_method', 'tfidfavg')
    cluster_method = kargs.get('cluster_method', 'kmeans')
    identifier = 'C%s-P%s-D2V%s-G%s' % (cluster_method, seq_ptype, d2v_method, cohort_name)
    print('loadClusters> cohort: %s, ctype: %s -> %s, d2v: %s, cluster method: %s' % \
        (cohort_name, ctype, seq_ptype, d2v_method, cluster_method))

    # [params]
    header = ['id', 'cluster_id', ]
    fname = 'cluster_ids-%s.csv' % identifier
    fpath = os.path.join(basedir, fname)
    assert os.path.exists(fpath), "Invalid cluster file path: %s" % fpath
    df_cluster = pd.read_csv(fpath, sep='|', header=0, index_col=False, error_bad_lines=True)
    print('io> load precomputed cluster assignments from %s' % fpath)

    return df_cluster 
def loadDocuments(**kargs): 
    cohort_name = kargs.get('cohort', 'PTSD')
    ctype = content_type = kargs.get('ctype', 'mixed')  
    seq_ptype = seqparams.normalize_ptype(ctype)  # 'regular', 'mixed' all mapped to regular

    include_timestamps = kargs.get('include_timestamps', True)
    
    # [params] misc 
    verify_seq = True 

    timestamps = []

    print('loadDocuments> cohort: %s, time? %s, ctype: %s->%s' % (cohort_name, include_timestamps, ctype, seq_ptype))
    
    # [todo] this reads from the source that contains all code types ... 08.17
    ret = sr.readTimedDocPerPatient(load_=False, simplify_code=False, 
                                verify_=verify_seq, include_timestamps=include_timestamps, 
                                seq_ptype=seq_ptype, cohort=cohort_name) 
    # res = sr.readToCSVDoc(load_=False, simplify_code=False, 
    #                             verify_=verify_seq, include_timestamps=include_timestamps, 
    #                             seq_ptype=seq_ptype, cohort=cohort_name)
    assert not include_timestamps or len(ret) == 2
    if include_timestamps: 
        documents, timestamps = ret 
    else: 
        documents = ret
    
    return (documents, timestamps)

def selectMembersByDocLength(cluster_ngram=None, **kargs):
    """
    Input
    -----
    cluster_ngram: a map from cluster ID to its internal represntations (e.g. ngrams) that serve as the basis 
                   for selecting cluster members. 

    Output
    ------
    filtered cluster ngrams 
        number of n-grams for each cluster has an upperbound

    """
    import math
    # import algorithms, seqAlgo
    # import seqAnalyzer as sa
    # import seqReader as sr
    # from pattern import medcode as pmed 
    
    tFilterDoc = True
    cohort_name = kargs.get('cohort', 'PTSD')

    # [params]
    n_classes = kargs.get('n_classes', 1)
    n_clusters = kargs.get('n_clusters', 10) 
    cIDs = range(0, n_clusters)   # CID

    documents, timestamps = loadDocuments(**kargs)  # [params] cohort, ctype ('mixed'), include_timestamps (True)
    if len(timestamps) > 0: assert len(documents) == len(timestamps)

    # retain only necessary information in patient doc
    nD = len(documents)
    print('info> n_docs=%d | cohort=%s, filter doc? %s' % (nD, cohort_name, tFilterDoc))
    
    # build cluster map
    DCluster = {cid: [] for cid in cIDs}
    df_cluster = loadClusters(**kargs) # [params] cohort, ctype, d2v_method, cluster_method

    cmap = zip(df_cluster['id'].values, df_cluster['cluster_id'].values)
    for i, cid in cmap:
        DCluster[cid].append(i)  # cid -> document (positional) IDs

    # [verify]
    idx = DCluster[random.sample(cIDs, 1)[0]] # foreach cluster
    assert isinstance(documents[idx[0]], list), "documents have not been converted to lists of tokens: %s" % documents[idx[0]]

    ### ensure that all documents are 'tokenized' 
    scores = {}  
    nDMax = max(1000, kargs.get('n_docs', 3000))
    nDMaxPerCluster = int(math.ceil(nDMax/n_clusters)) # max number of docs allowed per cluster
    nMatchMin, rMatchMin = 5, 0.5  # absolute minimum vs min ratio (of matched n-grams within queried patient doc)
    if nDMax is None or (nD < nDMaxPerCluster): tFilterDoc = False  # no need to filter

    cluster_seqx = {}
    nT = 0
    if tFilterDoc: 
        for cid, idx in DCluster.items(): 
            nc = len(idx)
            if nc < nDMaxPerCluster: 
                cluster_seqx[cid] = [documents[i] for i in idx]
                nT += nc
            else: 
                # ordered from longest to shortest
                rankedlist = sorted([(i, len(documents[i])) for i in idx], key=lambda x:x[1], reverse=True)[:nDMaxPerCluster]
                cluster_seqx[cid] = [documents[i] for i, _ in rankedlist]
                nT += len(cluster_seqx[cid])

    else: 
        print('status> Consider all documents for each cluster (n_clusters=%d)' % n_clusters)
        for cid, idx in DCluster.items(): # foreach cluster
            cluster_seqx[cid] = [documents[i] for i in idx]
            nT += len(idx)
    print('verify> number of cluster members in total (where n_clusters=%d): %d' % (n_clusters, nT))
    return cluster_seqx

def selectClusterMembers(cluster_ngram=None, **kargs): 
    # criterion 
    #    ngrams
    #    lengths 
    policy = kargs.get('policy', 'length')
    if policy.startswith('ng'): # ngram
        return selectMembers(cluster_ngram, **kargs)  

    return selectMembersByDocLength(cluster_ngram, **kargs)

def selectMembers(cluster_ngram=None, **kargs): 
    """
    Select cluster members by matching with popular cluster n-grams (e.g. ranked by tf-idf) as a filter, where
    the cluster members are the patient temporal documents (e.g. pre-diagnositc, post-diagnostic or full document). 

    Output
    ------
    cluster ID -> documents that satifify the selection criteria (i.e. those that match at least some 
                'popular' n-grams, defined via evalMatch, evalMatchRatio, etc.)

    Input
    -----
    cluster_ngram: cid -> ngrams
    
    cluster-to-ngram mapping derived from rankClusterMotifs() or other rank-related functions (rank*). 
    e.g. pathway_cluster_ranked-GPTSD-COPdiagnosis-total-prior.csv (params: diagnosis, total, prior)

    Use
    ---
    1. Select the represenative patients for deriving LCSs.  

    """
    def tokenize(ngr, sep=' '):
        return ngr.split(sep)
    def evalMatch(ngrams, seq):
        n_matched = 0 
        for ngram in ngrams: 
            if seqAlgo.isSubsequence(ngram, seq): # is this n-gram pattern a subsequence of the patient doc (seq)? 
                n_matched += 1
        return n_matched
    def evalMatchRatio(ngrams, seq):
        N = len(ngrams)
        n_matched = 0 
        for ngram in ngrams: 
            if seqAlgo.isSubsequence(ngram, seq): # is this n-gram pattern a subsequence of the patient doc (seq)? 
                n_matched += 1
        
        return n_matched/(N+0.0)

    import algorithms, seqAlgo
    import seqAnalyzer as sa
    import seqReader as sr
    from pattern import medcode as pmed

    tFilterDoc = False

    # [params] cohort (e.g. diabetes, PTSD)
    cohort_name = kargs.get('cohort', 'PTSD')

    # [params] I/O
    outputdir = basedir = seqparams.get_basedir(cohort=cohort_name)  # os.path.join(os.getcwd(), cohort_name)
    documents = kargs.get('documents', None)
    timestamps = [] 
    include_timestamps = True

    # [params]
    n_classes = kargs.get('n_classes', 1)
    n_clusters = kargs.get('n_clusters', 10) 
    cIDs = range(0, n_clusters)   # CID

    # load clusters
    # [input] e.g. cluster_ids-Ckmeans-Pdiag-D2Vtfidfavg-GPTSD.csv
    ctype = content_type = kargs.get('ctype', 'mixed')  
    seq_ptype = seqparams.normalize_ptype(ctype)  # 'regular', 'mixed' all mapped to regular
    d2v_method = kargs.get('d2v_method', 'tfidfavg')
    cluster_method = 'kmeans'
    identifier = 'C%s-P%s-D2V%s-G%s' % (cluster_method, seq_ptype, d2v_method, cohort_name)

    # [params]
    header = ['id', 'cluster_id', ]
    fname = 'cluster_ids-%s.csv' % identifier
    fpath = os.path.join(basedir, fname)
    assert os.path.exists(fpath), "Invalid cluster file path: %s" % fpath
    df_cluster = pd.read_csv(fpath, sep='|', header=0, index_col=False, error_bad_lines=True)
        
    # [input]
    div('Select members; each member = document (ctype=%s)')
    if documents is None: 
        # [todo] this reads from the source that contains all code types ... 08.17
        res = sr.readToCSVDoc(load_=False, simplify_code=False, 
                                verify_=True, include_timestamps=include_timestamps, 
                                seq_ptype=seq_ptype, cohort=cohort_name)
        assert not include_timestamps or len(res) == 2
        if include_timestamps: 
            documents, timestamps = res 
        else: 
            documents = res

    # retain only necessary information in patient doc
    nD = len(documents)
    print('info> n_docs=%d | cohort=%s' % (nD, cohort_name))
    
    # build cluster map
    DCluster = {cid: [] for cid in cIDs}
    cmap = zip(df_cluster['id'].values, df_cluster['cluster_id'].values)
    for i, cid in cmap:
        DCluster[cid].append(i)

    # [verify]
    idx = DCluster[random.sample(cIDs, 1)[0]] # foreach cluster
    assert isinstance(documents[idx[0]], list), "documents have not been converted to lists of tokens: %s" % documents[idx[0]]
    
    scores = {}  
    nDMax = 5000 
    nDMaxPerCluster = int(math.ceil(nDMax/n_clusters)) # max number of docs allowed per cluster
    nMatchMin, rMatchMin = 5, 0.5  # absolute minimum vs min ratio (of matched n-grams within queried patient doc)
    if nD < nDMaxPerCluster: tFilterDoc = False  # no need to filter

    sep_input = ' '
    cluster_seqx = {}  # maps cluster ID to patient document (instead of top-ranked n-grams)
    if tFilterDoc: 

        # input cluster_ngram[cid] tokenized (i.e. turned into a list of codes)? 
        is_tokenized = True
        ngrams = cluster_ngram[random.sample(cIDs, 1)[0]]
        if isinstance(ngrams[0], str): # still in string format
            for cid, ngrams in cluster_ngram.items(): 
                cluster_ngram[cid] = [tokenize(ngram, sep=sep_input) for ngram in ngrams]  
        else: 
            # sid = random.sample(range(len(ngrams)), 1)[0]
            assert isinstance(ngrams[0], list), "invalid ngram format: %s" % ngrams[0]

        print('verify> example n-gram patterns:\n%s\n' % cluster_ngram[random.sample(cIDs, 1)[0]][:5]) 

        # [filter]
        n_selected = 0
        for cid, idx in DCluster.items(): # foreach cluster
            n = len(idx)
            if n < nDMaxPerCluster: 
                cluster_seqx[cid] = [documents[i] for i in idx]
                n_selected += n 
            else: 
                for i in idx: # foreach patient doc 
                    seq = documents[i]  # a list of tokens
                    assert isinstance(seq, list)
                
                    # [policy] how many n-grams match? consider a match if say more than 50% is matched
                    r_match = evalMatchRatio(cluster_ngram[cid], seq) # match ngrams with seq and see how many matched
                    if r_match >= rMatchMin: 
                        cluster_seqx[cid].append(seq)
                        n_selected += 1 
                         
            print('progress> Selected %d documents/seqx for cluster ID=%s' % (len(cluster_seqx[cid]), cid))
        print('status> Selected a total of %d documents (n_clusters=%d)' % (n_selected, n_clusters))
            
        # [policy] rank members by number of match 
        # cluster_ngram[cid]     # the list of 'important' n-grams
    else: 
        print('status> Consider all documents for each cluster (n_clusters=%d)' % n_clusters)
        for cid, idx in DCluster.items(): # foreach cluster
            cluster_seqx[cid] = [documents[i] for i in idx]

    return cluster_seqx


def pClusterPathway2(**kargs): # d: derived
    return pClusterPathwayLCS(**kargs) 
def pClusterPathwayLCS(**kargs):  
    """
    Organize derived motfis files obtained from pClusterPathway.  

    Related 
    -------
    pClusterPathway

    Memo
    ----
    1. LCS consists of largely just duplicates (e.g. consecutive diagnoses of the same code)
       use itertools.groupy to remove duplicates 

    Input
    -----
    1a filtered cluster ngrams after applying the following operations
        pClusterPathway
        selectMembersByDocLength

    1b from file 
        tpheno/seqmaker/data/<cohort>/pathway/pathway_cluster_ranked-*


    Output
    ------


    Reference
    ---------
    1. itertools.groupy
       http://menno.io/posts/itertools_groupby/

    Old Memo
    --------
    1. The LCSs derived from pairwise comparisons within cluster-level n-grams do no seem to match 
    with the real patient documents ... fixed

    

    """
    def tokenize(ngr, sep=' '):
        return ngr.split(sep)

    def filterLength(lcs_counts, min_length=1, max_length=np.inf, sep=' '): # lcsMinLength, lcsFMap
        if min_length is not None: # screen shorter LCS (not as useful for observing disease progression)
            # ngr_cnts = [(s, cnt) for s, cnt in lcsFMap.items()]
            ngr_cnts = []
            for s in lcs_counts.keys(): 
                ntok = len(s.split(sep))  # number of tokens
                if ntok >= min_length and ntok <= max_length: 
                    ngr_cnts.append((s, lcs_counts[s]))
        else: 
            ngr_cnts = [(s, cnt) for s, cnt in lcs_counts.items()]
        
        return ngr_cnts

    import motif as mf
    import algorithms

    # [params]
    cohort_name = kargs.get('cohort', 'PTSD')
    basedir = seqparams.getCohortDir(cohort) # seqparams.get_basedir(cohort=cohort_name) 
    sep = '|'  # column separator

    # each cluster has its own pathway file (which is derived from cluster motifs & which includes tf-idf scores, etc.)
    n_classes = kargs.get('n_classes', 1)
    n_clusters = kargs.get('n_clusters', 10) 
    cIDs = range(0, n_clusters)   # CID

    # label = kargs.get('label', 1)
    content_type = ctype = kargs.get('ctype', 'diagnosis') # content type (what's in the sequence?): medication, mixed
    order_type = otype = kargs.get('otype', 'total')
    policy_type = ptype = kargs.get('ptype', 'prior')  # cut point: pre-diagnosis (prior), post-diagnosis (posterior), whole sequence (mixed)
    # cluster_method = kargs.get('cluster_method', 'kmeans')
    seq_ptype = kargs.get('seq_ptype', seqparams.normalize_ptype(ctype))  # diag, regular, med
    tFilterByContentType = False if seq_ptype is 'regular' else True
    # d2v_method = kargs.get('d2v_method', 'tfidfavg')

    # identifier = 'G%s-COP%s-%s-%s-nC%s-nL%s-D2V%s' % (cohort_name, ctype, order_type, policy_type, n_clusters, n_classes, d2v_method)

    # [params][policy] LCS 
    # options: frequency/local (f), global frequency (g), length (l), diversity (d)
    lcsSelectionPolicy = seqparams.normalize_lcs_selection(kargs.get('lcs_selection_policy', 'freq'))
    # lcsMinLength = 5

    cluster_ngram = kargs.get('candidate_motifs', {})  # cluster ID -> ngrams
    identifier1 = 'G%s-COP%s-%s-%s' % (cohort_name, ctype, order_type, policy_type)  # smaller file ID

    if cluster_ngram is None:  # take input from pClusterPathway() by default
        cluster_ngram = {cid: [] for cid in cIDs}

        div(message='Loading cluster-level target motifs from files ...')
        # [params]
        # focused_lengths = [1, 2, 4, 8]
        # cluster_freq_max = 1
        # topn_per_cluster = 20
        # topn_per_length = topn_per_n = 10
        header = ['cid', 'ngram', 'tfidf', ]

        # [params][todo] add parameters? cluster_freq_max, topn_per_cluster
        # identifier1 = 'G%s-COP%s-%s-%s' % (cohort_name, ctype, order_type, policy_type)
        # [input] output of rankClusterMotifs
        fname_in = 'pathway_cluster_ranked-%s.csv' % identifier1 # identifier: (tset_type, cluster_method, seq_ptype, d2v_method)
        basedir1 = os.path.join(basedir, 'pathway')  # derived base directory
        fpath_in = os.path.join(basedir1, fname_in)

        df = pd.read_csv(fpath_in, sep='|', header=0, index_col=False, error_bad_lines=True)
        nrow0 = df.shape[0]

        # consolidate n-grams
        gcol = ['cid']
        for i, (g, dfc) in enumerate(df.groupby(gcol)):  # foreach cluster
            assert g in cIDs
            cluster_ngram[g].extend(dfc['ngram'].values) 

    else: 
        print('info> Take cluster_ngram (cid -> ngrams) from input ...') 

    print('verify> number of entries (cluster_ngram): %d, number of total ngrams or documents: %d' % \
        (len(cluster_ngram), sum(len(ngrx) for _, ngrx in cluster_ngram.items()) ))
    assert len(cluster_ngram) > 0, 'No cluster ngram mapping available to further analyze.'
    
    ### reprsentative n-grams
    #      > find 1. tokenize + most frequent unigram
    #             2. longest common subsequences within each cluster
    div(message="1. Derive longest common subsequences (LCSs) (ctype:%s, otype:%s, ptype: %s, LCS selection: %s)" % \
        (ctype, otype, ptype, lcsSelectionPolicy))
    topn_lcs = kargs.get('topn_lcs', 20)   # but predominent will be shorter ones
    header = ['cid', 'lcs', 'length', 'count', 'diversity']  # frequency? no need because these lcs are selected from 'prevalent' n-grams
    adict = {h: [] for h in header}
    cluster_lcs = {cid: [] for cid in cIDs}
    sep = ' '  # [note] that sep pathway_cluster_derived* and the source (e.g. condition_drug_timed_seq-PTSD-regular.csv) may not be the same

    # peek data type 
    ngrams = cluster_ngram[random.sample(cIDs, 1)[0]]
    if isinstance(random.sample(ngrams, 1)[0], str): 
        print('test> find the ngram data type and its delimiter ...')
        # infer seperator? 
        sep_candidates = [',', ' ']
        found_sep = False
        the_sep = ' '
        for cid, ngrams in cluster_ngram.items(): 
            assert len(ngrams) > 0
            longest_ngram = sorted([(i, len(ngram)) for i, ngram in enumerate(ngrams)], key=lambda x:x[1], reverse=True)[0][1]

            # try splitting it
            for tok in sep_candidates: 
                # select hte longest one 
                if longest_ngram.find(tok) > 0:  # e.g. '300.01 317 311' and tok = ' '
                    if len(longest_ngram.split(tok)) > 1: 
                        the_sep = tok
                        found_sep = True
                        break 
            if found_sep: break 
        print("status> determined cluster_ngram seperator to be '%s' =?= '%s' (default) ... " % (the_sep, sep))
        sep = the_sep

        # now tokenize
        for cid, ngrams in cluster_ngram.items(): 
            cluster_ngram[cid] = [tokenize(ngram, sep=the_sep) for ngram in ngrams]

    else: 
        assert isinstance(random.sample(ngrams, 1)[0], list)

    # [filter]
    div(message='1a. Filtering sequence contents according to sequence type:%s (content type:%s)' % (seq_ptype, ctype))
    if tFilterByContentType: 
        for cid, ngrams in cluster_ngram.items(): 
            ngrams2 = []
            for ngram in ngrams: 
                ngrams2.append(transformByType(ngram, seq_ptype=seq_ptype))
            cluster_ngram[cid] = ngrams2

    # [params]
    maxNPairs = kargs.get('max_n_pairs', 100000) 
    removeDups = kargs.get('remove_duplicates', True) # remove duplicate codes with consecutive occurrences (and preserve only the first one)
    lcsSep = sep

    # only see LCS of length >= 5; set to None to prevent from filtering
    lcsMinLength, lcsMaxLength = kargs.get('min_length', 2), kargs.get('max_length', np.inf) 
    lcsMinCount = kargs.get('min_count', 10)  # min local frequency (used for 'diveristy' policy)
    # lcsSelectionPolicy  # options: frequency (f), global frequency (g), longest (l)

    lcsFMapGlobal = {}  # LCS -> count
    lcsPersonMap = {}   # LCS -> person ids  # can be very large
    lcsMap = {} # cid -> canidate LCS
    
    for cid, ngrams in cluster_ngram.items(): 
        n = len(ngrams)
        
        lcsx = set()  # x: set
        lcsFMap = {}

        # pairwise lcs   
        n_total_pairs = (n*(n-1))/2.  # approx.
        pairs = [(i, j) for i in range(n) for j in range(i+1, n)]  # likely data source: condition_drug_timed_seq-PTSD-regular.csv
        if maxNPairs is not None: 
            npt = len(pairs)
            pairs = random.sample(pairs, min(maxNPairs, npt))
            n_total_pairs = len(pairs)
            print('  + [cluster %d] choose %d out of %d ngram combos' % (cid, n_total_pairs, npt))

        n_pairs = 0
        for pair in pairs:  # likely data source: condition_drug_timed_seq-PTSD-regular.csv
            i, j = pair

            ngr1 = ngrams[i]
            ngr2 = ngrams[j]
            
            if len(ngr1) == 0 or len(ngr2) == 0: continue 
            assert isinstance(ngr1, list) and isinstance(ngr2, list), "ngr1=%s, ngr2=%s" % (ngr1, ngr2)

            sl = mf.lcs(ngr1, ngr2)  # sl is a list of codes since ngr1: list, ngr2: list
            if removeDups: 
                sl = [e[0] for e in itertools.groupby(sl)]
            # if i >= n-2 and j == n-1: print "   + lcs: %s" % sl

            # convert list of codes to a string for indexing
            s = sep.join(sl)  

            # count local frequencies
            if s: # don't add emtpy strings
                if (not s in lcsx): lcsx.add(s) 

                if not s in lcsFMap: lcsFMap[s] = 0
                lcsFMap[s] += 1 

                # count global frequencies
                if not s in lcsFMapGlobal: lcsFMapGlobal[s] = 0
                lcsFMapGlobal[s] += 1

                # if not s in lcsPersonMap: lcsPersonMap[s] = set()
                # lcsPersonMap[s].update([i, j])

            n_pairs += 1 
            r = n_pairs/(n_total_pairs+0.0)
            # r_percent = int(math.ceil(r*100))
            # percentages = {interval: 0 for interval in range(0, 100, 10)}
            if n_pairs % 500 == 0: 
                print('  + [cluster %s] finished computing (%d out of %d ~ %f%%) pairwise LCS ...' % \
                    (cid, n_pairs, n_total_pairs, r*100))
        
        lcsMap[cid] = lcsx
        print('verify> lcsx: %s' % list(lcsx)[-3:])  # a list of strings (of ' '-separated codes)
        
        # find topn longest LCS
        ranked_ngrams, ranked_ngrams_lengths, ranked_ngrams_counts, ranked_ngrams_nutoks = [], [], [], []
        # ranked_ngrams_nutoks: diversity i.e. number of unique tokens within the LCS

        # [filter]
        if lcsSelectionPolicy.startswith('len'): # longest
            # header = ['cid', 'lcs', 'length', 'count']  # or ['cid', 'lcs', 'length', ]
            
            # [filter] length
            if lcsMinLength is not None: # screen shorter LCS (not as useful for observing disease progression)
                ngram_lengths = []
                for s in lcsFMap.keys(): 
                    ntok = len(s.split(sep))
                    if ntok >= lcsMinLength and ntok <= lcsMaxLength: 
                        ngram_lengths.append((s, ntok))
            else: 
                ngram_lengths = [(s, len(s.split(sep))) for s in lcsFMap.keys()]
            
            assert len(ngr_cnts) > 0, "Could not find (in cluster %d) any LCS of length > lcsMinLength: %d" % (cid, lcsMinLength)
            ngrams_length_sorted = sorted(ngram_lengths, key=lambda x: x[1], reverse=True)[:topn_lcs] # descending order in length
            
            print('  + LCS sorted by length => %s' % (ngrams_length_sorted[:5]))

            # dataframe only keeps nrow = len(ranked_ngrams)
            # ranked_ngrams = [ngram for ngram, _ in ngrams_length_sorted][:topn_lcs]
            for s, ntok in ngrams_length_sorted: 
                ss = s.split(sep)
                ranked_ngrams.append(s)
                ranked_ngrams_lengths.append(ntok)
                ranked_ngrams_counts.append(lcsFMap[s])
                ranked_ngrams_nutoks.append(len(set(ss)))
            
        elif lcsSelectionPolicy.startswith( ('f', 'loc') ): # local frequency 
            # header = ['cid', 'lcs', 'length', 'count']

            # [filter] length
            if lcsMinLength is not None: # screen shorter LCS (not as useful for observing disease progression)
                # ngr_cnts = [(s, cnt) for s, cnt in lcsFMap.items()]
                ngr_cnts = []
                for s in lcsFMap.keys(): 
                    ntok = len(s.split(sep))  # number of tokens
                    if ntok >= lcsMinLength and ntok <= lcsMaxLength: 
                        ngr_cnts.append((s, lcsFMap[s]))
            else: 
                ngr_cnts = [(s, cnt) for s, cnt in lcsFMap.items()]

            assert len(ngr_cnts) > 0, "Could not find (in cluster %d) any LCS of length > lcsMinLength: %d" % (cid, lcsMinLength)

            ngr_cnts = sorted(ngr_cnts, key=lambda x: x[1], reverse=True)[:topn_lcs]
            # ranked_ngrams = [ngr for ngr, _ in ngr_cnts][:topn_lcs]
            
            for i, (ngr, cnt) in enumerate(ngr_cnts): 
                ss = ngr.split(sep)
                ranked_ngrams.append(ngr)
                ranked_ngrams_lengths.append( len(ss) )
                ranked_ngrams_counts.append(cnt)
                ranked_ngrams_nutoks.append( len(set(ss)) )  # diversity: number of unique tokens within the LCS

        elif lcsSelectionPolicy.startswith('div'): # diversity: as many different codes as possible
            # header = ['cid', 'lcs', 'length', 'count']

            # ensure that the LCS are not just some outliers
            # [filter] local count (lcsMinCount = 10)
            if lcsMinCount is not None: # screen shorter LCS (not as useful for observing disease progression)
                # ngr_cnts = [(s, cnt) for s, cnt in lcsFMap.items()]
                ngr_scores = []
                for s, cnt in lcsFMap.items(): 
                    if cnt >= lcsMinCount: 
                        ss = s.split(sep)
                        
                        # relative diversity score or absolute? 
                        # ntok = len(ss)  # number of tokens
                        n_utok = len(set(ss))
                        divScore = n_utok               # absolute score: number of unique tokens
                        # divRScore = ntok/(n_utok+0.0) # relative score: may end up getting a lot of unigrams
                        
                        ngr_scores.append((s, divScore))  # (s, divRScore)
            else: 
                ngr_scores = [(s, len(set(s.split(sep)))) for s, cnt in lcsFMap.items()]
                # ngr_scores = []
                # for s, cnt in lcsFMap.items(): 
                #     ss = s.split(sep)
                #     n_utok = len(set(ss))
                #     divScore = n_utok # absolute
                #     # divRScore = len(ss)/(n_utok+0.0) 
                #     ngr_scores.append((s, divScore))
            assert len(ngr_cnts) > 0, "Could not find (in cluster %d) any LCS with min. local count: %d" % (cid, lcsMinCount)
            
            ngr_scores = sorted(ngr_cnts, key=lambda x: x[1], reverse=True)[:topn_lcs]
            for i, (ngr, score) in enumerate(ngr_scores):
                ss = ngr.split(sep)
                ranked_ngrams.append(ngr)
                ranked_ngrams_lengths.append(len(ss))
                ranked_ngrams_counts.append(cnt)
                ranked_ngrams_nutoks.append( len(set(ss)) )  # diversity: number of unique tokens within the LCS

        elif lcsSelectionPolicy.startswith('g'): # global count
            pass  # noop just yet
    
        if ranked_ngrams is not None: 
            n2 = len(ranked_ngrams)
            # find top lcs
            adict['cid'].extend([cid] * n2)
            adict['lcs'].extend(ranked_ngrams)

            assert ranked_ngrams_lengths is not None
            adict['length'].extend(ranked_ngrams_lengths)

            if ranked_ngrams_counts: 
                adict['count'].extend(ranked_ngrams_counts)
            else: 
                adict['count'].extend([ lcsFMap[s] for s in ranked_ngrams])

            if ranked_ngrams_nutoks: 
                adict['diversity'].extend(ranked_ngrams_nutoks)
            else: 
                adict['diversity'].extend([ len(set(s.split(sep))) for s in ranked_ngrams])

            # each ngram in ranked_ngrams is a "string"
            cluster_lcs[cid] = ranked_ngrams # for computing n-gram statistics 
    
    ### end foreach cluster ngram set 

    if len(adict['cid']) == 0: 
        assert lcsSelectionPolicy.startswith('g') and (len(lcsFMapGlobal) > 0 and len(lcsMap) > 0)
        print("policy> select LCSs that are 'globally popular' for each cluster (n_total=%d)" % len(lcsFMapGlobal))
        
        # header = ['cid', 'lcs', 'length', 'count', 'diversity']
        for cid, candidates in lcsMap.items():  # cid -> a set of candidate LCSs 
            # ngr_cnts = sorted([(candidate, lcsFMapGlobal[candidate]) for candidate in candidates], key=lambda x: x[1], reverse=True)
            
            if lcsMinLength is not None: # screen shorter LCS (not as useful for observing disease progression)
                ngr_cnts = []
                for s in candidates: 
                    ntok = len(s.split(sep)) 
                    if ntok >= lcsMinLength and ntok <= lcsMaxLength: 
                        ngr_cnts.append((s, lcsFMapGlobal[s]))

                assert len(ngr_cnts) > 0, "Could not find (in global scope) any LCS of length > lcsMinLength: %d" % lcsMinLength
                ngr_cnts = sorted(ngr_cnts, key=lambda x: x[1], reverse=True)
            else: 
                ngr_cnts = sorted([(candidate, lcsFMapGlobal[candidate]) for candidate in candidates], key=lambda x: x[1], reverse=True)

            ranked_ngrams = [ngr for ngr, _ in ngr_cnts][:topn_lcs]
            ranked_ngrams_counts = [cnt for _, cnt in ngr_cnts][:topn_lcs]
            ranked_ngrams_lengths = [len(ngr.split(sep)) for ngr in ranked_ngrams]
            ranked_ngrams_nutoks = [len( set(ngr.split(sep)) ) for ngr in ranked_ngrams]

            n2 = len(ranked_ngrams)
            # find top lcs
            adict['cid'].extend([cid] * n2)
            adict['lcs'].extend(ranked_ngrams)
            adict['count'].extend(ranked_ngrams_counts)
            adict['length'].extend(ranked_ngrams_lengths)
            adict['diversity'].extend(ranked_ngrams_nutoks)
            cluster_lcs[cid] = ranked_ngrams # for computing n-gram statistics
    
    df = DataFrame(adict, columns=header)
    
    # save data
    # identifier1 = 'G%s-COP%s-%s-%s' % (cohort_name, ctype, order_type, policy_type)  # smaller file ID
    fname_out = 'pathway_cluster-lcs-%s-%s.csv' % (lcsSelectionPolicy, identifier1) # identifier: (tset_type, cluster_method, seq_ptype, d2v_method)
    basedir1 = os.path.join(basedir, 'pathway')
    fpath_out = os.path.join(basedir1, fname_out)

    df = DataFrame(adict, columns=header)
    df.to_csv(fpath_out, sep='|', index=False, header=True)
    print('io> saved derived cluster-level lcs data (dim: %s) to %s' % (str(df.shape), fpath_out))

    ### compute n-gram statistics within these LCSs 
    div(message="2. Compute n-gram statistics witin the spac of the derived LCSs ...")
    ng_max = 4  # focus on only short motifs
    delimit = ' ' # code separator
    tPartial = True if order_type.startswith('part') else False
    header = ['cid', 'length', 'ngram', 'count']  # use 'global_count' to be consistent with that in cluster_motifs* 
    adict = {h: [] for h in header}

    for cid, seqx in cluster_lcs.items(): 

        # ngrams: n -> Counter: (tupled) ngram -> count
        ngrams = algorithms.count_ngrams2(seqx, min_length=1, max_length=ng_max, partial_order=tPartial)  
        print('  + seqx:   %s' % seqx)  # '282.60 282.60 296.30 296.30 296.30 296.30 296.30 296.30 296.30'
        print('  + ngrams: %s' % ngrams)

        n_patterns_cluster = 0
        for n, ngr_cnts in ngrams.items():  # ngr_cnts is a map: length -> list of counters
            n_patterns = len(ngr_cnts)
            n_patterns_cluster += n_patterns

            adict['length'].extend([n] * n_patterns)  # [redundancy]
            
            ngram_reprx, local_counts = [], []
            counts = ngr_cnts.most_common(None)  # sorted in descending order 
            for i, (ngt, ccnt) in enumerate(counts):  # ccnt: cluster-level count 
                if i == 0: assert isinstance(ccnt, int)
                # from tuple (ngt) to string (ngstr) 
                if isinstance(ngt, str): 
                    ngstr = ngt 
                else: 
                    assert isinstance(ngt, tuple), "invalid ngram format: %s" % str(ngt)
                    ngstr = delimit.join(str(e) for e in ngt)
                    
                    # [test]
                    if i == 0: assert isinstance(ngt, tuple), "invalid ngram: %s" % str(ngt)

                ngram_reprx.append(ngstr) # foreach element in n-gram tuple
                local_counts.append(ccnt)
            ### end foreach ngram counter associated with the same length

            adict['ngram'].extend(ngram_reprx)
            adict['count'].extend(local_counts)

        ### end foreach length
        adict['cid'].extend([cid] * n_patterns_cluster)

    ### end foreach cluster
    fpath_out2 = os.path.join(basedir1, 'pathway_cluster-lcs_stats-%s-%s.csv' % (lcsSelectionPolicy, identifier1))
    df = DataFrame(adict, columns=header)
    df.to_csv(fpath_out2, sep='|', index=False, header=True)
    print('io> saved lcs stats data (dim: %s) to %s' % (str(df.shape), fpath_out2))

    return 

def tokenize(documents, timestamps=None, infer_sep=False): # convert strings to lists of tokens
    sep_candidates = [',', ' ']
    delimit = sep_candidates[0]

    print('test> find the ngram data type and its delimiter ...')
    # infer seperator? 
    nD = len(documents)
    if timestamps is not None: assert nD == len(timestamps)
 
    is_tokenized = True
    for i, document in enumerate(documents): 
        if isinstance(document, str): 
            is_tokenized = False
            break
        else: 
            assert len(timestamps[i]) == len(document)
            if i > 100: break # early stop

    # early return
    if is_tokenized: 
        return (documents, timestamps)  # done! 

    if infer_sep: 
        found_sep = False
        nBase = 20

        # select random subset of docs and sort them ~ lengths
        sDocs = sorted([(doc, len(doc)) for i, doc in enumerate(random.sample(documents, min(nD, nBase)))], key=lambda x:x[1], reverse=True)
        docSubset = [doc for doc, _ in sDocs]

        the_sep = ' '
        for document in docSubset: 
            assert len(document) > 0

            # try splitting it
            for tok in sep_candidates: 
                # select hte longest one 
                if document.find(tok) > 0:  # e.g. '300.01 317 311' and tok = ' '
                    if len(document.split(tok)) > 1: 
                        the_sep = tok
                        found_sep = True
                        break 
            if found_sep: break 
        print("status> determined document seperator to be '%s' =?= '%s' (default) ... " % (the_sep, sep))
        delimit = the_sep

    for i, document in enumerate(documents): 
        documents[i] = document.split(delimit)
        timestamps[i] = timestamps[i].split(delimit)
        assert len(documents[i]) == len(timestamps[i])

    return (documents, timestamps)

def loadDocuments(**kargs):
    """
    Load documents (and labels)

    Params
    ------
    1. for reading source documents
        cohort
        use_surrogate_label: applies only when no labels found in the dataframe

        result_set: a dictionary with keys ['sequence', 'timestamp', 'label', ]
        ifiles: paths to document sources
    2. document transformation 
        seq_ptype 
        predicate 
        simplify_code

    Output: a 3-tuple: (D, T, l) where 
            D: a list of documents (in which each document is a list of strings/tokens)
            T: a list of timestamps
            l: labels in 1-D array

    Use 
    ---
    loadDocuments(cohort, use_surrogate=kargs.get('use_surrogate', False))

    """
    def load_docs(): 
        ret = kargs.get('result_set', {}) # 'sequence', 'timestamp', 'label' 
        if not ret: 
            ifiles = kargs.get('ifiles', [])
            ret = sa.readDocFromCSV(cohort=cohort_name, inputdir=docSrcDir, ifiles=ifiles, complete=True) # [params] doctype (timed)
        return ret

    # import matplotlib.pyplot as plt
    # from sklearn.feature_extraction.text import TfidfVectorizer
    # from pattern import diabetes as diab 
    from seqparams import TSet
    from labeling import TDocTag
    import seqTransform as st
    # import seqAnalyzer as sa 
    # import vector

    # [input] cohort, labels, (result_set), d2v_method, (w2v_method), (test_model)
    #         simplify_code, filter_code
    #         

    # [params] cohort   # [todo] use classes to configure parameters, less unwieldy
    composition = seq_compo = kargs.get('composition', 'condition_drug') # filter by content? 
    cohort_name = kargs['cohort'] # kargs.get('cohort', 'diabetes')
    tSingleLabelFormat = True  # single label format ['l1', 'l2', ...] instead of multilabel format [['l1', 'l2', ...], [], [], ...]
    tSurrogateLabels = kargs.get('use_surrogate', False) # use surrogate/nosiy labels if the dataframe itself doesn't not carry labeling information

    # [params]
    read_mode = seqparams.TDoc.read_mode  # assign 'doc' (instead of 'seq') to form per-patient sequences
    docSrcDir = sys_config.read('DataExpRoot')  # document source directory

    ### load model
    # 1. read | params: cohort, inputdir, doctype, labels, n_classes, simplify_code
    #         | assuming structured sequencing files (.csv) have been generated
    div(message='1. Read temporal doc files ...')

    # [note] csv header: ['sequence', 'timestamp', 'label'], 'label' may be missing
    # [params] if 'complete' is set, will search the more complete .csv file first (labeled > timed > doc)
    
    # if result set (sequencing data is provided, then don't read and parse from scratch)
    ret = load_docs()

    # D: documents, L: labels, T: timestamps
    D, T, L = ret['sequence'], ret.get('timestamp', []), ret.get('label', [])
    nDoc = len(seqx); print('verify> number of docs: %d' % nDoc)  

    # transform (e.g. diagnostic codes only?)

    # [params] seq_ptype, predicate, simplify_code
    seq_ptype = kargs.get('seq_ptype', 'regular')
    predicate = kargs.get('predicate', None)
    simplify_code = kargs.get('simplify_code', False)
    if not simplify_code and seq_ptype.startswith('reg'): 
        print('loadDocuments> Effectively no trasformation ...')
        pass # noop
    else: 
        # note the order of return values
        D, L, T = st.transformDocuments2(D, labels=L, items=T, policy='empty_doc', 
            seq_ptype=seq_ptype, predicate=predicate, simplify_code=simplify_code)

    return (D, T, L)

def pClusterPathway3(**kargs): 
    return pClusterPathwayLCSTimeSeries(**kargs)
def pClusterPathwayLCSTimeSeries(**kargs): 
    """
    Given Cluser-level LCSs (pClusterPathway2: pClusterPathwayLCS), find matching patients who 
    share these LCSs and subsequently, plot the corresponding time series of these codes. 
    
    Proposal time series format: 
    1) for each patient 
         generate a dataframe (times/row vs codes/columns)
            where each cell represents an accumulated count 
         plot 

    Input
    -----
        tpheno/seqmaker/data/<cohort>/pathway/pathway_cluster-lcs-* 

    Output
    ------
        tpheno/seqmaker/data/<cohort>/pathway/pathway_cluster-lcs_stats*

    Memo
    ----
    1. Use seqmaker.seqMaker2 to generate med coding sequences tagged with timestamps by setting 'include_timestamps' to True: 

       t_make_seq(save_intermediate=True, include_timestamps=True, 
           condition_table='condition_occurrence-query_ids-PTSD.csv', 
           drug_table='drug_exposure-query_ids-PTSD.csv', cohort='PTSD')

           + file pattern for times med coding sequences 
                f_tdoc = '%s_timed_seq-%s.dat' % (tdoc_prefix, cohort_name) // tdoc_prefix tells us which types of codes are included in the doc
                => condition_drug_timed_seq-PTSD.dat   


           + content format <date>|<code> 
                2900-10-02|920;2901-11-11|296.4;2902-01-04|296.82; ... 2908-01-19|300.4$ 

 

    """
    def tokenize(ngr, sep=' '):
        return ngr.split(sep)

    import motif as mf
    import algorithms, seqAlgo
    import seqAnalyzer as sa
    import seqReader as sr

    random.seed(10)

    # [params]
    cohort_name = kargs.get('cohort', 'PTSD')
    basedir = seqparams.get_basedir(cohort=cohort_name) 
    sep = '|'  # column separator

    # [params] coding sequence 
    read_mode = mode = kargs.get('read_mode', 'doc') # or 'seq'
    seq_compo = kargs.get('composition', 'condition_drug') # what does the sequence consist of? 
    default_ifile = '%s_seq-%s.dat' % (seq_compo, cohort_name) if cohort_name is not None else '%s_seq.dat' % seq_compo
    ifile = kargs.get('ifile', default_ifile) 
    ifiles = kargs.get('ifiles', None) # [todo]

    # [params] each cluster has its own pathway file (which is derived from cluster motifs & which includes tf-idf scores, etc.)
    n_classes = kargs.get('n_classes', 1)
    n_clusters = kargs.get('n_clusters', 10) 
    cIDs = range(0, n_clusters)   # CID

    # label = kargs.get('label', 1)
    content_type = ctype = kargs.get('ctype', 'diagnosis') # content type (what's in the sequence?): medication, mixed
    order_type = otype = kargs.get('otype', 'total')
    policy_type = ptype = kargs.get('ptype', 'prior')  # cut point: pre-diagnosis (prior), post-diagnosis (posterior), whole sequence (mixed)
    # cluster_method = kargs.get('cluster_method', 'kmeans')
    seq_ptype = kargs.get('seq_ptype', seqparams.normalize_ptype(ctype))  # diag, regular, med
    # d2v_method = kargs.get('d2v_method', 'tfidfavg')

    # identifier = 'G%s-COP%s-%s-%s-nC%s-nL%s-D2V%s' % (cohort_name, ctype, order_type, policy_type, n_clusters, n_classes, d2v_method)

    cluster_ngram = kargs.get('candidate_motifs', {})  # cluster ID -> ngrams
    identifier1 = 'G%s-COP%s-%s-%s' % (cohort_name, ctype, order_type, policy_type)  # smaller file ID

    # continuing from 2. Compute n-gram statistics for the derived LCSs
    div(message="3. Compute LCS occurrences ...") 

    # [params] pathway LCS file (e.g. pathway_cluster-lcs-GPTSD-COPdiagnosis-total-prior.csv)
    print("input> 3.1 first load two files: 1) original medical coding sequences and 2) pathway LCS data obtained from pClusterPathwayLCS()") 
    
    # [params] LCS 
    # options: frequency/local (f), global frequency (g), length (l), diversity (d)
    lcsSelectionPolicy = seqparams.normalize_lcs_selection(kargs.get('lcs_selection_policy', 'freq'))  

    header = ['cid', 'lcs', 'length', 'count'] 
    sep_lcs = ' '
    fname_in = 'pathway_cluster-lcs-%s-%s.csv' % (lcsSelectionPolicy, identifier1) # identifier: (tset_type, cluster_method, seq_ptype, d2v_method)
    basedir1 = os.path.join(basedir, 'pathway')
    fpath_in = os.path.join(basedir1, fname_in)
    df_lcs = pd.read_csv(fpath_in, sep='|', header=0, index_col=False, error_bad_lines=True)
    dim0 = df_lcs.shape
    print('io> loaded cluster lcs data (dim: %s) from %s' % (str(dim0), fpath_in))

    print("input> 3.2 Reading input sequences ...")

    # [params] load input temporal sequences (in order to cross reference timestamps)
    # read_from_source = True
    # include_timestamps = True
    # tdoc_prefix = seq_compo
    # tdoc_stem = 'timed_seq'
    # basedir_tdoc = kargs.get('basedir_tdoc', sys_config.read('DataExpRoot'))  
    # documents, timestamps = [], [] 

    documents, timestamps, labels = loadDocuments(cohort=cohort_name, seq_ptype=seq_ptype, 
        predicate=kargs.get('predicate', None),
        simplify_code=kargs.get('simplify_code', False), 
        ifiles=kargs.get('ifiles', []))
    
    seqx = documents
    nD, nT = len(documents), len(timestamps)
    assert nD == nT
    assert nD > 100, "Not enough temporal documents (nD=%d): formatting issue?" % nD
    print('status> Read %d sequences from the input %s' % (nD, ifile)) # e.g. PTSD: ~5000

    # [params] lcs and document formatting
    delimit = ','
    delimit_lcs = ' '

    # verify documents to ensure that they are converted to lists of tokens/codes
    test_doc = random.sample(documents, 1)[0]
    if isinstance(test_doc, str): # not yet tokenized 
        print('status> tokenizing the documents (i.e. str to list) ...')
        test_tdoc = random.sample(documents, 1)[0]
        assert isinstance(test_tdoc, str), "document data types not consistent (doc: %s but time: %s)" % (type(test_doc), type(test_tdoc))
        for i, document in enumerate(documents): 
            documents[i] = tokenize(document, sep=delimit)
            timestamps[i] = tokenize(timestamps[i], sep=delimit)
            assert len(documents[i]) == len(timestamps[i])

    print('step> 3.3 Find matching persons whose records contain a given LCS; do this for each LCS ...')
    lcsmap = {}
    maxNIDs = 100  # retain at most only this number of patient IDs
    matched_idx = set()
    n_personx = []
    for j, lcs in enumerate(df_lcs['lcs'].values): 
        # linear search (expansive): how many patients contain this LCS? 
        if not lcsmap.has_key(lcs): lcsmap[lcs] = []
        lcs_seq = lcs.split(delimit_lcs)  

        # [test]
        # if j == 0: print('  + tokenized LCS: %s' % lcs_seq)  # ok. 

        n_subs = 0
        for i, document in enumerate(documents): 
            if len(lcs_seq) > len(document): continue 

            # doc_seq = document.split(delimit) # ','
            assert isinstance(document, list), "document has not been tokenized:\n%s\n" % document
            if seqAlgo.isSubsequence(lcs_seq, document): # if LCS was derived from patient doc, then at least one match must exist
                lcsmap[lcs].append(i)  # add the person index
                matched_idx.add(i)
                n_subs += 1 
        n_personx.append(n_subs)  # number of persons sharing the same LCS

        if maxNIDs is not None and len(lcsmap[lcs]) > maxNIDs: 
            lcsmap[lcs] = random.sample(lcsmap[lcs], maxNIDs)

    print('status> Skipped %d persons/documents (without any matched LCS)' % len(set(range(nD))-set(matched_idx)))
    matched_idx = None; gc.collect()

    # [test]
    # list documents that never had a match 
    # set(range(nD))-set(matched_idx)

    df_lcs['n_matched'] = n_personx
    df_lcs.to_csv(fpath_in, sep='|', index=False, header=True)
    print('io> saved lcs counts back to input: %s' % fpath_in)
    n_no_match = sum(1 for n in n_personx if n == 0)
    print('info> number of LCSs without a single match: %d' % n_no_match)

    div(message="4. Define and plot time series for selected LCSs ...")  
    # convert lcs-to-index map to a dataframe where indices are timestamps, columns are codes and cell values correspond to 
    # accumulated counts (of code occurrences up to that time point)

    # find freuqent n-grams witin the LCSs 
    # [input] e.g. pathway_cluster-lcs_stats-GPTSD-COPdiagnosis-total-prior.csv
    print('info> Find freuqent n-gram counts witin the space of cluster-level LCSs ...')  # where LCSs are derived from patient documents

    # identifier1 = 'G%s-COP%s-%s-%s' % (cohort_name, ctype, order_type, policy_type)  # smaller file ID
    header = ['cid', 'length', 'ngram', 'count', ]
    sep_lcs = ' '
    fname_in = 'pathway_cluster-lcs_stats-%s-%s.csv' % (lcsSelectionPolicy, identifier1) # identifier: (tset_type, cluster_method, seq_ptype, d2v_method)
    # basedir1 = os.path.join(basedir, 'pathway')
    fpath_in = os.path.join(basedir1, fname_in)
    df_stats = pd.read_csv(fpath_in, sep='|', header=0, index_col=False, error_bad_lines=True)
    dim0 = df_stats.shape
    print('io> loaded cluster lcs stats data (dim: %s) from %s' % (str(dim0), fpath_in))

    delimit = ','
    delimit_lcs = ' '
    ucodes_total = set()
    to_accumulated_count = True

    # select top N unigrams and observe their waveforms? 
    # 1. prediagnosis: normalize the time of the first occurrence of the diagnostic code
    # 2. 

    nLCSMax, nPersonMax = 10, 10
    popular_codes = ['296.30', ]

    for i, lcs in enumerate(lcsmap): # foreach LCS in the selected set of matching patients (e.g. maxNIDs=100)
        if i > nLCSMax: break
        idx = lcsmap[lcs]  # lcs (string type) -> the set of person (positional) IDs with this 'lcs' in their temporal documents

        # find unique med codes
        lcs_seq = lcs.split(delimit_lcs)
        lcs_length = len(lcs_seq)
        ucodes = sorted(set(lcs_seq))
        ucodes_total.update(ucodes) # keep track of unique tokens of LCSs

        # foreach person id, find his coding sequence and timestamps
        for j in idx: 
            if j > nPersonMax: break
            seq, tseq = documents[j], timestamps[j]  # full documents # both should have been converted to lists tokens

            assert isinstance(seq, list), "documents have not been converted to LISTS OF TOKENS: \n%s\n" % str(seq)
            # seq, tseq = s.split(delimit), t.split(delimit)
            tlen = len(seq)
            assert tlen == len(tseq) 

            # filter by most frequent code: look up from lcs-stats (e.g. pathway_cluster-lcs_stats-GPTSD-COPdiagnosis-total-prior.csv)
            times = []  # time as indices
            pIDs = seqAlgo.traceSubsequence(lcs_seq, seq)  # position in Doc (pInDoc) => matching positions in the docuement 

            assert pIDs is not None, "LCS %s found to be NOT in doc: %s" % (lcs, seq[:100])
            assert len(pIDs)==lcs_length, "matching position vector not of the same length as LCS: %d <> len(pIDs)=%d" % (lcs_length, len(pIDs))

            times = [tseq[p] for p in pIDs] # matching positions -> real times

            slots = {c: [0] * lcs_length for c in ucodes} # find accumulated counts (of occurrences over time) for each code
            for pl, code in enumerate(lcs_seq): 
                slots[code][pl] = 1  # set position pl to 1 to mark code as being active
                pt = tseq[pIDs[pl]]  # find its timestamp: pIDs[pl]: lcs pos -> matching pos in doc -> time
            
            # see concept sheet for an example of visualizing multiple codes and their occurrences in the time course. 
            if to_accumulated_count: 
                # convert into accumulated counts

                for code, cntseq in slots.items(): 
                    t = 1
                    while t < lcs_length: 
                        cntseq[t]+=cntseq[t-1]
                        t += 1
                    slots[code] = cntseq

            for code in popular_codes: 
                if slots.has_key(code): 
                    print('+ [%d] %s' % (j, slots[code]))
                    print('+ [%d] %s' % (j, times))                  
 
            if j % 2 == 0: 
                print('  + LCS: %s' % lcs)
                print('      + codes: %s' % ucodes)
                print('      + pID=%d (among %d matched the given LCS)' % (j, len(idx)))
                for ci, code in enumerate(ucodes): 
                    if ci < 3: 
                        print('      + code:        %s => %s' % (code, slots[code]))  # should be increasing
                        print('      + time:        %s => %s' % (code, times))
                        pts = [times[st] for st in slots[code] if st == 1]
                        print('      + active time: %s' % pts)
    
    print('info> temporal sequence (cohot=%s), n_uniq_tokens: %d' % (cohort_name, len(ucodes_total)))

    # one df per matching patient 
    # ID vs codes where each cell represents an accumuated count

    # 1. most representative codes? 
    # 2. for each representative code, what are their common coreferences (histograms)
    # 3. find matching patients (average?)
    # 

    return

def vInputData(**kargs): # v: verify
    """

    Memo
    ----
    1. use seqmaker.seqMaker2 to generate coding sequence data 

        t_make_seq(save_intermediate=True, include_timestamps=True, 
            condition_table='condition_occurrence-query_ids-PTSD.csv', 
            drug_table='drug_exposure-query_ids-PTSD.csv', cohort='PTSD')
    """
 
    # input coding sequences 
    basedir = source_dir = kargs.get('source_dir', sys_config.read('DataExpRoot'))  

    # cohort_name = kargs.get('cohort', 'PTSD')
    read_mode = mode = kargs.get('read_mode', 'doc') # options: 'seq', 'doc', 'timed'
    default_seq_compo = kargs.get('composition', 'condition_drug') # what does the sequence consist of? 
    
    # [parmas] source
    default_ifile = '%s_seq-%s.dat' % (seq_compo, cohort_name) if cohort_name is not None else '%s_seq.dat' % seq_compo
    
    # [params] sequence types and contents
    # content_type = ctype = kargs.get('ctype', 'diagnosis') # content type (what's in the sequence?): medication, mixed
    # order_type = otype = kargs.get('otype', 'total')
    # policy_type = ptype = kargs.get('ptype', 'prior')  # cut point: pre-diagnosis (prior), post-diagnosis (posterior), whole sequence (mixed)
    # cluster_method = kargs.get('cluster_method', 'kmeans')
    # seq_ptype = kargs.get('seq_ptype', seqparams.normalize_ptype(ctype))  # diag, regular, med

    tdoc_prefixes = ['condition_drug', ]  # sequence content (which includes ALL types of medical codes)
    tdoc_stems = ['seq', 'timed_seq', ]  # original sequences or with addition info (e.g. timestamps)
    cohorts = ['PTSD', 'diabetes', ]  # patient cohorts
    content_types = ['diag', 'med', 'mixed']  # focused sequence content (in terms of specific types of medical codes in each file)
    seq_ptypes = [seqparams.normalize_ptype(ct) for ct in content_types] 
    
    for params in itertools.product(tdoc_prefixes, tdoc_stems, cohorts, seq_ptypes): 
        tdoc_prefix, tdoc_stem, cohort_name, seq_ptype = params[0], params[1], params[2], params[3]
        f_tdoc = '%s_%s-%s-%s.dat' % (tdoc_prefix, tdoc_stem, cohort_name, seq_ptype)   
        basedir_tdoc = kargs.get('basedir_tdoc', sys_config.read('DataExpRoot'))  

    return

def evalClusterDistribution(**kargs):

    # find frequent n-grams in each cluster and plot their frequency distributions
    evalMotifFreq(**kargs)

    # quantitative measures of cluster differences 
    evalClusterDiff(**kargs)

    return

def clusterDiff(**kargs): 
    return evalClusterDiff(**kargs)
def evalClusterDiff(**kargs): 
    """
    Load cluster motifs, which come from two possible sources: 
       1. pathway 
          attributes: length|ngram|tfidf|tf_cluster|tf_global|cluster_occurrence
       2. cluster_motifs
          attributes: length|ngram|count|global_count|ratio

    Memo
    ----
    1. distribution of n-grams in different clusters, are they different? 
       unigram distribution 
       bigram distribution 

       problem with higher n is the duplicates (e.g. different only by the reading frame)


    """
    def hamming(s1, s2):  # dissimilarity
        """Calculate the Hamming distance between two bit strings"""
        assert len(s1) == len(s2)
        return sum(c1 != c2 for c1, c2 in zip(s1, s2))
    def hamming_ratio(s1, s2):  # similarity
        assert len(s1) == len(s2)
        r = 1 - sum(c1 != c2 for c1, c2 in zip(s1, s2)) / (len(s1)+0.0) 
        return r       
    def binarize(avec):
        avec2 = [0] * len(avec)
        for i, v in enumerate(avec): 
            if v > 0: 
                avec2[i] = 1 
        return avec2

    # unigram distribution
    import types, itertools
    import plotUtils as pu
        
    # [params]
    cohort_name = kargs.get('cohort', 'PTSD')
    basedir = seqparams.get_basedir(cohort=cohort_name) 
    experimentdir = 'pathway'

    # [params] cluster, class 
    n_clusters = kargs.get('n_clusters', 10)
    n_classes = kargs.get('n_classes', 1)
    cIDs = range(0, n_clusters)   # CID

    # [params] file ID
    content_type = ctype = kargs.get('ctype', 'diagnosis') # content type (what's in the sequence?): medication, mixed
    order_type = otype = kargs.get('otype', 'total')
    policy_type = ptype = kargs.get('ptype', 'prior')  # cut point: pre-diagnosis (prior), post-diagnosis (posterior), whole sequence (mixed)
    cluster_method = kargs.get('cluster_method', 'kmeans')
    seq_ptype = kargs.get('seq_ptype', seqparams.normalize_ptype(ctype))  # diag, regular, med
    d2v_method = kargs.get('d2v_method', 'tfidfavg')
    identifier = 'G%s-COP%s-%s-%s-nC%s-nL%s-D2V%s' % (cohort_name, ctype, order_type, policy_type, n_clusters, n_classes, d2v_method)

    documents, timestamps = loadDocuments(seq_ptype=seq_ptype, cohort=cohort_name)  # [params] cohort, ctype ('mixed'), include_timestamps (True)
    if len(timestamps) > 0: assert len(documents) == len(timestamps)

    # [params] (cohort_name, content_type, order_type, policy_type, n_clusters, n_classes, d2v_method)
    # [option] load_selected_motifs: raw cluster motifs (cluster_motifs-) vs processed/selected (pathway-)

    # 1. collect ngrams
    focused_lengths = [1, 2, 3, ]
    tSelectedMotifs = False

    # filter n-gram counts by their global usage (to simplify the frequency distribution)
    tFilterByGlobalUsage = kargs.get('filter_by_global_usage', False) 

    # find out all n-grams 
    ngramList = {fl:[] for fl in focused_lengths} # length -> ngrams (sorted)
    for cid in cIDs: 
        dfc_gen = loadClusterMotifs(cID=cid, cohort=cohort_name, load_selected_motifs=tSelectedMotifs, 
                ctype=ctype, otype=otype, ptype=ptype, n_clusters=n_clusters, d2v_method=d2v_method) 
        dfc = list(dfc_gen)[0]
        # isinstance(dfc, types.GeneratorType)
        print('io> successfully loaded motifs for cid=%s => dim: %s' % (cid, str(dfc.shape)))
        
        for fl in focused_lengths: 
        
            # select by length 
            dfcl = dfc.loc[dfc['length']==fl]
            if not dfcl.empty: 
                ngramList[fl].extend(dfcl['ngram'])

        for fl in focused_lengths: 
            ngramList[fl] = wlist = sorted(np.unique(ngramList[fl]))   # ascending ordr
            print('  + vocaculary size %d | ngram length=%d' % (len(wlist), fl))

    # compute n-gram frequencies
    #    [init] ngramFreq: cid -> (length -> [f1, f2, ...])  where f<i>: frequency of ith-ngram from ngramList
    ngramFreq = {cid: {} for cid in cIDs} # order is important
    for cid in cIDs: 
        ngramFreq[cid] = {fl: [0] * len(ngramList[fl]) for fl in focused_lengths}

    for cid in cIDs: 
        dfc_gen = loadClusterMotifs(cID=cid, cohort=cohort_name, load_selected_motifs=tSelectedMotifs, 
                ctype=ctype, otype=otype, ptype=ptype, n_clusters=n_clusters, d2v_method=d2v_method) 
        dfc = list(dfc_gen)[0] 
        
        # query (local, raw) counts or scores 
        qa = 'tfidf' if tSelectedMotifs else 'count'
        tb = dict(zip(dfc['ngram'], dfc[qa]))

        for fl in focused_lengths: 
            ngramFreq[cid][fl] = [tb.get(ngr, 0) for ngr in ngramList[fl]] 
        
        # [test]
        for fl in focused_lengths: 
            assert sum(ngramFreq[cid][1]) > 0

    # filter n-grams with smaller global usage rate/count
    if tFilterByGlobalUsage: 
        maxN = 100
        for fl in focused_lengths: 
            wcount = np.array([0] * len(ngramList[fl]))
            for cid in cIDs: 
                wcount += np.array(ngramFreq[cid][fl])
        
            # [filter] filter by global usage/counts
            wcs = sorted( zip(ngramList[fl], wcount), key=lambda x:x[1], reverse=True)   # word count sorted
            print('  ++ %d-gram count descending order (n=%d): %s' % (fl, len(wcs), wcs[:20]))
        
            selected_ngrams = set([ngr for ngr, _ in wcs[:maxN]])
        
            # new vocabulary (V') 
            ngramList[fl] = wlist2 = [ngr for ngr in ngramList[fl] if ngr in selected_ngrams]

            # preserve only counts for V'  
            for cid in cIDs: 
                newCounts = []
                for i, ngr in enumerate(ngramList[fl]): 
                    if ngr in selected_ngrams: 
                        newCounts.append(ngramFreq[cid][fl][i])
                ngramFreq[cid][fl] = newCounts
            assert len(ngramFreq[cIDs[0]][fl]) == len(ngramFreq[cIDs[-1]][fl]) 

    # save data for plotting (perhaps filter n-grams with low usage rate first)
    #   + save for each length n-gram frequencies vs cluster (IDs)
    basedir2 = os.path.join(basedir, experimentdir)  # e.g. .../seqmaker/data/<disease>/pathway
    header = ['C_%s' % cid for cid in cIDs]

    for fl in focused_lengths: # distribution for each file in column vectors (ngram counts: row vs clusters: column)
        adict = {cid: None for cid in cIDs}

        for cid in cIDs: 
            adict[cid] = ngramFreq[cid][fl] 

        df = DataFrame(adict, columns=header)
        
        fname = 'freq_distribution-n%s-%s.csv' % (fl, identifier)
        df.to_csv(os.path.join(basedir2, fname), sep='|', index=False, header=True)

        # plot n-gram frequency distribution for all clusters (and perhaps compare them?)
        fname = 'freq_distribution-n%s-%s.pdf' % (fl, identifier)
        
        # pu.plot_multiple_hist(df, fpath=os.path.join(basedir2, fname))  # panda
        pu.plot_multiple_hist_plotly(df, cols=None, outputdir='tpheno', fname=fname) # plotly, cufflinks

    # run sigfinicance test on these n-gram frequencies

    # 1. similarity measure
    div(message='Compute similarities between clusters ...')

    simx = {}
    tl = 1  # target length
    for (c1, c2) in itertools.combinations(cIDs, 2):  # compare similarity of all pairs
        simx[(c1, c2)] = sc = hamming_ratio(binarize(ngramFreq[c1][tl]), binarize(ngramFreq[c2][tl]))
        simx[(c2, c1)] = sc 

    dissimilarList = [] # keep track of which clusters are being selected as most dissimilar to the other clusters
    for cid in cIDs: 
        # select the least similar one for each cluster
        otherCIDs = set(cIDs)-set([cid])   # other clusters

        # set reverse=False => ascending => from dissimilar to similar
        dlist = sorted([(oid, simx[(oid, cid)]) for oid in otherCIDs], key=lambda x:x[1], reverse=False) 
        print('  + cid=%d, most dissimilar clusters: %s' % (cid, dlist[:3]))

        dissimilarList.append([o for o, s in dlist])

    # now which clusters are more likely to be anomalies? (also see significance testing below)
    nMostDissim = 3
    maxVotes = collections.Counter() # cid -> # of times selected for being dissimilar
    for dlist in dissimilarList: 
        counts = collections.Counter(dlist)
        maxVotes.update(counts)
    anomalousClusters = maxVotes.most_common(nMostDissim)
    print('  + most anamalous clusters (~ majority votes): %s' % anomalousClusters)

    # [Q] plot freq distribution only for these anomalous clusters? 

    # plot
    focused_lengths2 = [1,  ] 
    # header = ['C_%s' % cid for cid in cIDs]
    for fl in focused_lengths2: # or focused_length
        fname = 'freq_distribution-n%s-%s.csv' % (fl, identifier)
        fpath = os.path.join(basedir2, fname)
        df = pd.read_csv(fpath, sep='|', header=0, index_col=False, error_bad_lines=True)
        assert not df.empty
        print('io> loaded freq distribution file: %s' % fname)
    
    # 3. significance testing 

      

    return

def evalMotifFreqBatch(**kargs):
    pass 
def evalMotifFreq(**kargs):
    """
    Compare cluster motif frequencies (towards operational definition of 
    disease subtypes)

    Memo
    ----
    1. filtering 
            cond_c = df['code'] == c 
            cond_m = df['model'] == m
            subset = df.loc[cond_c & cond_m]
    """
    # import seaborn as sns
    # import matplotlib.pyplot as plt

    sns.set(style="whitegrid")
    
    # [params] 
    #     ctype: sequence type 
    #     otype: strictly ordered, partial 
    #     ptype: prior (to diagnosis), mixed, post

    # index sequence parameters to the apprpriate set of cluster motif files 
    # foreach top n-gram, find their frequencies in other clusters => histogram

    # rank top-ngram (across clusters) in terms of their tf-idf scores 

    # general naming pattern of cluster motif file: 
    #     stem = cluster_motifs
    #     id   = 'CIDL%s-%s-COP%s-%s-%s-C%s-S%s-D2V%s' % (mtype, label, ctype, otype, ptype, cluster_method, seq_ptype, d2v_method) 
    #     e.g. 
    #             cluster_motifs-CIDL9-1-COPdiagnosis-part-noop-Ckmeans-Sdiag-D2Vtfidfavg.csv

    cohort_name = kargs.get('cohort', 'PTSD')
    basedir0 = basedir = seqparams.get_basedir(cohort=cohort_name) 

    # [params] cluster motif files (one particular configuraiton)
    n_clusters = kargs.get('n_clusters', 10) 
    cIDs = range(0, n_clusters)   # CID
    
    label = kargs.get('label', 1)
    content_type = ctype = kargs.get('ctype', 'diagnosis') # content type (what's in the sequence?): medication, mixed
    order_type = otype = kargs.get('otype', 'total')
    policy_type = ptype = kargs.get('ptype', 'prior')  # cut point: pre-diagnosis (prior), post-diagnosis (posterior), whole sequence (mixed)
    cluster_method = kargs.get('cluster_method', 'kmeans')
    seq_ptype = kargs.get('seq_ptype', seqparams.normalize_ptype(ctype))  # diag, regular, med
    d2v_method = kargs.get('d2v_method', 'tfidfavg')
 
    div(message='1. Load global motif file (params> ct=%s, ot=%s, pt=%s, cm:%s, d2v:%s)' % \
        (ctype, otype, ptype, cluster_method, d2v_method))

    # motifs-CMOPdiagnosis-global-total-prior-Sdiag-D2Vtfidfavg.csv
    stem = 'motifs'
    mtype = 'global'
    header_global = ['length', 'ngram', 'global_count']
    identifier = 'CMOP%s-%s-%s-%s-S%s-D2V%s' % (ctype, mtype, otype, ptype, seq_ptype, d2v_method)
    fname = '%s-%s.csv' % (stem, identifier)
    basedir1 = os.path.join(basedir, 'global_motifs')  # use either basedir1 (organized ~ motif type) or basedir 

    fpath = os.path.join(basedir1, fname)
    dfg = pd.read_csv(fpath, sep='|', header=0, index_col=False, error_bad_lines=True)
    print('io> loaded global motif file (dim: %s, fname: %s)' % (str(dfg.shape), fname))

    div(message='2. Load all the cluster motifs (params> ct=%s, ot=%s, pt=%s, cm:%s, d2v:%s)' % \
        (ctype, otype, ptype, cluster_method, d2v_method))

    stem = 'cluster_motifs'
    filter_low_freq, min_freq = True, 3

    apply_group_sort, select_topn = True, True
    save_byproduct = True # [I/0]

    header = ['length', 'ngram', 'count', 'global_count', 'ratio']
    topn_per_n = 15  # select n-grams based on local/cluster counts 
    pivots = ['length', 'count', 'ratio', ]
    max_length = 10  # focus on n-grams that are not too long
    
    countmapl = {cid: {} for cid in cIDs}  # ngram -> (local) count
    countmapg = {} # ngram -> (global) count

    lengths = []

    short_chain = [2, 3, 4]  # only focus on these n-grams in the PLOT
    long_chain = [8, 9, 10]

    # consider short, medium and long motifs
    length_targets_set = [short_chain, long_chain]

    basedir2 = os.path.join(basedir, 'cluster_motifs')  # use either basedir1 (organized ~ motif type) or basedir 

    for i, cid in enumerate(cIDs): 

        # example: cluster_motifs-CIDL1-1-COPdiagnosis-total-prior-Ckmeans-Sdiag-D2Vtfidfavg.csv
        # [I/O]
        identifier = 'CIDL%s-%s-COP%s-%s-%s-C%s-S%s-D2V%s' % (cid, label, ctype, otype, ptype, cluster_method, seq_ptype, d2v_method) 
        fname = "%s-%s.csv" % (stem, identifier)
        # basedir2 = os.path.join(basedir, 'cluster_motifs')  # use either basedir1 (organized ~ motif type) or basedir 
        fpath = os.path.join(basedir2, fname)
        assert os.path.exists(fpath), "invalid path? %s" % fpath

        dfc = pd.read_csv(fpath, sep='|', header=0, index_col=False, error_bad_lines=True)
        dim0 = dfc.shape
        print('io> loaded cluster motif file (cid: %s, dim: %s, fname: %s)' % (cid, str(dim0), fname))

        # init
        if not lengths: 
            lengths = sorted(set(dfc['length'].values))
            print('info> Found n-grams where n = %s' % lengths)

        # filter low frequency (could be anomaly)
        if filter_low_freq:
            c_global = dfc['global_count'] >= min_freq 
            dfc = dfc.loc[c_global]

        # filter: focus only on n-grams no longer than 'max_length'
        if max_length is not None: 
            c_length = dfc['length'] <= max_length
            dfc = dfc.loc[c_length]

        ### N-gram representatitiveness 
        #    policy #1: high local counts i.e. select 'representative' cluster n-grams (foreach n) ~ (local) count
        #               Note that high local counts does not necessarily imply high local-to-global frequency ratios
        #    policy #2: low cluster frequencies i.e. representative in terms of # of clusters with given n-grams present; the lower the better
        #               this can be calculated via pathway files (see pathway_global-GPTSD-COPdiagnosis-total-prior-nC10-nL1-D2Vtfidfavg.csv)
        #                  > See pClusterPathway* 
        if select_topn: # select topn ngrams foreach n
            pivots_group = ['length', ]
            pivots_local = ['count', ]   # 'count': cluster count, 'global_count'
            dfcx = []
            for g, dfi in dfc.groupby(pivots_group): # foreach length
                dfcx.append(dfi.nlargest(topn_per_n, pivots_local)) # highest local counts (e.g. top 15)
            dfc = pd.concat(dfcx, ignore_index=True)

        if apply_group_sort: 
            # cluster_motifs files were sorted (in descending order) wrt ['length', 'count', 'ratio']
            dfc = dfc.sort_values(pivots, ascending=False)
 
        # save the curated dataframe for a record 
        if save_byproduct: 
            # create subdirectory 
            subdir = 'cluster_motifs'
            basedir2 = os.path.join(basedir, subdir)
            if not os.path.exists(basedir2): 
                print('io> creating new data directory at %s' % basedir2)
                os.makedirs(basedir_prime) # test directory

            stem1 = 'high_freq_ngrams'
            identifier = 'CIDL%s-%s-COP%s-%s-%s-C%s-S%s-D2V%s' % (cid, label, ctype, otype, ptype, cluster_method, seq_ptype, d2v_method) 
            fname = "%s-%s.csv" % (stem1, identifier)

            fpath_out = os.path.join(basedir2, fname)
            print('io> saving selected cluster motif dataframe of dim: %s to %s' % (str(dfc.shape), fpath_out))
            dfc.to_csv(fpath_out, sep='|', index=False, header=True)

        # [verification]

        print('io> after curating cluster motifs (cid: %s), dim: %s => %s' % (cid, str(dim0), str(dfc.shape)))
        # [log] io> after curating cluster motifs (cid: 0), dim: (285805, 5) => (40, 5)

        # map 
        for length in lengths: 
            # local/cluster: use 2-tuple list to preserve the ordering
            countmapl[cid][length] = zip(dfc['ngram'].values, dfc['count'].values)
            
            # global
            countmapg[length] = zip(dfc['ngram'].values, dfc['global_count'].values)

            # [todo] double check global counts > see if this local global count is consistent with that in global motif file
               
        print('io> plot cluster distribution over the topn motifs (local counts) ...') 
        # generate histograms (foreach n: ngrams vs local and global counts)
        # e.g. do 8-gram, 4-gram, and 2-gram, 1-gram => filter by lengths 
        
        for length_targets in length_targets_set: 

            # seaborn
            # plot_horizontal_bar(df=dfc, cluster_id=cid, length_targets=length_targets, 
            #     ctype=ctype, otype=otype, ptype=ptype, outputdir=None, basedir=basedir)

            # plotly
            # dfc: ['length', 'ngram', 'count', 'global_count', 'ratio']
            plot_horizontal_bar2(df=dfc, cluster_id=cid, length_targets=length_targets, 
                ctype=ctype, otype=otype, ptype=ptype, outputdir=None, basedir=basedir2)

            print('status> completed cluster (cid=%s)' % cid)

    ### end foreach cluster (cid)

    div(message='3. Evaluate cluster representativeness (the fraction of n-grams captured by a cluster) ...')
    # cases: 1. considering ordering or 2. not
    print('info> use evalClusterRecall() ...')

    return

def evalClusterRecall(**kargs):
    """
    Evaluate cluster representativeness in terms of the fraction of n-grams captured by a cluster. 

    Memo
    ----
    Among all n-grams captured in a cluster, how important they are? frequency ratio, tf-idf, ...

    """
    # import seaborn as sns
    # import matplotlib.pyplot as plt

    sns.set(style="whitegrid")
    
    # [params] 
    #     ctype: sequence type 
    #     otype: strictly ordered, partial 
    #     ptype: prior (to diagnosis), mixed, post

    # general naming pattern of cluster motif file: 
    #     stem = cluster_motifs
    #     id   = 'CIDL%s-%s-COP%s-%s-%s-C%s-S%s-D2V%s' % (mtype, label, ctype, otype, ptype, cluster_method, seq_ptype, d2v_method) 
    #     e.g. 
    #             cluster_motifs-CIDL9-1-COPdiagnosis-part-noop-Ckmeans-Sdiag-D2Vtfidfavg.csv

    cohort_name = kargs.get('cohort', 'PTSD')
    basedir0 = basedir = seqparams.get_basedir(cohort=cohort_name) 

    # [params] cluster motif files (one particular configuraiton)
    n_clusters = kargs.get('n_clusters', 10) 
    cIDs = range(0, n_clusters)   # CID
    
    label = kargs.get('label', 1)
    content_type = ctype = kargs.get('ctype', 'diagnosis') # content type (what's in the sequence?): medication, mixed
    
    # order type is 'marginalized'
    order_types = otypes = kargs.get('otypes', ['total', 'part'])
    
    policy_type = ptype = kargs.get('ptype', 'prior')  # cut point: pre-diagnosis (prior), post-diagnosis (posterior), whole sequence (mixed)
    cluster_method = kargs.get('cluster_method', 'kmeans')
    seq_ptype = kargs.get('seq_ptype', seqparams.normalize_ptype(ctype))  # diag, regular, med
    d2v_method = kargs.get('d2v_method', 'tfidfavg')
 
    div(message='1. Load global motif file (params> ct=%s, ot=(total+partial), pt=%s, cm:%s, d2v:%s)' % \
        (ctype, ptype, cluster_method, d2v_method))

    # motifs-CMOPdiagnosis-global-total-prior-Sdiag-D2Vtfidfavg.csv
    stem = 'motifs'
    mtype = 'global'
    header_global = ['length', 'ngram', 'global_count']

    basedir1 = os.path.join(basedir, 'global_motifs')  # use either basedir1 (organized ~ motif type) or basedir 
    
    gmap = {}
    for otype in order_types: 
        identifier = 'CMOP%s-%s-%s-%s-S%s-D2V%s' % (ctype, mtype, otype, ptype, seq_ptype, d2v_method)
        fname = '%s-%s.csv' % (stem, identifier)
        fpath = os.path.join(basedir1, fname)
        dfg = pd.read_csv(fpath, sep='|', header=0, index_col=False, error_bad_lines=True)
        print('io> loaded global motif file (dim: %s, fname: %s)' % (str(dfg.shape), fname))
        gmap[otype] = dfg

    div(message='2. Load all the cluster motifs (params> ct=%s, ot=(total+partial), pt=%s, cm:%s, d2v:%s)' % \
        (ctype, ptype, cluster_method, d2v_method))

    stem = 'cluster_motifs'
    filter_low_freq, min_freq = True, 3
    apply_group_sort, select_topn = True, True
    save_byproduct = True # [I/0]

    header = ['length', 'ngram', 'count', 'global_count', 'ratio']
    topn_per_n = 15
    pivots = ['length', 'count', 'ratio', ]
    max_length = 10  # focus on n-grams that are not too long
    
    countmapl = {cid: {} for cid in cIDs}  # ngram -> (local) count
    countmapg = {} # ngram -> (global) count

    lengths = []

    short_chain = [2, 3, 4]  # only focus on these n-grams in the PLOT
    long_chain = [8, 9, 10]

    # consider short, medium and long motifs
    length_targets_set = [short_chain, long_chain]

    basedir2 = os.path.join(basedir, 'cluster_motifs')  # use either basedir1 (organized ~ motif type) or basedir 
    # example: cluster_motifs-CIDL1-1-COPdiagnosis-total-prior-Ckmeans-Sdiag-D2Vtfidfavg.csv
    #          cluster_motifs-CIDL8-1-COPdiagnosis-part-prior-Ckmeans-Sdiag-D2Vtfidfavg.csv
    #              + header: ['length', 'ngram', 'count', 'global_count', 'ratio']
    # [I/O]

    min_ratio = 0.9
    for i, cid in enumerate(cIDs): 
        # ordered 
        cmap = {} # cluster-level statistics (to be compared with those in global scope)
        fractions = {}  # fractions of exclusive n-grams (controlled by min_ratio)
        for otype in order_types: 
            identifier = 'CIDL%s-%s-COP%s-%s-%s-C%s-S%s-D2V%s' % (cid, label, ctype, otype, ptype, cluster_method, seq_ptype, d2v_method) 
            fname = "%s-%s.csv" % (stem, identifier)
            
            fpath = os.path.join(basedir2, fname) # fpath = os.path.join(basedir, fname)
            assert os.path.exists(fpath), "invalid path? %s" % fpath

            dfc = pd.read_csv(fpath, sep='|', header=0, index_col=False, error_bad_lines=True)
            dim0 = dfc.shape; nrow = dim0[0]
            print('io> loaded (%s-ordered) cluster motif file (cid: %s, dim: %s, fname: %s)' % (otype, cid, str(dim0), fname))  
            cmap[otype] = dfc; fractions[otype] = {}

            ### 1. compute {average, *weigthed average} 'recalls': number of n-grams captured by a cluster / number of n-grams in the global scope 
            # weighted by ratios between cluster and global frequencies so that relatively frequent n-grams are given higher weights 
    
            dfg = gmap[otype]

            # dfc/dfg 

            ### 2. compute fractions of n-grams within a cluster that occur exclusively in that cluster 
            df_exclusive = dfc.loc[dfc['ratio']>=min_ratio]
            n_exclusive = df_exclusive.shape[0]
            fractions[otype][cid] = n_exclusive/(nrow+0.0)

    ### plot 
    div(message="Plotting n-gram exclusiveness histograms (clusters [cids] vs ratios) ... ")
    
    # use PyData?

    return

def plot_horizontal_bar2(df, **kargs): 
    import plotly.plotly as py
    import plotly.graph_objs as go

    # df = data  # data has another meaning (list of traces)

    # [filter] only focus on n-grams of certain lengths
    nx = length_targets = kargs.get('length_targets', None)
    if nx is not None: 
        assert hasattr(nx, '__iter__')
        df = df.loc[df['length'].isin(nx)]
    else: 
        nx = length_targets = sorted(set(df['length'].values))

    plt.clf()

    # [params]
    label = "Global count"
    col_y, col_x = "ngram", "global_count"

    y_vals = df[col_y].values 
    x_vals = df[col_x].values
    
    # [params]
    save_fig = kargs.get('save_', True)
    cID = kargs.get('cluster_id', None)

    # [test]
    if cID == 5: 
        print('  + lengths=%s' % nx)
        print('  + x_vals:\n%s\n' % x_vals)
        print('  + y_vals:\n%s\n' % y_vals)


    ### plotly code ### 

    trace1 = go.Bar(
        y=y_vals,   # names
        x=x_vals,   # counts
        name=label,
        orientation = 'h',
        # font=dict(size=10),
        marker = dict(
            color = 'rgba(246, 78, 139, 0.6)',
            line = dict(
                color = 'rgba(246, 78, 139, 1.0)',
                width = 3)
        )
    )

    # [params]
    label2 = "Cluster count"
    col_y, col_x = "ngram", "count"

    y_vals = df[col_y].values 
    x_vals = df[col_x].values

    trace2 = go.Bar(
        y=y_vals,
        x=x_vals,
        name=label2,
        orientation = 'h',
        # font=dict(size=10),
        marker = dict(
            color = 'rgba(58, 71, 80, 0.6)',
            line = dict(
                color = 'rgba(58, 71, 80, 1.0)',
                width = 3)
        )
    )

    data = [trace1, trace2]
    layout = go.Layout(
        barmode='stack'
    )

    fig = go.Figure(data=data, layout=layout)
    py.iplot(fig, filename='marker-h-bar')

    # [params]
    sep = ', '
    length_targets_str = sep.join(str(n) for n in length_targets)
    xlabel = "Top n-grams distribution in cluster %s (where n = %s)" % (cID, length_targets_str)
    print('info> plotting %s' % xlabel)

    if save_fig: 
        stem = 'ngrams_freq'
        ctype, otype, ptype = kargs.get('ctype', None), kargs.get('otype', None), kargs.get('ptype', None) 
        basedir = kargs.get('basedir', os.getcwd())

        subdir = 'plot'
        outputdir = os.path.join(basedir, subdir)
        if not os.path.exists(outputdir): 
            print('io> creating new graphic directory %s' % outputdir)
            os.makedirs(outputdir) # test directory
 
        identifier = 'CID%s' % cID
        if ctype is not None: identifier = 'CID%s-COP%s-%s-%s' % (cID, ctype, otype, ptype) 
        
        ext_fig = 'pdf'  # Plotly-Python package currently only supports png, svg, jpeg, and pdf
        fname = "%s-%s.%s" % (stem, identifier, ext_fig)
        fpath = os.path.join(outputdir, fname)

        print('io> saving %s-figure to %s' % (stem, fpath))
        # sns.savefig(fpath)
        # plt.savefig(fpath, dpi=300)

        # (@) Send to Plotly and show in notebook
        # py.iplot(fig, filename=fname)
        # (@) Send to broswer 
        plot_url = py.plot(fig, filename=fname)
        py.image.save_as({'data': data}, fpath)

    plt.close() 

    return 

def plot_horizontal_bar(df, **kargs): # seaborn

    # Load the example car crash dataset
    # df = data

    # [params]
    save_fig = kargs.get('save_', True)

    figsize = (6, 15)
    cID = kargs.get('cluster_id', None)

    nx = length_targets = kargs.get('length_targets', None)
    if nx is not None: 
        assert hasattr(nx, '__iter__')
        df = df.loc[df['length'].isin(nx)]
    else: 
        nx = length_targets = sorted(set(df['length'].values))

    plt.clf()

    # Initialize the matplotlib figure
    f, ax = plt.subplots(figsize=figsize)

    # [params]
    color_codes = "pastel"
    label = "Global count"
    col_x, col_y = "ngram", "global_count"
    color_bar = "b"

    # e.g. the global count
    # sns.set_color_codes(color_codes)
    sns.barplot(x=col_x, y=col_y, data=df,
                label=label, color=color_bar)

    # [params]
    color_codes = "muted"
    # label = "Cluster count" if cID is None else "Cluster count (CID=%s)" % cID  # or local count
    label = "Cluster count"
    col_x, col_y = "ngram", "count"
    color_bar = "g"
    
    # e.g. the local count (of a given cluster)
    # sns.set_color_codes(color_codes)
    sns.barplot(x=col_x, y=col_y, data=df,
                  label=label, color=color_bar)

    # [params]
    sep = ', '
    length_targets_str = sep.join(str(n) for n in length_targets)
    xlabel = "Top n-grams distribution in cluster %s (where n = %s)" % (cID, length_targets_str)
    print('info> plotting %s' % xlabel)
    
    # Add a legend and informative axis label
    ax.legend(ncol=2, loc="lower right", frameon=True)
    ax.set(xlim=(0, 24), ylabel="",
             xlabel=xlabel)
    sns.despine(left=True, bottom=True) 

    if save_fig: 
        stem = 'ngrams_freq'
        ctype, otype, ptype = kargs.get('ctype', None), kargs.get('otype', None), kargs.get('ptype', None) 
        basedir = kargs.get('basedir', os.getcwd())

        subdir = 'plot'
        outputdir = os.path.join(basedir, subdir)
        if not os.path.exists(outputdir): 
            print('io> creating new graphic directory %s' % outputdir)
            os.makedirs(outputdir) # test directory
 
        identifier = 'CID%s' % cID
        if ctype is not None: identifier = 'CID%s-COP%s-%s-%s' % (cID, ctype, otype, ptype) 
        
        ext_fig = 'tif'
        fname = "%s-%s.%s" % (stem, identifier, ext_fig)
        fpath = os.path.join(outputdir, fname)

        print('io> saving %s-figure to %s' % (stem, fpath))
        # sns.savefig(fpath)
        plt.savefig(fpath, dpi=300)
    plt.close() 

    return

def t_purity(**kargs): 

    # find purest clusters associated with cluster labels (majority votes)
    cluster_purity(**kargs)

    return

# [cluster]
def t_pipeline(**kargs):
    """

    Memo
    ----
    (*) Questions that distinguish Clusters
        four main questions that char a cluster 
        1. hist of codes vs freq (evalClusterDistribution() -> plot_horizontal_bar2)
            + short chain 
            + long chain
        2. list of lcs (pClusterPathwayLCS() -> {pathway_cluster-lcs, pathway_cluster-lcs_stats})
        3. example time series mentioned earlier freq vs time 

        4. bigram cooccurrence 

    """ 
    # import itertools
    
    ### horizontal bar plot 
    # plot_horizontal_bar2()

    ### find purest clusters associated with cluster labels (majority votes)
    # t_purity(**kargs)

    ### sort n-grams accorindg to tf-idf scores 
    # eval_tfidf(**kargs)

    ### adjust motif statistics 
    # adjust_motifs()

    ### modify (e.g. sort) lookup table
    # processLookupTable(**kargs)

    ### cluster distribution: frequency distributions of n-grams across clusters (and how they're compared to the global scope)
    # evalMotifFreq(**kargs)

    ### select representative cluster motifs (towards defining meaningful disease subtypes)
    cohort_name = 'PTSD'
    # 1. default 
    # pClusterPathway(**kargs)

    # 2. combinations
    
    otypes = ['total', ]  # n-gram with ordering considered? {'partial', 'total',}
    ctypes = ['diagnosis', 'medication', 'mixed',]  # {'diagnosis', 'medication', 'mixed'}
    ptypes = ['prior', ]   # {'prior', 'noop', 'posterior'} | noop: complete sequence  
    
    lcs_selection_policy = 'freq'  # high local frequencies
    
    # test_combo = ['partial', 'diagnosis', 'prior']
    # for params in test_combo:
    for params in itertools.product(otypes, ctypes, ptypes): 
        order_type, content_type, policy_type = params[0], params[1], params[2] 

        # look into global motif and cluster motif files 
        # and retain top N cluster n-grams by a given metric (e.g. tf-idf scores) => pathway_cluster_derived-* 
        # This is done to facilitate *** manual insepections *** (VS rankClusterMotifs is for automatic methods)
        findRepresentativeNGrams(ctype=content_type, otype=order_type, ptype=policy_type) # for manual inspection
        print('  + completed n-gram analysis')
        # find n-gram frequencies based instead on the number of patients having these n-grams in their sequences 
        
        # find LCS from among all the top-ranked cluster motifs (~ tfidf) 
        # [operation ] use { rankClusterMotifs, } as criteria to find LCS 
        # [output] pathway_cluster_ranked
        cluster_ngram = rankClusterMotifs(ctype=content_type, otype=order_type, ptype=policy_type, 
                                            cluster_freq_max=1, topn_per_length=100)  # ranked by tfidf by default
        
        # [filter] 'screen' people 
        # use the top ranked motifs to select patients from which to derive LCS 
        # cluster_ngram = selectClusterMembers(policy='length', n_docs=3000, cohort=cohort_name, ctype=content_type)
        cluster_ngram = selectMembersByDocLength(n_docs=5000, cohort=cohort_name, ctype=content_type) # [params] d2v_method, cluster_method

        # [note] only 'representative' n-grams are retained for computing their LCSs (e.g. top 100 tfidf-ranked n-grams for ngram length)
        # [output] pathway_cluster-lcs, pathway_cluster-lcs_stats
        pClusterPathwayLCS(candidate_motifs=cluster_ngram, topn_lcs=30, 
            lcs_selection_policy='freq',     # select n highest frequencies (with minimal length) where n=topn_lcs 
            min_length=5, max_n_pairs=100000,     # selected LCS must be longer than 'min_length'; number of pairwise cmp <= 'max_n_pairs'
            ctype=content_type, otype=order_type, ptype=policy_type)

        # time series 
        pClusterPathwayLCSTimeSeries(ctype=content_type, otype=order_type, ptype=policy_type, lcs_selection_policy='freq')

        # find topn represenatative short-chain n-grams (n=1~4)? 
        pClusterPathwayLength(topn_per_length=20, ctype=content_type, otype=order_type, ptype=policy_type)

    # cluster n-gram uniqueness (order type is 'marginalized')
    for params in itertools.product(ctypes, ptypes): 
        content_type, policy_type = params[0], params[1]
        evalClusterRecall(ctype=content_type, ptype=policy_type)

    return

def t_selected_stage(**kargs):
    """
    Main use: 
        Skip time-consuming steps such as pClusterPathwayLCS() by simply assuming that data have been generated. 
    """
    
    ### select representative cluster motifs (towards defining meaningful disease subtypes)
    cohort_name = 'PTSD'
    # 1. default 
    # pClusterPathway(**kargs)

    # 2. combinations
    
    otypes = ['total', ]  # n-gram with ordering considered? {'partial', 'total',}
    ctypes = ['diagnosis', 'medication', 'mixed',]  # {'diagnosis', 'medication', 'mixed'}
    ptypes = ['prior', ]   # {'prior', 'noop', 'posterior'} | noop: complete sequence  
    test_combo = ['partial', 'diagnosis', 'prior']
    
    # for params in test_combo:
    for params in itertools.product(otypes, ctypes, ptypes): 
        order_type, content_type, policy_type = params[0], params[1], params[2] 

        # time series 
        pClusterPathwayLCSTimeSeries(ctype=content_type, otype=order_type, ptype=policy_type, lcs_selection_policy='freq')

        # find topn represenatative short-chain n-grams (n=1~4)? 
        pClusterPathwayLength(topn_per_length=20, ctype=content_type, otype=order_type, ptype=policy_type)

    ### cluster distribution: frequency distributions of n-grams across clusters (and how they're compared to the global scope)
    evalMotifFreq(**kargs)

    # cluster n-gram uniqueness (order type is 'marginalized')
    for params in itertools.product(ctypes, ptypes): 
        content_type, policy_type = params[0], params[1]
        evalClusterRecall(ctype=content_type, ptype=policy_type)

    return 

def t_evaluation(**kargs): 

    # [params]
    cohort_name = kargs.get('cohort', 'PTSD')
    basedir = seqparams.get_basedir(cohort=cohort_name) 

    # [params] cluster, class 
    n_clusters = kargs.get('n_clusters', 10)
    n_classes = kargs.get('n_classes', 1)
    cIDs = range(0, n_clusters)   # CID

    # [params] file ID
    content_type = ctype = kargs.get('ctype', 'diagnosis') # content type (what's in the sequence?): medication, mixed
    order_type = otype = kargs.get('otype', 'total')
    policy_type = ptype = kargs.get('ptype', 'prior')  # cut point: pre-diagnosis (prior), post-diagnosis (posterior), whole sequence (mixed)
    cluster_method = kargs.get('cluster_method', 'kmeans')
    seq_ptype = kargs.get('seq_ptype', seqparams.normalize_ptype(ctype))  # diag, regular, med
    d2v_method = kargs.get('d2v_method', 'tfidfavg')

    kargs['filter_by_global_usage'] = True
    evalClusterDiff(**kargs)

    return

def test(**kargs): 

    # full stages 
    t_pipeline(**kargs)

    return

def test2(**kargs): 
    """
    
    Memo
    ----
    (*) Questions that distinguish Clusters
        four main questions that char a cluster 
        1. hist of codes vs freq (evalMotifFreq() -> plot_horizontal_bar2)
            + short chain 
            + long chain
        2. list of lcs (pClusterPathwayLCS() -> {pathway_cluster-lcs, pathway_cluster-lcs_stats})
        3. example time series mentioned earlier freq vs time 

        4. bigram cooccurrence 
    """
    # Selective stages for pathway analysis from the template t_pipeline()
    # t_selected_stage(**kargs)

    t_evaluation(**kargs)

    return

if __name__ == "__main__": 
    # test()
    test2()  
