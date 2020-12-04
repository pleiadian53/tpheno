# encoding: utf-8

# import seqMaker2 as smk2  # this interfaces DB and create time series documents
import seqReader as sr    # this intefaces time series documents themselves
import seqAnalyzer as sa 
import seqCluster as sc

# [todo] __init__
import pandas as pd
import numpy as np
from pandas import DataFrame
import os, gc, sys, random 
from os import getenv 
import time, re
import timeit
try:
    import cPickle as pickle
except:
    import pickle

# local modules 
from batchpheno import icd9utils, sampling, qrymed2
from config import seq_maker_config, sys_config
from batchpheno.utils import div
from pattern import medcode as pmed
import seqparams
import analyzer, vector

from cluster import analyzer as canalyzer

###################################################################
#  
#  Sampling utilities specifically for sequence data. 
# 
#  For general sampling utitlies, use tpheno.sampler module. 
#  
#
#


# [todo] factor to seqpamaters
def get_default_codes(**kargs):
    from batchpheno import icd9utils

    # 1. diabetes without complications 
    code_def = "24900 25000 25001 7902 79021 79022 79029 7915 7916 V4585 V5391 V6546"

    # codes = icd9utils.preproc_code(code_def, base_only=True, no_ve_code=True) # loose 
    codes = icd9utils.preproc_code(code_def, base_only=False, no_ve_code=True) # exact

    # 2. Diabetes mellitus with complications
    # code_def = """24901 24910 24911 24920 24921 24930 24931 24940 24941 24950 24951 24960 24961 24970 24971 24980 24981 24990 24991 25002
    #               25003 25010 25011 25012 25013 25020 25021 25022 25023 25030 25031 25032 25033 25040 25041 25042 25043 25050 25051 25052
    #               25053 25060 25061 25062 25063 25070 25071 25072 25073 25080 25081 25082 25083 25090 25091 25092 25093"""
    # codes = icd9utils.preproc_code(code_def, base_only=False, no_ve_code=True)


    # 3. childbirth
    # code_def = "64800 64801 64802 64803 64804 64880 64881 64882 64883 64884"

    return codes

def run_clutering(X=None, y=None, cluster_method='kmeans', **kargs): 
    if cluster_method.startswith('km'): 
        return sc.cluster_kmeans(X=None, y=None, **kargs)
    else:  
        NotImplementedError
    return 

def get_cluster_representative(**kargs):  # [subsumed]
    """

    Related 
    -------
    seqUtils.sample_class2()
       Run clustering first to get cluster_labels, which can then be passed as 'y'

    """

    basedir = sys_config.read('DataExpRoot') 

    # [params] cluster sampling
    n_sample = kargs.get('n_sample', 100)
    n_doc = kargs.get('n_doc', None) # only used for debugging

    cluster_method = kargs.get('cluster_method', 'kmeans')  # hc, etc.  # only for cluster-based sampling
    force_clustering = kargs.get('force_clustering', False) # force re-computing the clustering

    f_cluster_id = 'cluster_id'
    f_data = 'data'

    read_mode = kargs.get('mode', kargs.get('read_mode', 'doc')) # or 'seq'
    seq_ptype = kargs.get('seq_ptype', 'regular') # sequence pattern type: regular, random, diag, med, lab

    # see cluster_kmeans(), cluster_hc() in seqCluster 
    default_ifile = '%s_map-%s.csv' % (cluster_method, seq_ptype) # linked to seqCluster.build_data_matrix()
    ifile = kargs.get('ifile', default_ifile)
    fpath = os.path.join(basedir, ifile)
    if force_clustering or not os.path.exists(fpath): 
        div(message="Clustering-on-document operation has not been performed | clustering method: %s" % cluster_method, symbol='%')
        if cluster_method.startswith('km'): 
            sc.cluster_kmeans(X=None, y=None, **kargs)
        else:  
            NotImplementedError
        assert os.path.exists(fpath)

    df = pd.read_csv(fpath, sep=',', header=None, index_col=False, error_bad_lines=True)  
    print('info> loaded %s-cluster map file of dim: %s from:\n%s\n' % (cluster_method, str(df.shape), fpath))
    assert not df.empty
    if n_doc is not None: 
        assert df.shape[0] == n_doc, "numbers of documents are not consistent: input %d vs hint %s" % (df.shape[0], n_doc)

    # [assumption] doc positions ~ build_data_matrix() ~ sa.loadModel ~ sa.read
    return sampling.sample_cluster(df[f_cluster_id].values, n_sample=n_sample)


# [todo] merge into read routine
def get_representative(docs=None, **kargs): 
    """
    Given input sequences/docs, filter out docs that do not satisfy 
    representativeness criteria: 
       1. mininal frequency of code mentions 
       2. a set of codes 

    Memo
    ----
    1. Diabetes with complications 
       24900 25000 25001 7902 79021 79022 79029 7915 7916 V4585 V5391 V6546

    2. Diabetes mellitus with complications 
       24901 24910 24911 24920 24921 24930 24931 24940 24941 24950 24951 24960 24961 24970 24971 24980 24981 24990 24991 25002
       25003 25010 25011 25012 25013 25020 25021 25022 25023 25030 25031 25032 25033 25040 25041 25042 25043 25050 25051 25052
       25053 25060 25061 25062 25063 25070 25071 25072 25073 25080 25081 25082 25083 25090 25091 25092 25093 
    """
    policy = kargs.get('policy', 'cluster')
    target_codes = kargs.get('codes', kargs.get('targets', get_default_codes(**kargs))) # diabetes with or without complications
    topn, min_count = 10, 3
    n_sample = kargs.get('n_sample', 100)
    
    read_mode = kargs.get('mode', kargs.get('read_mode', 'doc')) # or 'seq'
    seq_ptype = kargs.get('seq_ptype', 'regular') # sequence pattern type: regular, random, diag, med, lab

    # load sequence 
    n_doc = None if docs is None else len(docs)
    if docs is None and policy.startswith(('diag', 'code', )): 
        if read_mode.startswith('doc'): 
            docs = read_doc(load_=True, simplify_code=False, seq_ptype=seq_ptype)
        else: 
            docs = read(load_=True, simplify_code=False, seq_ptype=seq_ptype)
        n_doc = len(docs)
        if kargs.has_key('n_doc'): assert kargs['n_doc'] == n_doc, "expected %d documents but got %d" % (kargs['n_doc'], n_doc)
    assert n_doc is not None
    
    # [params] cluster sampling
    cluster_method = kargs.get('cluster_method', 'kmeans')  # hc, etc.  # only for cluster-based sampling
    force_clustering = kargs.get('force_clustering', False)
    f_cluster_id = 'cluster_id'
    f_data = 'data'

    if policy.startswith('cluster') and force_clustering: 
        run_clutering(cluster_method=cluster_method)

    # frequency-based candidate criteria
    # 1. among the top 10 (most frequent) diagnostic codes, targets are included among them
    # 2. min_count: target codes appear more than min_count (v)
    candidates = []
    if policy.startswith(('diag',)): # use a set of diagnostic codes collectively as a criteria
        # target_docs = [] 
        # for doc in docs: 
        #     counter = collections.Counter(doc)
        #     for tc in target_codes: 
        #         # min-count criteria
        raise NotImplementedError

    elif policy.startswith('cluster'): # cluster representative 
        candidates = get_cluster_representative(**kargs)

    elif policy.startswith('rand'):
        candidates = random.sample(range(n_doc), n_sample)
    
    return candidates 
