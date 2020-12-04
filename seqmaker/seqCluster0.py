# encoding: utf-8

import matplotlib
matplotlib.use('Agg') # use a non-interactive backend such as Agg (for PNGs), PDF, SVG or PS.
from matplotlib import pyplot as plt
import matplotlib.cm as cm  # silhouette test
import seaborn as sns

# from gensim.models import doc2vec
from collections import namedtuple
import collections

import csv
import re
import string
import sys, os, random, gc 

from pandas import DataFrame, Series
import pandas as pd
import numpy as np

try:
    import cPickle as pickle
except:
    import pickle

# doc2vec experiments 
# from gensim.models import Doc2Vec
# import gensim.models.doc2vec
# from collections import OrderedDict

# tpheno modules (assuming that PYTHONPATH has been set)
from batchpheno import icd9utils, qrymed2, utils, dfUtils  # batchpheno.sampling is obsolete
from config import seq_maker_config, sys_config
from batchpheno.utils import div
from pattern import medcode as pmed

import sampler  # sampling utilities
import seqparams
import analyzer, vector
# import seqAnalyzer as sa 
import labeling 
import seqUtils, plotUtils


import evaluate  # classification
import algorithms, seqAlgo  # count n-grams, sequence-specific algorithms
# from seqparams import TSet

# multicore processing 
import multiprocessing

# clustering algorithms 
from sklearn.cluster import KMeans, MiniBatchKMeans, AgglomerativeClustering, AffinityPropagation, SpectralClustering, DBSCAN
# from sklearn.cluster import KMeans

# from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import kneighbors_graph

# from sklearn.cluster import AffinityPropagation

# evaluation 
from sklearn import metrics
from scipy.spatial import distance
from sklearn.neighbors import NearestNeighbors  # kNN

##### Set module variables ##### 
GNFeatures = vector.W2V.n_features # [note] descendent: seqparams.W2V -> vector.W2V
GWindow = vector.W2V.window
GNWorkers = vector.W2V.n_workers
GMinCount = vector.W2V.min_count

def print_cluster(word_centroid_map, n_clusters=10):
    def to_str(seq, sep=','): 
        return sep.join(str(e) for e in seq)

    cidx = word_centroid_map.values() # cluster ids
    widx = word_centroid_map.keys()  # words 

    # For the first 10 clusters
    print('\n')

    n_displayed = 0
    n_map = len(word_centroid_map)

    # labeling
    cutoff = 3
    lsep = '_'

    for cluster in range(0, n_map):
        if n_displayed >= n_clusters: break

        # Find all of the words for that cluster number, and print them out
        # word_centroid_map: word2index => cluster id
        words = []
        
        # linear search
        for i in range(n_map):  # foreach entry
            if ( cidx[i] == cluster ):  # find all words that belong to this cluster
                words.append(widx[i])

        # label
        # label = labelSeq(words, topn=1)
        clist = [str(word) for word in words if pmed.isICD(word)][:cutoff]
        label = lsep.join(clist) 
        if clist: 
            print "Cluster #%d (label %s):\n    > %s\n" % (cluster, label, to_str(words))
            n_displayed += 1 

    if n_displayed < n_clusters: 
        for cluster in range(0, n_clusters-n_displayed): 
            
            # linear search
            words = [widx[i] for i in range(n_map) if cidx[i] == cluster]  

            # label 
            # mlabel = labelSeq(words, topn=3)
            mlabel = words[0]
            print "Cluster #%d (label %s):\n    > %s\n" % (cluster, mlabel, to_str(words)) 

    return

def labelSeq(sequence, **kargs):
    def to_str(seq, sep=','): 
        return sep.join(str(e) for e in seq)
    
    label = 'unknown'
    topn = kargs.get('topn', 1) # use topn most frequent elements as label
    sortby = kargs.get('sortby', 'freq')  
    lsep = '_'

    # [policy] frequency 
    counter = collections.Counter(sequence) 
    ltuples = counter.most_common(topn)
    
    if len(sequence) > 0: 
        if topn == 1: 
            label = ltuples[0][0]
        else: 
            if sortby.startswith('freq'): 
                sl = sorted(ltuples, key=lambda x: x[1], reverse=True) # sort ~ count from high to low
                label = to_str([l[0] for l in sl], sep=lsep)
            else: 
                label = to_str(sorted([l[0] for l in ltuples]), sep=lsep)

    return label

def makeD2VLabels(sequences, **kargs): 
    """
    Label sequences/sentences for the purpose of using Doc2Vec. 

    Related 
    -------
    * labelDoc()

    """ 
    import labeling
    return labeling.makeD2VLabels(sequences, **kargs)

def getDocVec(docs, **kargs):  # model will be stored ~ outputdir
    """
    Input: documents (where each document is a list of tokens)
    Output: document vectors

    Params
    ------
    d2v_method
    outputdir: the directory in which d2v model is saved. 

    """
    # import vector

    # [params] main
    cohort_name = kargs.get('cohort', 'diabetes') # 'diabetes', 'PTSD'
    kargs['outputdir'] = kargs.get('outputdir', seqparams.getCohortLocalDir(cohort=cohort_name))

    # [params] d2v_method, (w2v_method), test_, load_model 
    return vector.getDocVec(docs, **kargs)  # [output] docuemnt vectors
def getDocVecModel(labeled_docs, **kargs):
    """
    Input: documents (where each document is a list of tokens)
    Output: d2v model (NOT document vectors, for which use getDocVec())
    """
    return vectorize2(labeled_docs, **kargs) # [output] model (d2v)

def vectorize_d2v(labeled_docs, **kargs):
    return vectorize2(labeled_docs, **kargs)
def vectorize2(labeled_docs, **kargs): # wrapper on top of vector.vectorize2() 
    """
    Compute sentence embeddings. 


    Input 
    ----- 
    labeled_doc: labeled sequences (i.e. makeD2VLabels() was invoked)

    Output
    ------
    d2v model

    Related 
    -------
    seqAnalyzer.vectorize (word2vec)

    Memo
    ----
    1. Example settings 

        # PV-DM w/concatenation - window=5 (both sides) approximates paper's 10-word total window size
        Doc2Vec(dm=1, dm_concat=1, size=100, window=5, negative=5, hs=0, min_count=2, workers=cores),
        
        # PV-DBOW 
        Doc2Vec(dm=0, size=100, negative=5, hs=0, min_count=2, workers=cores),
        
        # PV-DM w/average
        Doc2Vec(dm=1, dm_mean=1, size=100, window=10, negative=5, hs=0, min_count=2, workers=cores),

    """
    # from gensim.models import Doc2Vec

    # [params]
    cohort_name = kargs.get('cohort', 'diabetes') # 'diabetes', 'PTSD'
    # seq_compo = composition = kargs.get('composition', 'condition_drug')
    

    # [params] d2v model: use seqparams and vector
    d2v_method = vector.D2V.d2v_method
    vector.D2V.show_params()

    # [params] IO
    outputdir = basedir = kargs.get('outputdir', seqparams.getCohortDir(cohort=cohort_name))
    

    # Mikolov pointed out that to reach a better result, you may either want to shuffle the 
    # input sentences or to decrease the learning rate alpha. We use the latter one as pointed
    # out from the blog provided by http://rare-technologies.com/doc2vec-tutorial/

    # negative: if > 0 negative sampling will be used, the int for negative specifies how many "noise words"
    #           should be drawn (usually between 5-20).

    model = vector.getDocModel(labeled_docs, outputdir=outputdir) # [output] model (d2v)

    # [test]
    tag_sample = [ld.tags[0] for ld in labeled_docs][:10]
    print('verify> example tags (note: each tag can be a string or a list):\n%s\n' % tag_sample)
    # [log] ['V70.0_401.9_199.1', '746.86_426.0_V72.19', '251.2_365.44_369.00', '362.01_599.0_250.51' ... ] 

    return model

def phenotypeDoc(sequences=None, **kargs):
    """
    Input
    -----
    documents of symbolic time sequences (list of lists or 2D array of strings)

    Output
    ------
    labeled documents/sequences
    labels/label sets are in a 3-digit format of {0: False, 1: True}
        [type I?, type II?, gestational?] 

    """
    def save(fname=None): 
        # header = ['type_1', 'type_2', 'gestational']
        # given labelsets 
        adict = {h:[] for h in header}
        for i, lset in enumerate(labelsets): 
            adict['type_1'].append(lset[0])
            adict['type_2'].append(lset[1])
            adict['gestational'].append(lset[2])

        df = DataFrame(adict, columns=header)
        if fname is None: 
            fname = '%s_labels.%s' % (doc_basename, doctype)
        fpath = os.path.join(basedir, fname)
        print('output> saving labels to %s' % fpath)
        df.to_csv(fpath, sep=fsep, index=False, header=True)
        return df
    def load(fname=None):  
        # header = ['type_1', 'type_2', 'gestational']
        if fname is None: 
            fname = '%s_labels.%s' % (doc_basename, doctype)
        fpath = os.path.join(basedir, fname)
        if not os.path.exists(fpath): 
            print('load> label set does not exist at %s > recomputing label sets' % fpath)
            return []
        df = pd.read_csv(fpath, sep=fsep, header=0, index_col=False, error_bad_lines=True)
        assert df is not None and not df.empty 

        labelsets = []
        for r, row in df.iterrows(): 
            labelsets.append(tuple(row.values))
        print('output> loaded %d label sets' % len(labelsets))
        return labelsets

    from pattern import diabetes as diab

    # [todo] given a list of composition, create its string regr for file naming
    seq_compo = kargs.get('composition', 'condition_drug')
    read_mode = kargs.get('read_mode', 'doc')  # doc: per-patient documents; seq: per-visit documents/sentences
    seq_ptype = seqparams.normalize_ctype(**kargs) # values: regular, random, diag, med, lab ... default: regular
    cohort_name = kargs.get('cohort', 'diabetes')

    # identifier: seq_compo + seq_ptype + cohort 
    if not seq_ptype.startswith('reg'):
        doc_basename = '%s-%s' % (doc_basename, seq_ptype)
    if cohort_name is not None: 
        doc_basename = '%s-%s' % (doc_basename, cohort_name)

    identifier = doc_basename   # use identifier for file naming instead [todo]
    simplify_code = kargs.get('simplify_code', False)
    save_label = kargs.get('save_', True)
    doctype = 'csv'

    # header = ['type_1', 'type_2', 'gestational']
    basedir = sys_config.read('DataExpRoot')
    fsep = ','

    if kargs.get('load_', True): 
        labelsets = load()  # access doc_basename
        if len(labelsets) > 0: 
            return labelsets

    # [input]
    if sequences is None: 
        # [note] sa.read takes care of the ifiles (i.e. appropriate medical sequence input files)
        sequences = sa.read(load_=False, simplify_code=simplify_code, mode=read_mode, seq_ptype=seq_ptype, cohort=cohort_name)

    n_doc = len(sequences)
    print('phenotype> read %d doc of type %s' % (n_doc, seq_ptype))
    
    if cohort_name is None or cohort_name.startswith('dia'):  # diabetes 
        labelsets = diab.phenotypeDoc(sequences, **kargs)
    else: # one class 
        print('phenotype> Assuming we are in one-class problem (e.g. PTSD cohort without particular subtypes) ...')
        labelsets = []
        save_label = False

    if save_label: save()

    return labelsets

def labelBy(sequences, **kargs):  # [refactor] labeling module 
    """

    Related
    -------
    phenotypeDoc(sequences=None, **kargs)
      : returns label sets in the order of the input sequendes
    """
    policy = kargs.pop('policy', 'frequency')
    if policy.startswith('freq'):
        return labelDocsByFreq(sequences, **kargs) 
    elif policy.startswith('med'):  # [todo]
        raise NotImplementedError
    else: 
        print('default> use the diag code frequency for labeling ...')
    return labelDocsByFreq(sequences, **kargs) 
def labelDocsByFreq(sequences=None, **kargs):
    return labelDoc(sequences, **kargs)
def labelDoc(sequences=None, **kargs):  # refactor this to labeling
    """
    Label (patient) documents via heuristics (e.g. highest code frequencies) 

    Output
    ------
    1. df(label, sequence)
       where label is in diagnostic code-based multilabel format and 
             sequence constains only a subset of the orignial sequence 
             with length determined by 'topn_repr'

    """
    def to_str(seq, sep=','): 
        return sep.join(str(e) for e in seq)

    def save(adict, header, fname=None): 
        df = DataFrame(adict, columns=header)
        if fname is None: 
            fname = '%s_labeled.%s' % (doc_basename, doctype)
        fpath = os.path.join(basedir, fname)
        print('output> saving %s' % fpath)
        df.to_csv(fpath, sep=fsep, index=False, header=True)
        return df

    import seqAnalyzer as sa

    seq_compo = composition = kargs.get('composition', 'condition_drug')
    read_mode = kargs.get('read_mode', 'doc')  # doc: per-patient documents; seq: per-visit documents/sentences
    seq_ptype = seqparams.normalize_ctype(**kargs) # values: regular, random, diag, med, lab ... default: regular
    cohort_name = kargs.get('cohort', 'diabetes')

    doc_basename = seq_compo if seq_ptype.startswith('reg') else '%s-%s' % (seq_compo, seq_ptype) 
    if cohort_name is not None: 
        doc_basename = '%s-%s' % (doc_basename, cohort_name)

    simplify_code = kargs.get('simplify_code', False)
    load_label = kargs.get('load_', False)
    doctype = 'csv' 
    
    basedir = sys_config.read('DataExpRoot')  # [I/O] global data directory (cf: local data directory /data)
    fsep = '|'
    lsep = '_' # label separator (alternatively, '+')

    seqr_type = kargs.get('seqr', 'full') # full vs diag 
    sortby = kargs.get('sortby', 'freq') # sort labels by their frequencies or alphabetic order? only applicable to multilabeling

    # [output] 
    ofile_slabel = '%s_unilabel.%s' % (doc_basename, doctype)
    ofile_mlabel = '%s_multilabel.%s' % (doc_basename, doctype)
    # labels, mlabels = [], []

    # [load results]
    if load_label:
        res = [] #[None] * 2 
        
        for ifile in (ofile_mlabel, ):  # ofile_slabel,
            has_data = True
            fpath = os.path.join(basedir, ifile)
            if not os.path.exists(fpath): 
                print('io> labeled data does not exist yet at %s' % fpath)
                has_data = False
                break

            df = None
            if has_data: 
                df = pd.read_csv(fpath, sep=fsep, header=0, index_col=False, error_bad_lines=True)
                print('io> loaded labeled data (dim=%s) from: %s' % (str(df.shape), fpath))
            
                # labels = df['label'].values
                # seqr = df['sequence'].values
                # d = dict(zip(labels, seqr))

                res.append(df)

        if len(res) > 0: 
            # return res[0]
            return pd.concat(res, ignore_index=True)
        else: 
            div(message='Waring: Could not load pre-computed dataset. Labeling the data now ...', symbol='%')

    # [input]
    if sequences is None: 
        sequences = sa.read(load_=False, simplify_code=simplify_code, mode=read_mode, seq_ptype=seq_ptype, cohort=cohort_name)

    n_doc = len(sequences)
    print('info> read %d doc' % n_doc)
    
    # [policy] label by diagnostic codes

    repr_seqx = [None] * n_doc  # representative sequence (formerly diag_seqx)
    n_anomaly = 0
    if seq_ptype in ('regular', 'random', 'diag', ):  # only use diagnostic codes for labeling in these cases
        for i, sequence in enumerate(sequences): 
            # [c for c in sequence if pmed.isICD(e)] 
            repr_seqx[i] = filter(pmed.isICD, sequence)  # use diagnostic codes to label the sequence
            if len(repr_seqx[i]) == 0: 
                print('warning> No diagnostic code found in %d-th sequence/doc:\n%s\n' % (i, to_str(sequence)))
                n_anomaly += 1 
        div(message='A total of %d out of %d documents have valid diag codes.' % (n_doc-n_anomaly, n_doc), symbol='%')
    elif seq_ptype in ('med', ): 
        for i, sequence in enumerate(sequences): 
            # [c for c in sequence if pmed.isICD(e)] 
            repr_seqx[i] = filter(pmed.isMed, sequence)
            if len(repr_seqx[i]) == 0: 
                print('warning> No med code found in %d-th sequence/doc:\n%s\n' % (i, to_str(sequence)))
                n_anomaly += 1 
        div(message='A total of %d out of %d documents have valid med codes.' % (n_doc-n_anomaly, n_doc), symbol='%')
        # div(message='Only examining non-diagnostic codes > seq_ptype: %s' % seq_ptype, symbol='%')

    ### I. single lable 
    sldict = {}  # single-label dictionary

    # save two labeling formats: single label & multilabel format
    # 1a. label + top 10 in whole sequence (.csv) 
    # 1b. label + top 10 in diag sequence (.csv) 
    # 2?. label + whole doc (.txt)

    topn, topn_repr = 1, 10
    header = ['label', 'sequence']
    freq_cseq_map = {h: [] for h in header}  # most frequent from complete sequences
    freq_diag_map = {h: [] for h in header}  # most frequent from diag sequences

    for i, dseq in enumerate(repr_seqx): 
        counter_diag = collections.Counter(dseq)
        counter_full = collections.Counter(sequences[i])

        # labeling 
        if dseq: 
            label = counter_diag.most_common(1)[0][0]
        else: 
            label = 'unknown' # no diagnostic codes
        # labels.append(label)

        # complete sequence
        seqr = to_str([pair[0] for pair in counter_full.most_common(topn_repr)])
        freq_cseq_map['label'].append(label)
        freq_cseq_map['sequence'].append(seqr)

        # diag sequence
        seqr = to_str([pair[0] for pair in counter_diag.most_common(topn_repr)])
        freq_diag_map['label'].append(label)
        freq_diag_map['sequence'].append(seqr)

    # [output]
    # save result 1a
    ofile = '%s-unilabel.%s' % (doc_basename, doctype)
    save(freq_cseq_map, header=header, fname=ofile)
    # save result 1b 
    ofile = '%s-unilabel_diag.%s' % (doc_basename, doctype)
    save(freq_diag_map, header=header, fname=ofile)

    ### II. multi-label 
    mldict = {}  # multi-label dictionary
    topn, topn_repr = 3, 10  # use 'topn' most frequent labels
    # header = ['label', 'sequence']
    freq_cseq_map = {h: [] for h in header}  # most frequent from complete sequences
    freq_diag_map = {h: [] for h in header}  # most frequent from diag sequences

    for i, dseq in enumerate(repr_seqx): 
        counter_diag = collections.Counter(dseq)
        counter_full = collections.Counter(sequences[i])

        # labeling 
        # topn_eff = min(len(dseq), topn)  # topn should be smaller than total length 
        if dseq: 
            # use frequent diag codes as label 
            ltuples = counter_diag.most_common(topn)  # if topn > # unique token, will only show topn (symbol, count)-tuples
            # if len(ltuples) < topn: 
               #  print('warning> Diag sequence only has %d unique labels while topn=%d > dseq: \n%s\n' % \
                  #   (len(ltuples), topn, to_str(dseq)))

            # sort according to frequencies? 
            if sortby.startswith('freq'): 
                sl = sorted(ltuples, key=lambda x: x[1], reverse=True) # sort ~ count from high to low
                label = to_str([l[0] for l in sl], sep=lsep)
            else: 
                label = to_str(sorted([l[0] for l in ltuples]), sep=lsep)
        else: 
            label = 'unknown'

        # mlabels.append(label)

        # complete sequence
        seqr = to_str([pair[0] for pair in counter_full.most_common(topn_repr)])
        freq_cseq_map['label'].append(label)
        freq_cseq_map['sequence'].append(seqr)

        # diag sequence
        seqr = to_str([pair[0] for pair in counter_diag.most_common(topn_repr)])
        freq_diag_map['label'].append(label)
        freq_diag_map['sequence'].append(seqr)

    # [output]
    # save result 1a
    ofile = '%s-multilabel.%s' % (doc_basename, doctype)
    df = save(freq_cseq_map, header=header, fname=ofile) # content shows all (frequent) medical codes
    # save result 1b 
    ofile = '%s-multilabel_diag.%s' % (doc_basename, doctype)
    df2 = save(freq_diag_map, header=header, fname=ofile)  # content shows only (frequent) diagnostic codes

    # return (labels, mlabels)
    if seqr_type.startswith('full'): 
        return df 
    return df2

def load_w2v(**kargs): 
    pass 
def load_d2v(**kargs): 
    pass

def t_cluster(**kargs):
    """
    Create clusters by mapping medical codes to clusters. 

    Scheme: Cluster of clusters

    Related 
    -------
    * build_data_matrix() 
    * map_clusters() 
    * config_doc2vec_model

    Memo
    ----
    * parameters for loading model 
        base_only:  
        load_model: load the pre-computed word2vec model
        load_seq: load the processed sequences
        load_lookuptb: load symbol lookup table (takes long to compute due to querying via REST)

        simplify_code 
    """ 
    def to_str(alist, sep=','):  
        return sep.join(str(e) for e in alist)

    def display_cluster(adict, n_sample=None): # where adict: cluster id => members 
        if n_sample is not None: 
            # change on adict is not going to affect the input 'adict' because new copy is created
            adict2 = utils.sample_dict(adict, n_sample=n_sample) # can assign return value to adict 
        for k, vals in adict2.items(): 
            # div(message='Cluster (%d):\n%s\n' % (k, v))
            msg = 'Cluster (%s):\n' % k 
            for v in vals: 
                msg += '    + %s\n' % v 
            div(message=msg, symbol='*', adaptive=False)
        return

    # from sklearn.cluster import KMeans
    import time
    import seqAnalyzer as sa

    # [params]
    cohort_name = kargs.get('cohort', 'diabetes')
    seq_compo = composition = kargs.get('composition', 'condition_drug')
    read_mode = 'doc'  # assign 'doc' (instead of 'seq') to form per-patient sequences
    seq_ptype = kargs.get('seq_ptype', 'regular') # sequence pattern type: regular, random, diag, med, lab

    # [params] training 
    n_features = kargs.get('n_features', GNFeatures)
    window = kargs.get('window', GWindow)
    min_count = kargs.get('min_count', GMinCount)

    n_cores = multiprocessing.cpu_count()
    # print('info> number of cores: %d' % n_cores)
    n_workers = kargs.get('n_workers', GNWorkers)

    # [params] load pre-computed data
    load_word2vec_model = kargs.get('load_w2v', True)
    test_model = kargs.get('test_model', False)
    load_lookuptb = kargs.get('load_lookuptb', True)
    load_token_cluster = kargs.get('load_token_cluster', True) and load_word2vec_model
    load_doc_cluster = kargs.get('load_doc_cluster', True) and load_word2vec_model # per-patient document clusters
    load_visit_cluster = kargs.get('load_visit_cluster', False) and load_word2vec_model # per-visit sentence clusters
    load_label = kargs.get('load_label', True)
    
    load_doc2vec_model = kargs.get('load_d2v', True) # simple (first attempt) doc2vec model; induced by vectorize2

    # [params] cluster document
    doctype = 'txt' 
    doc_basename = seq_compo if seq_ptype.startswith('reg') else '%s-%s' % (seq_compo, seq_ptype) 
    if cohort_name is not None: 
        doc_basename = '%s-%s' % (doc_basename, cohort_name)

    # [params]
    basedir = outputdir = sys_config.read('DataExpRoot')

    # [params] test 

    # [params] document labeling 
    lsep = seqparams.lsep  # label separator (e.g. '_')

    # [input]
    # read_mode: {'seq', 'doc', 'csv'}  # 'seq' by default
    # cohort vs temp doc file: diabetes -> condition_drug_seq.dat
    ifile = '%s_seq-%s.dat' % (seq_compo, cohort_name) if cohort_name is not None else '%s_seq.dat' % seq_compo  
    result = sa.loadModel(n_features=n_features, window=window, min_count=min_count, n_workers=n_workers, 
                            seq_ptype=seq_ptype, read_mode=read_mode, test_model=test_model, 
                                ifile=ifile, cohort=cohort_name, bypass_lookup=True,
                                load_seq=False, load_model=load_word2vec_model, load_lookuptb=load_lookuptb) # attributes: sequences, lookup, model
    
    if load_word2vec_model: 
        div(message='Successfully loaded learned model (cohort=%s) ... :)' % cohort_name, symbol='%')

    # [params]
    # codeset = t_diabetes_codes(simplify_code=False)
    # codeset_root = t_diabetes_codes(simplify_code=True)
    # n_root = len(codeset_root)
    # topn = 3

    # example symbol vectors 
    map_tokens = result['token_type']
    c_tokens = map_tokens['diag']  # Condition tokens 
    p_tokens = map_tokens['drug']  # Prescription tokens
    # l_tokens = map_tokens['lab']
    o_tokens = map_tokens['other']  # Other tokens
    n_diag, n_drug, n_other = len(c_tokens), len(p_tokens), len(o_tokens)

    # for code in random.sample(condition_tokens, min(3, n_diag)): 
    #     print('code: %s -> fisrt 5 vector components:\n%s\n' % (code, model[code][:5]))
   
    # compute averaged feature vector for each patient document
    sequences = result['sequences']
    model = result['model']
    lookuptb = result['symbol_chart']

    print('t_cluster> number of documents: %d' % len(sequences))
    # avgfvec = analyzer.getAvgFeatureVecs(sequences, model, n_features)

    # Set "k" (num_clusters) to be 1/5th of the vocabulary size, or an
    # average of 5 words per cluster
    word_vectors = model.syn0  # a word is a medical code here
    print('info> number of word vectors: %d, dim: %s' % (len(word_vectors), word_vectors.shape[1]))  # [log] number of word vectors: 200310

    # [params] stats 
    n_doc = len(sequences)

    n_tokens = word_vectors.shape[0]
    n_clusters = min(word_vectors.shape[0] / 10, 20)
    print('info> n_tokens: %d > n_doc: %d > n_clusters: %d' % (n_tokens, n_doc, n_clusters))

    # index 2 words 
    print('test> index to words(100):\n%s\n' % model.index2word[:100])
    # [log] test> index to words(100): ['401.9', '250.00', '62439', 'V22.1', 'V65.44', 'unknown', '61895', '62934', ... ] 
    
    # [params][output]
    token_cluster_file = word_centroid_map_file = 'token_cluster-w%sc%s.pkl' % (n_tokens, n_clusters)
    doc_cluster_file = '%s_cluster.%s' % (doc_basename, 'csv')
    doc_cluster_file_labeled = '%s_cluster_labeled.%s' % (doc_basename, 'csv')
    # doc_label_centroid_map_file = 'doc_label_centroid_map.pkl'
    doc_mlabel_centroid_map_file = 'doc_mlabel_centroid_map.pkl'
    doc_sqr_centroid_map_file = 'doc_sqr_centroid_map_file.pkl'
    cluster_method = 'kmeans'

    div(message='Clustering word vectors ...', symbol='*')
    
    fpath = os.path.join(basedir, token_cluster_file)
    use_precomputed_data = load_token_cluster and os.path.exists(fpath)
    word_centroid_map = {}

    if use_precomputed_data: 
        word_centroid_map = pickle.load(open(fpath, 'rb'))
        print('input> loaded symbol map of %d entries' % len(word_centroid_map))

        assert n_clusters == max( word_centroid_map.values() ) + 1
    else: 
        start = time.time() # Start time
        
        # Initalize a k-means object and use it to extract centroids
        kmeans_clustering = KMeans( n_clusters = n_clusters )
        idx = kmeans_clustering.fit_predict( word_vectors )

        end = time.time()
        elapsed = end - start
        print "Time taken for K Means clustering: ", elapsed, "seconds."

        # Create a Word / Index dictionary, mapping each vocabulary word to
        # a cluster number    

        # [output]   
        # assuming that same (repeated) word belong to the same cluster (id)                                                                                     
        word_centroid_map = dict(zip( model.index2word, idx )) # word -> cluster ID
        if kargs.get('save_', True): 
            fpath = os.path.join(outputdir, token_cluster_file)
            print('output> saving token clusters to %s' % fpath)
            pickle.dump(word_centroid_map, open(fpath, "wb" ))

    assert len(word_centroid_map) > 0
    print_cluster(word_centroid_map, n_clusters=15)

    ### Re-express symbolic time series in terms of cluster centroids
    
    # labels, mlabels = labelDoc(sequences, load_=load_label)  # [input]
    div(message='Labeling document using heuristics (e.g. frequency) ')

    # [note] seqr='full': use all medical codes (but most frequent) to represent sequences
    #        sortby='freq': sort label according to 'label frequencies'
    df_ldoc = labelDocsByFreq(sequences, load_=load_label, seqr='full', sortby='freq', seq_ptype=seq_ptype)
    mlabels = list(df_ldoc['label'].values)
    labels = [mlabel.split(lsep)[0] for mlabel in mlabels] # take most frequent (sub-)label as the label
    print('verify> labels (dtype=%s): %s' % (type(labels), labels[:10]))

    # consider each sequence as a doc 
    train_centroids = np.zeros( (n_doc, n_clusters), dtype="float32" )  # very sparse!
    for i, sequence in enumerate(sequences):  # [log] 432000 documents
        fset = vector.create_bag_of_centroids( sequence, word_centroid_map )
        train_centroids[i] = fset  # [c1, c2, c2] = [freq1, freq2, freq3]

    # [output][byproduct] generate feature representation for external library e.g. ClusterEng
    # feature vectors based on k-means clustering word embedding
    fpath = os.path.join(outputdir, 'fset_%s_%s.csv' % (cluster_method, n_clusters)) # very sparse! 
    df = DataFrame(train_centroids)
    print('verify> sparse dataframe > average active values: %f' % dfUtils.getAvgEffAttributes(df, trivial_val=0))
    df.to_csv(fpath, sep=',', index=False, header=False)
    print('byproduct> saving (%s-) clustered feature set to %s' % (cluster_method, fpath))

    # [output][byproduct]
    assert len(labels) == df.shape[0]
    df.index = mlabels   # assign labels to data points   
    fpath = os.path.join(outputdir, 'fset_%s_%s_labeled.csv' % (cluster_method, n_clusters)) # very sparse! 
    df.to_csv(fpath, sep=',', index=True, header=False)   
    print('byproduct> saving (%s-) clustered feature set (labeled) to %s' % (cluster_method, fpath))
    df_subset = df.sample(min(10000, df.shape[0]))
    fpath = os.path.join(outputdir, 'fset_%s_%s_labeled_subset.csv' % (cluster_method, n_clusters)) 
    df_subset.to_csv(fpath, sep=',', index=True, header=False) 
    print('byproduct> saving a SUBSET (size=%d) of (%s-) clustered feature set (labeled) to %s' % (df_subset.shape[0], cluster_method, fpath))

    # [output] also generate feature representation for external library e.g. ClusterEng
    fpath = os.path.join(basedir, doc_sqr_centroid_map_file)
    idx = []
    n_doc_clusters = min(n_doc / 10, 100)
    if load_doc_cluster and os.path.exists(fpath): 
        # [i/o]
        # df = pd.read_csv(fpath, sep=',', header=None, index_col=False, error_bad_lines=True) # no header
        # assert df.shape[0] == n_doc and df.shape[1] == n_clusters
        # train_centroids = df.values
        doc_mlabel_centroid_map = pickle.load(open(os.path.join(basedir, doc_mlabel_centroid_map_file), "rb"))
        doc_sqr_centroid_map = pickle.load(open(os.path.join(basedir, doc_sqr_centroid_map_file), "rb"))
        idx = [e[1] for e in doc_mlabel_centroid_map]
    else: 
        # cluster documents 
        div(message='Clustering re-expressed documents in cluster centroids ...', symbol='*')
        start2 = time.time() # Start time
    
        print('info> n_doc_clusters: %d' % n_doc_clusters)
        
        kmeans_clustering = KMeans( n_clusters = n_doc_clusters )
        idx = kmeans_clustering.fit_predict( train_centroids )

        # [test]
        print('verify> idx (dtype=%s): %s' % (type(idx), idx[:10]))

        # [output]
        # repeated labels could have different underlying contents and thus correspond to different clusters
        # doc_label_centroid_map = zip( labels, idx ) # word -> cluster ID
        doc_mlabel_centroid_map = zip( mlabels, idx ) # don't use dict() because labels are not necessarily unique
        doc_sqr_centroid_map = zip( list(df_ldoc['sequence'].values), idx )

        pickle.dump(doc_mlabel_centroid_map, open(os.path.join(basedir, doc_mlabel_centroid_map_file), "wb" ))
        pickle.dump(doc_sqr_centroid_map, open(os.path.join(basedir, doc_sqr_centroid_map_file), "wb" ))

        # assuming that labelDoc has been called, and we have obtained labels for each patient doc ...
    
        end2 = time.time()
        elapsed2 = end2 - start2
        print('info> Took %d sec to cluster documents (based on frequency of clusters)' % elapsed2)

    # analyze clusters 
    n_cluster_sample = n_doc_clusters/10
    cidx_set = set(idx) # range(0, n_doc_clusters), random.sample(set(idx), n_cluster_sample)

    doc_mlabel_centroid_map = utils.pair_to_hashtable(doc_mlabel_centroid_map, key=1)
    doc_sqr_centroid_map = utils.pair_to_hashtable(doc_sqr_centroid_map, key=1)

    adict = {}
    for cid in cidx_set: 
        mlx = doc_mlabel_centroid_map[cid] # assuming that ordering is preserved
        seqrx = doc_sqr_centroid_map[cid]
        assert len(mlx) == len(seqrx)
        mlseqr = ['%s: %s' % (e, seqrx[i]) for i, e in enumerate(mlx)]
        adict[cid] = mlseqr  # [format] xyz: xxxyyyuuz

    div(message='Show example clusters (kmeans on cluster_tokens) display_cluster')
    display_cluster(adict, n_sample=n_cluster_sample) # via multiple-(sub)label scheme 
    
    ### document embedding, paragraph vector
    div(message='Test document embedding using paragraph vector ...')

    # [params]
    doc2vec_method = 'PVDM'  # distributed memory
    doctype = 'd2v' 
    doc_basename = 'condition_drug'
    descriptor = kargs.get('meta', doc_basename)  # or algorithm type: PV-DBOW, DM
    # basedir = sys_config.read('DataExpRoot')

    labeled_seqx = makeD2VLabels(sequences=sequences, labels=mlabels)
    # save it?

    model = makeDocVec(labeled_docs=labeled_seqx, load_model=load_doc2vec_model, 
                        seq_ptype=seq_ptype, 
                        n_features=n_features, window=window, min_count=min_count, n_workers=n_workers)
    # [log] output> saving doc2vec models (on 432000 docs) to <prefix>/tpheno/data-exp/condition_drug_test.doc2vec

    div(message='Testing most similar documents ...')  # => t_doc2vec1

    
    return
### end t_cluster()

def getSurrogateLabels(docs, **kargs): 
    import labeling 
    return labeling.getSurrogateLabels(docs, **kargs)

def make_tset_labeled(**kargs):
    raise ValueError, "Redundant: Use makeTSet()"
def make_tset_unlabeled(**kargs): 
    raise ValueError, "Redundant: Use makeTSet()"

def makeTSet0(**kargs): 
    """
    Make training set data (based on w2v and d2v repr) for classification and clusterig
    
    Usage 
    -----
    1. try to call load_tset() first. 

    Related
    -------
    * t_preclassify*
    make_tset_labeled (n_classes >=2)

    Operations 
    --------- 
    loadModel: read (temp docs: D) -> analyze -> vectorize (: D -> w2v)

    Output
    ------ 
    1. identifier: 
         identifier = '%s-%s-%s' % (seq_ptype, w2v_method, d2v_method)
         'tset_nC%s_ID%s-G%s.csv' % (n_classes, identifier, cohort_name)

    """
    def v_index(res, n=5): 
        type1_idx = res['type_1']
        type2_idx = res['type_2']
        type3_idx = res['gestational']
        criteria = [('type_1', diab.has_type1), ('type_2', diab.has_type2), ('gestational', diab.has_gestational), ]

        for j, idx in enumerate([type1_idx, type2_idx, type3_idx, ]):
            for i in random.sample(idx, min(n, len(idx))): 
                assert criteria[j][1](sequences[i]), "%d-th sequence does not satisfy %s criteria!" % (i, criteria[j][0])
        return
    def apply_approx_pheno(docs): 
        # label sequences in a manner that that adheres to the  phenotyping criteria ~ diagnostic codes
        labelsets = phenotypeDoc(sequences=docs, seq_ptype=seq_ptype, load_=load_labelsets)
        if labelsets is None: # no labeling given 
            print('warning> Phenotyping is a no-op for cohort: %s' % cohort_name)
            return {}
        # [params]
        #     res key: ['type_0', 'type_1', 'type_2', 'gestational', 'type_3', 'type_1_2', 'type_1_3', 'type_2_3', 'type_1_2_3']
        res = diab.phenotypeIndex(labelsets)
        v_index(res, n=6)
        return res  # disease types to indices
    def load_fset(fpath, fsep=','):
        if os.path.exists(fpath) and os.path.getsize(fpath) > 0: 
            ts = pd.read_csv(fpath, sep=fsep, header=0, index_col=False, error_bad_lines=True)
            return ts 
        return None

    # import matplotlib.pyplot as plt
    # from sklearn.feature_extraction.text import TfidfVectorizer
    # from pattern import diabetes as diab 
    from seqparams import TSet
    from labeling import TDocTag
    import seqTransform as st
    import seqAnalyzer as sa 
    # import vector

    # [input] cohort, labels, (result_set), d2v_method, (w2v_method), (test_model)
    #         simplify_code, filter_code
    #         

    # [params] cohort   # [todo] use classes to configure parameters, less unwieldy
    composition = seq_compo = kargs.get('composition', 'condition_drug')
    cohort_name = kargs['cohort'] # kargs.get('cohort', 'diabetes')
    # doc_basename = seq_compo if seq_ptype.startswith('reg') else '%s-%s' % (seq_compo, seq_ptype) 
    # if cohort_name is not None: 
    #     doc_basename = '%s-%s' % (doc_basename, cohort_name)

    # [params]
    read_mode = seqparams.TDoc.read_mode  # assign 'doc' (instead of 'seq') to form per-patient sequences
    seq_ptype = seqparams.normalize_ctype(kargs.get('seq_ptype', 'regular')) # sequence pattern type: regular, random, diag, med, lab

    # document embedding wrt all information
    assert seq_ptype in ['regular', ], "seq_ptype=%s > document vectors will not contain complete coding information." % seq_ptype

    # testdir = seqparams.get_testdir() # prefix is None by default => os.getcwd(), topdir='test'

    # assumption: w2v has been completed
    # os.path.join(os.getcwd(), 'data/%s' % cohort_name) 

    docSrcDir = sys_config.read('DataExpRoot')  # document source directory

    # model, and any outputs produced by this module
    
    # [note] dir_type is used to distingush training from test but for cluster analysis, dir_type is 'train' by default
    #        other possible basedir: seqparams.getCohortLocalDir(cohort=cohort_name), sys_config.read('DataExpRoot')
    basedir = outputdir = TSet.getPath(cohort=cohort_name) # [params] dir_type='train', create_dir=True
    print('make_tset> basedir: %s' % basedir)
    
    # [params] D2V use seqparams and vector modules to configure (or simply use their defaults)
    # n_features = kargs.get('n_features', GNFeatures)
    # window = kargs.get('window', GWindow)
    # min_count = kargs.get('min_count', GMinCount)  # set to >=2, so that symbols that only occur in one doc don't count
    # n_cores = multiprocessing.cpu_count()
    # # print('info> number of cores: %d' % n_cores)
    # n_workers = kargs.get('n_workers', GNWorkers)

    # configure vector module 
    # vector.config(n_features=GNFeatures, window=GWindow, min_count=GMinCount)

    # doctype = 'timed'  # ['doc', 'timed', 'labeled']; see seqparams.TDoc

    # coding semantics lookup
    bypass_code_lookup = kargs.get('bypass_lookup', True)  # this is time consuming involving qymed MED codes, etc. 

    # [input] sequences and word2vec model 
    # [note] seqAnalyzer expects a "processed" input sequences (i.e. .pkl or .csv) instead of the raw input (.dat)
    # ifiles = TDoc.getPaths(cohort=cohort_name, doctype=doctype, ifiles=kargs.get('ifiles', []), ext='csv', verfiy_=False) 
    # print('make_tset> loadModel> (cohort: %s => ifiles: %s)' % (cohort_name, ifiles)) 

    ### read + (analyze) + vectorize
    # [note] the 'analyze' step can be performed independently
    tSimplifyCode = kargs.get('simplify_code', False)
    tFilterCode = kargs.get('filter_code', False)
    # result = sa.loadModel(n_features=n_features, window=window, min_count=min_count, n_workers=n_workers, 
    #                         w2v_method=w2v_method, 
    #                         seq_ptype=seq_ptype, read_mode=read_mode, test_model=test_model, 
    #                             ifile=ifile, cohort=cohort_name, 
    #                             load_seq=False, load_model=load_word2vec_model, load_lookuptb=load_lookuptb, 
    #                             bypass_lookup=bypass_code_lookup) # attributes: sequences, lookup, model

    ### load model
    # 1. read | params: cohort, inputdir, doctype, labels, n_classes, simplify_code
    #         | assuming structured sequencing files (.csv) have been generated
    div(message='1. Read temporal doc files ...')

    # [note] csv header: ['sequence', 'timestamp', 'label'], 'label' may be missing
    # [params] if 'complete' is set, will search the more complete .csv file first (labeled > timed > doc)
    
    # if result set (sequencing data is provided, then don't read and parse from scratch)
    ret = kargs.get('result_set', {}) # 'sequence', 'timestamp', 'label' 
    if not ret: 
        ifiles = kargs.get('ifiles', [])
        ret = sa.readDocFromCSV(cohort=cohort_name, inputdir=docSrcDir, ifiles=ifiles, complete=True) # [params] doctype (timed)
    seqx = ret['sequence'] # must have sequence entry
    tseqx = ret.get('timestamp', [])

    # precedence: user-provided (class) labels > precomputed labels in .csv
    tags = kargs['labels'] if len(kargs.get('labels', [])) > 0 else ret.get('label', [])
    print('makeTSet.test> tags (multilabel): %s' % tags[:10])  # multilabel formt [[], [], []]

    nDoc = n_docs = len(seqx)
    print('verify> number of docs: %d' % nDoc)

    ### determine labels and n_classes 
    div(message='1.1 Document labeling ...')
    # check user input -> check labels obtained from within he labeled_seq file
    if not tags: 
        # disease module: pattern.medcode or pattern.<cohort> (e.g. diabetes cohort => pattern.diabetes)
        print("info> No data found from 1) user input 2) sequencing data > call getSurrogateLabels() from disease module.")  
        tags  = getSurrogateLabels(seqx, cohort=cohort_name)  # arg: cohort_name
   
        ### candidate labeling functions 
        # tags = labeling.labelDocByFreqDiag(seqx)
        # tags = labeling.labelDocsByFreq(seqx)
    else: 
        assert len(tags) == nDoc
        print('test> labels/tags: %s' % tags[:10])
    labeling = True if len(tags) > 0 else False
    # [condition] len(seqx) == len(tseqx) == len(labels) if all available

    # labels is not the same as tags (a list of lists)
    slabels = labels = TDocTag.toSingleLabel(tags, pos=0) # to single label format (for single label classification)
    print('makeTSet.test> cohort=%s, labels (single label): %s' % (cohort_name, labels[:100])) 

    n_classes = 1  # seqparams.arg(['n_classes', ], default=1, **kargs) 
    if labeling: n_classes = len(set(labels))
    # [condition] n_classes determined
    print('stats> n_docs: %d, n_classes: %d | cohort: %s, composition: %s' % (nDoc, n_classes, cohort_name, seq_compo))

    ### data subset selection and sequence transformation 
    div(message='1.2 Data Transformation and subset selection ...')
    # [control] simplify code?
    if tSimplifyCode: 
        print('    + simply the codes')
        seqx = seqAlgo.simplify(seqx)  # this will not affect medication code e.g. MED:12345

    ### [control] train specific subset of codes (e.g. diagnostic codes only)
    print('    + select code according to a predicate.')
    if tFilterCode: 
        predicate_routine = kargs.get('predicate', None)

        # precedence: predicate > seq_ptype
        idx = st.filterCodes(seqx, seq_ptype=seq_ptype, predicate=predicate_routine)  # if predicate is given, seq_ptype is ignoreed
        seqx = st.indexToDoc(seqx, idx)  # only preserve those documents with indices in idx
        if tseqx: tseqx = st.indexToDoc(tseqx, idx)
        if labels: labels = st.indexToDoc(labels, idx)

    div(message='2. Compute document embedding (params: )') # [note] use W2V and D2V classes in seqparams and vector to adjust d2v parameters
    d2v_method = kargs.get('d2v_method', 'pv-dm')
    
    # model = vector.getDocVecModel(d2v_method=d2v_method)
    # [output] document vectors to local cohort-specific directory i.e. <this module/data/<cohort>

    # very labeling schemes 
    ilabels = TDocTag.getIntIDs(nDoc) # TDocTag.toMultiLabel(range(nDoc))
    
    # [params] debug: cohort 
    X = getDocVec(seqx, d2v_method=d2v_method, labels=ilabels, 
                    outputdir=outputdir,
                    test_=kargs.get('test_model', True), 
                    load_model=kargs.get('load_model', True), 
                    cohort=cohort_name) # [params] w2v_method, outputdir, outputfile

    assert X.shape[0] == nDoc and X.shape[1] == vector.D2V.n_features

    # lookuptb = ret['symbol_chart']
    print('status> Model computation complete.')
    
    div(message='3. Save training set')
    # e.g. .../data/PTSD/tset_nC1_IDregular-sg-tfidfavg-GPTSD.csv
    fname = TSet.getName(cohort=cohort_name, d2v_method=d2v_method) # seq_ptype ('regular'), w2v_method ('sg')
    fpath = os.path.join(outputdir, fname) # nC: n_classes, G: group ie cohort

    # can also use 'tags' (in multilabel format)
    ts = TSet.to_csv(X, y=labels) # [params] labels, if not given => all positive (1)

    # fpath = os.path.join(basedir, 'tset_%s_%s_t1t2-P%s.csv' % (doc2vec_method, n_doc0, seq_ptype)) 
    print('io> saving (n_classes=%d, d2v_method=%s) training data to:\n%s\n' % (n_classes, d2v_method, fpath))
    ts.to_csv(fpath, sep=',', index=False, header=True) # cmp: previously, we use ts.index to keep labels => index=True

    return ts

def makeTSet(**kargs): 
    # yield (X_train, X_test, y_train, y_test)
    return makeTSetCombined(**kargs)  # [output] X
def makeTSetCombined(**kargs): # same as seqClassify.makeTSet but does not distinguish train and test split in d2v
    """
    Similar to makeTSet but this routine does not distinguish train and test split in d2v
    The idea is to separate CV process in evaluating classifiers (modeled over document vectors) from 
    the training of d2v models. 

    Output
    ------
    Output location: e.g. .../data/<cohort>/cv/

    Use
    ---
    1. set dir_type to 'combined' to load the pre-computed training data
       e.g. TSet.load(cohort=cohort_name, d2v_method=d2v_method, dir_type='combined') to load

    """
    import seqClassify
    
    # [params] d2v_method, cohort, seq_ptype
    return seqClassify.makeTSetCombined(**kargs) 

def loadTSet(**kargs): 
    """
    Use
    ---
    ts = loadTSetCombined(cohort=cohort_name, d2v_method=d2v_method, seq_ptype=seq_ptype)   
    X, y = TSet.toXY(ts)
    """
    return loadTSetCombined(**kargs) # output: ts
def loadTSetCombined(**kargs): 
    import seqClassify
    return seqClassify.loadTSetCombined(**kargs) # output: ts

def data_matrix(**kargs):  
    """
    Get documents and their corresponding document vectors. 

    Input
    Output: (D, ts) where 
            D is a list of documuents in list-of-token format
            ts is the document vectors in 2D array

    """
    from seqparams import TSet
    import seqAnalyzer as sa

    # [params] makeTSet 
    #          cohort, labels, (result_set), d2v_method, (w2v_method), (test_model)
    #         simplify_code, filter_code
    #         

    # [params] read the document first followed by specifying training set based on seq_ptype and d2v method
    cohort_name = kargs['cohort'] # kargs.get('cohort', 'diabetes')
    seq_ptype = seqparams.normalize_ctype(kargs.get('seq_ptype', 'regular'))

    w2v_method = word2vec_method = kargs.get('w2v_method', vector.W2V.w2v_method)
    d2v_method = doc2vec_method = kargs.get('d2v_method',  vector.D2V.d2v_method) # 'pv-dm' for distributed memory
    
    docSrcDir = sys_config.read('DataExpRoot')  # document source directory
    basedir = TSet.getPath(cohort=cohort_name)  # cohort-specific local (output) directory

    # tset_version = 'new'  # or 'old'
    # tset_type = kargs.get('tset_type', 'binary') 
    # if n_classes is not None: seqparams.normalize_ttype(n_classes)
    ret = kargs.get('result_set', {})
    if not ret: 
        ifiles = kargs.get('ifiles', [])
        ret = sa.readDocFromCSV(cohort=cohort_name, inputdir=docSrcDir, ifiles=ifiles, complete=True) # [params] doctype (timed)
    D = sequences = ret['sequence']  # ['sequence', 'timestamp', 'label', ] 
    nDoc = len(sequences)

    ### try loading first, if non-existent, then make it (read sequences + compute doc vectors)
    ts = loadTSet(cohort=cohort_name, d2v_method=d2v_method, seq_ptype=seq_ptype)  # subset_idx: subsetting training data by indices
    if ts is None:
        # [options]
        # makeTSet: train test split (randomized subsampling, n_trials is usu. 1)
        # makeTSetCombined: no separation of train and test data on the d2v level 
        # makeTSetCV: 
        ts = makeTSet(cohort=cohort_name, d2v_method=d2v_method, seq_ptype=seq_ptype, 
                            model_id=kargs.get('model_id', seq_ptype), # distinguish models based on sequence contents
                            test_model=kargs.get('test_model', True), 
                            load_model=kargs.get('load_model', False))  # [note] this shouldn't revert back and call load_tset() again 

    assert ts is not None and not ts.empty, 'data_matrix> Warning: No training set found!'
    assert len(D) == ts.shape[0]
    
    return (D, ts)

def load_tset(**kargs):
    """

    Chain
    -----
    data_matrix -> load_tset <- make_tset

    Reference
    ---------
    1. search files ~ regex
       https://stackoverflow.com/questions/6798097/find-regex-in-python-or-how-to-find-files-whose-whole-name-path-name
    """
    from seqparams import TSet 
    # import fnmatch, re

    # [params] cohort
    # composition = seq_compo = kargs.get('composition', 'condition_drug')
    cohort_name = kargs['cohort'] # kargs.get('cohort', 'diabetes')
    seq_ptype = seqparams.normalize_ctype(**kargs) # sequence pattern type: regular, random, diag, med, lab
    read_mode = kargs.get('read_mode', 'doc') # assign 'doc' (instead of 'seq') to form per-patient sequences
    fsep=','

    basedir = outputdir = seqparams.get_basedir(cohort=cohort_name)  # ./data/<cohort>
    print('load_tset> basedir: %s' % basedir)
    
    w2v_method = word2vec_method = kargs.get('w2v_method', 'sg') # e.g. sg; not case-sensitive 
    d2v_method = doc2vec_method = kargs.get('d2v_method', 'tfdifavg')  # options: average, PVDM i.e. distributed memory

    # tset_type = kargs.get('tset_type', 'unary')
    n_classes = seqparams.arg(['n_classes', 'n_labels'], 1, **kargs)
    tset_type = seqparams.normalize_ttype(n_classes)

    seq_ptype_eff = 'regular' # because clustering is always computed in the context of all medical codes
    if w2v_method is not None: 
        identifier = '%s-%s-%s' % (seq_ptype_eff, w2v_method, d2v_method)  # no need to include w2v_method (which is subsumed by d2v_method)
    else: 
        identifier = '%s-%s' % (seq_ptype_eff, d2v_method)
    fpath = os.path.join(basedir, 'tset_nC%s_ID%s-G%s.csv' % (n_classes, identifier, cohort_name)) # nC: n_classes, G: group ie cohort
    
    # [todo] should be able to use the training set to identify number of classes

    # if only a subset is needed
    subset_idx = seqparams.arg(['idx', 'subset_idx'], None, **kargs)

    # [todo] use file pattern match fnmatch or glob

    ts = None
    if os.path.exists(fpath) and os.path.getsize(fpath) > 0: 
        print('io> loading training set (n_classes=%d, cohort=%s) from:\n%s\n' % (n_classes, cohort_name, fpath))
        ts = pd.read_csv(fpath, sep=fsep, header=0, index_col=False, error_bad_lines=True)
    
    # example path: tpheno/seqmaker/data/PTSD/tset_nC1_IDregular-sg-tfidfavg-GPTSD.csv
    if ts is None: 
        print('load_tset> Warning: No training set found at %s' % fpath)
        return None 
    if subset_idx is not None: 
        assert hasattr(subset_idx, '__iter__')
        print('subset> Only return a subset (n=%d) out of total %d' % (len(subset_idx), ts.shape[0]))
        return ts.loc[ts[TSet.index_field].isin(subset_idx)]
            
    return ts

def t_preclassify_weighted(**kargs):  # count based method for docuemnt vector
    """
    

    Log
    ---
    1. example tfidf maps:
       {'74300000189': 11.673598089044336, 'prescription112847': 12.589888820918492, 
        'prescription911': 13.283036001478438, ... '136325': 11.491276532250383, 'T40.5X1A': 13.976183182038383, 
        '955.2': 13.976183182038383, 'prescription126266': 13.283036001478438 ...}

    """
    def v_index(res, n=5): 
        type1_idx = res['type_1']
        type2_idx = res['type_2']
        type3_idx = res['gestational']
        criteria = [('type_1', diab.has_type1), ('type_2', diab.has_type2), ('gestational', diab.has_gestational), ]

        for j, idx in enumerate([type1_idx, type2_idx, type3_idx, ]):
            for i in random.sample(idx, min(n, len(idx))): 
                assert criteria[j][1](sequences[i]), "%d-th sequence does not satisfy %s criteria!" % (i, criteria[j][0])
        return
    def apply_approx_pheno(docs): 
        # label sequences in a manner that that adheres to the  phenotyping criteria ~ diagnostic codes
        labelsets = phenotypeDoc(sequences=docs, seq_ptype=seq_ptype, load_=load_labelsets)
        # [params]
        #     res key: ['type_0', 'type_1', 'type_2', 'gestational', 'type_3', 'type_1_2', 'type_1_3', 'type_2_3', 'type_1_2_3']
        res = diab.phenotypeIndex(labelsets)
        v_index(res, n=6)
        return res  # disease types to indices
    def load_fset(fpath, fsep=','):
        if os.path.exists(fpath) and os.path.getsize(fpath) > 0: 
            ts = pd.read_csv(fpath, sep=fsep, header=0, index_col=False, error_bad_lines=True)
            return ts 
        return None

    # import matplotlib.pyplot as plt
    from sklearn.feature_extraction.text import TfidfVectorizer
    from pattern import diabetes as diab 
    import seqAnalyzer as sa
    # import vector
    from seqparams import TSet

    # [params]
    # sequences and word2vec model
    cohort_name = kargs.get('cohort', 'diabetes')
    doc_basename = 'condition_drug' if cohort_name is None else 'condition_drug-%s' % cohort_name

    read_mode = 'doc'  # assign 'doc' (instead of 'seq') to form per-patient sequences
    seq_ptype = seqparams.normalize_ctype(**kargs) # sequence pattern type: regular, random, diag, med, lab
    word2vec_method = 'SG'
    doc2vec_method = 'PVDM'  # default: distributed memory
    testdir = os.path.join(os.getcwd(), 'test')
    if not os.path.exists(testdir): os.makedirs(testdir) # test directory

    # [params] training 
    n_features = kargs.get('n_features', GNFeatures)
    window = kargs.get('window', GWindow)
    min_count = kargs.get('min_count', GMinCount)  # set to >=2, so that symbols that only occur in one doc don't count

    n_cores = multiprocessing.cpu_count()
    # print('info> number of cores: %d' % n_cores)
    n_workers = kargs.get('n_workers', GNWorkers)
    labeling = kargs.get('labeling', True)  # [operation] apply labeling?

    # [params] data set
    f_prefix = seqparams.generic_feature_prefix  # feature prefix
    f_label = seqparams.TSet.label_field
    fsep = ','  # feature separator

    doctype = 'd2v' 
    doc_basename = 'condition_drug'
    descriptor = kargs.get('meta', doc_basename)  # algorithm type: PV-DBOW, DM

    # assumption: w2v has been completed
    basedir = outputdir = os.path.join(os.getcwd(), 'data/%s' % cohort_name) # [I/O] sys_config.read('DataExpRoot')
    if not os.path.exists(basedir): os.makedirs(basedir) # base directory

    load_word2vec_model = kargs.get('load_w2v', True)
    test_model = kargs.get('test_model', True) # independent from loading or (re)computing w2v
    load_lookuptb = kargs.get('load_lookuptb', True) and load_word2vec_model
    # load_doc2vec_model = kargs.get('load_doc2vec_model', False) # simple (first attempt) doc2vec model; induced by vectorize2 
    # load_label = kargs.get('load_label', True) and load_doc2vec_model
    load_labelsets = kargs.get('load_labelsets', False)

    # [input] medical coding sequences
    ifile = 'condition_drug_seq-%s.dat' % cohort_name if cohort_name is not None else 'condition_drug_seq.dat' 

    # op: read, analyze, vectorize
    result = sa.loadModel(n_features=n_features, window=window, min_count=min_count, n_workers=n_workers, 
                            w2v_method=word2vec_method, 
                            seq_ptype=seq_ptype, read_mode=read_mode, test_model=test_model, 
                                ifile=ifile, cohort=cohort_name, bypass_lookup=True,
                                load_seq=False, load_model=load_word2vec_model, load_lookuptb=load_lookuptb) # attributes: sequences, lookup, model

    sequences = result['sequences']
    model = result['model']
    lookuptb = result['symbol_chart']
    n_doc = n_doc0 = len(sequences)
    print('verify> number of docs: %d' % n_doc0)

    # sequence labeling (via frequecy-based heuristics)
    # df_ldoc = labelDoc(sequences, load_=load_label, seqr='full', sortby='freq', seq_ptype=seq_ptype)
    # labels = list(df_ldoc['label'].values) # multicode labels by default

    # find out the disease types (type 1, 2 for diabetes) and their corresponding sequence indices 
    res = getSurrogateLabels(sequences, cohort=cohort_name)
    # labeling convention: type 1: 0, type 2: 1, type 3 (gestational): 2
    t1idx, t2idx, t3idx = res['type_1'], res['type_2'], res['gestational']  

    ### Document Embedding as a Function of Word Embedding ### 

    # Method 1: doc vectors by averating

    ### Binary Classification ### 
    doc2vec_method = 'average'
    fpath = os.path.join(basedir, 'tset_%s_%s_t1t2-P%s.csv' % (doc2vec_method, n_doc0, seq_ptype)) 
    ts = load_fset(fpath, fsep=fsep)
    if ts is None: 
        X = vector.byAvg(sequences, model=model, n_features=n_features)
        assert X.shape[0] == n_doc0
        X_t1, y_t1 = X[t1idx], np.repeat(0, len(t1idx))  # type 1 is labeled as 0
        X_t2, y_t2 = X[t2idx], np.repeat(1, len(t2idx))  # type 2 is labeled as 1  
        Xb = np.vstack([X_t1, X_t2]) 
        yb = np.hstack([y_t1, y_t2])
        idxb = np.hstack([t1idx, t2idx])
        assert Xb.shape[0] == len(yb)
        n_docb = Xb.shape[0]
        
        print('output> preparing ts for binary classification > method: %s, n_doc: %d' % (doc2vec_method, n_docb))
        header = ['%s%s' % (f_prefix, i) for i in range(Xb.shape[1])]
        ts = DataFrame(Xb, columns=header)
        ts[TSet.target_field] = yb
        ts[TSet.index_field] = idxb
        ts = ts.reindex(np.random.permutation(ts.index)) 

        # fpath = os.path.join(basedir, 'tset_%s_%s_t1t2-P%s.csv' % (doc2vec_method, n_doc0, seq_ptype)) 
        print('output> saving (binary classification, d2v method=%s) training data to %s' % (doc2vec_method, fpath))
        ts.to_csv(fpath, sep=fsep, index=False, header=True) # cmp: previously, we use ts.index to keep labels => index=True

    ### multiclass: class including gestational
    doc2vec_method = 'average'
    fpath = os.path.join(basedir, 'tset_%s_%s_t1t2t3-P%s.csv' % (doc2vec_method, n_doc0, seq_ptype)) 
    ts = load_fset(fpath, fsep=fsep)
    if ts is None: 
        X_t3, y_t3 = X[t3idx], np.repeat(2, len(t3idx))  # gestational is labeled 2
        Xm = np.vstack([Xb, X_t3]) # m: multiclass
        ym = np.hstack([yb, y_t3])
        idxm = np.hstack([idxb, t3idx])
        n_docm = Xm.shape[0]
    
        print('output> preparing ts for multiclass classification > method: %s, n_doc: %d ...' % (doc2vec_method, n_docm))
        header = ['%s%s' % (f_prefix, i) for i in range(Xm.shape[1])]
        ts = DataFrame(Xm, columns=header)
        ts[TSet.target_field] = ym 
        ts[TSet.index_field] = idxm
        ts = ts.reindex(np.random.permutation(ts.index))   

        # fpath = os.path.join(basedir, 'tset_%s_%s_t1t2t3-P%s.csv' % (doc2vec_method, n_doc0, seq_ptype)) 
        print('output> saving (3-label classification) training data to %s' % fpath)
        ts.to_csv(fpath, sep=',', index=False, header=True) # cmp: previously, we use ts.index to keep labels => index=True  

    # # now compute TD-IDF scores
    # X_mdoc = np.asarray(mseqx)
    # vectorizer = TfidfVectorizer(min_df=1)    
    # X_tfidf = vectorizer.fit_transform(mseqx) # Learn vocabulary and idf, return term-document matrix.
    # print('> X_tfidf dim: %s =?= modeled sequence dim: %s' % (str(X_tfidf.shape), str(X_mdoc.shape)))

    ###### weight w2v via td-idf scores 

    ### Binary Classification ### 
    doc2vec_method = 'tfidfavg'
    ts = load_fset(fpath, fsep=fsep)
    fpath = os.path.join(basedir, 'tset_%s_%s_t1t2-P%s.csv' % (doc2vec_method, n_doc0, seq_ptype)) 
    if ts is None: 
        X = vector.getTfidfAvgFeatureVecs(sequences, model, n_features) # optional: min_df, max_df, max_features
        assert X.shape[0] == n_doc0

        X_t1, y_t1 = X[t1idx], np.repeat(0, len(t1idx))  # type 1 is labeled as 0
        X_t2, y_t2 = X[t2idx], np.repeat(1, len(t2idx))  # type 2 is labeled as 1  
        Xb = np.vstack([X_t1, X_t2])  # b: binary classification
        yb = np.hstack([y_t1, y_t2])
        idxb = np.hstack([t1idx, t2idx])
        assert Xb.shape[0] == len(yb)
        n_docb = Xb.shape[0]

        print('output> preparing ts for binary classification > method: %s, n_doc: %d' % (doc2vec_method, n_docb))
        header = ['%s%s' % (f_prefix, i) for i in range(Xb.shape[1])]
        ts = DataFrame(Xb, columns=header)
        ts[TSet.target_field] = yb
        ts[TSet.index_field] = idxb
        ts = ts.reindex(np.random.permutation(ts.index)) 

        # fpath = os.path.join(basedir, 'tset_%s_%s_t1t2-P%s.csv' % (doc2vec_method, n_doc0, seq_ptype)) 
        print('output> saving (binary classification, d2v method=%s) training data to %s' % (doc2vec_method, fpath))
        ts.to_csv(fpath, sep=',', index=False, header=True) # cmp: previously, we use ts.index to keep labels => index=True

    ### multiclass: class including gestational
    doc2vec_method = 'tfidfavg'
    ts = load_fset(fpath, fsep=fsep)
    fpath = os.path.join(basedir, 'tset_%s_%s_t1t2t3-P%s.csv' % (doc2vec_method, n_doc, seq_ptype)) 
    if ts is None: 
        X_t3, y_t3 = X[t3idx], np.repeat(2, len(t3idx))  # gestational is labeled 2
        Xm = np.vstack([Xb, X_t3]) # m: multiclass
        ym = np.hstack([yb, y_t3])
        idxm = np.hstack([idxb, t3idx])
        n_docm = Xm.shape[0]

        print('output> preparing ts for multiclass classification > method: %s, n_doc: %d ...' % (doc2vec_method, n_docm))
        header = ['%s%s' % (f_prefix, i) for i in range(Xm.shape[1])]
        ts = DataFrame(Xm, columns=header)
        ts[TSet.target_field] = ym 
        ts[TSet.index_field] = idxm
        ts = ts.reindex(np.random.permutation(ts.index))   

        # fpath = os.path.join(basedir, 'tset_%s_%s_t1t2t3-P%s.csv' % (doc2vec_method, n_doc, seq_ptype)) 
        print('output> saving (3-label classification) training data to %s' % fpath)
        ts.to_csv(fpath, sep=fsep, index=False, header=True) # cmp: previously, we use ts.index to keep labels => index=True  
    
    return

def t_tdidf(**kargs): 
    # import vector
    import seqAnalyzer as sa

    # [params] training 
    n_features = kargs.get('n_features', GNFeatures)
    window = kargs.get('window', GWindow)
    min_count = kargs.get('min_count', GMinCount)  # set to >=2, so that symbols that only occur in one doc don't count
    
    n_cores = multiprocessing.cpu_count()
    # print('info> number of cores: %d' % n_cores)
    n_workers = kargs.get('n_workers', GNWorkers)

    word2vec_method = 'SG'
    doc2vec_method = 'PVDM'  # default: distributed memory
    read_mode = kargs.get('read_mode', 'doc')  
    seq_ptype = seqparams.normalize_ctype(**kargs)
    test_model = test_w2v_model = False

    # sequences and word2vec model
    result = sa.loadModel(n_features=n_features, window=window, min_count=min_count, n_workers=n_workers,
                            w2v_method=word2vec_method,  
                            read_mode=read_mode, seq_ptype=seq_ptype, test_model=test_model,
                                load_seq=False, load_model=True, load_lookuptb=True) 
    sequences = result['sequences']
    model = result['model']
    lookuptb = result['symbol_chart']
    n_doc = n_doc0 = len(sequences)
    print('verify> number of docs: %d' % n_doc0)   
  
    doc2vec_method = 'tfidfavg'
    X = vector.getTfidfAvgFeatureVecs(sequences, model, n_features, test_=True) # optional: min_df, max_df, max_features
    assert X.shape[0] == n_doc0

    return

def t_doc2vec1(**kargs):  # [old]
    """
    A template function that generates document vectors. 
    
    Note
    ----
    1. model induced by vectorize2
          model = doc2vec.Doc2Vec(alpha=0.025, min_alpha=0.025)  # use fixed learning rate
          ... 
          for epoch in range(10):
              model.train(labeled_docs)
               model.alpha -= 0.002  # decrease the learning rate
               ...

    Related 
    -------
    * build_data_matrix() 
    * map_clusters() 
    * config_doc2vec_model

    """
    def v_norm_distribution(X, y=None, p=2): 
        # [params] testdir

        # plot histogram of distancesto the origin for all document vectors
        plt.clf() 
        norms = [np.linalg.norm(x, p) for x in X]
        n, bins, patches = plt.hist(norms)
        print('verfiy_norm> n: %s, n_bins: %s, n_patches: %s' % (n, len(bins), len(patches)))
 
        fpath = os.path.join(testdir, 'd2v_%snorm_distribution-P%s.tif' % (p, seq_ptype))
        print('output> saving d2v data %s-norm-distribution (of sequence type %s) to %s' % (p, seq_ptype, fpath))
        plt.savefig(fpath)

    # import matplotlib.pyplot as plt
    import seqAnalyzer as sa
    cohort_name = kargs.get('cohort', 'diabetes')

    # [params]
    read_mode = 'doc'  # assign 'doc' (instead of 'seq') to form per-patient sequences
    seq_ptype = seqparams.normalize_ctype(**kargs) # sequence pattern type: regular, random, diag, med, lab
    word2vec_method = 'SG'
    doc2vec_method = 'PVDM'  # distributed memory
    testdir = os.path.join(os.getcwd(), 'test')
    if not os.path.exists(testdir): os.makedirs(testdir) # test directory

    # [params] training 
    n_features = kargs.get('n_features', GNFeatures)
    window = kargs.get('window', GWindow)
    min_count = kargs.get('min_count', GMinCount)  # set to >=2, so that symbols that only occur in one doc don't count

    n_cores = multiprocessing.cpu_count()
    # print('info> number of cores: %d' % n_cores)
    n_workers = kargs.get('n_workers', GNWorkers)

    # [params] data set
    f_prefix = seqparams.generic_feature_prefix  # feature prefix
    f_label = seqparams.TSet.label_field

    doctype = 'd2v' 
    doc_basename = 'condition_drug'

    descriptor = kargs.get('meta', doc_basename)  # algorithm type: PV-DBOW, DM

    basedir = outputdir = seqparams.get_basedir(cohort=cohort_name) # sys_config.read('DataExpRoot')

    load_w2v = load_word2vec_model = kargs.get('load_w2v', True)
    test_model = kargs.get('test_model', False) # test w2v model 
    load_lookuptb = kargs.get('load_lookuptb', True) and load_word2vec_model
    load_d2v = load_doc2vec_model = kargs.get('load_d2v', True) # simple (first attempt) doc2vec model; induced by vectorize2 
    load_label = kargs.get('load_label', True) and load_doc2vec_model

    ### sequences and word2vec model
    ifile = 'condition_drug_seq-%s.dat' % cohort_name if cohort_name is not None else 'condition_drug_seq.dat'
    result = sa.loadModel(n_features=n_features, window=window, min_count=min_count, n_workers=n_workers, 
                            w2v_method=word2vec_method, 
                            seq_ptype=seq_ptype, read_mode=read_mode, test_model=False, 
                                ifile=ifile, cohort=cohort_name, bypass_lookup=True,
                                load_seq=False, load_model=load_word2vec_model, load_lookuptb=load_lookuptb) # attributes: sequences, lookup, model
    sequences = result['sequences']
    model = result['model']
    lookuptb = result['symbol_chart']
    n_doc = len(sequences)
    print('verify> number of docs: %d' % n_doc)

    ### sequence labeling (via frequecy-based heuristics)
    
    div(message='Assign labels to documents/sequences ...')
    df_ldoc = labelDocsByFreq(sequences, load_=load_label, seqr='full', sortby='freq', seq_ptype=seq_ptype)

    # can also obtain labels directly from the dataframe (.csv) in the case of CKD
    # df_ldoc = labelDocsByDataFrame(sequences, seq_ptype=seq_ptype)

    labels = list(df_ldoc['label'].values) # multicode labels by default
    # labels = [mlabel.split(lsep)[0] for mlabel in mlabels] # take most frequent (sub-)label as the label
    print('t_doc2vec1> labels (dtype=%s): %s' % (type(labels), labels[:10]))
    labeled_seqx = makeD2VLabels(sequences=sequences, labels=labels)
    # save? 

    # doc2vec model
    # ofile = kargs.get('output_file', '%s-%s_f%dw%d.%s' % (descriptor, doc2vec_method, n_features, window, doctype))
    # fpath = os.path.join(basedir, ofile)
    print('t_doc2vec1> load previous d2v model? %s' % load_d2v)
    model = vectorize(seqx, load_model=load_model, seq_ptype=seq_ptype, 
                            w2v_method=w2v_method, test_model=test_model,
                            n_features=n_features, window=window, min_count=min_count, n_workers=n_workers, cohort=cohort_name)
    d2v_model = makeDocVec(labeled_docs=labeled_seqx, load_model=load_doc2vec_model, 
                                seq_ptype=seq_ptype, 
                                n_features=n_features, window=window, min_count=min_count, n_workers=n_workers)

    n_doc2 = d2v_model.docvecs.count
    print('verify> number of docs (from model): %d =?= %d' % (n_doc2, n_doc))

    lseqx = random.sample(labeled_seqx, 10)
    for i, lseq in enumerate(lseqx): 
        tag = lseq.tags[0]
        vec = d2v_model.docvecs[tag]  # model.docvecs[doc.tags[0]]
        print('[%d] label: %s => vec (dim: %s)' % (i, tag, str(vec.shape))) 
        sim_docs = d2v_model.docvecs.most_similar(tag, topn=10)

        print('  + most similar docs: %s' % str(sim_docs[0]))
        sim_docs_res = [sim_doc[0] for sim_doc in sim_docs[1:]]
        print('     ++ other similar docs:\n%s\n' % '> '.join(sim_docs_res))

    # [output][byproduct]
    # save feature vector file (for external use e.g. ClusterEng)
    # train_docvec = np.zeros( (n_doc, n_features), dtype="float32" )  # very sparse!
    X, y = np.array([d2v_model.docvecs[label] for label in labels]), labels
    print('verify> doc vec set X > dim: %s | (n_labels: %d, n_docs: %d)' % (str(X.shape), len(labels), n_doc))

    header = ['%s%s' % (f_prefix, i) for i in range(X.shape[1])]
    df = DataFrame(X, columns=header) 
    df[f_label] = labels

    # [todo] target: multiclass

    # [old]
    # df = DataFrame(X)     
    # df.index = labels   # assign labels to data points   

    # assert len(labels) == df.shape[0] and n_doc == df.shape[0]
    fpath = os.path.join(outputdir, 'tset_%s_%s_labeled-P%s.csv' % (doc2vec_method, n_doc, seq_ptype)) # very sparse! 
    df.to_csv(fpath, sep=',', index=True, header=False)   
    print('byproduct> saving (%s-) doc2vec feature set (labeled) to %s | feature set dim: %s' % \
        (doc2vec_method, fpath, str(df.shape)))

    # verify data
    v_norm_distribution(X, y)  # data variation
    print('t_doc2vec1> completed')

    return

def load_XY(**kargs): 
    """

    Related
    -------
    1. X, y, D <- build_data_matrix2
       where X is computed via d2v model (e.g. PVDM)
    2. load_ts
    3. getXY, getXYD
    """
    import evaluate

    standardize_ = kargs.get('standardize_', 'minmax')
    ts = load_ts(**kargs)
    X, y = evaluate.transform(ts, standardize_=standardize_) # default: minmax

    return (X, y)
def getXY(**kargs): 
    return load_XY(**kargs)
def getXYD(**kargs): 
    import evaluate

    # [params]
    cohort_name = kargs.get('cohort', 'diabetes')
    read_mode = 'doc'  # assign 'doc' (instead of 'seq') to form per-patient sequences
    kargs['seq_ptype'] = seq_ptype = seqparams.normalize_ctype(**kargs) # sequence pattern type: regular, random, diag, med, lab
    doc2vec_method = kargs.get('d2v_method', 'tfidfavg')   # average, tfidfavg, or PVDM (distributed memory)
    load_labelsets = kargs.get('load_labelsets', False)
    standardize_ = kargs.get('standardize_', 'minmax')
    tset_type = kargs.get('tset_type', None)  # tertiary, mutliple
    n_classes = kargs.get('n_classes', None)

    assert not (tset_type is None and n_classes is None), "tset_type and n_classes cannot be both unknown."
    D, ts = data_matrix(read_mode=read_mode, seq_ptype=seq_ptype, w2v_method=w2v_method, d2v_method=doc2vec_method, 
                            tset_type=tset_type, n_classes=n_classes, cohort=cohort_name)
    # return build_data_matrix2(**kargs) 
    X, y = evaluate.transform(ts, standardize_=standardize_) # default: minmax
    return (X, y, D)


def t_classify(**kargs): 
    import evaluate
    import seqClassify 
    return seqClassify.t_classify(**kargs)

def t_tsne(**kargs):
    # from cluster import tsne
    import tsne

    ts = loadTSet(tset_type='binary')
    tsne.run(ts=ts, **kargs)

    return

def t_analysis(**kargs):
    def v_norm_distribution(X, y=None, p=2): 
        # [params] testdir

        # plot histogram of distancesto the origin for all document vectors
        plt.clf() 
        norms = [np.linalg.norm(x, p) for x in X]
        n, bins, patches = plt.hist(norms)
        print('verfiy_norm> (seq_ptype: %s, d2v: %s) n: %s, n_bins: %s, n_patches: %s' % \
            (seq_ptype, d2v_method, n, len(bins), len(patches)))

        identifier = '%s-%s' % (seq_ptype, d2v_method)
        fpath = os.path.join(testdir, 'd2v_%snorm_distribution-P%s.tif' % (p, identifier))
        print('output> saving d2v data %s-norm-distribution to %s' % (p, fpath))
        plt.savefig(fpath)
    def v_count_nonzero(X, y=None): 
        # count nonzeros across column (axis=1)
        # e.g. np.count_nonzero([[0,1,7,0,0],[3,0,0,2,19]], axis=1)

        n_nonzeros = []
        for x in X: 
            n_nonzeros.append(np.count_nonzero(x))
        print('verify_nonzero> (seq_ptype: %s, d2v: %s) number of features ~ X: %d >=? max of nonzeros: %d' % \
            (seq_ptype, d2v_method, X.shape[1], max(n_nonzeros)))
        assert len(n_nonzeros) == X.shape[0]
        
        n, bins, patches = plt.hist(n_nonzeros)
        # n: counts 
        print('verify> counts:\n%s\n' % n)

        identifier = '%s-%s' % (seq_ptype, d2v_method)
        fpath = os.path.join(testdir, 'd2v_nonzero_distribution-P%s.tif' % identifier)
        print('output> saving d2v data nonzero-distribution to %s' % fpath)
        plt.savefig(fpath)

    # from numpy.linalg import norm
    # from sklearn.cluster import KMeans

    n_clusters = 50
    kargs['seq_ptype'] = seq_ptype = kargs.get('seq_ptype', 'regular') # sequence pattern type: regular, random, diag, med, lab
    d2v_method = '?'
    # d2v_method = kargs.get('d2v_type', kargs.get('d2v_method', 'PVDM'))  # i.e. doc2vec_method
    testdir = os.path.join(os.getcwd(), 'test')
    if not os.path.exists(testdir): os.makedirs(testdir) # test directory

    # load data 
    # X, y, D = build_data_matrix2(**kargs) # seq_ptype, n_features, etc. 
    # 1. read_doc, seq_ptype => document type (random, regular, etc.)
    # 2. doc2vec_method, n_doc, seq_ptype => training data set
    d2v_methods = ['average', 'tfidfavg', ] # 'PVDM'
    for d2v_method in d2v_methods: 
        kargs['d2v_method'] = d2v_method
        X, y, D = getXYD(**kargs) 
        n_doc = len(D)

        # label y via phenotypeDoc()
        n_labels = len(set(y))
        assert X.shape[0] == y.shape[0] == n_doc
        print('verify> setting: (seq_ptype: %s, d2v: %s) > n_doc: %d, embedded dim: %d' % (seq_ptype, d2v_method, n_doc, X.shape[1]))
        print('verify> example composite labels:\n%s\n' % y[:10])

        print('verify> total number of labels: %d > number of unique labels: %d' % (len(y), n_labels))

        # 0. compute variations of vectors (for examining paragraph vectors)
        # v_norm_distribution(X, y)  # X: all close to 0
        # v_count_nonzero(X, y)
    
        # 1. k-means,  
        cluster_analysis(X=X, y=y, n_clusters=n_clusters, save_=True, seq_ptype=seq_ptype, d2v_method=d2v_method, cluster_method='kmeans')

        # cluster_spectral(X=X, y=y, n_clusters=n_clusters, save_=True)

        # 2. t-SNE 

        # (approximate) classification

        # end
        div(message='Completed analysis on seq_ptype: %s, d2v_method: %s' % (seq_ptype, d2v_method))

    # [old] labeled determined via frequency of code occurrences
    # lsep = '_'
    # yp = [mlabel.split(lsep)[0] for mlabel in y] # take most frequent (sub-)label as the label
    # print('verify> example single labels:\n%s\n' % yp[:10])
    # n_unique_labels = len(set(yp))
    # [log] number of unique multilabels: 264650 > unique single labels: 6234
    #       n_doc: 432000, fdim: 200
    # print('verify> total number of labels: %d > number of unique multilabels: %d > unique single labels: %d' % \
    #   (len(y), n_unique_mlabels, n_unique_slabels))

    return

def labelize(docs, class_labels=[], label_type='doc'): # essentially a wrapper of labeling.labelize 
    """
    Label docuemnts. Note that document labels are not the same as class labels (class_labels)

    Params
    ------
    labels
    """
    import vector
    return vector.labelDocuments(docs, class_labels=class_labels, label_type=label_type) # overwrite_=Fal

def build_data_matrix2(**kargs):
    """
    Compute (X, y, D) where X consists of PV-based doc vectors
    """
    # import seqAnalyzer as sa

    # [params] training parameters 
    n_features = kargs.get('n_features', GNFeatures)
    window = kargs.get('window', GWindow)
    min_count = kargs.get('min_count', GMinCount)  # set to >=2, so that symbols that only occur in one doc don't count
    n_cores = multiprocessing.cpu_count()
    n_workers = kargs.get('n_workers', GNWorkers) # n_cores: 30

    # [params] document type and document source 
    seq_ptype = kargs.get('seq_ptype', 'regular') # sequence pattern type: regular, random, diag, med, lab
    prefix = kargs.get('input_dir', sys_config.read('DataExpRoot'))

    cohort_name = kargs.get('cohort', 'diabetes')
    ifile = 'condition_drug_seq-%s.dat' % cohort_name if cohort_name is not None else 'condition_drug_seq.dat'
    ifiles = kargs.get('ifiles', [ifile, ])

    load_seq, save_seq = False, False

    # [note] default source document files: ['condition_drug_seq.dat', ] from DataExpRoot 
    #        use 'prefix' to change source (base) directory
    #        use 'ifiles' to change the document source file set
    print('source> reading from %s' % os.path.join(prefix, ifiles[0]))
    seqx = sa.read(simplify_code=False, mode='doc', verify_=False, seq_ptype=seq_ptype, ifile=ifile, 
                       load_=load_seq, save_=save_seq)

    df_ldoc = labelDoc(seqx, load_=True, seqr='full', sortby='freq', seq_ptype=seq_ptype, cohort=cohort_name)

    labels = list(df_ldoc['label'].values) 
    lseqx = makeD2VLabels(sequences=seqx, labels=labels, cohort=cohort_name)
    model = makeDocVec(labeled_docs=lseqx, load_model=True, 
                         seq_ptype=seq_ptype, 
                         n_features=n_features, window=window, min_count=min_count, n_workers=n_workers)

    # [params] inferred 
    n_doc = len(lseqx)
    n_features = model.docvecs[lseqx[0].tags[0]].shape[0]
    print('verify> n_doc: %d, fdim: %d' % (n_doc, n_features))

    # [test]
    lseqx_subset = random.sample(lseqx, 10)
    for i, lseq in enumerate(lseqx_subset): # foreach labeld sequence
        tag = lseq.tags[0]
        vec = model.docvecs[tag]  # model.docvecs[doc.tags[0]]
        print('[%d] label: %s => vec (dim: %s)' % (i, tag, str(vec.shape)))

    # dmatrix = [model.docvecs[lseq.tags[0]] for lseq in lseqx]
    dmatrix = np.zeros( (n_doc, n_features), dtype="float32" )  # alloc mem to speed up
    for j, lseq in enumerate(lseqx): 
        dmatrix[i] = model.docvecs[lseq.tags[0]]    
    
    return (dmatrix, np.array(labels), seqx) # (X, y, D)

def build_data_matrix(lseqx=None, **kargs): 
    """
    Compute (X, y) where X consists of PV-based doc vectors
    lseqx: labeled sequences/documents 

    Memo
    ----
    Assuming that d2v models were obtained, call this routine to get matrix and labels 

    Related
    -------
    build_data_matrix2() returns (X, y, D) where D consists of coding sequences that corresponds to X

    """
    # import seqAnalyzer as sa

    # [params] cohort 
    cohort_name = kargs.get('cohort', 'diabetes')

    # [params] training parameters 
    n_features = kargs.get('n_features', GNFeatures)
    window = kargs.get('window', GWindow)
    min_count = kargs.get('min_count', GMinCount)  # set to >=2, so that symbols that only occur in one doc don't count
    n_cores = multiprocessing.cpu_count()
    n_workers = kargs.get('n_workers', GNWorkers) # n_cores: 30

    # [params] document type and document source 
    prefix = kargs.get('input_dir', sys_config.read('DataExpRoot'))

    # read_mode: {'seq', 'doc', 'csv'}  # 'seq' by default
    # cohort vs temp doc file: diabetes -> condition_drug_seq.dat
    ifile = 'condition_drug_seq-%s.dat' % cohort_name if cohort_name is not None else 'condition_drug_seq.dat'
    ifiles = kargs.get('ifiles', [ifile, ])
    
    seq_ptype = kargs.get('seq_ptype', 'regular') # sequence pattern type: regular, random, diag, med, lab

    load_seqx = False; labels = []
    if lseqx is None: 
        # [note] default source document files: ['condition_drug_seq.dat', ] from DataExpRoot 
        #        use 'prefix' to change source (base) directory
        #        use 'ifiles' to change the document source file set
        print('source> reading from %s' % os.path.join(prefix, ifiles[0]))
        
        # read(load_=load_seq, simplify_code=simplify_code, mode=read_mode, verify_=verify_seq, seq_ptype=seq_ptype, ifile=ifile)
        seqx = sa.read(load_=True, simplify_code=False, mode='doc', verify_=False, seq_ptype=seq_ptype, ifiles=ifiles)

        df_ldoc = labelDoc(seqx, load_=True, seqr='full', sortby='freq', seq_ptype=seq_ptype)
        labels = list(df_ldoc['label'].values) 
        lseqx = makeD2VLabels(sequences=seqx, labels=labels)
        load_seqx = True
    else: 
        labels = [lseq.tags[0] for lseq in lseqx]  # [config] configure labels here

    model = makeDocVec(labeled_docs=lseqx, load_model=True, 
                         seq_ptype=seq_ptype, 
                         n_features=n_features, window=window, min_count=min_count, n_workers=n_workers)

    # [params] inferred 
    n_doc = len(lseqx)
    n_features = model.docvecs[lseqx[0].tags[0]].shape[0]
    print('verify> n_doc: %d, fdim: %d' % (n_doc, n_features))

    # [test]
    lseqx_subset = random.sample(lseqx, 10)
    for i, lseq in enumerate(lseqx_subset): # foreach labeld sequence
        tag = lseq.tags[0]
        vec = model.docvecs[tag]  # model.docvecs[doc.tags[0]]
        print('[%d] label: %s => vec (dim: %s)' % (i, tag, str(vec.shape)))

    # dmatrix = [model.docvecs[lseq.tags[0]] for lseq in lseqx]
    dmatrix = np.zeros( (n_doc, n_features), dtype="float32" )  # alloc mem to speed up
    for j, lseq in enumerate(lseqx): 
        dmatrix[i] = model.docvecs[lseq.tags[0]]    
    
    return (dmatrix, np.array(labels)) # (X, y)


def cluster_hc(X=None, y=None, **kargs):
    """

    Related 
    -------
    1. demo/hc.py  ... demo of hierarchical clustering algorithms

    """
    # [params]
    n_clusters = kargs.get('n_clusters', 100)

    if X is None or y is None: 
        X, y  = build_data_matrix(**kargs)
    lablels = y

    return

def map_clusters(cluster_labels, X):
    """
    Map cluster/class labels to cluster IDs. 

    Read the labels array and clusters label and return the set of words in each cluster

    Input
    -----
    cluster_labels: cluster IDs ~ oredering the training instances (X)
    X: any inputs (X, y, D) consistent with the labeling

    Output
    ------
    A hashtable: cluster IDs -> labels
   
    e.g. 

    data = [x1, x2, x3, x4 ]  say each x_i is a 100-D feature vector
    labels = [0, 1, 0, 0]  # ground truth labels
    clusters = [0, 1, 1, 0]   say there are only two clusters 
    => cluster_to_docs: 
       {0: [0, 0], 1: [1, 0]}
    """
    cluster_to_docs = utils.autovivify_list()

    # mapping i-th label to cluster cid
    for i, cid in enumerate(cluster_labels):  # cluster_labels is listed in the order of (sequences ~ labels)
        cluster_to_docs[ cid ].append( X[i] )
    return cluster_to_docs

def map_clusters2(cluster_labels, docs): 
    """
    Input
    -----
    docs: can be labels, true docuemnts, other representations characterizing the documents

    """
    def repr(doc): 
        # [todo] policy plug-in 
        return doc 

    assert len(cluster_labels) == len(docs)
    cluster_to_docs = utils.autovivify_list()

    # mapping i-th label to cluster cid
    for i, cid in enumerate(cluster_labels):  # cluster_labels is listed in the order of (sequences ~ labels)
        cluster_to_docs[ cid ].append( docs[i] )
    return cluster_to_docs

def evalNClusters(X=None, y=None, **kargs):
    """
    Find 'optimal' number of clusters. 
    """ 
    pass

def eval_cluster(clusters, labels, cluster_to_labels=None, **kargs): # [refactor] cluster.analyzer
    """
    Input
    -----
    clusters: list of cluster labels (whose size is the same as X, i.e. the data from which clusters
              were derived)
    labels (y): cluster labels in the order of the original data set (X, y) from which clusters were 
              derived. 
    cluster_to_labels: generated by applying map_cluster()
    """
    def hvol(tb): # volume of hashtable
        return sum(len(v) for v in tb.values())

    # [params]
    topn_clusters = kargs.get('topn_clusters', None)
        
    y = labels
    assert y is not None, "Could not evaluate purity without ground truths given."
        
    # cluster_to_labels = map_clusters(clusters, y)
    # if 'cluster_to_labels' in locals():
    if cluster_to_labels is None: 
        cluster_to_labels = map_clusters(clusters, y)

    N = n_total = hvol(cluster_to_labels)
        
    # [output] purity_score/score, cluster_labels/clabels, ratios, fractions, topn_ratios
    res = {}

    ulabels = sorted(set(y))
    n_labels = len(ulabels)

    res['unique_label'] = res['unique_labels'] = res['ulabels'] = res['ulabel'] = ulabels

    maxx = []
    clabels = {}  # cluster/class label by majority vote
    for cid, labels in cluster_to_labels.items():
        counts = collections.Counter(labels)
        l, cnt = counts.most_common(1)[0]  # [policy]
        clabels[cid] = l            
        maxx.append(max(counts[ulabel] for ulabel in ulabels))

    res['purity_score'] = res['score'] = sum(maxx)/(n_total+0.0)
    res['cluster_label'] = res['cluster_labels'] = res['clabels'] = clabels
        
    # cluster ratios for each (unique) label 
    ratios = {ulabel:[] for ulabel in ulabels}
    fractions = {ulabel:[] for ulabel in ulabels}
    for ulabel in ulabels: # foreach unique label 
        for cid, labels in cluster_to_labels.items(): # foreach cluster (id)
            counts = collections.Counter(labels)
            r = counts[ulabel]/(len(labels)+0.0) # given a (true) label, find the ratio of that label in a cluster
            rf = (counts[ulabel], len(labels)) # fraction format
            ratios[ulabel].append((cid, r))
            fractions[ulabel].append((cid, rf))
    res['ratio'] = res['ratios'] = ratios # cluster purity ratio for each label: cluster_label -> [(cid, ratio)]
    res['fraction'] = res['fractions'] = fractions # cluster_label -> [(cid, (n_label, n_total))]

    # ratio of the label determined by majority votes 
    ratios_max_votes = {}  # cid -> label -> ratio
    for cid, lmax in clabels.items():  # cid, label of cluster by max vote
        ratios_max_votes[cid] = dict(res['ratios'][lmax])[cid]
    res['ratio_max_vote'] = res['ratios_max_votes'] = ratios_max_votes

    # rank purest clusters for each label and find which clusters to study
    # args: only analyze 'topn_clusters' 
    ranked_ratios = {}
    if topn_clusters is not None: 
        for ulabel in ulabels: 
            ranked_ratios[ulabel] = sorted(ratios[ulabel], key=lambda x: x[1], reverse=True)[:topn_clusters]  # {(cid, r)}
    else: 
        for ulabel in ulabels: 
            ranked_ratios[ulabel] = sorted(ratios[ulabel], key=lambda x: x[1], reverse=True) # {(cid, r)}            
    res['ranked_ratio'] = res['ranked_ratios'] = ranked_ratios
    
    return res # keys: ['unique_label', 'purity_score', 'cluster_label', 'ratio', 'fraction', 'ratio_max_vote', 'ranked_ratio', ]

def cluster_documents(X=None, y=None, D=None, clusters=None, **kargs):
    """
    Produce a mapping from cluster ID to documents (coding sequences). 

    """
    from pattern import diabetes, ptsd  # and possibly other diseases 
    import seqTransform as st   # transform sequences
    from sklearn.feature_extraction.text import TfidfTransformer
    # import algorithms  # count n-grams
    # from itertools import chain  # faster than flatten lambda (nested list comprehension)

    cohort_name = kargs['cohort'] # kargs.get('cohort', 'diabetes')

    # [params] I/O
    basedir = outputdir = kargs.get('outputdir', seqparams.get_basedir(cohort=cohort_name))  # [output] local data directory

    # [params] training data
    w2v_method = kargs.get('w2v_method', 'sg')  # cbow  (sg=1 or sg=0 <- cbow)
    d2v_method = kargs.get('d2v_method', 'tfidfavg')  # options: PVDM, average
    std_method = kargs.get('std_method', 'minmax')  # standardizing feature values
    
    # [params] cluster analysis
    cluster_method = kargs.get('cluster_method', 'unknown') # debug only
    n_clusters = kargs.get('n_clusters', 10)
    
    cluster_label_policy = 'max_vote'  # policy for labeling cluster members (via heuristics-based labeling)
    min_clusterID = 0 # default cluster ID (only one cluster, e.g. PTSD chorot without subtyping)

    # [params] pathway mining, ngram results
    tApplyGroupbySort = True

    # [params] training data
    n_classes = seqparams.arg(['n_classes', 'n_labels'], 1, **kargs) # None => don't know but need to know it to loading proper tset
    tset_type = seqparams.normalize_ttype(n_classes)  # convert to a canoical training set type     

    # [params] pathways
    order_type = seqparams.arg(['order_type', 'otype'], None, **kargs)
    partial_order = kargs.get('partial_order', True)
    if order_type is None: 
        order_type = 'partial' if partial_order else 'total'
    else: 
        partial_order = True if order_type.startswith('part') else False

    ctype = content_type = kargs.get('ctype', 'mixed', ) # 'diagnosis', 'medication', 'mixed'  # => seq_ptype: (diag, med, regular)
    policy_type = policy = seqparams.arg(['policy_type', 'ptype'], 'noop', **kargs) #  prior,  noop: (no cut), posterior: seq after diagnosis

    # predicate for verifying if it is a target disease for each element in the sequence
    seq_ptype = seqparams.normalize_ctype(seq_ptype=ctype)  # overwrites global value
    save_gmotif = True

    # [params] sequence
    read_mode = kargs.get('read_mode', 'doc') # documents/doc (one patient one sequence) or sequences/seq
    # seq_ptype = kargs.get('seq_ptype', 'regular')  # global default but varies according to pathway analyses
    # policy = kargs.get('policy', 'prior') # posterior, noop (i.e. no cut and preserve entire sequence)
    tLoadMotif = kargs.get('load_motif', True)# load global motif

    ## Fetch training data 
    div(message='1. Loading training data (document vectors) ...')
    if X is None or y is None or D is None: 
        # assert not (n_classes is None and tset_type is None), \
        #     "n_classes and tset_type cannot both be unknown => cannot index into proper training set."
        print('io> reading training set file (cohort:%s, n_classes:%s)' % (cohort_name, n_classes))
        D, ts = data_matrix(read_mode=read_mode, seq_ptype=seq_ptype, w2v_method=w2v_method, d2v_method=d2v_method, 
                                tset_type=tset_type, n_classes=n_classes, cohort=cohort_name)
        X, y = evaluate.transform(ts, standardize_=std_method) # default: minmax

    print('params> n_clusters: %d, n_classes: %d | clusters given? %s' % (n_clusters, n_classes, clusters))
    
    ### perform cluster analysis

    if clusters is None and n_clusters > 1: 
        div(message='2. Running cluster analysis ...')
        # run cluster analysis

        # [params] cluster 
        range_n_clusters = kargs.get('range_n_clusters', None) # silhouette scores
        min_n_clusters = kargs.get('min_n_clusters', None)
        max_n_clusters = kargs.get('max_n_clusters', None)
        optimize_k = kargs.get('optimize_k', False)
        print('params> optimize_k? %s | min_k: %s, max_k: %s, k_range: %s' % (optimize_k, min_n_clusters, max_n_clusters, str(range_n_clusters)))

        # [todo] use cluster.cluster_analysis
        # [log] 
        clusters, cmetrics = cluster_analysis(X=X, y=y, n_clusters=n_clusters, 
                                cohort=cohort_name, 
                                cluster_method=cluster_method, optimize_k=optimize_k, 
                                    range_n_clusters=range_n_clusters, 
                                    min_n_clusters=min_n_clusters, max_n_clusters=max_n_clusters, 
                                    seq_ptype=seq_ptype, d2v_method=d2v_method, 
                                        save_=True, load_=False, outputdir=outputdir)  # signal outputdir for run_silhouette_analysis and gap_stats
    else: 
        div(message='Condition: Single cluster analysis (i.e. No clustering).')
        clusters = [min_clusterID] * X.shape[0]  # min_cluster_ID
        cmetrics = None

    # [params] pathways: derived 
    n_classes_verified = len(set(y))
    if n_classes is not None: assert n_classes == n_classes_verified
    # tset_type = seqparams.normalize_ttype(n_classes)

    ### evaluate cluster purity 
    clabels = ratios_max_votes = lR = ulabels = None
    # topn_clusters = None # a number < n_clusters or None: analyze all clusters 
    
    if n_clusters > 1 and n_classes > 1:

        # cluster_to_labels = map_clusters(clusters, y) 
        # keys: ['unique_label', 'purity_score', 'cluster_label', 'ratio', 'fraction', 'ratio_max_vote', 'ranked_ratio', ]
        res = eval_cluster(clusters, labels=y, cluster_to_labels=None, **kargs)

        div(message='Result: %s clustering > purity: %f' % (cluster_method, res['purity_score']), symbol='#')
        
        # [I/O] save
        clabels = res['cluster_label']  # cluster (id) => label (by majority vote)
        ratios_max_votes = res['ratio_max_vote']

        lR = res['ranked_ratio'] # topn ratios by labels
        ulabels = res['unique_label']
        # res_motifs = {ulabel: {} for ulabel in ulabels}
    else: 
        # [todo] noop

        # [test] 
        if n_clusters > 1: 
            assert clusters is not None

            # keys: ['unique_label', 'purity_score', 'cluster_label', 'ratio', 'fraction', 'ratio_max_vote', 'ranked_ratio', ]
            res = eval_cluster(clusters, labels=y, cluster_to_labels=None, **kargs)
            assert res['purity_score'] == 1.0
            clabels = res['cluster_label']  # cluster (id) -> label (by majority vote)
            div(message='Result: %s clustering > purity: %f' % (cluster_method, res['purity_score']), symbol='#')
        else: 
            res = {}
            clables = {cid:1 for cid in range(n_clusters)}  # cluster 0, label=1
            res['purity_score'] = 1.0
            res['unique_label'] = [1]

    #### Pathway analysis by types: otype, ctype, ptype
    # document set D must be available to continue 
    # assert D is not None and len(D) > 0
    if D is None or len(D) == 0: 
        print('Warning: No input coding sequences available. Exiting ...')
        return None

    # [params] experimental settings 
    min_length, max_length = (1, 10)  
    # n_clusters

    # [params] I/O
    identifier = 'G%s-COP%s-%s-%s-nC%s-nL%s-D2V%s' % (cohort_name, ctype, order_type, policy_type, n_clusters, n_classes, d2v_method)
    # fname_tfidf = 'tfidf-%s-%s.csv' % (ctype, identifier)    

    # [test]
    n_doc0 = X.shape[0]
    assert len(D) == X.shape[0], "n_doc: %d while nrows of X: %d" % (len(D), n_doc0)
    n_cluster_id = len(clusters) if clusters is not None else 1
    
    print('test> inspecting via cluster method: %s' % cluster_method)
    assert n_doc0 == n_cluster_id or clusters is None, "n_doc: %d while number of cluster IDs: %d (nrow of X: %d)" % (n_doc0, n_cluster_id, X.shape[0])
    
    ### build cluster maps 

    cluster_to_docs, DCluster = {}, {0: [], }
    cut_policy = policy_type # [synomym]
   
    # D must be available to continue 
    condition_predicate = seqparams.arg(['predicate', 'condition_predicate', ], None, **kargs) 
    # condtiion: if predicate is not provided, then by default, it'll go to seqTransform.getDiseasePredicate

    if clusters is not None:  # n_clusters > 1
        cluster_to_docs = map_clusters(clusters, D) # cluster ID => a set of documents (d in D) in the same cluster    
        DCluster = {cid:[] for cid in cluster_to_docs.keys()}  # cid -> doc segments

        # before the first diagnosis, inclusive
        for cid, docs in cluster_to_docs.items(): # cluster with the ORIGINAL documents
            for doc in docs: 
                DCluster[cid].append(st.transform(doc, cut_policy=cut_policy, inclusive=True, 
                                                    seq_ptype=seq_ptype, cohort=cohort_name, predicate=condition_predicate))
    else: # only one cluster 
        for doc in D: 
            DCluster[0].append(st.transform(doc, cut_policy=cut_policy, inclusive=True, 
                                                seq_ptype=seq_ptype, cohort=cohort_name, predicate=condition_predicate)) 
   
    # cidx = DCluster.keys() # cluster IDs
    # DCD = DCluster if n_clusters == 1 else merge_cluster(DCluster) 
    # DCD = merge_cluster(DCluster) # PROCESSED document set (DCD: Document Clustered Derivative)
    # n_doc_global = len(DCD); assert n_doc_global > 0, "Empty (global) input sequences!"

    return DCluster

def analyze_pathway(X=None, y=None, D=None, clusters=None, **kargs): # [refactor] pathwayAnalyzer.py
    """
    Analyze pathways (or loosely speaking n-grams) per sequence types; that is,
    this routine inspects n-grams patterns given the following conditions specifiied (to be used in a 
    loop): 

    Params 
    ------

    otype (order type)  : {partial_order, total_order, }   <todo> varying degree of partial ordering 
    ptype (policy type) : {'prior', 'posterior', 'noop', }
    ctype (cluster type): {'diagnosis', 'medication', 'mixed', }   ... analogus to seq_ptype (see normalized_ptype)
        + ctype -> content type or canonical sequence pattern type)     <todo> renaming? 


    Note
    ----
    1. TdidfTransformer
       The formula that is used to compute the tf-idf of term t is tf-idf(d, t) = tf(t) * idf(d, t), 
       and the idf is computed as idf(d, t) = log [ n / df(d, t) ] + 1 (if smooth_idf=False), 
       where n is the total number of documents and df(d, t) is the document frequency; the document frequency 
       is the number of documents d that contain term t. The effect of adding “1” to the idf in the equation above 
       is that terms with zero idf, i.e., terms that occur in all documents in a training set, will not be entirely ignored. 

    """ 
    def merge_cluster(clusters, cids=None): 
        assert isinstance(clusters, dict)
        member = clusters.itervalues().next()
        assert hasattr(member, '__iter__')

        data = []
        if cids is None: # merge all 
           for cid, members in clusters.items(): # members can be a list or list of lists
               data.extend(members)
        else: 
            for cid in cids: 
                try: 
                    members = clusters[cid]
                    data.extend(members)
                except: 
                    raise ValueError, "Unknown cluster ID: %s" % cid
        if not data:
            assert cids is not None 
            print('warning> No data selected given cids:\n%s\n' % cids)
        return data
    def motif_stats0(cmotifs, cid=None):  # [input] cluster id, cluster motifs
        # rset = {}  # result set 
        # tfidf_stats[ngr]['cids']: which clusters does an ngram appear? 
        if cid is None: 
            for n, counts in cmotifs.items(): # foreach n, count object
                for ngr, c in counts:  # foreach n-gram, count
                    if not tfidf_stats.has_key(ngr): 
                        tfidf_stats[ngr] = {}
                        tfidf_stats[ngr]['count'] = 0
                        # tfidf_stats[ngr]['cids'] = []
                    tfidf_stats[ngr]['count'] += c
                    
        else: 
            for n, counts in cmotifs.items(): 
                for ngr, c in counts: 
                    if not tfidf_stats.has_key(ngr): 
                        tfidf_stats[ngr] = {}
                        tfidf_stats[ngr]['count'] = 0  # appeared in one cluster
                        tfidf_stats[ngr]['cids'] = set() 
                    
                    tfidf_stats[cid]    
                    tfidf_stats[ngr]['count'] += c     # global count (which should be available in global_motifs)
                    tfidf_stats[ngr]['cids'].add(cid)  # document/cluster frequency
                    # ngram_cidx[ngr].append(cid)
        return tfidf_stats
    def motif_stats(cmotifs, cid=0, topn=None):  # use 'topn' to set an upper limit of motifs to analyze

        # [test]
        assert 'idf_stats' in locals(), "(inverse) document/cluster frequency dictionary has not been defined."
        assert 'tfidf_stats' in locals(), "dictionary 'tfidf_stats' should have been defined prior to this call" 
        # assert 'global_motifs' in locals(), "need 'global_motifs' to get global statistics of n-grams"
        assert 'global_motifs_test' in locals(), "need 'global_motifs_test' to test ngram existence in clusters."
        
        if not tfidf_stats.has_key(cid): tfidf_stats[cid] = {}
        # if not tfidf_stats.has_key('global'): tfidf_stats['global'] = {}

        for n, counts in cmotifs.items(): 
            for ngr, c in counts:   
                assert ngr in global_motifs_test, "found %s in cluster but not in global scope" % ngr
                # if not idf_stats.has_key(ngr): idf_stats[ngr] = set()  # can be expensive
                assert ngr in idf_stats, "idf_stats not initialized yet."
                tfidf_stats[cid][ngr] = c 
                # tfidf_stats['global'][ngr] = global_motifs[n][ngr]  # global count of 'ngr'
                idf_stats[ngr].add(cid) # so the size of idf_stats[ngr] is the DF (or cluster frequency, CF, in our case)

        return 
    def motif_stats2(topn=10, min_freq=1, min_n=1):  # compute tf-idf scores,etc. 

        res = {}  # I/O: result set

        # [precond]
        cids = sorted(DCluster.keys()) # ascending order
        assert len(cids) <= n_clusters, "More than requested number of clusters? %d > n_clusters: %d" % (len(cids), n_clusters)
        assert  set(tfidf_stats.keys()) <= set(cids), "Not all cluster IDs are in tfdif table.\n + cids: %s vs tfidf keys: %s" % \
            (cids, tfidf_stats.keys())
        assert 'global_motifs' in locals(), "need 'global_motifs' to get global statistics of n-grams."


        # [def] topn for each length
        topn_motifs = []
        for n, ngr_cnts in global_motifs.items(): 
            if n < min_n: continue # only look at n-gram where n >= min_n (in global scope)
            for ngr, cnt in ngr_cnts: # [(ng, count)]
                if cnt >= min_freq:  # ignore (very) infrequent n-grams 
                    topn_motifs.append(ngr)
        n_motifs = len(topn_motifs)

        # [log] Found 207026 n-grams (with min_n: 1 & min_freq: 2)
        print('info> Found %d n-grams (with min_n: %d & min_freq_global: %d)' % (n_motifs, min_n, min_freq))

        ### [def] topN ngrams: this strategy results in mostly unigrams
        # populate n-grams and compute tf-idf scores for (topn) n-grams
        # topn_motifs2 = set()
        # nC = 0
        # for cid, counts in tfidf_stats.items(): # ngr: n-gram in tuple repr
        #     counter = collections.Counter(counts) 
            
        #     # [def] this definition leads to predominantly unigrams
        #     freq_items = counter.most_common(topn)  # topn are mostly unigrams

        #     if cid == random.sample(cids, 1)[0]: 
        #         print('test> freq_items (cid=%s):\n%s\n' % (cid, freq_items))
        #     topn_motifs2.update([ngr for ngr, cnt in freq_items])
        #     nC += 1 

        # compute tf-idf scores beyond cluster boundaries
        # n_motifs = len(topn_motifs)
        # print('info> Found %d n-grams (union of topn=%d of all clusters)' % (n_motifs, topn))
        
        ### Compute n-grams within the context of topn_motifs ### 

        # sort ~ length 
        res['motif'] = topn_motifs = sorted(topn_motifs, key=lambda s: len(s), reverse=False)
        print('info> final global motif examples:\n  + %s\n  + %s\n' % (str(topn_motifs[0]), str(topn_motifs[-1])))

        # cluster occurences: which clusters contain this n-gram? 
        res['ngram_to_clusters'] = {}
        if not 'idf_stats' in locals(): 
            for ngr in topn_motifs: # ngr: n-gram in tuple repr
                if not res['ngram_to_clusters'].has_key(ngr): res['ngram_to_clusters'][ngr] = []
                for cid, counts in tfidf_stats.items(): 
                    n = counts[ngr]
                    if n > 0: 
                        res['ngram_to_clusters'][ngr].append(cid)
        else: 
            assert len(idf_stats) > 0
            for ngr in topn_motifs: # ngr: n-gram in tuple repr
                if not res['ngram_to_clusters'].has_key(ngr): res['ngram_to_clusters'][ngr] = []
                res['ngram_to_clusters'][ngr] = idf_stats[ngr] # cids with non-zero occurrences of 'ngr'
        
        # tfx = np.zeros((nC, n_motifs))
        print('test> topn_motifs:\n%s\n' % list(topn_motifs)[-10:])

        # build temporary ngram frequency from global_motifs 
        # res['gtf'] = gtfx = [dict(global_motifs[len(ngr)])[ngr] for ngr in topn_motifs]  # this is slow
       
        if not 'global_motifs_test' in locals(): 
            gdict = {} # map directly from ngr to count
            for n, ngr_cnts in global_motifs.items(): 
                for ngr, cnt in ngr_cnts: 
                    gdict[ngr] = cnt
            res['gtf'] = gtfx = [gdict[ngr] for ngr in topn_motifs]  # global term frequency
            gdict = None; gc.collect()
        else: 
            res['gtf'] = gtfx = [global_motifs_test[ngr] for ngr in topn_motifs]

        ctfx = []
        # cids = sorted(tfidf_stats.keys())  # ascending order
        # cids = range(n_clusters)
        print("info> compute 'term frequency (tf)' for each cluster (n_clusters: %d)" % len(cids))
        for cid in cids:
            counts = tfidf_stats[cid]  # [(ngr, count)] 

            # 'topn_motifs' consists of union of motifs from across all clusters, therefore each pattern may not always have a count
            ctfx.append([counts.get(ngr, 0) for ngr in topn_motifs])  

        res['ctf'] = res['tf'] = ctfx  # cluster-specific term frequencies
        print("info> compute tf-idf scores (# of documents=n_clusters: %d)" % len(cids))
        transformer = TfidfTransformer(smooth_idf=False) # see [1]
        tfidf_scores = transformer.fit_transform(ctfx) # [[c1], [c2], ..., [cn]]
        res['tfidf'] = tfidf_scores.toarray()

        # the result can be sent to pathwayAnalyzer for further analysis
        return res  # ['motif', 'gtf', 'ctf', 'tf', 'tfidf']
                 
    import motif as mf
    from seqparams import TSet 
    import seqTransform as st   # transform sequences
    from pattern import diabetes, ptsd  # and possibly other diseases 
    from sklearn.feature_extraction.text import TfidfTransformer
    
    # import algorithms  # count n-grams
    # from itertools import chain  # faster than flatten lambda (nested list comprehension)
    cohort_name = kargs['cohort'] # kargs.get('cohort', 'diabetes')

    # [params] I/O
    #          seqparams.get_basedir(cohort=cohort_name)
    basedir = outputdir = kargs.get('outputdir', TSet.getPath(cohort=cohort_name))  # [output] local data directory

    # [params] training data
    w2v_method = kargs.get('w2v_method', vector.W2V.w2v_method)  # cbow  (sg=1 or sg=0 <- cbow)
    d2v_method = kargs.get('d2v_method', vector.D2V.d2v_method)  # options: pv-dm, pv-dbow, average
    std_method = kargs.get('std_method', 'minmax')  # standardizing feature values
    
    # [params] cluster analysis
    cluster_method = kargs.get('cluster_method', 'unknown') # debug only
    n_clusters = kargs.get('n_clusters', 10)
    topn_clusters_only = kargs.get('topn_clusters_only', False) # analyze topn clusters only
    topn_ngrams = 10 # analyze only top n ngrams (for each n)
    topn_ngrams_global = 100 # topn across all n 
    
    cluster_label_policy = 'max_vote'  # policy for labeling cluster members (via heuristics-based labeling)
    min_clusterID = 0 # default cluster ID (only one cluster, e.g. PTSD chorot without subtyping)

    # [params] pathway mining, ngram results
    tApplyGroupbySort = True

    # [params] classification on (surrogate) labels
    # n_classes = seqparams.arg(['n_classes', 'n_labels'], None, **kargs)

    # [params] training data
    n_classes = 1     

    # [params] pathways
    order_type = seqparams.arg(['order_type', 'otype'], None, **kargs)
    partial_order = kargs.get('partial_order', True)
    if order_type is None: 
        order_type = 'partial' if partial_order else 'total'
    else: 
        partial_order = True if order_type.startswith('part') else False

    ctype = content_type = kargs.get('ctype', 'mixed', ) # 'diagnosis', 'medication', 'mixed'  # => seq_ptype: (diag, med, regular)
    policy_type = policy = seqparams.arg(['policy_type', 'ptype'], 'noop', **kargs) #  prior,  noop: (no cut), posterior: seq after diagnosis

    # predicate for verifying if it is a target disease for each element in the sequence
    seq_ptype = seqparams.normalize_ctype(seq_ptype=ctype)  # overwrites global value
    save_gmotif = True

    # [params] sequence
    read_mode = kargs.get('read_mode', 'doc') # documents/doc (one patient one sequence) or sequences/seq
    # seq_ptype = kargs.get('seq_ptype', 'regular')  # global default but varies according to pathway analyses
    # policy = kargs.get('policy', 'prior') # posterior, noop (i.e. no cut and preserve entire sequence)
    tLoadMotif = kargs.get('load_motif', True)# load global motif

    ## Fetch training data 
    div(message='Step 1. Getting training data (document vectors) ...')
    if X is None or y is None or D is None: 
        # assert not (n_classes is None and tset_type is None), \
        #     "n_classes and tset_type cannot both be unknown => cannot index into proper training set."
        print('io> No input data (X, y, D), load or compute a new model')
        D, ts = data_matrix(cohort=cohort_name, seq_ptype=seq_ptype, w2v_method=w2v_method, d2v_method=d2v_method, 
                    test_model=kargs.get('test_model', True), 
                    load_model=kargs.get('load_model', True))
        X, y = evaluate.transform(ts, standardize_=std_method) # default: minmax
        n_classes = len(ts[TSet.target_field].unique())
        print('io> reading training set file (cohort:%s, n_classes:%s)' % (cohort_name, n_classes))

    # condition: (X, y), D, n_classes
    print('params> n_clusters: %d, n_classes: %d | clusters given? %s' % (n_clusters, n_classes, clusters))
    
    ### perform cluster analysis
    div(message='Step 2: Running cluster analysis ...')
    if clusters is None and n_clusters > 1: 
        # run cluster analysis

        # [params] cluster 
        range_n_clusters = kargs.get('range_n_clusters', None) # silhouette scores
        min_n_clusters = kargs.get('min_n_clusters', None)
        max_n_clusters = kargs.get('max_n_clusters', None)
        optimize_k = kargs.get('optimize_k', False)
        print('params> optimize_k? %s | min_k: %s, max_k: %s, k_range: %s' % (optimize_k, min_n_clusters, max_n_clusters, str(range_n_clusters)))

        # [todo] use cluster.cluster_analysis
        # [log] 
        clusters, cmetrics = cluster_analysis(X=X, y=y, n_clusters=n_clusters, 
                                cohort=cohort_name, 
                                cluster_method=cluster_method, optimize_k=optimize_k, 
                                    range_n_clusters=range_n_clusters, 
                                    min_n_clusters=min_n_clusters, max_n_clusters=max_n_clusters, 
                                    seq_ptype=seq_ptype, d2v_method=d2v_method, 
                                        save_=True, load_=False, outputdir=outputdir)  # signal outputdir for run_silhouette_analysis and gap_stats
    else: 
        div(message='Condition: Single cluster analysis (i.e. No clustering).')
        clusters = [min_clusterID] * X.shape[0]  # min_cluster_ID
        cmetrics = None

    # [params] pathways: derived 
    n_classes_verified = len(set(y))
    if n_classes is not None: assert n_classes == n_classes_verified
    tset_type = seqparams.normalize_ttype(n_classes)

    ### evaluate cluster purity 
    
    clabels = ratios_max_votes = lR = ulabels = None
    # topn_clusters = None # a number < n_clusters or None: analyze all clusters 
    
    div(message='Step 2a: Evaluate Clusters (intrinsic) ...')
    if n_clusters > 1 and n_classes > 1:

        # cluster_to_labels = map_clusters(clusters, y) 
        # keys: ['unique_label', 'purity_score', 'cluster_label', 'ratio', 'fraction', 'ratio_max_vote', 'ranked_ratio', ]
        res = eval_cluster(clusters, labels=y, cluster_to_labels=None, **kargs)

        div(message='Result: %s clustering > purity: %f' % (cluster_method, res['purity_score']), symbol='#')
        
        # [I/O] save
        clabels = res['cluster_label']  # cluster (id) => label (by majority vote)
        ratios_max_votes = res['ratio_max_vote']

        lR = res['ranked_ratio'] # topn ratios by labels
        ulabels = res['unique_label']
        # res_motifs = {ulabel: {} for ulabel in ulabels}
    else: 
        # [todo] noop

        # [test] 
        if n_clusters > 1: 
            assert clusters is not None

            # keys: ['unique_label', 'purity_score', 'cluster_label', 'ratio', 'fraction', 'ratio_max_vote', 'ranked_ratio', ]
            res = eval_cluster(clusters, labels=y, cluster_to_labels=None, **kargs)
            assert res['purity_score'] == 1.0
            clabels = res['cluster_label']  # cluster (id) -> label (by majority vote)
            div(message='Result: %s clustering > purity: %f' % (cluster_method, res['purity_score']), symbol='#')
        else: 
            res = {}
            clables = {cid:1 for cid in range(n_clusters)}  # cluster 0, label=1
            res['purity_score'] = 1.0
            res['unique_label'] = [1]

    #### Pathway analysis by types: otype, ctype, ptype
    # document set D must be available to continue 
    if D is None: 
        print('Warning: No input coding sequences available. Exiting ...')
        return None

    # [params] experimental settings 
    min_length, max_length = (1, 10)  
    # n_clusters

    # [params] I/O
    identifier = 'G%s-COP%s-%s-%s-nC%s-nL%s-D2V%s' % (cohort_name, ctype, order_type, policy_type, n_clusters, n_classes, d2v_method)
    # fname_tfidf = 'tfidf-%s-%s.csv' % (ctype, identifier)    

    # [test]
    n_doc0 = X.shape[0]
    assert len(D) == X.shape[0], "n_doc: %d while nrows of X: %d" % (len(D), n_doc0)
    n_cluster_id = len(clusters) if clusters is not None else 1
    
    print('test> inspecting via cluster method: %s' % cluster_method)
    assert n_doc0 == n_cluster_id or clusters is None, "n_doc: %d while number of cluster IDs: %d (nrow of X: %d)" % (n_doc0, n_cluster_id, X.shape[0])
    
    ### build cluster maps 
    div(message='Step 2b. Build cluster map (associate documents with cluster IDS) ...')
    cluster_to_docs, DCluster = {}, {0: [], }
    cut_policy = policy_type # [synomym]
   
    # D must be available to continue 
    condition_predicate = seqparams.arg(['predicate', 'condition_predicate', ], None, **kargs) 
    # condtiion: if predicate is not provided, then by default, it'll go to seqTransform.getDiseasePredicate

    if clusters is not None:  # n_clusters > 1
        cluster_to_docs = map_clusters(clusters, D) # cluster ID => a set of documents (d in D) in the same cluster    
        DCluster = {cid:[] for cid in cluster_to_docs.keys()}  # cid -> doc segments

        # before the first diagnosis, inclusive
        for cid, docs in cluster_to_docs.items(): # cluster with the ORIGINAL documents
            for doc in docs: 
                DCluster[cid].append(st.transform(doc, cut_policy=cut_policy, inclusive=True, 
                                                    seq_ptype=seq_ptype, cohort=cohort_name, predicate=condition_predicate))
    else: # only one cluster 
        for doc in D: 
            DCluster[0].append(st.transform(doc, cut_policy=cut_policy, inclusive=True, 
                                                seq_ptype=seq_ptype, cohort=cohort_name, predicate=condition_predicate)) 
   
    cidx = DCluster.keys() # cluster IDs
    # DCD = DCluster if n_clusters == 1 else merge_cluster(DCluster) 
    DCD = merge_cluster(DCluster) # PROCESSED document set (DCD: Document Clustered Derivative)
    n_doc_global = len(DCD); assert n_doc_global > 0, "Empty (global) input sequences!"

    # [params]
    div(message='Step 3. Evaluate global motifs ...')
    tLoadMotif = False # load global motif
    global_motifs = eval_motif(DCD, topn=None, ng_min=min_length, ng_max=max_length, 
                                    partial_order=partial_order, 
                                    mtype='global', ctype=ctype, ptype=cut_policy,
                                        seq_ptype=seq_ptype, d2v_method=d2v_method, 
                                        cohort=cohort_name, 
                                        save_=True, load_=tLoadMotif, outputdir=outputdir)  # n (as in n-gram) -> topn counts [(n_gram, count), ]
    
    # [test]
    global_motifs_test = {}  # map directly from ngr to count 
    for n, ngr_cnts in global_motifs.items(): 
        for ngr, cnt in ngr_cnts: 
            global_motifs_test[ngr] = cnt

    print('params> ctype: %s, policy: %s, n_clusters: %d, n_doc: %d' % (ctype, policy_type, len(cidx), n_doc_global))

    tfidf_stats = {} # {'global': {}} # counts of cluster occurrences for each ngram => compute idf score and combined ngram statistics
    
    # idf_stats = {ngr:set() for n, counts in global_motifs for ngr, count in counts}  # ngr -> cids
    idf_stats = {}
    for n, ngr_cnts in global_motifs.items(): 
        for ngr, cnt in ngr_cnts: 
            idf_stats[ngr] = set()
    print('info> total num of global ngrams found: %d' % len(idf_stats))
    
    div(message='status> iterating over all (%s) clusters (N=%d)' % (cluster_method, len(cidx)))
    # if topn_clusters_only set to False, then it'll iterate through all clusters

    # selective tests 
    n_cluster_loo_test = 5 
    cidx_loo = random.sample(cidx, min(len(cidx), n_cluster_loo_test)) # leave-one-out cluster ID set

    # [params] control 
    tLoadMotif = False
    tLOOTest = False

    # tdict_local = {1: 20, 2: 20, }  # n -> topn number of n-grams to analyze, if not specified, use default (5)
    div(message='Step 3a. Evaluate cluster motifs (local, i.e. cluster by cluster) ...')

    for cid, seqx in DCluster.items(): # foreach (set of) clustered documents, analyze their motifs  
        clabel = 1 if n_classes == 1 else clabels[cid]  # cluster label based on majority votes

        # set context motifs to global_motifs to find out how cluster motifs counts compared to global level 
        div(message='Result: Label %s + Cluster #%d > cluster-wise statistics ... ' % (clabel, cid))
        n_doc_cluster = len(seqx) # a list of sequences
        cluster_motifs = eval_cluster_motif(seqx, topn=None, ng_min=min_length, ng_max=max_length, 
                                                partial_order=partial_order, ctype=ctype, ptype=cut_policy,  
                                                cid=cid, label=clabel, context_motifs=global_motifs, 
                                                cluster_method=cluster_method, seq_ptype=seq_ptype, d2v_method=d2v_method, 
                                                    cohort=cohort_name, 
                                                    save_=True, load_=tLoadMotif, outputdir=outputdir) # n -> (n-gram -> count)

        # [test] leave one (cluster) out test
        # [note] topn: None => keep all counter elements
        if n_clusters > 1: 
            if cid in cidx_loo: 
                div(message='Test: Leave-one-out test for cluster #%d' % cid)

                # set focused_motifs to cluster motifs above to observe their counts in the cluster complement (i.e. all but this cluster)
                seqx_compl = merge_cluster(DCluster, cids=set(cidx)-set([cid]))
                n_doc_cluster_compl = len(seqx_compl)
                assert n_doc_global == n_doc_cluster + n_doc_cluster_compl, "total: %d != cluster: %d + complement: %d" % \
                            (n_doc_global, n_doc_cluster, n_doc_cluster_compl)

                if tLOOTest: 
                    loo_motifs = eval_cluster_motif(seqx_compl,  
                                                        topn=None, ng_min=min_length, ng_max=max_length, 
                                                        partial_order=partial_order, context_motifs=global_motifs, focused_motifs=cluster_motifs,
                                                            cid='%s-complement' % cid, label=clabel, ctype=ctype, ptype=cut_policy,
                                                            cluster_method=cluster_method, seq_ptype=seq_ptype, d2v_method=d2v_method,   
                                                                cohort=cohort_name, 
                                                                save_=True, load_=tLoadMotif, outputdir=outputdir)  # n (as in n-gram) -> topn counts [(n_gram, count), ]

        # find counts of global motifs/ngrams within this cluster
        n_ngrx0 = len(tfidf_stats)  # : cid -> ngram -> count 
        # establish n-gram counts per cluster
        motif_stats(cluster_motifs, cid=cid)  #  tfidf_stats: {cid} -> {ngr} -> ngr_counts
        assert cid in tfidf_stats and len(tfidf_stats) > 0 and len(tfidf_stats) >= n_ngrx0

        ### end foreach cluster 
    div(message='status> Cluster motifs completed.')

    # [I/O] save tfidf_stats 
    if len(tfidf_stats) > 0:   #  tfidf_stats: {cid} -> {ngr} -> ngr_counts
        # assert n_clusters > 1, "No tdidf stats for only one cluster." 

        if n_clusters > 1: 
            assert len(clabels) > 0 and isinstance(clabels, dict)

        header = ['cid', 'length', 'ngram', 'tfidf', 'tf_cluster', 'tf_global', 'cluster_freq', 'label', 'cluster_occurrence', ]
        if n_classes > 1: 
            # cluster_freq: number of clusters that a given n-gram appear
            # tf: n-gram frequency within a cluster
            header_local = ['length', 'ngram', 'tfidf', 'tf_cluster', 'tf_global', 'label', 'cluster_freq', 'cluster_occurrence', ]
        else: 
            header_local = ['length', 'ngram', 'tfidf', 'tf_cluster', 'tf_global', 'cluster_freq', 'cluster_occurrence', ]
        assert set(header_local) < set(header)
                
        wsep = ' '  # separator for list objects 
                
        # ref_entries = set(DCluster.keys()); ref_entries.add('global')
        topn = topn_ngrams  # topn for each n (of n-gram)
        
        min_freq_global = 1  # min_freq at global scope
        min_freq_local = 3  # min_freq at cluster level

        res = motif_stats2(min_freq=min_freq_global, min_n=1)  # keys: ['motif', 'gtf', 'ctf', 'tf', 'tfidf']
        n_motifs = len(res['motif'])
        print('info> Found %d n-grams (min_freq_global: %d, min_freq_local: %d)' % (n_motifs, min_freq_global, min_freq_local))
        
        # [params] groupby    
        pivots = ['length', 'tfidf', ]  # sort 
        pivots_group = ['length', ]  # groupby
        apply_groupby_sort = tApplyGroupbySort

        # dfg = []
        div(message='Step 4. Create per-cluster pathway files: n-gram motifs ...')
        for cid, counts in tfidf_stats.items(): # ngr: n-gram in tuple repr
            fname_tfidf_local = 'pathway_C%s-%s.csv' % (cid, identifier)  # [I/O] local
            assert fname_tfidf_local.find(cohort_name) > 0
            fpath_local = os.path.join(outputdir, fname_tfidf_local) 

            adict = {h:[] for h in header}
           
            # foreach common motif across all clusters
            # counter = collections.Counter(counts)
            # freq_motifs = counter.most_common(topn)
            assert len(res['motif']) == len(res['gtf']) 

            topn_prime = 0
            for i, ngr in enumerate(res['motif']): # ordered topn motifs/ngrams
               
                tf_ci = res['ctf'][cid][i]  # TF(i-th ngram) in cluster 'c' 

                # keep out ngrams with zero counts 
                if tf_ci >= min_freq_local:  # if this ngram appears in cluster 'cid' frequently enough
                    topn_prime += 1
                    tfidf_ci = res['tfidf'][cid][i]
                    assert tfidf_ci > 0, "term/cluster freq (%d > 0) but tfidf (%d < 0???)" % (tf_ci, tfidf_ci)
                    
                    adict['cid'].append(cid)
                    adict['length'].append(len(ngr))
                    adict['ngram'].append(mf.normalize(ngr))  # normalize motif
                    adict['tfidf'].append(tfidf_ci)
                    adict['tf_cluster'].append(tf_ci)
                    adict['tf_global'].append(res['gtf'][i])

                    cidset = res['ngram_to_clusters'].get(ngr, [])  # cluster-level frequency (~ document frequency)
                    adict['cluster_freq'].append(len(cidset))

                    cidstr = [str(e) for e in res['ngram_to_clusters'][ngr]]
                    adict['cluster_occurrence'].append(wsep.join(cidstr))
                    clabelstr = [str(clabels.get(e, -1)) for e in res['ngram_to_clusters'][ngr]]
                    adict['label'].append(wsep.join(clabelstr)) 
                
            # assert topn_prime <= topn, "N-grams with non-zero counts (%d) not consistent with topn=%d" % (topn_prime, topn)
            df = DataFrame(adict, columns=header)

            # [postcond] sort and groupby 
            # [todo] better way to convert back to dataframe from groupby object?
            if apply_groupby_sort: 
                # print('status> applying sort (%s) + groupby (%s)' % (str(pivots), pivots_group))
                # dfx = []
                # for g, dfi in df.sort_values(pivots, ascending=False).groupby(pivots_group):
                #     dfx.append(dfi)
                # df = pd.concat(dfx, ignore_index=True) 

                print('status> applying sort (%s)' % str(pivots))
                df = df.sort_values(pivots, ascending=False)

            print('io> saving tfidf stats for cluster #%d (dim: %s) > %s' % (cid, str(df.shape), fpath_local))
            dfc = df[header_local]
            dfc.to_csv(fpath_local, sep='|', index=False, header=True)

            # dfg.append(df)
        ### end foreach cluster 
        
        # use pathwayAnalyzer to further select top N n-grams within each cluster 

        # global statistics 
        # dfg = pd.concat(dfg, ignore_index=True)
        
        # [I/O] global
        div(message='Step 4. Create global pathway files: n-gram motifs ...')
        fname_tfidf = 'pathway_global-%s.csv' % identifier # identifier: (tset_type, cluster_method, seq_ptype, d2v_method)
        fpath = os.path.join(outputdir, fname_tfidf) # fname_tdidf(ctype, )

        # n_clusters: count number of cluster occurrences and cluster IDs
        header = ['length', 'ngram', 'tf_global', 'cluster_freq', 'cluster_occurrence', ] 
        adict = {h:[] for h in header}
        for i, ngr in enumerate(res['motif']): # ordered topn motifs/ngrams
            adict['length'].append(len(ngr))
            adict['ngram'].append(mf.normalize(ngr))  # normalize motif
            adict['tf_global'].append(res['gtf'][i])       

            cidstr = [str(e) for e in res['ngram_to_clusters'][ngr]]   
            adict['cluster_occurrence'].append(wsep.join(cidstr))

            cidset = res['ngram_to_clusters'].get(ngr, [])  # cluster-level frequency (~ document frequency)
            adict['cluster_freq'].append(len(cidset))
        
        dfg = DataFrame(adict, columns=header)
        
        pivots = ['length', 'tf_global', ]  # sort 
        # pivots_group = ['length', ]  # groupb
        if apply_groupby_sort: 
            # print('status> applying sort (%s) + groupby (%s)' % (str(pivots), pivots_group))
            # # dfg.assign(n_clusters = dfg['cluster_occurrence']).sort_values('f').drop('f', axis=1)
            # dfx = []
            # for g, dfi in dfg.sort_values(pivots, ascending=False).groupby(pivots_group):
            #     dfx.append(dfi)
            # dfg = pd.concat(dfx, ignore_index=True) 
            print('status> applying sort (%s)' % str(pivots))
            dfg = dfg.sort_values(pivots, ascending=False, kind='mergesort')  # need stable sort
            dfg = dfg.sort_values(['cluster_freq', ], ascending=True, kind='mergesort')  # least frequent across clusters => more informative

        print('io> saving global ngram stats (dim: %s) to %s' % (str(dfg.shape), fpath))
        dfg.to_csv(fpath, sep='|', index=False, header=True)

        # any n-grams that occur in majority of the clusters? (say, 80%)? 
        # fname_derived = 'pathway_global-sortnc-%s.csv' % identifier # identifier: (tset_type, cluster_method, seq_ptype, d2v_method)
        # fpath = os.path.join(outputdir, fname_derived) # fname_tdidf(ctype, )
        # dfg = dfg.sort_values(['n_clusters'], ascending=False)
        # print('io> saving derived ngram stats (dim: %s) to %s' % (str(dfg.shape), fpath))
        # dfg.to_csv(fpath, sep='|', index=False, header=True)

        # most common n-grams across all clusters? 
        # dfg.sort_values()

        div(message='Tf-idf ranked cluster motifs and pathway analyses completed.')
        
    if n_classes > 1 and len(clabels) > 0: 
        # [I/O] save cluster summary statistics (e.g. within cluster purity)
        fname_cluster_stats = 'cluster_stats-LP%s-%s.csv' % (cluster_label_policy, identifier)
        fpath = os.path.join(outputdir, fname_cluster_stats)
        header = ['cid', 'label', 'ratio']
        adict = {h: [] for h in header}
        # clabels:  cid -> label 
        # ratios_max_votes: cid -> ratio (~ label by majority votes)
        for cid, lmax in clabels.items(): 
            adict['cid'].append(cid)
            adict['label'].append(lmax)
            adict['ratio'].append(ratios_max_votes[cid])
        df = DataFrame(adict, columns=header)
        print('io> saving cluster stats (dim: %s) to %s' % (str(df.shape), fpath))
        df.to_csv(fpath, sep='|', index=False, header=True)

    return global_motifs

def eval_given_motif(seqx, motifs, partial_order=True): # [old]
    # assuming that the desired motifs are known 
    # import algorithms 
    if not motifs:
        print('warning> no input motifs given.')
        return {} 

    # motifs_prime = motifs
    if isinstance(motifs, dict):  # n -> counts: [(ngram, count)] i.e. counts is an ordered list/dict 
        motifs_prime = []
        clist = motifs.itervalues().next()  # [log] n=1 => [('401.9', 441985), ('250.00', 286408),...]
        if isinstance(clist, dict):   
            for ng, counts in motifs.items(): # ng: n as in n-gram, counts: {(ngram, count)}
                motifs_prime.extend(counts.keys()) 
        else: 
            if len(clist) > 0: assert len(clist[0]) == 2, "not (ngram, count)? %s" % str(clist)
            for ng, counts in motifs.items(): # ng: n as in n-gram, counts: [(ngram, count), ]
                motifs_prime.extend(ngm for ngm, cnt in counts)
                # counts_prime = dict(counts)
                # motifs_prime.extend(counts_prime.keys()) 
    elif isinstance(motifs, list) or isinstance(motifs, tuple): # ordered dictionary
        me = random.sample(motifs, 1)[0]  # me: motif element
        assert len(me) == 2 and isinstance(me[1], int)
        motifs_prime = []
        for ngram, count in motifs:
            motifs_prime.append(ngram)
    else: 
        assert hasattr(motifs, '__iter__'), "Invalid motif input: %s" % str(motifs)   
        motifs_prime = motifs

    # print('test> input motifs:\n%s\n' % motifs_prime[:100])
    # [input] motifs is a set/list of n-grams (in tuples)
    return algorithms.count_given_ngrams2(seqx, motifs_prime, partial_order=partial_order)  # n -> counts: {(ngram, count)}

def eval_motif(seqx, topn=None, ng_min=1, ng_max=8, partial_order=True, **kargs):
    def eval_bigrams(words):
        wprev = 'start'  # special beginning token
        for w in words:
            yield (wprev, w)
            wprev = w
        return 
    def rel_motif_stats(gc_ng_count): # global over cluster map: n -> (n-gram -> count)
        assert global_motifs is not None
        res = {}
        # for n, ccounts in cluster_motifs: 
        ratios = {}  # n-gram -> ratio (cluster wrt global)
        fractions = {} 
        
        for n, counts in global_motifs.items(): # foreach n as in n-gram, descending order in counts
            local_ngrams = gc_ng_count[n] # (ngram -> count): occurrences of global n-grams in this cluster
            
            for ngram, gcnt in counts: # foreach n-gram, gcnt (global count)
                # lcnt = local_ngrams.get(ngram, 0)
                lcnt = local_ngrams.get(ngram, 0)  # all ngram in global should also have been evaluated in the context of cluster
                ratios[ngram] = lcnt/(gcnt+0.0)
                fractions[ngram] = (lcnt, gcnt)
        res['ratios'] = ratios 
        res['fractions'] = fractions
        return res

    import algorithms  # count n-grams
    from itertools import chain  # faster than flatten lambda (nested list comprehension)

    res = {}  # result set
    n_docs0 = len(seqx)

    # [note] for cluster-level motif, eval_cluster_motif() is suggested
    cohort_name = kargs.get('cohort', None)

    mtype = kargs.get('mtype', 'global')  # motif type: global, cluster-level (use cid) or from arbitrary sequence? 
    ctype = kargs.get('ctype', 'diagnosis') # code type: diagnosis, medication, lab, ... 
    save_motif = kargs.get('save_', True)
    load_motif = kargs.get('load_', True) 

    # [params] file 
    seq_ptype = kargs.get('seq_ptype', 'regular')
    d2v_method = kargs.get('d2v_method', 'tfidfavg')
    cluster_method = kargs.get('cluster_method', 'na')
    n_clusters = kargs.get('n_clusters', None)

    # [note] use otype or order_type instead of setting partial_order to specify more ordering types
    otype = seqparams.arg(['otype', 'order_type'], None, **kargs)  # part: partial
    if otype is None: 
        otype = 'part' if partial_order else 'total'
    else: 
        # no-op 
        print('params> use order type (while partial_order=%s): %s' % (partial_order, otype))

    ptype = seqparams.arg(['ptype', 'policy_type'], 'prior', **kargs)   # kargs.get('ptype', kargs.get('policy_type', 'prior'))
    print('eval_motif> input D, policy=%s | n_doc: %d, n_tokens(seqx[0]): %d' % (ptype, n_docs0, len(seqx[0])))

    # identifier = 'T%s-O%s-C%s-S%s-D2V%s' % (tset_type, order_type, cluster_method, seq_ptype, d2v_method)

    # [note] ctype and seq_ptype are correlated (ro=1.0)
    if n_clusters is None: 
        identifier = 'CMOP%s-%s-%s-%s-S%s-D2V%s' % (ctype, mtype, otype, ptype, seq_ptype, d2v_method)
    else: 
        identifier = 'CMOP%s-%s-%s-%s-nC%s-S%s-D2V%s' % (ctype, mtype, otype, ptype, n_clusters, seq_ptype, d2v_method)

    # identifier = 'C%s-M%s-S%s-D2V%s' % (ctype, mtype, seq_ptype, d2v_method)

    basedir = os.path.join(os.getcwd(), 'data') if cohort_name is None else seqparams.get_basedir(cohort=cohort_name)
    outputdir = kargs.get('outputdir', basedir)
    load_path = os.path.join(outputdir, 'motifs-%s.pkl' % identifier)

    print('params> coordinate (seq:%s, d2v: %s) | order: %s | type (c:%s, m:%s) => input n_docs:%d' % \
        (seq_ptype, d2v_method, otype, ctype, mtype, n_docs0))

    if load_motif and os.path.exists(load_path): 
        res = pickle.load(open(load_path, 'rb'))
        if len(res) > 0:
            print('io> loaded pre-computed %s motifs from %s' % (mtype, load_path)) 
            return res

    if ng_min < 1: ng_min = 1
    if ng_max < ng_min: ng_max = ng_min

    topnmap = {n: None for n in range(ng_min, ng_max+1)} # n_doc0  # max 
    tKeepAll = False
    if topn is None: # no limit, don't use most_common() to filter n-grams
        tKeepAll = True # no-op 
    elif isinstance(topn, dict):  # topn: n -> number of m most common n-grams
        # topn can be either None or a number
        for n in range(ng_min, ng_max+1): 
            topnmap[n] = topn.get(n, None) # default no upperbound
    else: 
        assert isinstance(topn, int)
        for n in range(ng_min, ng_max+1): 
            topnmap[n] = topn

    # common unigrams
    # print('input> seqx:\n%s\n' % seqx[:10])
    # tokens = list(chain.from_iterable(seqx)) # this flattens out entire docs, may cause extra counts of n-grams straddling between lines
    # counter = collections.Counter(tokens)  # cluster/population wise frequency
    # res[1] = counter.most_common(topn[1])

    # n = topn.get(1, topn_default)
    # print('verify> top %d unigram-frequencies (%s):\n%s\n' % (n, mtype, res[1]))  #  [('c', 2), ('b', 2)]
    # common_codes = [e[0] for e in res[1]]
    # print('verify> top %d codes (%s):\n%s\n' % (n, mtype, common_codes))

    # bigrams 
    # ng = 2 
    # bigrams = eval_bigrams(tokens)   # (), (), ... 
    # counter = collections.Counter(bigrams)  # population wise frequency
    # topn_counts = counter.most_common(topn)     
    # print('verify> top %d %s bi-frequencies:\n%s\n' % (topn, ctype, topn_counts))  #  [('c', 2), ('b', 2)]
    # common_ngrams = [e[0] for e in topn_counts]
    # print('        top %d %s bigrams:\n%s\n' % (topn, ctype, common_ngrams))  
    
    # if ng_max >= 2: # bi-grams+  
    # length: counters (n-gram: counts)  
    ngrams = algorithms.count_ngrams2(seqx, min_length=1, max_length=ng_max, partial_order=partial_order) 
    for ng in range(1, ng_max+1):  # +1: inclusive
        if not ngrams.has_key(ng): 
            print('> %d-gram motif does not exist in input sequences' % ng)
            continue
        counter = ngrams[ng] # length -> Counter: {(ngr, count)}

        # most_common results in {(ngr, count)}, a list of 2-tuples
        res[ng] = counter.most_common(topnmap[ng]) # if topn is None, then return all

        print("verify> top global %d-gram frequencies(topn=%s | ct=%s, ot=%s, pt=%s):\n%s\n" % \
            (ng, topnmap[ng], ctype, otype, ptype, str(res[ng][:10])) )  #  [('c', 2), ('b', 2)]
        # common_ngrams = [e[0] for e in topn_counts]

    res_df = None
    apply_groupby_sort = True 
    if save_motif: 
        # save both csv and pkl
        ### 1. csv
        # global path (can also save one file for each n-gram)
        fpath = os.path.join(outputdir, 'motifs-%s.csv' % identifier)
        header = ['length', 'ngram', 'global_count']  # use 'global_count' to be consistent with that in cluster_motifs*    
        adict = {h: [] for h in header} 
        wsep = ' ' # word separator

        for n, ngr_cnts in res.items(): 
            n_patterns = len(ngr_cnts)
            adict['length'].extend([n] * n_patterns)  # [redundancy]
            
            ngram_reprx, local_counts = [], []
            for i, (ngt, ccnt) in enumerate(ngr_cnts):  # ccnt: cluster-level count 
                # from tuple (ngt) to string (ngstr) 
                # ngstr = wsep.join(str(e) for e in ngt)

                # from tuple (ngt) to string (ngstr) 
                if isinstance(ngt, str): 
                    ngstr = ngt 
                else: # bigram+
                    assert isinstance(ngt, tuple), "invalid ngram format: %s" % str(ngt)
                    ngstr = wsep.join(str(e) for e in ngt)
                    
                    # [test]
                    if i == 0: assert isinstance(ngt, tuple), "invalid ngram: %s" % str(ngt)

                ngram_reprx.append(ngstr) # foreach element in n-gram tuple
                local_counts.append(ccnt)
            ### end for

            adict['ngram'].extend(ngram_reprx)
            adict['global_count'].extend(local_counts)

        ### end for

        df = DataFrame(adict, columns=header)
        
        pivots = ['length', 'global_count']
        if apply_groupby_sort: 
            df = df.sort_values(pivots, ascending=False)

        print('io> saving motif dataframe of dim: %s to %s' % (str(df.shape), fpath))
        df.to_csv(fpath, sep='|', index=False, header=True)

        ### 2. pickle file 
        fpath = load_path # os.path.join(outputdir, 'motifs-%s.pkl' % identifier)
        pickle.dump(res, open(fpath, "wb" ))

    ### end if save_motif


    return res # n -> counts: [(ngram, count)] // n as in ngram maps to counts which is an ordered list of (ngram, count)

def eval_cluster_motif(seqx, topn=None, ng_min=1, ng_max=8, partial_order=True, **kargs): 
    """
   
    Note
    ----
    1. topn n-grams 
       by default, eval_motif (usually used for finding global motifs) sets topn to None, which keeps all 
       elements in the counter 
       by contrary, eval_cluster_motif only keeps top_default (small number, not None)

       
    2. lookup_motif()
       Frequencies of cluster-level motifs in the global context. 
        

    Log
    ---
    1. .../tpheno/seqmaker/data/motifs-Cmedication-CID38-kmeans-Sregular-D2Vtfidf.csv

    """
    def size(motifs): 
        if motifs is None: return 0
        return utils.size_hashtable(motifs)

    import algorithms  # count n-grams
    from itertools import chain  # faster than flatten lambda (nested list comprehension)

    # res = {}  # result set

    cohort_name = kargs.get('cohort', None)

    n_docs0 = len(seqx)
    mtype = seqparams.arg(['cid', 'motif_type', 'mtype'], 'cluster', **kargs) # motif type (e.g. CID)
    # if mtype is None: mtype = kargs.get('cid_minus', 'unknown')
    ctype = kargs.get('ctype', 'diagnosis')  # code type: diagnosis, medication, lab, ... 
    otype = seqparams.arg(['otype', 'order_type'], None, **kargs)  # part: partial
    if otype is None: 
        otype = 'part' if partial_order else 'total'
    else: 
        # no-op 
        print('params> use order type (while partial_order=%s): %s' % (partial_order, otype))

    ptype = seqparams.arg(['ptype', 'policy_type'], 'prior', **kargs)   # kargs.get('ptype', kargs.get('policy_type', 'prior'))
    save_motif = kargs.get('save_', True)
    load_motif = kargs.get('load_', False) 

    # exclusion mode? if so, don't assert ccnt < gcnt (for the counts of ngrams)
    # exclusion_mode = True if mtype.lower().find('loo') >= 0 else False

    # comparison with global motifs if given 
    global_motifs = seqparams.arg(['global_motifs', 'context_motifs'], None, **kargs) # kargs.get('global_motifs', kargs.get('context_motifs', None))
    n_global_motifs = size(global_motifs)
    if global_motifs is None: 
        return eval_motif(seqx, topn=topn, ng_min=ng_min, ng_max=ng_max, partial_order=partial_order, **kargs)

    focused_motifs = kargs.get('focused_motifs', None) # if set, only keep track of these motifs
    n_focused_motifs = size(focused_motifs)
    is_cluster_complement = kargs.get('is_complement', focused_motifs is not None)

    # [params] file 
    label = seqparams.arg(['label', 'cluster_label'], None, **kargs)  # kargs.get('label', kargs.get('cluster_label', None))
    seq_ptype = kargs.get('seq_ptype', 'regular')
    d2v_method = kargs.get('d2v_method', 'tfidfavg')
    cluster_method = kargs.get('cluster_method', 'unknown')

    # [params] ID 
    identifier = 'CID%s-COP%s-%s-%s-C%s-S%s-D2V%s' % (mtype, ctype, otype, ptype, cluster_method, seq_ptype, d2v_method)  
    if label is not None: 
        identifier = 'CIDL%s-%s-COP%s-%s-%s-C%s-S%s-D2V%s' % (mtype, label, ctype, otype, ptype, cluster_method, seq_ptype, d2v_method) 
        # identifier = 'C%s-L%s-CID%s-O%s-C%s-S%s-D2V%s' % (ctype, label, mtype, otype, cluster_method, seq_ptype, d2v_method)
       
    basedir = os.path.join(os.getcwd(), 'data') if cohort_name is None else seqparams.get_basedir(cohort=cohort_name) 
    outputdir = kargs.get('outputdir', basedir)
    load_path = os.path.join(outputdir, 'motifs-%s.pkl' % identifier)

    print('params> coordinate (l:%s, seq:%s, d2v: %s) | order: %s | method: %s | type (c:%s, m:%s) => input n_docs:%d' % \
        (label, seq_ptype, d2v_method, 'partial' if partial_order else 'total', cluster_method, ctype, mtype, n_docs0))
    print('stats> size of global_motifs: %d, size of focused_motifs: %d (complement? %s)' % \
        (n_global_motifs, n_focused_motifs, n_focused_motifs > 0))

    res = {}
    if load_motif and os.path.exists(load_path): 
        res = pickle.load(open(load_path, 'rb'))
        if len(res) > 0:
            print('io> loaded pre-computed %s motifs from %s' % (mtype, load_path)) 
            # return res

    if not res: 
        if ng_min < 1: ng_min = 1
        if ng_max < ng_min: ng_max = ng_min

        topnmap = {n: None for n in range(ng_min, ng_max+1)} # n_doc0  # max 
        tKeepAll = False
        if topn is None: # no limit, don't use most_common() to filter n-grams
            tKeepAll = True # no-op 
        elif isinstance(topn, dict):  # topn: n -> number of m most common n-grams
            # topn can be either None or a number
            for n in range(ng_min, ng_max+1): 
                topnmap[n] = topn.get(n, None) # default no upperbound
        else: 
            assert isinstance(topn, int)
            for n in range(ng_min, ng_max+1): 
               topnmap[n] = topn

        # common unigrams
        # tokens = list(chain.from_iterable(seqx)) # this flattens out entire docs, may cause extra counts of n-grams straddling between lines
        # counter = collections.Counter(tokens)  # cluster/population wise frequency
        # res[1] = counter.most_common(topn[1])

        # n = topn.get(1, topn_default)
        # print('verify> top %d unigram-frequencies (%s):\n%s\n' % (n, mtype, res[1]))  #  [('c', 2), ('b', 2)]
        # common_codes = [e[0] for e in res[1]]
        # print('verify> top %d codes (%s):\n%s\n' % (n, mtype, common_codes))
    
        # if ng_max >= 2: # bi-grams+  
        # length: counters (n-gram: counts) 

        if n_focused_motifs > 0: print('verify> number of doc as a %s-complement: %d' % (mtype, n_docs0)) # assuming mtype holds cid
        ngrams = algorithms.count_ngrams2(seqx, min_length=1, max_length=ng_max, partial_order=partial_order) 
        for ng in range(1, ng_max+1):  # +1: inclusive
            if not ngrams.has_key(ng): 
                print('> %d-gram motif does not exist in input sequences' % ng)
                continue
            counter = ngrams[ng]
            res[ng] = topn_counts = counter.most_common(topnmap[ng])
            print("verify> top cluster %d-gram frequencies(cid=%s, topn=%s | ct=%s, ot=%s, pt=%s):\n%s\n" % \
                (ng, mtype, topnmap[ng], ctype, otype, ptype, str(topn_counts[:10])))  #  [('c', 2), ('b', 2)]
            
            # common_ngrams = [e[0] for e in topn_counts]
            # print('verify> top %s %d-grams(%s):\n%s\n' % (n_docs0 if n is None else n, ng, mtype, common_ngrams[:5])) # ok 
    ### end if-else 
        
    apply_groupby_sort, filter_low_freq = True, True
    if save_motif:   # saving CLUSTER motifs
        # save both csv and pkl

        ### 1. csv
        # global path (can also save one file for each n-gram)
        fpath = os.path.join(outputdir, 'cluster_motifs-%s.csv' % identifier)
        header = ['length', 'ngram', 'count', 'global_count', 'ratio']  # 'ratio': ccnt/gcnt

        # count global motifs in the context of the cluster-scoped sequences
        # global_motifs_in_cluster = eval_given_motif(seqx, motifs=global_motifs)  # n -> {(n-gram, count)}, a dict of dictionaries
        adict = compare_motif(lmotifs=res, gmotifs=global_motifs, fmotifs=focused_motifs, header=header, 
                    mtype=mtype, ctype=ctype, otype=otype, ptype=ptype, outputdir=outputdir) # [input] res->lmotifs, global_motifs->gmotifs
        if n_focused_motifs > 0:  # just for testing, don't save anything
            # no-op 
            pass 
        elif len(res) > 0:  # cluster motifs
            assert size(adict) > 0 
            try: 
                df = DataFrame(adict, columns=header)
            except Exception, e: 
                print('dump> adict:\n%s\n' % adict)
                raise ValueError, e

            # sort n-grams that appear more exclusively within a cluster
            # use pathwayAnalyzer to further organize the data
            if apply_groupby_sort: 
                df = df.sort_values(['length', 'ratio', ], ascending=[False, False, ])

            # if filter_low_freq: 
            #     pass

            print('verify> saving cluster motif dataframe (dim=%s, cid=%s, ct=%s, ot=%s, pt=%s) to %s' % \
                (str(df.shape), mtype, ctype, otype, ptype, fpath))
            df.to_csv(fpath, sep='|', index=False, header=True)

            # group by n-gram length and sort 

        else: 
            print('warning> coordinate (l:%s, seq:%s, d2v: %s) > No motifs found.' % (label, seq_ptype, d2v_method))

        ### 2. pickle file 
        # fpath = load_path # os.path.join(outputdir, 'motifs-%s.pkl' % identifier)
        # pickle.dump(res, open(fpath, "wb" ))

    return res # n -> [(ngr, count)]

# [refactor] from lookup_motif() in eval_cluster_motif()
def compare_motif(lmotifs, gmotifs, fmotifs=None, **kargs): # lookp cluster motif in global scope
    """
    Helper functon for eval_cluster_motif()

    Given lmotifs (i.e. local motifs: ngrams and counts within a cluster) and gmotifs (i.e. global motifs: ngrams and counts in global scope), 
    compare their counts and compute summary statistics such as ratios

    Input
    -----
    lmotifs 
    gmotifs
    fmotifs: focused motifs if given, only focus on the ngrams defined in it (a cluster)
       usage example: 
          lmotifs: cluster complement, C_bar
          gmotifs: entire documens, D  
          fmotifs: cluster C 

          where Union(C_bar, C) = D 
    """
    import heapq 

    # [input] cluster_motifs, global_motifs 

    # assert 'global_motifs_test' in locals(), "need 'global_motifs_test' to compare with cluster/local motifs."
        
    #         (get) counts of cluster motifs in the global context
    header = kargs.get('header', ['length', 'ngram', 'count', 'global_count', 'ratio'])

    adict = {h: [] for h in header} 
    wsep = ' ' # word separator   # or use ' -> '  

    assert len(gmotifs) > 0 and len(lmotifs) > 0
    n_lmotifs = utils.size_hashtable(lmotifs)
    n_gmotifs = utils.size_hashtable(gmotifs)

    cid = mtype = kargs.get('mtype', 'unknown_cid') # motif type (usually a CID)

    # [filter]
    if fmotifs is not None: 
        n_fmotifs = utils.size_hashtable(fmotifs)
        print('compare> Complement: size of local %d, focused: %d, global motifs: %d' % (n_lmotifs, n_fmotifs, n_gmotifs))

        for n, ngr_cnts in fmotifs.items(): # foreach focused motifs (e.g. pre-computed cluster motifs): n (as in n-gram), counts (a dict)
            lcounts = dict(lmotifs[n]) if lmotifs.has_key(n) else {}  # the complement
            if not lcounts: print('compare> Warning: Could not find %d-gram in local motifs (mtype: %s, ctype: %s)' % (n, mtype, ctype))
            gcounts = dict(gmotifs[n]) if gmotifs.has_key(n) else {}
            if not gcounts: print('compare> Warning: Could not find %d-gram in global motifs (mtype: %s, ctype: %s)' % (n, mtype, ctype))

            # n_patterns = len(ngr_cnts)
            # adict['length'].extend([n] * n_patterns)  # [redundancy]

            lengthx, ngram_reprx, local_counts, global_counts, ratios = [], [], [], [], []
            for i, (ngt, fcnt) in enumerate(ngr_cnts):  # foreach ngram and count

                ccnt = lcounts.get(ngt, 0) # how many times does it appear in this cluster? (fcnt is usually the counts from cluster complement)
                gcnt = gcounts.get(ngt, 0) # how many times does it appear in global context?    
                    
                # [verify] sum assertion does not hold
                tcnt = ccnt + fcnt 
                # assert tcnt == gcnt, "fcnt: %d + ccnt: %d = %d != gcnt: %d??" % (fcnt, ccnt, tcnt, gcnt)
                if tcnt != gcnt: print("lookup> Not a cluster complement? fcnt: %d + ccnt: %d = %d != gcnt: %d??" % (fcnt, ccnt, tcnt, gcnt))
                assert ccnt <= gcnt, "pattern %s | (ccnt: %d > gcnt: %d, fcnt: %d) > global %d-gram patterns:\n%s\n" % \
                        (str(ngt), ccnt, gcnt, fcnt, n, str(gcounts))    

                # ratio between cluster and global frequencies
                if ccnt > 0 and gcnt > 0: 
                    ratios.append(ccnt/(gcnt+0.0))
                else: 
                    if gcnt == 0: 
                        assert ccnt == 0
                        ratios.append(-1) # as in unknown
                    else: # gcnt > 0  
                        ratios.append(0.0)
                        # raise ValueError, "gcnt %d > 0 but ccnt == 0? %d" % (gcnt, ccnt)

                # from tuple (ngt) to string (ngstr) 
                if isinstance(ngt, str): 
                    ngstr = ngt 
                else: 
                    assert isinstance(ngt, tuple), "invalid ngram format: %s" % ngt
                    ngstr = wsep.join(str(e) for e in ngt)
                    
                    # [test]
                    if i == 0: assert isinstance(ngt, tuple), "invalid ngram: %s" % str(ngt) 
                    
                lengthx.append(n)
                ngram_reprx.append(ngstr) # foreach element in n-gram tuple
                local_counts.append(ccnt)
                global_counts.append(gcnt)
            ### end foreach count object

            # add rows
            adict['length'].extend(lengthx)
            adict['ngram'].extend(ngram_reprx)
            adict['count'].extend(local_counts)
            adict['global_count'].extend(global_counts)
            adict['ratio'].extend(ratios)

        ### end foreach count in cluster scope    
  
    else: 
        r = n_lmotifs/(n_gmotifs+1.0)
        print('compare> Cluster (%s): size of local %d <? global motifs: %d (ratio: %f)' % (mtype, n_lmotifs, n_gmotifs, r))

        n_eq = n_uneq = 0 # [test] any differences between cluster and global frequencies? yes
        
        # [test]
        # 1. among all the n-grams (with 'n' fixed), which subset has smallest ratio between the cluster freq and global freq 
        #    => more unique to the cluster?  can use heap, priority queue  
        minRatios = {}
        for n, ngr_cnts in lmotifs.items(): # foreach cluster motif: n (ngram), counts (a dict)
            gcounts = dict(gmotifs[n]) if gmotifs.has_key(n) else {}
            if not gcounts: print('lookup> Warning: Could not find %d-gram in global motifs (mtype: %s, ctype: %s)' % (n, mtype, ctype))

            print('test> cluster #%d %d-gram > size: %d vs global size: %d' % (mtype, n, len(ngr_cnts), len(gcounts)))

            # smallest ratios 
            minRatios[n] = []

            # n_patterns = len(ngr_cnts)
            # adict['length'].extend([n] * n_patterns)  # [redundancy]

            # record these attributes
            lengthx, ngram_reprx, local_counts, global_counts, ratios = [], [], [], [], []

            # n_uneq_per_n = 0
            for i, (ngt, ccnt) in enumerate(ngr_cnts):  # foreach n-gram in cluster motifs | ccnt: cluster-level count 

                # [lookup]
                gcnt = gcounts.get(ngt, 0) # how many times does it appear in global context? 
                assert ccnt <= gcnt, "cluster count > global counts? (ccnt: %d > gcnt: %d) > global %d-gram patterns:\n%s\n" % \
                            (ccnt, gcnt, n, str(gcounts))

                # ratio between cluster and global frequencies
                r = 0.0 # float('nan') 
                if ccnt > 0 and gcnt > 0: 
                    # ratios.append(ccnt/(gcnt+0.0))
                    r = ccnt/(gcnt+0.0)
                else: 
                    if gcnt == 0: 
                        assert ccnt == 0
                        # ratios.append(-1.0)  # should be float('nan')
                        # r = -1.0 
                    else: # gcnt > 0 => ccnt > 0 
                        # ratios.append(0.0)
                        r = 0.0 
                        # raise ValueError, "gcnt %d > 0 but ccnt == 0? %d" % (gcnt, ccnt)
                ratios.append(r)
                heapq.heappush(minRatios[n], (r, ngt))

                # [test]
                if ccnt == gcnt: 
                    n_eq += 1 
                else: 
                    n_uneq += 1
                    # n_uneq_per_n += 1 # refreshed for each 'n'
                    
                    # print('   + found (ccnt <> gcnt) case | n=%d, ccnt=%d < gcnt=%d' % (n, ccnt, gcnt)) 
                    # print('   ++ ngr: %s' % str(ngt))
                    # heapq.heappush(minRatios[n], r)  # if equal r = 1.0

                # from tuple (ngt) to string (ngstr) 
                if isinstance(ngt, str): 
                    ngstr = ngt 
                else: 
                    assert isinstance(ngt, tuple), "invalid ngram format: %s" % ngt
                    ngstr = wsep.join(str(e) for e in ngt)
                    
                lengthx.append(n)
                ngram_reprx.append(ngstr) # foreach element in n-gram tuple
                local_counts.append(ccnt)
                global_counts.append(gcnt)

            
            if len(minRatios[n]): 
                rmin, ngr_min = minRatios[n][0] # minQ, reading smallest takes const time
                print("    + smallest ratio %s-gram (r, ngr)=(%f, %s)" % (n, rmin, ' '.join(ngr_min)))

            ### end foreach count object (i.e. n-grams, counts in cluster motifs )

            # add rows
            adict['length'].extend(lengthx)
            adict['ngram'].extend(ngram_reprx)
            adict['count'].extend(local_counts)
            adict['global_count'].extend(global_counts)
            adict['ratio'].extend(ratios)
        ### end foreach count in cluster scope 

        # [byproduct] find rare patterns
        nm = 5
        cid = mtype
        # wsep = ' '
    
        header_temp = ['length', 'ngram', 'count', 'global_count', 'ratio']
        byproduct = {h: [] for h in header_temp}
        for n in minRatios.keys(): 
            min_ratios = heapq.nsmallest(nm, minRatios[n])
            nitems = len(min_ratios)
            byproduct['length'].extend([n] * nitems)

            cntmap = dict(lmotifs[n])
            cntmapG = dict(gmotifs[n])
            for r, ngr in min_ratios: 
                ngstr = wsep.join(str(e) for e in ngr)
                byproduct['ngram'].append(ngstr)
                byproduct['count'].append(cntmap[ngr])
                byproduct['global_count'].append(cntmapG[ngr])
                byproduct['ratio'].append(r)
              
        # [params] file
        content_type, order_type, policy_type = kargs.get('ctype', 'regular'), kargs.get('otype', 'partial'), kargs.get('ptype', 'prior')
        identifier = 'CID%s-COP%s-%s-%s' % (mtype, content_type, order_type, policy_type) 
        fname = 'smallRatioNGrams-%s.csv' % identifier
        outputdir = kargs.get('outputdir', os.path.join(os.getcwd(), 'data'))
        fpath = os.path.join(outputdir, fname) 
   
        df = DataFrame(byproduct, columns=header_temp)
        print('io> saving the n-grams with the smallest ratios between the cluster frequency and its global frequency to:\n%s' % fpath)
        df.to_csv(fpath, sep='|', index=False, header=True)

    ### end if-then-else: focused motifs (external) vs local motifs in this cluster

    # [test]
    print('test> Cluster #%s| n_eq=%d, n_uneq=%d' % (mtype, n_eq, n_uneq))
    

    return adict

def analyze_pathway_batch(**kargs): # cohort, n_clusters, optimize_k, n_classes
    import itertools 

    # [params] cohort (e.g. diabetes, PTSD)
    cohort_name = kargs.get('cohort', 'diabetes')

    # [params] I/O
    outputdir = basedir = seqparams.get_basedir(cohort=cohort_name)  # os.path.join(os.getcwd(), cohort_name)

    # [params]
    seq_compo = composition = kargs.get('composition', 'condition_drug') # sequence composition
    read_mode = kargs.get('read_mode', 'doc') # documents/doc (one patient one sequence) or sequences/seq
    tset_type = kargs.get('tset_type', 'binary')
    std_method = kargs.get('std_method', 'minmax') # feature standardizing/scaling/preprocessing method
    n_sample, n_sample_small = 1000, 100

    # [params] cluster 
    n_clusters = kargs.get('n_clusters', 50)
    optimize_k = kargs.get('optimize_k', False)
    range_n_clusters = kargs.get('range_n_clusters', None) # None => to be determined by gap statistics 
    min_n_clusters, max_n_clusters = kargs.get('min_n_clusters', 1), kargs.get('max_n_clusters', 200)

    # [params] classification 
    n_classes = kargs.get('n_classes', 1)

    # [params] pathway 
    min_freq = kargs.get('min_freq', 2) # a dictionary or a number; this is different from 'min_count' used for w2v and d2v models

    w2v_method = kargs.get('w2v_method', 'sg')
    d2v_methods = ['tfidfavg', ]
    cluster_methods = ['kmeans', ]

    # training set: tset_nC1_IDregular-sg-tfidfavg-GPTSD.csv
    # make_tset_unlabeled(w2v_method=w2v_method, d2v_method=d2v_method, 
    #                         seq_ptype=seq_ptype, read_mode=read_mode, test_model=test_model,
    #                             cohort=cohort_name, bypass_lookup=bypass_lookup) 

    otypes = kargs.get('order_types', ['total', ])  # options: 'partial', 'total',   # n-gram with ordering considered? 
    ctypes = kargs.get('content_types', ['diagnosis', 'mixed',])  # 'medication'
    ptypes = kargs.get('policy_types', ['prior', 'noop', 'posterior',])  # cut policy types  {noop, regular, complete} => preserve entire sequence

    for d2v_method in d2v_methods: 
        # kargs['d2v_method'] = d2v_method

        for cluster_method in cluster_methods: 
            # kargs['cluster_method'] = cluster_method
            # [params] experiemntal settings for n-gram analysis: otype, ctype, ptype
            
            for params in itertools.product(otypes, ctypes, ptypes): 
                # cluster_method = params[0]
                order_type, ctype, policy_type = params[0], params[1], params[2]  

                analyze_pathway(order_type=order_type, policy_type=policy_type, ctype=ctype, 
                    n_classes=n_classes,  
                    d2v_method=d2v_method,
                    load_model=kargs.get('load_model', True),  # [chain] data_matrix -> loadTSet -> makeTSet
                    cohort=cohort_name, 
                        cluster_method=cluster_method,
                            n_clusters=n_clusters, optimize_k=optimize_k, 
                            min_n_clusters=min_n_clusters, max_n_clusters=max_n_clusters, 
                                min_freq=min_freq) 

    return 

def test_similarity(**kargs): 
    def profile(ts): 
        labels = list(set(ts[TSet.target_field].values))
        n_labels = len(labels)
        
        return

    import itertools, utils
    import learn_manifold as lman
    import hc, evaluate, plotUtils, seqUtils
    # import seqAnalyzer as sa 
    # from seqparams import TSet

    div(message='Test similarity (manifold, hc, sim matrix visualization)')
    # kargs['seq_ptype'] = seq_ptype = kargs.get('seq_ptype', 'regular') # sequence pattern type: regular, random, diag, med, lab
    # d2v_method = kargs.get('d2v_type', kargs.get('d2v_method', 'PVDM'))  # i.e. doc2vec_method
    testdir = os.path.join(os.getcwd(), 'test')
    if not os.path.exists(testdir): os.makedirs(testdir) # test directory

    # cohort (e.g. diabetes, PTSD)
    cohort_name = kargs.get('cohort', 'diabetes')
    outputdir = basedir = seqparams.get_basedir(cohort=cohort_name)  # os.path.join(os.getcwd(), cohort_name)
    # if not os.path.exists(outputdir): os.makedirs(outputdir) # output directory

    # load data 
    # X, y, D = build_data_matrix2(**kargs) # seq_ptype, n_features, etc. 
    # 1. read_doc, seq_ptype => document type (random, regular, etc.)
    # 2. doc2vec_method, n_doc, seq_ptype => training data set
    seq_compo = composition = kargs.get('composition', 'condition_drug') # sequence composition
    read_mode = kargs.get('read_mode', 'doc') # documents/doc (one patient one sequence) or sequences/seq
    tset_type = kargs.get('tset_type', 'binary')
    standardize_ = kargs.get('std_method', 'minmax') # feature standardizing/scaling/preprocessing method
    n_sample, n_sample_small = 1000, 100
    n_clusters = kargs.get('n_clusters', 50)
    n_classes = 1

    # [params] experimental settings
    seq_ptype = 'regular' # [default]
    seq_ptypes = ['regular', ] # ['random', 'regular', ]
    d2v_methods = ['tfidfavg', ]  # 'average',  'PVDM'

    # 'affinity_propogation': expensive (need subsetting)
    cluster_methods = ['kmeans', 'minibatch' ] # 'agglomerative', 'spectral', 'dbscan',

    cohort_name = kargs.get('cohort', 'diabetes')
    # doc_basename = seq_compo if cohort_name is None else '%s-%s' % (seq_compo, cohort_name)
    # original documents 
    # [params][input] temporal docs e.g. condition_drug_seq.dat, condition_drug_seq-PTSD.dat

    load_seq, save_seq = False, False
    default_ifile = '%s_seq-%s.dat' % (seq_compo, cohort_name) if cohort_name is not None else '%s_seq.dat' % seq_compo
    ifile = kargs.get('ifile', default_ifile)
    D0 = sa.read(simplify_code=False, mode=read_mode, verify_=False, seq_ptype=seq_ptype, ifile=ifile, 
                    load_=load_seq, save_=save_seq) # cohort=cohort_name
    n_doc0 = len(D0)
    print('verify> size of original docs: %d' % n_doc0)

    min_n_clusters, max_n_clusters = kargs.get('min_n_clusters', 1), kargs.get('max_n_clusters', 200)
    # [params] pathway 
    min_freq = kargs.get('min_freq', 2) 

    # for setting in itertools.product(seq_ptypes, d2v_methods)

    ### 1. binary class dataset (tset_type <- binary)
    for seq_ptype in seq_ptypes: 
        for d2v_method in d2v_methods: 
            kargs['d2v_method'] = d2v_method

            D, ts = data_matrix(read_mode=read_mode, seq_ptype=seq_ptype, d2v_method=d2v_method, tset_type=tset_type, cohort=cohort_name)
            X, y = evaluate.transform(ts, standardize_=standardize_) # default: minmax
            # X, y, D = getXYD(read_mode=read_mode, seq_ptype=seq_ptype, d2v_method=d2v_method, tset_type=tset_type) 
            n_doc0 = len(D)

            # label y via phenotypeDoc()
            labels = list(set(y))
            n_labels = len(labels)
            assert X.shape[0] == y.shape[0] == n_doc0
            print('verify> setting: (seq_ptype: %s, d2v: %s) > n_doc: %d, embedded dim: %d' % (seq_ptype, d2v_method, n_doc0, X.shape[1]))
            print('verify> example composite labels:\n%s\n' % y[:10])
            print('verify> total number of labels: %d > number of unique labels: %d' % (len(y), n_labels))

            # 1. k-means => use t_analysis()

            # cluster labels, clustering quality metrics: homogeneity, completeness, v_measure, ARI, AMI
            for cluster_method in cluster_methods: 
                clabels, cmetrics = cluster_analysis(X=X, y=y, n_clusters=n_clusters, cluster_method=cluster_method,
                                                        seq_ptype=seq_ptype, d2v_method=d2v_method, save_=True, load_=True, 
                                                        cohort=cohort_name, outputdir=outputdir)
                print('cluster> method: %s, metrics:\n%s\n' % (cluster_method, cmetrics))

                # analyze coding sequence
                # set topn_clusters_only to False in order to compute (inverse) document/cluster freq. 
                analyze_pathway(X=X, y=y, D=D, clusters=clabels,
                    order_type='partial', policy_type='prior', ctype='mixed', 
                    n_classes=n_classes,  
                    d2v_method=d2v_method,
                    cohort=cohort_name, 
                        cluster_method=cluster_method,
                            n_clusters=n_clusters, optimize_k=optimize_k, 
                            min_n_clusters=min_n_clusters, max_n_clusters=max_n_clusters, 
                                min_freq=min_freq)
            
            ### subsetting training data 
            ts_subset = seqUtils.sample_class(ts, n_sample=n_sample, ignore_index=True) # sample data according to the proportion of classes
            
            idx = ts_subset[TSet.index_field] # these indices correspond to the positions of the original documents
            Ds = sa.select(docs=D0, idx=idx)
            Xs, ys = evaluate.transform(ts_subset, standardize_=standardize_) # default: minmax
            n_doc_subset = len(Ds)
            labels = list(set(ys))
            n_labels_subset = len(labels)
            ### end subsetting
            print('verify> setting: (seq_ptype: %s, d2v: %s) > n_doc_subset: %d, embedded dim: %d' % (seq_ptype, d2v_method, n_doc_subset, Xs.shape[1]))
            print('verify> total number of labels: %d > number of unique labels: %d' % (len(ys), n_labels_subset))

            # 2. manifold embedding (e.g. locally linear, MDS, spectral t-SNE)
            lman.mds(Xs, y=ys, seq_ptype=seq_ptype, d2v_method=d2v_method, n_sample=n_sample, graph_mode='s')
            lman.spectral(Xs, y=ys, seq_ptype=seq_ptype, d2v_method=d2v_method, n_sample=n_sample, graph_mode='s')
            lman.tsne(Xs, y=ys, seq_ptype=seq_ptype, d2v_method=d2v_method, n_sample=n_sample, graph_mode='s')

            # 3. visualize ts (heatmap)
            tss = seqUtils.sample_class(ts, n_sample=n_sample_small, ignore_index=True) # sample data according to the proportion of classes
            idx = tss[TSet.index_field] # these indices correspond to the positions of the original documents
            Dss = sa.select(docs=D0, idx=idx)
            Xss, yss = evaluate.transform(tss, standardize_=standardize_) # default: minmax
            # plotUtils.t_heatmap_plotly(Xss, yss, **kargs)  # has a quota

            # 4. visualize similarity matrix (heatmap?)

            # 5. hc (not too many data points)
            # hc.run(Xss, yss, Dss, seq_ptype=seq_ptype, d2v_method=d2v_method, n_sample=n_sample)


            # end
            div(message='Completed similarity analysis on seq_ptype: %s, d2v_method: %s' % (seq_ptype, d2v_method))

    return

### cluster analysis code 

def cluster_agglomerative(X=None, y=None, **kargs): 
    kargs['cluster_method'] = 'agglomerative'
    return cluster_analysis(X, y, **kargs) 
def cluster_spectral(X=None, y=None, **kargs):
    kargs['cluster_method'] = 'spectral'
    return cluster_analysis(X, y, **kargs)      
def cluster_kmeans(X=None, y=None, **kargs):
    kargs['cluster_method'] = 'kmeans'
    return cluster_analysis(X, y, **kargs) 
def cluster_affinity(X=None, y=None, **kargs): 
    kargs['cluster_method'] = 'affinity_propogation'
    return cluster_analysis(X, y, **kargs)     

def cluster_analysis(ts=None, **kargs): 
    """

    Input
    -----
    * class_label_names: maps class labels to meanings

    Related
    -------
    1. more generic version of this function can be found in 
          tpheno.cluster.analyzer

    """
    def cluster_distribution(ldmap): # label to distribution map
        # [params] testdir
        # plot histogram of distancesto the origin for all document vectors
        for ulabel, distr in ldmap.items(): 
            plt.clf() 
           
            canonical_label = lmap.get(ulabel, 'unknown')
            
            f = plt.figure(figsize=(8, 8))
            sns.set(rc={"figure.figsize": (8, 8)})
            
            intervals = [i*0.1 for i in range(10+1)]
            sns.distplot(distr, bins=intervals)

            # n, bins, patches = plt.hist(distr)
            # print('verfiy_norm> Label: %s > seq_ptype: %s, d2v: %s > n: %s, n_bins: %s, n_patches: %s' % \
            #      (canonical_label, seq_ptype, d2v_method, n, len(bins), len(patches)))

            # identifier = 'L%s-%s' % (canonical_label, identifier) # [note] cannot reassign value local var
            identifier2 = 'L%s-%s' % (canonical_label, identifier)
            fpath = os.path.join(plotdir, 'cluster_distribution-%s.tif' % identifier2)
            print('output> saving cluster distribution to %s' % fpath)
            plt.savefig(fpath)

        return
    def evaluate_cluster(labels_cluster): 
        div(message='clustering algorithm: %s' % cluster_method)

        labels_true, labels = y, labels_cluster

        n_clusters_free = ('db', 'aff')  # these do not require specifying n_clusters
        if cluster_method.startswith(n_clusters_free): 
            print('(Estimated) number of clusters: %d' % n_clusters_est)
        else: 
            print('(Specified) number of clusters: %d' % n_clusters)

        mdict = {}
        mdict['homogeneity'] = metrics.homogeneity_score(labels_true, labels)
        print("Homogeneity: %0.3f" % mdict['homogeneity'])

        mdict['completeness'] = metrics.completeness_score(labels_true, labels)
        print("Completeness: %0.3f" % mdict['completeness'])

        mdict['v_measure'] = metrics.v_measure_score(labels_true, labels)
        print("V-measure: %0.3f" % mdict['v_measure'])

        mdict['ARI'] = metrics.adjusted_rand_score(labels_true, labels)
        print("Adjusted Rand Index: %0.3f" % mdict['ARI'])

        mdict['AMI'] = metrics.adjusted_mutual_info_score(labels_true, labels)
        print("Adjusted Mutual Information: %0.3f" % mdict['AMI'])

        # only silhouette coeff doesn't require ground truths (but can be very expensive due to pairwise computations)
        # cluster sampling
        # [log] could not broadcast input array from shape (1000,100) into shape (1000)
        n_sample = 1000 
        try: 
            Xsub = seqUtils.sample_class2(X, y=labels, n_sample=n_sample, replace=False) # [todo] without replacemet => replacement upon exceptions
        except: 
            print('evaluate> Could not sample X without replacement wrt cluster labels (dim X: %s while n_clusters: %d)' % \
                (str(X.shape), len(set(labels)) ))
            Xsub = seqUtils.sample_class2(X, y=labels, n_sample=n_sample, replace=True)
        try: 
            mdict['silhouette'] = metrics.silhouette_score(Xsub, np.array(labels), metric='sqeuclidean')
            print("Silhouette Coefficient: %0.3f" % mdict['silhouette'])
        except Exception, e: 
            print('evaluate> Could not compute silhouette score: %s' % e)

        return mdict 

    def summarize_cluster(cluster_labels):  # [refactor] also see cluster.analyzer
        clusters = map_clusters(cluster_labels, X)  # id -> {X[i]}
        
        res = {}
        # compute mean and medians
        cluster_means, cluster_medians, cluster_medoids = {}, {}, {}
        for cid, points in clusters.items(): 
            cluster_means[cid] = np.mean(points, axis=0)
            cluster_medians[cid] = np.median(points, axis=0)

        nrows = len(cluster_labels)
        assert X.shape[0] == nrows
        c2p = map_clusters(cluster_labels, range(nrows))  # cluster to position i.e. id -> indices 
        
        # k-nearest neighbors wrt mean given a distance metric 
        # [params]
        topk = 10
        dmetric = ssd.cosine # choose an appropriate metric 

        cluster_knn = {}
        for cid, mpoint in cluster_means.items(): # foreach cluster (id) and its centroid
            idx = c2p[cid]  # idx: all data indices in cluster cid 
            rankedDist = sorted([(i, dmetric(X[i], mpoint)) for i in idx], key=lambda x: x[1], reverse=True)[:topk]
            # if cid % 2 == 0: print('test> idx:\n%s\n\nranked distances:\n%s\n' % (idx, str(rankedDist)))
            cluster_knn[cid] = [i for i, d in rankedDist]

        cluster_knn_median = {}
        for cid, mpoint in cluster_medians.items(): # foreach cluster (id) and its centroid
            idx = c2p[cid]  # idx: all data indices in cluster cid 
            rankedDist = sorted([(i, dmetric(X[i], mpoint)) for i in idx], key=lambda x: x[1], reverse=True)[:topk]
            # if cid % 2 == 0: print('test> idx:\n%s\n\nranked distances:\n%s\n' % (idx, str(rankedDist)))
            cluster_knn_median[cid] = [i for i, d in rankedDist]

        # save statistics 
        res['cluster_means'] = cluster_means
        res['cluster_medians'] = cluster_medians
        res['cluster_knn'] = cluster_knn   # knn wrt mean
        res['cluster_knn_median'] = cluster_knn_median

        return res

    # from sklearn.cluster import AgglomerativeClustering
    # from sklearn.neighbors import kneighbors_graph
    # from sklearn.cluster import AffinityPropagation
    # from sklearn import metrics
    # rom scipy.spatial import distance
    # from sklearn.neighbors import NearestNeighbors  # kNN
    from cluster import cluster
    import scipy.spatial.distance as ssd
    # from seqparams import TSet 
    # import cluster.analyzer   # [use] analyzer.summarize_cluster()
    # ssd.cosine(arr[0], [1., 2., 3.])

    # [params]
    n_clusters = kargs.get('n_clusters', 20)
    n_clusters_est = -1

    cluster_method = kargs.get('cluster_method', 'kmeans')

    # [params] 
    cohort_name = kargs['cohort'] # kargs.get('cohort', 'diabetes')
    # inputdir = datadir = os.path.join(os.getcwd(), 'data/%s' % cohort_name) # [old] 'data', sys_config.read('DataExpRoot')
    outputdir = basedir = kargs.get('outputdir', seqparams.get_basedir(cohort=cohort_name)) # os.path.join(os.getcwd(), 'data/%s' % cohort_name) # 
    plotdir = seqparams.get_basedir(cohort=cohort_name, topdir='plot') # os.path.join(os.getcwd(), 'plot')
    lmap = kargs.get('class_label_names', {0: 'TD1', 1: 'TD2', 2: 'Gest', })  # mapping from class label to meanings

    # [params] identifier
    seq_compo = kargs.get('composition', 'condition_drug')
    seq_ptype = seqparams.normalize_ctype(kargs.get('seq_ptype', 'regular')) # sequence pattern type: regular, random, diag, med, lab

    w2v_method = kargs.get('w2v_method', 'sg')
    d2v_method = kargs.get('d2v_method', 'tfidfavg')

    # identifier = kargs['identifier'] if kargs.has_key('identifier') else '%s-%s' % (seq_ptype, d2v_method)
    identifier = kargs.get('identifier', None)
    if identifier is None: identifier = 'C%s-P%s-D2V%s-G%s' % (cluster_method, seq_ptype, d2v_method, cohort_name)

    map_label_only = kargs.get('map_label_only', True) # if False, then cluster map file will keep original docs in its data field
    doc_labeled = False
    
    save_cluster = kargs.get('save_cluster', True)

    # [params] input data (cf: cluster.analyzer)
    # [note] training set files e.g. 'tset_%s_%s_t1t2-P%s.csv' % (doc2vec_method, n_doc, seq_ptype)
    #        see data_matrix()
    tset_type = kargs.get('tset_type', None)
    n_classes = kargs.get('n_classes', None)
    D = kargs.get('D', None)
    X, y = kargs.get('X', None), kargs.get('y', None)
    ts_indices = None
    if ts is None:  # precedence: ts -> X -> load from existing data specified via parameters
        if X is None: 
            print('io> loading training data according to the parameter setting ...')
            # assert not (n_classes is None and tset_type is None), "n_classes and tset_type cannot be both unknown."  
            D, ts = data_matrix(read_mode=read_mode, seq_ptype=seq_ptype, w2v_method=w2v_method, d2v_method=doc2vec_method, 
                                    tset_type=tset_type, n_classes=n_classes, cohort=cohort_name)
            X, y = evaluate.transform(ts, standardize_=kargs.get('std_method', 'minmax')) # default: minmax
            ts_indices = ts[TSet.index_field]
            kargs['X'], kargs['y'] = X, y
        else: 
            ts_indices = kargs.get('instance_ids', None)
            if ts_indices is None: ts_indices = np.array(range(0, X.shape[0]))
    else: 
        ts_indices = ts[TSet.index_field].values  # ID field of the training data (e.g. person IDs)
        X, y = evaluate.transform(ts, standardize_=kargs.get('standardize_', 'minmax')) # default: minmax
        kargs['X'], kargs['y'] = X, y

    if y is None: y = np.repeat(1, X.shape[0]) # unlabeled data
    n_classes_verified = len(set(y))
    if n_classes is not None: assert n_classes == n_classes_verified
    if n_classes_verified > 1: 
        doc_labeled = True # each document is represented by its (surrogate) labels

    # X, y, seqx = build_data_matrix2(**kargs)
    # labels_true = y
    print('verify> algorithm: %s > doc2vec dim: %s > labeled? %s' % \
        (cluster_method, str(X.shape), True if n_classes > 1 else False))

    # [params] output 
    # cluster labels 
    load_path = os.path.join(outputdir, 'cluster_ids-%s.csv' % identifier) # id: seq_ptype, d2v_method
    load_cluster = kargs.get('load_', True) and (os.path.exists(load_path) and os.path.getsize(load_path) > 0)

    ### Run Cluster Analysis ### 
    model = n_cluster_est = None   # [todo] model persistance
    if not load_cluster: 
        model, n_clusters_est = cluster.run_cluster_analysis(**kargs)
        cluster_labels = model.labels_
        # cluster_inertia   = model.inertia_
    else: 
        print('io> loading pre-computed cluster labels (%s, %s, %s) ...' % (cluster_method, seq_ptype, d2v_method))
        try: 
            df = pd.read_csv(load_path, sep=',', header=0, index_col=False, error_bad_lines=True)
            cluster_labels = df['cluster_id'].values
        except: 
            print('io> Could not load pre-computed cluster labels for (%s, %s, %s), re-computing ...' % \
                (cluster_method, seq_ptype, d2v_method))
            model, n_clusters_est = cluster.run_cluster_analysis(**kargs)
            cluster_labels = model.labels_

        assert X.shape[0] == len(cluster_labels), "nrow of X: %d while n_cluster_ids: %d" % (X.shape[0], len(cluster_labels))

    # [note] alternatively use map_clusters2() to map cluster IDs to appropriate data point repr (including labels)
    # cluster_to_docs  = map_clusters(cluster_labels, seqx)
    print('status> completed clustering with %s' % cluster_method)

    # [test]
    cluster_to_docs = None
    if D is None: # label only for 'data'
        cluster_to_docs  = map_clusters(cluster_labels, y)  # y: pre-computed labels (e.g. heuristic labels)
    else: 
        cluster_to_docs  = map_clusters(cluster_labels, D)
    print('info> resulted n_cluster: %d =?= %d' % (len(cluster_to_docs), n_clusters))  # n_cluster == 2

    # [summary]
    # res_metrics = {}
    # if n_classes > 1: 
    print('status> evaluate clusters (e.g. silhouette scores) ...')
    res_metrics = evaluate_cluster(cluster_labels)

    print('status> summarize clusters (e.g. knn wrt centroids)')
    res_summary = summarize_cluster(cluster_labels)

    # save clustering result 
    if save_cluster: 
        header = ['id', 'cluster_id', ]
        adict = {h:[] for h in header}

        # [output]
        fpath = os.path.join(outputdir, 'cluster_ids-%s.csv' % identifier)
        for i, cl in enumerate(cluster_labels):
            adict['id'].append(ts_indices[i])
            adict['cluster_id'].append(cl)

        df = DataFrame(adict, columns=header) 
        if doc_labeled:
            header = ['id', 'cluster_id', 'label', ]
            df['label'] = y

        print('output> saving clusters (id -> cluster id) to %s' % fpath)
        df.to_csv(fpath, sep='|', index=False, header=True)  
        
        # [output]
        if cluster_to_docs is not None: 
            fpath = os.path.join(outputdir, 'cluster_map-%s.csv' % identifier)  
            header = ['cluster_id', 'data', ] # in general, this should include 'id'
            adict = {h:[] for h in header}
            size_cluster = 0
            for cid, content in cluster_to_docs.items():
                size_cluster += len(content)
            size_avg = size_cluster/(len(cluster_to_docs)+0.0)
            print('verify> averaged %s-cluster size: %f' % (cluster_method, size_avg))
            
            if D is not None: # save only 
                rid = random.sample(cluster_to_docs.keys(), 1)[0]  # [test]
                for cid, content in cluster_to_docs.items():
                    if cid == rid: 
                        print "log> cid: %s => %s\n" % (cid, str(cluster_to_docs[cid]))
                    # store 
                    adict['cluster_id'].append(cid)
                    adict['data'].append(content[0]) # label, sequence, etc. # [todo]
                df =  DataFrame(adict, columns=header)
                print('output> saving cluster content map (cid -> contents) to %s' % fpath)
 
        # [output] save knn 
        header = ['cluster_id', 'knn', ]
        cluster_knn_map = res_summary['cluster_knn']  # wrt mean, cid -> [knn_id]
        sep_knn = ','
        adict = {h: [] for h in header}
        for cid, knn in cluster_knn_map.items(): 
            adict['cluster_id'].append(cid) 
            adict['knn'].append(sep_knn.join([str(e) for e in knn]))
        
        fpath = os.path.join(outputdir, 'cluster_knnmap-%s.csv' % identifier)
        print('output> saving knn-to-centriod map (cid -> knn wrt centroid) to %s' % fpath)
        df = DataFrame(adict, columns=header)
        df.to_csv(fpath, sep='|', index=False, header=True)

    # optimal number of clusters?
    # step 2: Evalution 
    # compute silhouette scores (n_clusters vs silhouette scores)
    div(message='Testing cluster consistency with surrogate labels (%s) ...' % cluster_method)
    if doc_labeled: 
        labels_true = y
        ulabels = set(labels_true)
        n_labels = len(ulabels)
        ratios = {l:[] for l in ulabels}
        for ulabel in ulabels: 
            for cid, alist in cluster_to_docs.items():
                counts = collections.Counter(alist)
                r = counts[ulabel]/(len(alist)+0.0) # given a (true) label, find the ratio of that label in a cluster
                ratios[ulabel].append(r)
        print('result> cluster-label distribution (method: %s):\n%s\n' % (cluster_method, ratios))
        cluster_distribution(ratios)

    return (cluster_labels, res_metrics)  # cluster id to document members (labels or content)

def run_cluster_analysis(**kargs):  
    """
    Main routine for running cluster analysis. 

    Note
    ----
    Refactored to cluster.cluster

    Related
    -------
    cluster_analysis()

    """
    from cluster import cluster  # [todo] refector entire routine to cluster.cluster
    return cluster.run_cluster_analysis(**kargs)


def v_feature_vectors(): # v as verification
    """
    Verify
    ------
    1. feature vector dimensions
    2. subsetting 
       e.g. <prefix>/tpheno/data-exp/fset_kmeans_1000_labeled_subset.csv

    """
    basedir = sys_config.read('DataExpRoot')
    ifile = 'fset_kmeans_1000_labeled.csv'

    # [params]
    n_clusters = 1000
    n_subset = 10000 
    cluster_method = 'kmeans'

    fpath = os.path.join(basedir, ifile)
    df = pd.read_csv(fpath, sep=',', header=None, index_col=0, error_bad_lines=True)
    assert df.shape[1] == n_clusters  # k-means repr given in t_cluster()

    # [output]
    df_subset = df.sample(min(n_subset, df.shape[0]))
    fpath = os.path.join(basedir, 'fset_%s_%s_labeled_subset.csv' % (cluster_method, n_clusters)) 
    df_subset.to_csv(fpath, sep=',', index=True, header=False) 
    print('byproduct> saving a SUBSET (size=%d) of (%s-) clustered feature set (labeled) to %s' % (df_subset.shape[0], cluster_method, fpath))

    return

def test(**kargs): 
    # sa.t_similar(**kargs)

    cohort_name = 'PTSD' # diabetes

    # Given w2v, find d2v and clustering
    # t_cluster(load_w2v_model=True, load_lookuptb=True, 
    #           load_token_cluster=False, load_doc_cluster=False, load_label=False, load_d2v_model=False)

    # Given w2v, find d2v (more d2v-specific compared to t_cluster())
    # options: load_word2vec_model/load_model, load_lookuptb

    t_doc2vec1(load_doc2vec_model=True, seq_ptype='random', n_features=100, test_model=True) 
    t_doc2vec1(load_doc2vec_model=True, seq_ptype='regular', n_features=100, test_model=True)
    
    # Test: mini version of t_doc2vec()
    # v_mini_doc2vec()
    # v_feature_vectors(**kargs)

    # classification 
    t_preclassify_weighted(load_w2v_model=True, seq_ptype='random', n_features=100, test_model=True) 
    t_preclassify_weighted(load_w2v_model=True, seq_ptype='regular', n_features=100, test_model=True)

    t_preclassify(seq_ptype='random', n_features=100) # training data preparation
    t_preclassify(seq_ptype='regular', n_features=100)
    
    # Assuming that surrogate labels were computed (see diabetes.py and preclassify())
    t_classify(seq_ptype='random', n_features=100)
    t_classify(seq_ptype='regular', n_features=100)

    # Assuming d2v models are trained (t_doc2vec1() invoked)
    t_analysis(seq_ptype='random', n_features=100)
    t_analysis(seq_ptype='regular', n_features=100)

    # test_similarity()
    # t_tsne(seq_ptype='random', n_features=100)
    # t_tsne(seq_ptype='random', n_features=100)

    return 

def t_diabetes(**kargs): # PTSD: step one, training set 

    # [params]
    kargs['cohort']='diabetes'
    kargs['has_labels']=True
    kargs['n_clusters']=50; kargs['n_classes']=2
    # kargs['optimize_k']=False
    evalSubgroups(**kargs)

    return 

def classify(**kargs):  # use seqClassify
    return 

def evalSubgroups(**kargs): # PTSD: step one, training set 
    # import vector
    def get_policy_types(): # prior, posterior, or complete sequence (i.e. noop | regular | complete)
        if cohort_name in ['CKD', ]: 
            return ['noop', ]
        return ['prior', 'noop', 'posterior',]  # all 

    read_mode = 'doc'
    seq_ptype = 'regular'

    partial_order = True 
    # prefix = kargs.get('input_dir', sys_config.read('DataExpRoot'))
    cohort_name = kargs.get('cohort', 'PTSD')

    # [params] analyzer: attribute lookup etc
    # bypass_lookup = True

    # [params] training 
    w2v_method = kargs.get('w2v_method', vector.W2V.w2v_method)  # sg: skip-gram
    d2v_method = kargs.get('d2v_method', vector.D2V.d2v_method)  # tfidfavg
    std_method = kargs.get('std_method', 'minmax')  # standardizing feature values

    test_model = True
    
    n_clusters = kargs.get('n_clusters', 3)
    n_classes = '?'  # this is known only after loading training set data
    tOptNumClusters = kargs.get('optimize_k', False)

    tMakeTSet = True
    print('params> cohort=%s, w2v:%s, d2v:%s, n_classes:%s, n_clusters:%s (opt? %s)' % \
        (cohort_name, w2v_method, d2v_method, n_classes, n_clusters, tOptNumClusters))

    labels = kargs.get('labels', [])

    ### make training set 
    # [args] n_features=n_features, window=window, min_count=min_count, n_workers=n_workers,
    #        test_model: if True, test w2v model (similarity tests) after trained
    if tMakeTSet: 
        makeTSet(w2v_method=w2v_method, d2v_method=d2v_method, 
                    seq_ptype=seq_ptype, test_model=test_model,
                        cohort=cohort_name, 
                            labels=labels) 

    print('status> training set complete (cohort=%s, w2v=%s, d2v=%s, s:%s, mode:%s)' % \
        (cohort_name, w2v_method, d2v_method, seq_ptype, read_mode))

    # cluster analysis 
    # [log] status> best K (gap statistic): 40 in range (1, 200) | requested: 50

    # pathway analysis
    # [params] clusters 
    # range_n_clusters = range(20, 251, 10)
    # [note] if optimize_k is True, 'n_clusters' only serves as a reference but will likely be (re-)assigned to the optimized value
    analyze_pathway_batch(cohort=cohort_name, n_clusters=n_clusters, optimize_k=tOptNumClusters, 
        policy_types=get_policy_types())  # [note] for CKD, use entire sequences for now ... 10.23.17

    return 

def t_ptsd0(**kargs): # PTSD 
    import seqTransform as st
    # import seqAnalyzer as sa

    cohort_name = 'PTSD' # diabetes
    read_mode = 'doc'
    seq_ptype = 'regular'
    seq_compo = 'condition_drug'

    partial_order = True 
    # prefix = kargs.get('input_dir', sys_config.read('DataExpRoot'))
    basedir = seqparams.get_basedir(cohort=cohort_name) # load training set from this location
    
    ifile = '%S_seq-%s.dat' % (seq_compo, cohort_name) if cohort_name is not None else '%s_seq.dat' % (seq_compo)
    ifiles = kargs.get('ifiles', [ifile, ])
    
    # [params] auxiliary 
    cluster_method = kargs.get('cluster_method', 'unknown') # debug only

    # [params] training 
    w2v_method = kargs.get('w2v_method', 'sg')  # sg: skip-gram
    d2v_method = kargs.get('d2v_method', 'tfidfavg')  # tfidfavg
    std_method = kargs.get('std_method', 'minmax')  # standardizing feature values

    # [params] pathway 
    tset_type = 'unary'  # binary, multiclass
    order_type = 'partial'

    n_features = kargs.get('n_features', GNFeatures)
    window = kargs.get('window', GWindow)
    min_count = kargs.get('min_count', GMinCount)
    n_cores = multiprocessing.cpu_count()
    # print('info> number of cores: %d' % n_cores)
    n_workers = kargs.get('n_workers', max(n_cores-10, 15))
    test_model = False

    # t_doc2vec1(load_doc2vec_model=True, seq_ptype='regular', n_features=100, test_model=True, cohort=cohort_name)
    # t_preclassify_weighted0(**kargs) 

    # [note] default source document files: ['condition_drug_seq.dat', ] from DataExpRoot 
    #        use 'prefix' to change source (base) directory
    #        use 'ifiles' to change the document source file set
    # print('source> reading from %s' % os.path.join(prefix, ifiles[0]))
    # seqx = sa.read(load_=True, simplify_code=False, mode='doc', verify_=False, seq_ptype=seq_ptype, ifile=ifile)

    ifile = '%s_seq-%s.dat' % (seq_compo, cohort_name) if cohort_name is not None else '%s_seq.dat' % seq_compo  
    seqx = read(load_=True, simplify_code=False, mode=read_mode, verify_=True, seq_ptype=seq_ptype, ifile=ifile)

    # result = sa.loadModel(n_features=n_features, window=window, min_count=min_count, n_workers=n_workers, 
    #             w2v_method=w2v_method,
    #                 seq_ptype=seq_ptype, read_mode=read_mode, test_model=test_model,
    #                 ifile=ifile, cohort=cohort_name, bypass_lookup=True, 
    #                     load_seq=False, load_model=True, load_lookuptb=True) # attributes: sequences, lookup, model
    # seqx = result['sequences']

    # load training set 
    ts = load_tset(w2v_method=w2v_method, d2v_method=d2v_method, 
                        seq_ptype=seq_ptype, read_mode=read_mode, 
                            cohort=cohort_name, idx=None) # set idx to None => return entire tset

    assert ts is not None and not ts.empty, "No training set found!"

    
    # partial_order use global 
    partial_order = True 
    ng_min, ng_max = 1, 10
    topn = 10
    topn_dict = {n:topn for n in range(ng_min, ng_max+1)}

    for partial_order in (True, False, ): 
        otype = 'partial' if partial_order else 'total'

        for ctype in ('diagnosis', 'medication', 'mixed',):  # ~ seq_ptype: (diag, med, regular)
            for ptype in ('prior', 'noop', 'posterior', ): # policy type
                # D0 = {cid:[] for cid in cluster_to_docs.keys()}
                seq_ptype = seqparams.normalize_ctype(seq_ptype=ctype)  # overwrites global value
        
                D0 = []
                for seq in seqx: 
                    D0.append(st.transform(seq, cut_policy=ptype, inclusive=True, seq_ptype=seq_ptype, predicate=st.is_PTSD))
            
                # print('test2> n_doc0: %d' % len(D0))
                # seq_ptype (ctype) i.e. seq_ptype depends on ctype; assuming that d2v_method is 'global' to this routine
                # identifier = 'CM%s-%s-O%s-C%s-S%s-D2V%s' % (ctype, mtype, order_type, cluster_method, seq_ptype, d2v_method)
                
                # [memo] otype (partial_order or more subtle types)
                eval_motif(D0, topn=topn_dict, ng_min=1, ng_max=8, 
                            partial_order=partial_order, 
                                mtype='one_class_%s' % cohort_name, ctype=ctype, ptype=ptype, otype=otype,
                                seq_ptype=seq_ptype, d2v_method='null',
                                    save_=True, load_=True)  # n (as in n-gram) -> topn counts [(n_gram, count), ]
                
                print('status> finished [ctype: %s, seq_ptype: %s, ptype: %s, order: %s] >> n_doc: %d' % \
                    (ctype, seq_ptype, ptype, otype, len(D0)))

    return

def test2(**kargs): 
    # cluster_documents() 

    ### sequence labeling (via code usage frequencies, pre-labeled data, ...)
    # df = labelDocsByDataFrame()
    # labels = df['label'].values
    # labeled_seqx = makeD2VLabels(sequences=sequences, labels=labels)

    ### formulating d2v 
    # use seqmaker.analyzer

    return 

if __name__ == "__main__": 
    # test()
    # test2() 
    # test_similarity()  # pathway analysis
    
    evalSubgroups(cohort='CKD')  # make training data (and perform pathway/sequence analysis)
    # t_ptsd0()
    # t_diabetes()

    ### Misc test/template suite 
    # t_analysis(seq_ptype='regular', n_features=100)
    # t_tdidf(seq_ptype='regular')
    # t_preclassify_weighted(load_w2v_model=True, seq_ptype='random', n_features=100, test_model=True) 
    # t_preclassify_weighted(load_w2v_model=True, seq_ptype='regular', n_features=100, test_model=True)

