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

def labelDoc2(sequences=None, **kargs): 
    """
    Label sequences/sentences for the purpose of using Doc2Vec. 

    Related 
    -------
    * labelDoc()

    """ 
    read_mode = 'doc' # or 'seq'
    seq_ptype = kargs.get('seq_ptype', 'regular')  # values: regular, random, diag, med, lab ... default: regular
    # doc_basename = 'condition_drug'
    # if not seq_ptype.startswith('reg'):
    #     doc_basename = '%s-%s' % (doc_basename, seq_ptype) 

    simplify_code = kargs.get('simplify_code', False)

    load_label = False

    LabelDoc = namedtuple('LabelDoc', 'words tags') # [note] not the same as labelDoc() defined in this module
    exclude = set(string.punctuation)
    all_docs = []

    # [input]
    if sequences is None: sequences = sa.read(load_=True, simplify_code=simplify_code, mode=read_mode, seq_ptype=seq_ptype)

    labels = kargs.get('labels', None) # precomputed sentence labels 
    if labels is None:  
        df_ldoc = labelDoc(sequences, load_=load_label, seqr='full', sortby='freq', seq_ptype=seq_ptype)
        labels = mlabels = list(df_ldoc['label'].values)
        # labels = [mlabel.split(lsep)[0] for mlabel in mlabels] # take most frequent (sub-)label as the label

    for i, sen in enumerate(sequences):
        if isinstance(sen, str): 
            word_list = sen.split() 
        else: 
            word_list = sen  # split is already done

        # For every sentences, if the length is less than 3, we may want to discard it
        # as it seems too short. 
        # if len(word_list) < 3: continue   # filter short sentences
    
        # tag = ['SEN_' + str(i)]
        tag = labels[i]
        tagl = [tag, ]  # can assign a list of tags

        if isinstance(sen, str): 
            sen = ''.join(ch for ch in sen if ch not in exclude)  # filter excluded characters
            all_docs.append(LabelDoc(sen.split(), tagl))
        else:  
            all_docs.append(LabelDoc(sen, tagl)) # assuming unwanted char already filetered 

    # Print out a sample for one to view what the structure is looking like    
    # print all_docs[0:10]
    for i, doc in enumerate(all_docs[0:5]+all_docs[-5:]): 
        print('> doc #%d: %s' % (i, doc))
    # [log] e.g. doc #3: LabelDoc(words=['583.81', '250.41', 'V45.81', ... , '48003'], tags=['362.01_599.0_250.51'])

    return all_docs 

def vectorize_d2v(labeled_docs, **kargs):
    return vectorize2(labeled_docs, **kargs)  
def vectorize2(labeled_docs, **kargs):
    """
    Compute sentence embeddings. 


    Input 
    ----- 
    labeled_doc: labeled sequences (i.e. labelDoc2() was invoked)

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
    def show_params(): 
        msg = "Parameter Setting (d2v):\n"
        msg += '1.  number of features: %d\n' % n_features
        msg += '2.  window length: %d > total: 2 * window: %d\n' % (window, 2*window)
        msg += '3.  min count: %d\n' % min_count
        msg += '4.  PV method: %s\n' % "PV-DM" if dm == 1 else "PV-DBOW"
        msg += '4a. concat? %s\n' % str(dm_concat == 1)
        div(message=msg, symbol='%')
        return 

    # [params]
    doc2vec_method = 'PVDM'  # distributed memory

    n_features = kargs.get('n_features', 200)
    window = kargs.get('window', 7)
    min_count = kargs.get('min_count', 2)
    dm, dm_concat = 1, 1 

    n_cores = multiprocessing.cpu_count()
    print('info> number of cores: %d' % n_cores)
    n_workers = kargs.get('n_workers', max(n_cores-20, 15))

    # [params]
    doctype = 'd2v' 

    # [note] sequence pattern type only affects model file naming 
    seq_ptype = kargs.get('seq_ptype', 'regular')  # values: regular, random, diag, med, lab ... default: regular

    cohort_name = kargs.get('cohort', None) # 'diabetes', 'PTSD'
    doc_basename = 'condition_drug' if cohort_name is None else 'condition_drug-%s' % cohort_name
    # if not seq_ptype.startswith('reg'):
    doc_basename = '%s-P%s' % (doc_basename, seq_ptype) 

    descriptor = kargs.get('meta', doc_basename)  # algorithm type: PV-DBOW, DM
    basedir = sys_config.read('DataExpRoot')

    load_model = kargs.get('load_model', True)
    ofile = kargs.get('output_file', '%s-%s_f%dw%d.%s' % (descriptor, doc2vec_method, n_features, window, doctype))
    fpath = os.path.join(basedir, ofile)

    print('io> reading d2v model file from %s' % fpath)
    show_params()

    # Mikolov pointed out that to reach a better result, you may either want to shuffle the 
    # input sentences or to decrease the learning rate alpha. We use the latter one as pointed
    # out from the blog provided by http://rare-technologies.com/doc2vec-tutorial/

    # negative: if > 0 negative sampling will be used, the int for negative specifies how many "noise words"
    #           should be drawn (usually between 5-20).

    load_model = load_model and os.path.exists(fpath)
    # compute_model = not load_model
    if load_model:
        print('vectorize2> loading pre-computed model from %s' % fpath)

        try: 
            model = Doc2Vec.load(fpath)  # may also need .npy file
        except Exception, e: 
            print('error> failed to load model: %s' % e)
            load_model = False

    if not load_model: 
        # dm, dm_concat = 1, 1 
        print('vectorize2> computing d2v model (dm: %s, dm_concat: %d, n_features: %d, window: %d, min_count: %d, n_workers: %d)' % \
                (dm, dm_concat, n_features, window, min_count, n_workers))
        model = doc2vec.Doc2Vec(dm=1, dm_concat=1, size=n_features, window=window, negative=5, hs=0, min_count=min_count, workers=n_workers, 
                                alpha=0.025, min_alpha=0.025)  # use fixed learning rate
        model.build_vocab(labeled_docs) # [error] must sort before initializing vectors/weights

        for epoch in range(10):
            model.train(labeled_docs)
            model.alpha -= 0.002  # decrease the learning rate
            model.min_alpha = model.alpha  # fix the learning rate, no decay    

        # [output] Finally, we save the model
        print('output> saving doc2vec models (on %d docs) to %s' % (len(labeled_docs), fpath))
        model.save(fpath) 

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
    labels are in a 3-digit format of {0: False, 1: True}
        [type I?, type II?, gestational?] 

    """
    def normalize_ptype(): 
        ptype = kargs.get('seq_ptype', None)
        if ptype is None or ptype.startswith('reg'): 
            ptype = 'regular'
        elif ptype.startswith('rand'): 
            ptype = 'random'
        elif ptype.startswith('d'): 
            ptype = 'diag'
        elif ptype.startswith('m'): 
            ptype = 'med'
        elif ptype.startswith('l'): 
            ptype = 'lab'
        else: 
            print("warnig> unknown sequence pattern type: %s => set to 'regular'" % ptype)
            ptype = 'regular'
        return ptype

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

    read_mode = kargs.get('read_mode', 'doc')  # doc: per-patient documents; seq: per-visit documents/sentences
    seq_ptype = normalize_ptype() # values: regular, random, diag, med, lab ... default: regular

    cohort_name = kargs.get('cohort_name', None)
    doc_basename = 'condition_drug' if cohort_name is None else 'condition_drug-%s' % cohort_name

    if not seq_ptype.startswith('reg'):
        doc_basename = '%s-%s' % (doc_basename, seq_ptype)

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
        sequences = sa.read(load_=True, simplify_code=simplify_code, mode=read_mode, seq_ptype=seq_ptype)

    n_doc = len(sequences)
    print('phenotypeDoc> read %d doc of type %s' % (n_doc, seq_ptype))
    
    labelsets = diab.phenotypeDoc(sequences, **kargs)

    if save_label: save()

    return labelsets

def labelDoc(sequences=None, **kargs): 
    """
    Label (patient) documents via heuristics (e.g. highest code frequencies) 

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
    def normalize_ptype(): 
        ptype = kargs.get('seq_ptype', kargs.get('ptype', None))
        if ptype is None or ptype.startswith('reg'): 
            ptype = 'regular'
        elif ptype.startswith('rand'): 
            ptype = 'random'
        elif ptype.startswith('d'): 
            ptype = 'diag'
        elif ptype.startswith('m'): 
            ptype = 'med'
        elif ptype.startswith('l'): 
            ptype = 'lab'
        else: 
            print("warnig> unknown sequence pattern type: %s => set to 'regular'" % ptype)
            ptype = 'regular'
        return ptype

    read_mode = kargs.get('read_mode', 'doc')  # doc: per-patient documents; seq: per-visit documents/sentences

    seq_ptype = normalize_ptype() # values: regular, random, diag, med, lab ... default: regular

    cohort_name = kargs.get('cohort', None)
    doc_basename = 'condition_drug' if cohort_name is None else 'condition_drug-%s' % cohort_name

    if not seq_ptype.startswith('reg'):
        doc_basename = '%s-%s' % (doc_basename, seq_ptype) 

    simplify_code = kargs.get('simplify_code', False)
    load_label = kargs.get('load_', False)
    doctype = 'csv' 
    
    basedir = sys_config.read('DataExpRoot')
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
        sequences = sa.read(load_=True, simplify_code=simplify_code, mode=read_mode, seq_ptype=seq_ptype)

    n_doc = len(sequences)
    print('info> read %d doc' % n_doc)
    
    # [policy] label by diagnostic codes

    repr_seqx = [None] * n_doc  # representative sequence (formerly diag_seqx)
    n_anomaly = 0
    if seq_ptype in ('regular', 'random', 'diag', ): 
        for i, sequence in enumerate(sequences): 
            # [c for c in sequence if pmed.isICD(e)] 
            repr_seqx[i] = filter(pmed.isICDv2, sequence)  # use diagnostic codes to label the sequence
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

    # save two files
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
    ofile = '%s_unilabel.%s' % (doc_basename, doctype)
    save(freq_cseq_map, header=header, fname=ofile)
    # save result 1b 
    ofile = '%s_unilabel_diag.%s' % (doc_basename, doctype)
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
    ofile = '%s_multilabel.%s' % (doc_basename, doctype)
    df = save(freq_cseq_map, header=header, fname=ofile) # content shows all (frequent) medical codes
    # save result 1b 
    ofile = '%s_multilabel_diag.%s' % (doc_basename, doctype)
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

    # [params]
    read_mode = 'doc'  # assign 'doc' (instead of 'seq') to form per-patient sequences
    seq_ptype = kargs.get('seq_ptype', 'regular') # sequence pattern type: regular, random, diag, med, lab

    # [params] training 
    n_features = kargs.get('n_features', 200)
    window = kargs.get('window', 7)
    min_count = kargs.get('min_count', 2)

    n_cores = multiprocessing.cpu_count()
    # print('info> number of cores: %d' % n_cores)
    n_workers = kargs.get('n_workers', max(n_cores-20, 15))

    # [params] load pre-computed data
    load_word2vec_model = kargs.get('load_word2vec_model', kargs.get('load_w2v_model', True))
    test_model = kargs.get('test_model', kargs.get('test_w2v', False))
    load_lookuptb = kargs.get('load_lookuptb', True)
    load_token_cluster = kargs.get('load_token_cluster', True) and load_word2vec_model
    load_doc_cluster = kargs.get('load_doc_cluster', True) and load_word2vec_model # per-patient document clusters
    load_visit_cluster = kargs.get('load_visit_cluster', False) and load_word2vec_model # per-visit sentence clusters
    load_label = kargs.get('load_label', True)
    
    load_doc2vec_model = kargs.get('load_doc2vec_model', kargs.get('load_d2v_model', True)) # simple (first attempt) doc2vec model; induced by vectorize2

    # [params] cluster document
    doctype = 'txt' 
    cohort_name = kargs.get('cohort', None)
    doc_basename = 'condition_drug' if cohort_name is None else 'condition_drug-%s' % cohort_name

    # [params]
    basedir = outputdir = sys_config.read('DataExpRoot')

    # [params] test 

    # [params] document labeling 
    lsep = seqparams.lsep  # label separator (e.g. '_')

    # [input]
    # read_mode: {'seq', 'doc', 'csv'}  # 'seq' by default
    # cohort vs temp doc file: diabetes -> condition_drug_seq.dat
    ifile = kargs.get('ifile', 'condition_drug_seq-%s.dat' % cohort_name if cohort_name is not None else 'condition_drug_seq.dat')   
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
    df_ldoc = labelDoc(sequences, load_=load_label, seqr='full', sortby='freq', seq_ptype=seq_ptype)
    mlabels = list(df_ldoc['label'].values)
    labels = [mlabel.split(lsep)[0] for mlabel in mlabels] # take most frequent (sub-)label as the label
    print('verify> labels (dtype=%s): %s' % (type(labels), labels[:10]))

    # consider each sequence as a doc 
    train_centroids = np.zeros( (n_doc, n_clusters), dtype="float32" )  # very sparse!
    for i, sequence in enumerate(sequences):  # [log] 432000 documents
        fset = analyzer.create_bag_of_centroids( sequence, word_centroid_map )
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

    labeled_seqx = labelDoc2(sequences=sequences, labels=mlabels)
    # save it?

    model = vectorize2(labeled_docs=labeled_seqx, load_model=load_doc2vec_model, 
                        seq_ptype=seq_ptype, 
                        n_features=n_features, window=window, min_count=min_count, n_workers=n_workers)
    # [log] output> saving doc2vec models (on 432000 docs) to <prefix>/tpheno/data-exp/condition_drug_test.doc2vec

    div(message='Testing most similar documents ...')  # => t_doc2vec1

    
    return

def t_preclassify_weighted0(**kargs): 
    """
    No labeling 
    """
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
    import analyzer
    from seqparams import TSet

    # [params]
    read_mode = 'doc'  # assign 'doc' (instead of 'seq') to form per-patient sequences
    seq_ptype = seqparams.normalize_ptype(**kargs) # sequence pattern type: regular, random, diag, med, lab
    word2vec_method = 'SG'
    doc2vec_method = 'PVDM'  # default: distributed memory
    testdir = os.path.join(os.getcwd(), 'test')
    if not os.path.exists(testdir): os.makedirs(testdir) # test directory

    # [params] training 
    n_features = kargs.get('n_features', 100)
    window = kargs.get('window', 7)
    min_count = kargs.get('min_count', 2)  # set to >=2, so that symbols that only occur in one doc don't count

    n_cores = multiprocessing.cpu_count()
    # print('info> number of cores: %d' % n_cores)
    n_workers = kargs.get('n_workers', max(n_cores-20, 15))

    # [params] data set
    f_prefix = seqparams.generic_feature_prefix  # feature prefix
    f_label = seqparams.TSet.label_field
    fsep = ','  # feature separator

    doctype = 'd2v' 
    doc_basename = 'condition_drug'
    descriptor = kargs.get('meta', doc_basename)  # algorithm type: PV-DBOW, DM

    # assumption: w2v has been completed
    basedir = outputdir = os.path.join(os.getcwd(), 'data') # [I/O] sys_config.read('DataExpRoot')
    load_word2vec_model = kargs.get('load_model', kargs.get('load_word2vec_model', True))
    test_model = kargs.get('test_model', kargs.get('test_w2v', True)) # independent from loading or (re)computing w2v
    load_lookuptb = kargs.get('load_lookuptb', True) and load_word2vec_model
    # load_doc2vec_model = kargs.get('load_doc2vec_model', False) # simple (first attempt) doc2vec model; induced by vectorize2 
    # load_label = kargs.get('load_label', True) and load_doc2vec_model
    load_labelsets = kargs.get('load_labelsets', False)

    # sequences and word2vec model
    cohort_name = kargs.get('cohort', None)
    doc_basename = 'condition_drug' if cohort_name is None else 'condition_drug-%s' % cohort_name
    
    ifile = kargs.get('ifile', 'condition_drug_seq-%s.dat' % cohort_name if cohort_name is not None else 'condition_drug_seq.dat')   
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

    ### Binary Classification ### 
    # doc2vec_method = 'average'
    fpath = os.path.join(basedir, 'tset_%s_%s-C%s.csv' % (doc2vec_method, seq_ptype, cohort_name)) 
    ts = None
    if ts is None: 
        X = analyzer.getAvgFeatureVecs(sequences, model=model, n_features=n_features)
        assert X.shape[0] == n_doc0
        # X_t1, y_t1 = X[t1idx], np.repeat(0, len(t1idx))  # type 1 is labeled as 0
        # X_t2, y_t2 = X[t2idx], np.repeat(1, len(t2idx))  # type 2 is labeled as 1  
        # Xb = np.vstack([X_t1, X_t2]) 
        # yb = np.hstack([y_t1, y_t2])
        # idxb = np.hstack([t1idx, t2idx])
        # assert Xb.shape[0] == len(yb)
        # n_docb = Xb.shape[0]
        
        print('output> preparing ts for binary classification > method: %s, n_doc: %d' % (doc2vec_method, n_docb))
        header = ['%s%s' % (f_prefix, i) for i in range(X.shape[1])]
        ts = DataFrame(Xb, columns=header)
        ts[TSet.target_field] = 1  # only 1 class
        ts[TSet.index_field] = range(X.shape[0])
        ts = ts.reindex(np.random.permutation(ts.index)) 

        # fpath = os.path.join(basedir, 'tset_%s_%s_t1t2-P%s.csv' % (doc2vec_method, n_doc0, seq_ptype)) 
        print('output> saving (one-class classification, d2v method=%s) training data to %s' % (doc2vec_method, fpath))
        ts.to_csv(fpath, sep=fsep, index=False, header=True) # cmp: previously, we use ts.index to keep labels => index=True

    return

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
    import analyzer
    from seqparams import TSet

    # [params]
    read_mode = 'doc'  # assign 'doc' (instead of 'seq') to form per-patient sequences
    seq_ptype = seqparams.normalize_ptype(**kargs) # sequence pattern type: regular, random, diag, med, lab
    word2vec_method = 'SG'
    doc2vec_method = 'PVDM'  # default: distributed memory
    testdir = os.path.join(os.getcwd(), 'test')
    if not os.path.exists(testdir): os.makedirs(testdir) # test directory

    # [params] training 
    n_features = kargs.get('n_features', 100)
    window = kargs.get('window', 7)
    min_count = kargs.get('min_count', 2)  # set to >=2, so that symbols that only occur in one doc don't count

    n_cores = multiprocessing.cpu_count()
    # print('info> number of cores: %d' % n_cores)
    n_workers = kargs.get('n_workers', max(n_cores-20, 15))

    # [params] data set
    f_prefix = seqparams.generic_feature_prefix  # feature prefix
    f_label = seqparams.TSet.label_field
    fsep = ','  # feature separator

    doctype = 'd2v' 
    doc_basename = 'condition_drug'
    descriptor = kargs.get('meta', doc_basename)  # algorithm type: PV-DBOW, DM

    # assumption: w2v has been completed
    basedir = outputdir = os.path.join(os.getcwd(), 'data') # [I/O] sys_config.read('DataExpRoot')
    load_word2vec_model = kargs.get('load_model', kargs.get('load_word2vec_model', True))
    test_model = kargs.get('test_model', kargs.get('test_w2v', True)) # independent from loading or (re)computing w2v
    load_lookuptb = kargs.get('load_lookuptb', True) and load_word2vec_model
    # load_doc2vec_model = kargs.get('load_doc2vec_model', False) # simple (first attempt) doc2vec model; induced by vectorize2 
    # load_label = kargs.get('load_label', True) and load_doc2vec_model
    load_labelsets = kargs.get('load_labelsets', False)

    # sequences and word2vec model
    cohort_name = kargs.get('cohort', None)
    doc_basename = 'condition_drug' if cohort_name is None else 'condition_drug-%s' % cohort_name

    ifile = kargs.get('ifile', 'condition_drug_seq-%s.dat' % cohort_name if cohort_name is not None else 'condition_drug_seq.dat')   
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
    res = apply_approx_pheno(sequences)
    # labeling convention: type 1: 0, type 2: 1, type 3 (gestational): 2
    t1idx, t2idx, t3idx = res['type_1'], res['type_2'], res['gestational']  

    ### Document Embedding as a Function of Word Embedding ### 

    # Method 1: doc vectors by averating

    ### Binary Classification ### 
    doc2vec_method = 'average'
    fpath = os.path.join(basedir, 'tset_%s_%s_t1t2-P%s.csv' % (doc2vec_method, n_doc0, seq_ptype)) 
    ts = load_fset(fpath, fsep=fsep)
    if ts is None: 
        X = analyzer.getAvgFeatureVecs(sequences, model=model, n_features=n_features)
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

    # mseqx = mread(model, docs=sequences) # returns modeled parts of the input sequences
    # n_docmf = len(mseqx) # mf: model-filtered
    # assert n_docmf == n_docb
    # print('verify> number of docs > before: %d, after %d' % (n_doc0, n_docmf))

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
        X = analyzer.getTfidfAvgFeatureVecs(sequences, model, n_features) # optional: min_df, max_df, max_features
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
    import analyzer

    # [params] training 
    n_features = kargs.get('n_features', 100)
    window = kargs.get('window', 7)
    min_count = kargs.get('min_count', 2)  # set to >=2, so that symbols that only occur in one doc don't count
    
    n_cores = multiprocessing.cpu_count()
    # print('info> number of cores: %d' % n_cores)
    n_workers = kargs.get('n_workers', max(n_cores-20, 15))

    word2vec_method = 'SG'
    doc2vec_method = 'PVDM'  # default: distributed memory
    read_mode = kargs.get('read_mode', 'doc')  
    seq_ptype = seqparams.normalize_ptype(**kargs)
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
    X = analyzer.getTfidfAvgFeatureVecs(sequences, model, n_features, test_=True) # optional: min_df, max_df, max_features
    assert X.shape[0] == n_doc0

    return

def t_doc2vec1(**kargs):
    """
    
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
    def normalize_ptype(): 
        ptype = kargs.get('seq_ptype', kargs.get('ptype', None))
        if ptype is None or ptype.startswith('reg'): 
            ptype = 'regular'
        elif ptype.startswith('rand'): 
            ptype = 'random'
        elif ptype.startswith('d'): 
            ptype = 'diag'
        elif ptype.startswith('m'): 
            ptype = 'med'
        elif ptype.startswith('l'): 
            ptype = 'lab'
        else: 
            print("warnig> unknown sequence pattern type: %s => set to 'regular'" % ptype)
            ptype = 'regular'
        return ptype

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

    # [params]
    read_mode = 'doc'  # assign 'doc' (instead of 'seq') to form per-patient sequences
    seq_ptype = normalize_ptype() # sequence pattern type: regular, random, diag, med, lab
    word2vec_method = 'SG'
    doc2vec_method = 'PVDM'  # distributed memory
    testdir = os.path.join(os.getcwd(), 'test')
    if not os.path.exists(testdir): os.makedirs(testdir) # test directory

    # [params] training 
    n_features = kargs.get('n_features', 100)
    window = kargs.get('window', 7)
    min_count = kargs.get('min_count', 2)  # set to >=2, so that symbols that only occur in one doc don't count

    n_cores = multiprocessing.cpu_count()
    # print('info> number of cores: %d' % n_cores)
    n_workers = kargs.get('n_workers', max(n_cores-20, 15))

    # [params] data set
    f_prefix = seqparams.generic_feature_prefix  # feature prefix
    f_label = seqparams.TSet.label_field

    doctype = 'd2v' 
    doc_basename = 'condition_drug'

    descriptor = kargs.get('meta', doc_basename)  # algorithm type: PV-DBOW, DM

    basedir = outputdir = sys_config.read('DataExpRoot')

    load_word2vec_model = kargs.get('load_model', kargs.get('load_word2vec_model', True))
    test_model = kargs.get('test_model', kargs.get('test_w2v', False))
    load_lookuptb = kargs.get('load_lookuptb', True) and load_word2vec_model
    load_doc2vec_model = kargs.get('load_doc2vec_model', True) # simple (first attempt) doc2vec model; induced by vectorize2 
    load_label = kargs.get('load_label', True) and load_doc2vec_model

    # sequences and word2vec model
    result = sa.loadModel(n_features=n_features, window=window, min_count=min_count, n_workers=n_workers,
                            w2v_method=word2vec_method,  
                            read_mode=read_mode, seq_ptype=seq_ptype, test_model=test_model,
                                load_seq=False, load_model=load_word2vec_model, load_lookuptb=load_lookuptb) 
    sequences = result['sequences']
    model = result['model']
    lookuptb = result['symbol_chart']
    n_doc = len(sequences)
    print('verify> number of docs: %d' % n_doc)

    # sequence labeling (via frequecy-based heuristics)
    df_ldoc = labelDoc(sequences, load_=load_label, seqr='full', sortby='freq', seq_ptype=seq_ptype)
    labels = list(df_ldoc['label'].values) # multicode labels by default

    # labels = [mlabel.split(lsep)[0] for mlabel in mlabels] # take most frequent (sub-)label as the label
    print('t_doc2vec1> labels (dtype=%s): %s' % (type(labels), labels[:10]))

    labeled_seqx = labelDoc2(sequences=sequences, labels=labels)
    # save? 

    # doc2vec model
    # ofile = kargs.get('output_file', '%s-%s_f%dw%d.%s' % (descriptor, doc2vec_method, n_features, window, doctype))
    # fpath = os.path.join(basedir, ofile)
    print('t_doc2vec1> load previous d2v model? %s' % load_doc2vec_model)
    d2v_model = vectorize2(labeled_docs=labeled_seqx, load_model=load_doc2vec_model, 
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

def t_preclassify(**kargs): 
    """

    Note
    ----
    1.  Labeling convention: type 1: 0, type 2: 1, type 3 (gestational): 2


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

    from pattern import diabetes as diab 
    from seqparams import TSet

    # [params]
    read_mode = 'doc'  # assign 'doc' (instead of 'seq') to form per-patient sequences
    kargs['seq_ptype'] = seq_ptype = seqparams.normalize_ptype(**kargs) # sequence pattern type: regular, random, diag, med, lab
    doc2vec_method = 'PVDM'  # distributed memory
    load_labelsets = kargs.get('load_labelsets', False)
    
    n_clusters = 100
    f_prefix = seqparams.generic_feature_prefix  # feature prefix
    fsep = ','

    basedir = outputdir = os.path.join(os.getcwd(), 'data')  # global data dir: sys_config.read('DataExpRoot')
    if not os.path.exists(basedir): os.makedirs(basedir) # test directory

    # sequences and word2vec model
    X, y, D = build_data_matrix2(**kargs) # X according to d2v model (PVDM, etc.)
    sequences = D
    n_doc0 = len(sequences)
    print('preclassify> number of docs (of type %s): %d' % (seq_ptype, n_doc0))    

    res = apply_approx_pheno(sequences)
    
    # labeling convention: type 1: 0, type 2: 1, type 3 (gestational): 2
    t1idx, t2idx, t3idx = res['type_1'], res['type_2'], res['gestational']  
    
    ### binary classification ### 
    doc2vev_methods = 'PVDM'
    fpath = os.path.join(basedir, 'tset_%s_%s_t1t2-P%s.csv' % (doc2vec_method, n_doc0, seq_ptype)) 
    ts = load_fset(fpath, fsep=fsep) # try to load the data first
    if ts is None: 
        X_t1, y_t1 = X[t1idx], np.repeat(0, len(t1idx))  # type 1 is labeled as 0
        X_t2, y_t2 = X[t2idx], np.repeat(1, len(t2idx))  # type 2 is labeled as 1  

        Xb = np.vstack([X_t1, X_t2]) 
        yb = np.hstack([y_t1, y_t2])
        idxb = np.hstack([t1idx, t2idx]) # need to know where each feature vector corresponds to which doc

        assert Xb.shape[0] == len(yb)

        print('output> preparing ts for binary classification (with a header) ...')
        header = ['%s%s' % (f_prefix, i) for i in range(Xb.shape[1])]
        ts = DataFrame(Xb, columns=header)
        ts[TSet.target_field] = yb
        ts[TSet.index_field] = idxb

        ts = ts.reindex(np.random.permutation(ts.index)) 

        # fpath = os.path.join(basedir, 'tset_%s_%s_t1t2-P%s.csv' % (doc2vec_method, n_doc, seq_ptype)) 
        print('output> saving (binary classification) training data to %s' % fpath)
        ts.to_csv(fpath, sep=fsep, index=False, header=True) # cmp: previously, we use ts.index to keep labels => index=True

    ### multiclass: 3 class including gestational
    doc2vev_methods = 'PVDM'
    fpath = fpath = os.path.join(basedir, 'tset_%s_%s_t1t2t3-P%s.csv' % (doc2vec_method, n_doc0, seq_ptype)) 
    ts = load_fset(fpath, fsep=fsep) 
    if ts is None:    
        X_t3, y_t3 = X[t3idx], np.repeat(2, len(t3idx))  # gestational is labeled 2
        Xm = np.vstack([Xb, X_t3])
        ym = np.hstack([yb, y_t3])
        idxm = np.hstack([idxb, t3idx])
        print('output> preparing ts for multiclass classification (with a header) ...')
    
        header = ['%s%s' % (f_prefix, i) for i in range(Xm.shape[1])]
        ts = DataFrame(Xm, columns=header)
        ts[TSet.target_field] = ym 
        ts[TSet.index_field] = idxm
        ts = ts.reindex(np.random.permutation(ts.index))   

        # fpath = os.path.join(basedir, 'tset_%s_%s_t1t2t3-P%s.csv' % (doc2vec_method, n_doc, seq_ptype)) 
        print('output> saving (3-label classification) training data to %s' % fpath)
        ts.to_csv(fpath, sep=fsep, index=False, header=True) # cmp: previously, we use ts.index to keep labels => index=True  

    return

def load_ts(**kargs):
    """
    Load training set from pre-computed data files. 

    Related
    -------
    1. X, y, D <- build_data_matrix2
       where X is computed via d2v model (e.g. PVDM)
    2. load_XY 
    3. getXY, getXYD
    """
    # import evaluate

    # [params]
    read_mode = 'doc'  # assign 'doc' (instead of 'seq') to form per-patient sequences
    kargs['seq_ptype'] = seq_ptype = seqparams.normalize_ptype(**kargs) # sequence pattern type: regular, random, diag, med, lab
    doc2vec_method = kargs.get('d2v_method', kargs.get('doc2vec_method', 'PVDM'))   # average, tfidfavg, or PVDM (distributed memory)
    tset_type = kargs.get('tset_type', 'binary')  # tertiary, mutliple

    # [params] I/O
    basedir = outputdir = os.path.join(os.getcwd(), 'data')  # global data dir: sys_config.read('DataExpRoot')
    if not os.path.exists(basedir): os.makedirs(basedir) # test directory
    
    # sequences and word2vec model
    # X, y, D = kargs.get('X', None), kargs.get('y', None), kargs.get('D', None)
    if X is None: 
        # X, y, D = build_data_matrix2(**kargs) # n_features, etc. 
        D, ts = build_data_matrix_ts(read_mode=read_mode, seq_ptype=seq_ptype, d2v_method=doc2vec_method, tset_type=tset_type)
        # X, y = evaluate.transform(ts, standardize_=True) # default: minmax

    # sequences = D
    
    # pd.read_csv(fpath_default, sep=fsep, header=0, index_col=False, error_bad_lines=True)
    return ts
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

    standardize_ = kargs.get('standardize_ki', 'minmax')
    ts = load_ts(**kargs)
    X, y = evaluate.transform(ts, standardize_=standardize_) # default: minmax

    return (X, y)
def getXY(**kargs): 
    return load_XY(**kargs)
def getXYD(**kargs): 
    import evaluate

    # [params]
    read_mode = 'doc'  # assign 'doc' (instead of 'seq') to form per-patient sequences
    kargs['seq_ptype'] = seq_ptype = seqparams.normalize_ptype(**kargs) # sequence pattern type: regular, random, diag, med, lab
    doc2vec_method = kargs.get('d2v_method', kargs.get('doc2vec_method', 'PVDM'))   # average, tfidfavg, or PVDM (distributed memory)
    load_labelsets = kargs.get('load_labelsets', False)
    standardize_ = kargs.get('standardize_ki', 'minmax')
    tset_type = kargs.get('tset_type', 'binary')  # tertiary, mutliple

    D, ts = build_data_matrix_ts(read_mode=read_mode, seq_ptype=seq_ptype, d2v_method=doc2vec_method, tset_type=tset_type)
    # return build_data_matrix2(**kargs) 
    X, y = evaluate.transform(ts, standardize_=standardize_) # default: minmax
    return (X, y, D)

def t_classify(**kargs): 
    import evaluate

    # [params]
    read_mode = 'doc'  # assign 'doc' (instead of 'seq') to form per-patient sequences
    kargs['seq_ptype'] = seq_ptype = seqparams.normalize_ptype(**kargs) # sequence pattern type: regular, random, diag, med, lab
    doc2vec_method = 'PVDM'  # distributed memory
    load_labelsets = kargs.get('load_labelsets', False)
    fsep = ','
    tset_stem = 'tset'

    # [params] I/O
    basedir = outputdir = os.path.join(os.getcwd(), 'data')  # global data dir: sys_config.read('DataExpRoot')
    if not os.path.exists(basedir): os.makedirs(basedir) # test directory
    
    # sequences and word2vec model
    X, y, D = kargs.get('X', None), kargs.get('y', None), kargs.get('D', None)
    if X is None: 
        X, y, D = build_data_matrix2(**kargs) # n_features, etc. 
    sequences = D
    n_doc = len(sequences)    

    ### Binary class

    print('classify> run binary class classification ... ')
    # [todo] loop through cartesian product of seq_ptype and d2v_method? 

    for doc2vec_method in ('average', 'tfidfavg',  ): # 'PVDM',
        fpath = os.path.join(basedir, 'tset_%s_%s_t1t2-P%s.csv' % (doc2vec_method, n_doc, seq_ptype)) 
        assert os.path.exists(fpath), "dependency> fset for d2v_method=%s does not exist > Run t_preclassify(...) first!" % doc2vec_method
        ts = pd.read_csv(fpath, sep=fsep, header=0, index_col=False, error_bad_lines=True)
        print('info> d2v_method: %s > tset dim: %s' % (doc2vec_method, str(ts.shape)))
        evaluate.binary_classify(ts, seq_ptype=seq_ptype, d2v_type=doc2vec_method)

    ### Multiclass 

    # load data 
    print('classify> Run multiclass classification ... ')
    fpath = os.path.join(basedir, 'tset_%s_%s_t1t2t3-P%s.csv' % (doc2vec_method, n_doc, seq_ptype)) 
    assert os.path.exists(fpath), "dependency> Run t_preclassify(...) first!"

    return

def t_tsne(**kargs):
    # from cluster import tsne
    import tsne

    ts = load_ts(tset_type='binary')
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

def mread(model, **kargs): # model read
    """
    Read sequences in which only modeled codes are retained in each sequence.  

    Use 
    ---
    1. document embedding
       TD-IDF weighted document but excluding codes that fell below min_count and thus were not modeled. 
    """
    read_mode = kargs.get('read_mode', 'doc')
    seq_ptype = kargs.get('seq_ptype', 'regular') # sequence pattern type: regular, random, diag, med, lab
    load_seq = kargs.get('load_seq', True)
    seqx = kargs.get('docs', kargs.get('sequences', None))
    
    if seqx is None: 
        seqx = sa.read(load_=load_seq, simplify_code=False, mode=read_mode, verify_=False, seq_ptype=seq_ptype)

    seqx_prime = []
    n_removed = n_tokens = 0
    for seq in seqx: 
        seq_subset = []
        is_modeled = True
        for s in seq:
            n_tokens += 1 
            try: 
                model[s]
            except: 
                is_modeled = False 
            if is_modeled: 
                seq_subset.append(s)
            else: 
                n_removed += 1 
        seqx_prime.append(seq_subset)
    print('stats> removed %d tokens out of total %d (ratio: %f)' % (n_removed, n_tokens, n_removed/(n_tokens+0.0)))
    return seqx_prime

def build_data_matrix_ts(**kargs):  # [precond] t_preclassify()
    """
    Read training data from files where each file contains 
    (doc) feature vectors and (surrogate) labels derived from pre_classify with surrogate labeling and pre-computed d2v 

    """
    from seqparams import TSet

    # [params] read the document first followed by specifying training set based on seq_ptype and d2v method
    read_mode = kargs.get('read_mode', 'doc')  # doc, random, etc. 
    seq_ptype = seqparams.normalize_ptype(**kargs)
    doc2vec_method = kargs.get('d2v_method', kargs.get('doc2vec_method', 'PVDM'))
    tset_type = kargs.get('ts_type', kargs.get('tset_type', 'binary'))
    fsep = ','
    tset_stem = 'tset'
    
    load_seq = True
    
    basedir = outputdir = os.path.join(os.getcwd(), 'data')  # global data dir: sys_config.read('DataExpRoot')
    if not os.path.exists(basedir): os.makedirs(basedir) # test directory

    sequences = sa.read(load_=load_seq, simplify_code=False, mode=read_mode, verify_=False, seq_ptype=seq_ptype)
    n_doc = len(sequences)
    # but this loads all sequences!

    fpath_default = os.path.join(basedir, 'tset_%s_%s_t1t2-P%s.csv' % (doc2vec_method, n_doc, seq_ptype)) 
    if tset_type.startswith('bin'): 
        print('io> loading binary class training data ... ')
        fpath = os.path.join(basedir, 'tset_%s_%s_t1t2-P%s.csv' % (doc2vec_method, n_doc, seq_ptype)) 
        
    elif tset_type.startswith('t'):
        print('io> loading 3-label multiclass training data ... ')
        fpath = os.path.join(basedir, 'tset_%s_%s_t1t2t3-P%s.csv' % (doc2vec_method, n_doc, seq_ptype)) 
         
    else: 
        print('io> unknown tset_type: %s > use %s by default!' % (tset_type, os.path.basename(fpath_default)))
        fpath = fpath_default

    if not os.path.exists(fpath): 
        print("dependency> fset for d2v_method=%s does not exist > Run t_preclassify(...) first!") % doc2vec_method
        return None 

    ts = pd.read_csv(fpath, sep=fsep, header=0, index_col=False, error_bad_lines=True)
    print('output> loaded training data | d2v_method: %s > tset dim: %s' % (doc2vec_method, str(ts.shape)))

    idx = ts[TSet.index_field].values # ~ positions of sequences
    D = sa.select(docs=sequences, idx=idx)
    print('stats> subset original docs from size of %d to %d' % (n_doc, len(D)))

    return (D, ts)

def build_data_matrix0(**kargs): 
    """
    No label y 
    """
    # [params] training parameters 
    n_features = kargs.get('n_features', 200)
    window = kargs.get('window', 7)
    min_count = kargs.get('min_count', 2)  # set to >=2, so that symbols that only occur in one doc don't count
    n_cores = multiprocessing.cpu_count()
    n_workers = kargs.get('n_workers', max(n_cores-20, 15)) # n_cores: 30

    # [params] document type and document source 
    seq_ptype = kargs.get('seq_ptype', 'regular') # sequence pattern type: regular, random, diag, med, lab
    prefix = kargs.get('prefix', kargs.get('input_dir', sys_config.read('DataExpRoot')))

    cohort_name = kargs.get('cohort', None)
    ifile = kargs.get('ifile', 'condition_drug_seq-%s.dat' % cohort_name if cohort_name is not None else 'condition_drug_seq.dat') 
    ifiles = kargs.get('ifiles', [ifile, ])

    load_seqx = False

    # [note] default source document files: ['condition_drug_seq.dat', ] from DataExpRoot 
    #        use 'prefix' to change source (base) directory
    #        use 'ifiles' to change the document source file set
    print('source> reading from %s' % os.path.join(prefix, ifiles[0]))
    seqx = sa.read(load_=True, simplify_code=False, mode='doc', verify_=False, seq_ptype=seq_ptype, ifile=ifile)

    model = vectorize2(labeled_docs=lseqx, load_model=True, 
                         seq_ptype=seq_ptype, cohort=cohort_name, ifile=ifile, 
                         n_features=n_features, window=window, min_count=min_count, n_workers=n_workers)

    # [todo]

    return

def build_data_matrix2(**kargs):

    # [params] training parameters 
    n_features = kargs.get('n_features', 200)
    window = kargs.get('window', 7)
    min_count = kargs.get('min_count', 2)  # set to >=2, so that symbols that only occur in one doc don't count
    n_cores = multiprocessing.cpu_count()
    n_workers = kargs.get('n_workers', max(n_cores-20, 15)) # n_cores: 30

    # [params] document type and document source 
    seq_ptype = kargs.get('seq_ptype', 'regular') # sequence pattern type: regular, random, diag, med, lab
    prefix = kargs.get('prefix', kargs.get('input_dir', sys_config.read('DataExpRoot')))

    cohort_name = kargs.get('cohort', None)
    ifile = kargs.get('ifile', 'condition_drug_seq-%s.dat' % cohort_name if cohort_name is not None else 'condition_drug_seq.dat') 
    ifiles = kargs.get('ifiles', [ifile, ])

    load_seqx = False

    # [note] default source document files: ['condition_drug_seq.dat', ] from DataExpRoot 
    #        use 'prefix' to change source (base) directory
    #        use 'ifiles' to change the document source file set
    print('source> reading from %s' % os.path.join(prefix, ifiles[0]))
    seqx = sa.read(load_=True, simplify_code=False, mode='doc', verify_=False, seq_ptype=seq_ptype, ifile=ifile)

    df_ldoc = labelDoc(seqx, load_=True, seqr='full', sortby='freq', seq_ptype=seq_ptype, cohort=cohort_name)

    labels = list(df_ldoc['label'].values) 
    lseqx = labelDoc2(sequences=seqx, labels=labels, cohort=cohort_name)
    model = vectorize2(labeled_docs=lseqx, load_model=True, 
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
    
    return (dmatrix, np.array(labels), seqx) # (X, y, true_X)

def build_data_matrix(lseqx=None, **kargs): 
    """
    lseqx: labeled sequences/documents 

    Memo
    ----
    Assuming that d2v models were obtained, call this routine to get matrix and labels 

    Related
    -------
    build_data_matrix2() returns (X, y, trueX)

    """
    # [params]
    cohort_name = kargs.get('cohort', None)

    # [params] training parameters 
    n_features = kargs.get('n_features', 200)
    window = kargs.get('window', 7)
    min_count = kargs.get('min_count', 2)  # set to >=2, so that symbols that only occur in one doc don't count
    n_cores = multiprocessing.cpu_count()
    n_workers = kargs.get('n_workers', max(n_cores-20, 15)) # n_cores: 30

    # [params] document type and document source 
    prefix = kargs.get('prefix', kargs.get('input_dir', sys_config.read('DataExpRoot')))

    # read_mode: {'seq', 'doc', 'csv'}  # 'seq' by default
    # cohort vs temp doc file: diabetes -> condition_drug_seq.dat
    ifile = kargs.get('ifile', 'condition_drug_seq-%s.dat' % cohort_name if cohort_name is not None else 'condition_drug_seq.dat')
    ifiles = kargs.get('ifiles', [ifile, ])

    seq_ptype = kargs.get('seq_ptype', 'regular') # sequence pattern type: regular, random, diag, med, lab

    load_seqx = False; labels = []
    if lseqx is None: 
        # [note] default source document files: ['condition_drug_seq.dat', ] from DataExpRoot 
        #        use 'prefix' to change source (base) directory
        #        use 'ifiles' to change the document source file set
        print('source> reading from %s' % os.path.join(prefix, ifiles[0]))
        seqx = sa.read(load_=True, simplify_code=False, mode='doc', verify_=False, seq_ptype=seq_ptype)

        df_ldoc = labelDoc(seqx, load_=True, seqr='full', sortby='freq', seq_ptype=seq_ptype)
        labels = list(df_ldoc['label'].values) 
        lseqx = labelDoc2(sequences=seqx, labels=labels)
        load_seqx = True
    else: 
        labels = [lseq.tags[0] for lseq in lseqx]  # [config] configure labels here

    model = vectorize2(labeled_docs=lseqx, load_model=True, 
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
    fmotifs: focused motifs if given, only focus on the ngrams defined in it
       usage example: 
          lmotifs: cluster complement, C_bar
          gmotifs: entire documens, D  
          fmotifs: cluster C 

          where Union(C_bar, C) = D 
    """
    # [input] cluster_motifs, global_motifs 
        
    #         (get) counts of cluster motifs in the global context
    header = kargs.get('header', ['length', 'ngram', 'count', 'global_count', 'ratio'])

    adict = {h: [] for h in header} 
    wsep = ' ' # word separator   # or use ' -> '  

    assert len(gmotifs) > 0 and len(lmotifs) > 0
    n_lmotifs = utils.size_hashtable(lmotifs)
    n_gmotifs = utils.size_hashtable(gmotifs)

    # [filter]
    if fmotifs is not None: 
        n_fmotifs = utils.size_hashtable(fmotifs)
        print('compare> size of local %d, focused: %d, global motifs: %d' % (n_lmotifs, n_fmotifs, n_gmotifs))

        for n, ngr_cnts in fmotifs.items(): # foreach focused motifs (e.g. pre-computed cluster motifs): n (as in n-gram), counts (a dict)
            lcounts = dict(lmotifs[n]) if lmotifs.has_key(n) else {}  # the complement
            if not lcounts: print('compare> Warning: Could not find %d-gram in local motifs (mtype: %s, ctype: %s)' % (n, mtype, ctype))
            gcounts = dict(gmotifs[n]) if gmotifs.has_key(n) else {}
            if not gcounts: print('compare> Warning: Could not find %d-gram in global motifs (mtype: %s, ctype: %s)' % (n, mtype, ctype))

            # n_patterns = len(ngr_cnts)
            # adict['length'].extend([n] * n_patterns)  # [redundancy]

            lengthx, ngram_reprx, local_counts, global_counts, ratios = [], [], [], [], []
            for i, (ngt, fcnt) in enumerate(ngr_cnts): 

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
        print('compare> size of local %d <? global motifs: %d' % (n_lmotifs, n_gmotifs))

        for n, ngr_cnts in lmotifs.items(): # foreach cluster motif: n (ngram), counts (a dict)
            gcounts = dict(gmotifs[n]) if gmotifs.has_key(n) else {}
            if not gcounts: print('lookup> Warning: Could not find %d-gram in global motifs (mtype: %s, ctype: %s)' % (n, mtype, ctype))

            # n_patterns = len(ngr_cnts)
            # adict['length'].extend([n] * n_patterns)  # [redundancy]

            # record these attributes
            lengthx, ngram_reprx, local_counts, global_counts, ratios = [], [], [], [], []
            for i, (ngt, ccnt) in enumerate(ngr_cnts):  # foreach n-gram in cluster motifs | ccnt: cluster-level count 

                # [lookup]
                gcnt = gcounts.get(ngt, 0) # how many times does it appear in global context? 
    
                assert ccnt <= gcnt, "cluster count > global counts? (ccnt: %d > gcnt: %d) > global %d-gram patterns:\n%s\n" % \
                            (ccnt, gcnt, n, str(gcounts))

                # ratio between cluster and global frequencies
                if ccnt > 0 and gcnt > 0: 
                    ratios.append(ccnt/(gcnt+0.0))
                else: 
                    if gcnt == 0: 
                        assert ccnt == 0
                        ratios.append(-1)
                    else: # gcnt > 0 => ccnt > 0 
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
    ### end if-then-else: focused motifs (external) vs local motifs in this cluster

    return adict

def t_pathway(X=None, y=None, D=None, clusters=None, **kargs): 
    """
    Medical coding pathways analysis. 
    Use pathwayAnalyzer for further analyses. 

    Related
    -------
    map_clusters()


    Test
    ----
    1. copy number variations per cluster

    Reference 
    ---------
    1. Cluster purity 
       https://nlp.stanford.edu/IR-book/html/htmledition/evaluation-of-clustering-1.html

    """
    def hvol(tb): # volume of hashtable
        return sum(len(v) for v in tb.values())
   
    def rel_motif_stats(gc_ng_count): # global over cluster map: n -> (n-gram -> count)
        try: 
            global_motifs
        except: 
            tdict = {1: 15, 2: 10, } 
            global_motifs = eval_motif(D, topn=tdict, ng_min=1, ng_max=8, partial_order=partial_order, 
                 seq_ptype=seq_ptype, d2v_method=d2v_method, save_=True) # last two args only for file naming

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
    def df_motif_stats(cmotifs, tfidf_stats, cid=None):  # [input] cluster id, cluster motifs
        # rset = {}  # result set 
        # tfidf_stats[ngr]['cids']: which clusters does an ngram appear? 
        if cid is None: 
            for n, counts in cmotifs.items(): 
                for ngr, c in counts: 
                    if not tfidf_stats.has_key(ngr): 
                        tfidf_stats[ngr] = {}
                        tfidf_stats[ngr]['count'] = 1
                        tfidf_stats[ngr]['cids'] = [-1] 
                        # tfidf_stats[ngr] = 1  # appeared in one cluster
                    else: 
                        tfidf_stats[ngr]['count'] += 1
                        # tfidf_stats[ngr] += 1 
        else: 
            for n, counts in cmotifs.items(): 
                for ngr, c in counts: 
                    if not tfidf_stats.has_key(ngr): 
                        tfidf_stats[ngr] = {}
                        tfidf_stats[ngr]['count'] = 1  # appeared in one cluster
                        tfidf_stats[ngr]['cids'] = [cid] 
                        # ngram_cidx[ngr] = [cid]
                    else: 
                        tfidf_stats[ngr]['count'] += 1 
                        tfidf_stats[ngr]['cids'].append(cid)
                        # ngram_cidx[ngr].append(cid)
        return tfidf_stats

    def res_to_df(res): # populate the result set from eval_motifs()  
        header = ['length', 'ngrams', 'counts']
        adict = {h: [] for h in header} 
        wsep = ' ' # word separator

        for ng, ngr_cnts in res.items(): 
            # local_fpath = os.path.join(outputdir, '%sgram-%s.csv' % (ng, identifier))

            n_patterns = len(ngr_cnts)
            adict['length'].extend([ng] * n_patterns)  # [redundancy]
            ngram_tuples = [ngr for ngr, _ in ngr_cnts]

            # [test]
            if ng == 2 and len(ngram_tuples) > 0: assert isinstance(ngram_tuples[0], tuple), "invalid ngram: %s" % str(ngram_tuples[0]) 
            
            nt_reprx = []
            for nt in ngram_tuples: 
                nt_reprx.append(wsep.join(str(e) for e in nt))  # or use ' -> '

            adict['ngrams'].extend(nt_reprx)

            ngram_counts = [cnt for _, cnt in ngr_cnts]
            adict['counts'].extend(ngram_counts)

        df = DataFrame(adict, columns=header)
        return df 
    
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
     
    def eval_cluster(topn_clusters=None):
        assert y is not None, "Could not evaluate purity without ground truths given."
        # cluster_to_labels = map_clusters(clusters, y)
        # if 'cluster_to_labels' in locals():
        try: 
            cluster_to_labels
        except NameError:
            cluster_to_labels = map_clusters(clusters, y)

        n_total = hvol(cluster_to_labels)
        assert n_total == X.shape[0], "size of training data: %d != nrow(X): %d" % (n_total, X.shape[0])
        
        # [output] purity_score/score, cluster_labels/clabels, ratios, fractions, topn_ratios
        res = {}
        ulabels = sorted(set(y))
        n_labels = len(ulabels)

        res['unique_labels'] = res['ulabels'] = ulabels

        maxx = []
        clabels = {}  # cluster/class label by majority vote
        for cid, labels in cluster_to_labels.items():
            counts = collections.Counter(labels)
            l, cnt = counts.most_common(1)[0]  # [policy]
            clabels[cid] = l            
            maxx.append(max(counts[ulabel] for ulabel in ulabels))

        res['purity_score'] = res['score'] = sum(maxx)/(n_total+0.0)
        res['cluster_labels'] = res['clabels'] = clabels
        
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
        res['ratios'] = ratios # cluster purity ratio for each label
        res['fractions'] = fractions

        # [todo] ratios of label determined by majority votes 
        ratios_max_votes = {}  # cid -> label -> ratio
        for cid, lmax in clabels.items(): 
            ratios_max_votes[cid] = dict(res['ratios'][lmax])[cid]
        res['ratios_max_votes'] = ratios_max_votes

        # rank purest clusters for each label and find which clusters to study
        ranked_ratios = {}
        if topn_clusters is not None: 
            for ulabel in ulabels: 
                ranked_ratios[ulabel] = sorted(ratios[ulabel], key=lambda x: x[1], reverse=True)[:topn_clusters]  # {(cid, r)}
        else: 
            for ulabel in ulabels: 
                ranked_ratios[ulabel] = sorted(ratios[ulabel], key=lambda x: x[1], reverse=True) # {(cid, r)}            
        res['ranked_ratios'] = ranked_ratios

        return res  

    from pattern import diabetes as diab
    import seqTransform as st 
    # import algorithms  # count n-grams
    # from itertools import chain  # faster than flatten lambda (nested list comprehension)

    read_mode = kargs.get('read_mode', 'doc') # documents/doc (one patient one sequence) or sequences/seq
    tset_type = kargs.get('tset_type', 'binary')
    d2v_method = kargs.get('d2v_method', 'tfidfavg')  # tfidfavg
    std_method = kargs.get('std_method', 'minmax')  # standardizing feature values
    
    # [params] auxiliary 
    cluster_method = kargs.get('cluster_method', 'unknown') # debug only
    n_clusters = kargs.get('n_clusters', 10)
    topn_clusters_only = kargs.get('topn_clusters_only', False) # analyze topn clusters only
    topn_ngrams = 5  # analyze top N ngrams
    cluster_label_policy = 'max_vote'

    # [params] ordering 
    partial_order = kargs.get('partial_order', True)
    order_type = 'part' if partial_order else 'strict'

    # [params] sequence
    seq_ptype = kargs.get('seq_ptype', 'regular')  # global default but varies according to pathway analyses
    policy = kargs.get('policy', 'prior') # posterior, noop (i.e. no cut and preserve entire sequence)

    # [params] I/O
    identifier = 'T%s-O%s-C%s-S%s-D2V%s' % (tset_type, order_type, cluster_method, seq_ptype, d2v_method)

    outputdir = os.path.join(os.getcwd(), 'data')
    cohort_name = kargs.get('cohort', None)
    if cohort_name is not None: 
        outputdir = os.path.join(os.getcwd(), cohort_name)
        if not os.path.exists(outputdir): os.makedirs(outputdir) # test directory

    # fname_tfidf = 'tfidf-%s-%s.csv' % (ctype, identifier)    
    fname_cluster_stats = 'cluster_stats-%s-%s.csv' % (cluster_label_policy, identifier)
    load_motif = kargs.get('load_motif', kargs.get('load_', True)) # load global motif

    if X is None or y is None or D is None: 
        D, ts = build_data_matrix_ts(read_mode=read_mode, seq_ptype=seq_ptype, d2v_method=d2v_method, tset_type=tset_type)
        X, y = evaluate.transform(ts, standardize_=std_method) # default: minmax
        # X, y, D = getXYD(read_mode=read_mode, seq_ptype=seq_ptype, d2v_method=d2v_method, tset_type=tset_type) 
    if clusters is None: 
        # run cluster analysis
        clusters, cmetrics = cluster_analysis(X=X, y=y, n_clusters=n_clusters, cluster_method=cluster_method,
                                    seq_ptype=seq_ptype, d2v_method=d2v_method, 
                                    save_=True, load_=True)

    n_doc0 = X.shape[0]
    if D is not None: assert len(D) == X.shape[0], "n_doc: %d while nrows of X: %d" % (len(D), n_doc0)
    n_cluster_id = len(clusters)
    
    print('test> inspecting via cluster method: %s' % cluster_method)
    assert n_doc0 == n_cluster_id, "n_doc: %d while number of cluster IDs: %d (nrow of X: %d)" % (n_doc0, n_cluster_id, X.shape[0])
    
    ### build cluster maps 
    cluster_to_docs = map_clusters(clusters, D)  # cluster ID => a set of documents (d in D) in the same cluster 

    ### make temporal documents by types 

    # First, generate full global motifs as references

    # diagnositic coding sequences prior to the first ,ldiagnosis of the target disease 
    # [note] transform operations: cut, filter (according to seq_ptype)
    #        predicate: diab.is_diabetes by default

    tdict = {1: 500, 2: 500, }  # n -> topn number of n-grams to analyze, if not specified, use default (5)
    # partial_order use global 
    for ctype in ('diagnosis', 'medication', 'mixed',):  # ~ seq_ptype: (diag, med, regular)
        D0 = {cid:[] for cid in cluster_to_docs.keys()}
        seq_ptype = seqparams.normalize_ptype(seq_ptype=ctype)  # overwrites global value
        
        for cid, docs in cluster_to_docs.items(): 
            for doc in docs: 
                D0[cid].append(diab.transform(doc, policy='noop', inclusive=True, seq_ptype=seq_ptype)) # before the first diagnosis, inclusive

        # seq_ptype (ctype) i.e. seq_ptype depends on ctype; assuming that d2v_method is 'global' to this routine
        # identifier = 'CM%s-%s-O%s-C%s-S%s-D2V%s' % (ctype, mtype, order_type, cluster_method, seq_ptype, d2v_method)
        # [memo] otype (partial_order)
        eval_motif(merge_cluster(D0), topn=None, ng_min=1, ng_max=8, 
                            partial_order=partial_order, 
                                mtype='global', ctype=ctype, ptype='noop',
                                seq_ptype=seq_ptype, d2v_method=d2v_method,
                                    save_=True, load_=True)  # n (as in n-gram) -> topn counts [(n_gram, count), ]
    
    # # med coding sequences prior to the first diagnosis of the target disease 
    # D_med = {cid:[] for cid in cluster_to_docs.keys()}
    # for cid, docs in cluster_to_docs.items(): 
    #     for doc in docs: 
            
    #         # transform: cut => filter (according to seq_ptype)
    #         # predicate: diab.is_dies by default
    #         D_med[cid].append(diab.transform(doc, policy=policy, inclusive=True, seq_ptype='med')) # before the first diagnosis, inclusive

    # # mixed? set seq_ptype to regular
    # # med coding sequences prior to the first diagnosis of the target disease 
    # D_mixed = {cid:[] for cid in cluster_to_docs.keys()}
    # for cid, docs in cluster_to_docs.items(): 
    #     for doc in docs: 
    #         D_mixed[cid].append(diab.transform(doc, policy=policy, inclusive=True, seq_ptype='regular')) # before the first diagnosis, inclusive

    # find the most frequent diagnoses, medications ...
    
    # [tip] flattening list
    # flatten = lambda l: [item for sublist in l for item in sublist]
    # use itertools.chain instead: chain.from_iterable(seqx)

    # [params] motif, n-grams
    # topn_clusters = 5

    # res_motifs = {}  # [output] label -> {cids} -> motif stats (global on cluster)

    if y is not None: 
        cluster_to_labels = map_clusters(clusters, y) 

        # [query] purity_score/score, cluster_labels/clabels, ratios, fractions, topn_ratios
        res = eval_cluster(topn_clusters=None) # depends on cluster_to_labels

        div(message='Result: %s clustering > purity: %f' % (cluster_method, res['purity_score']), symbol='#')
        
        # [I/O] save
        clabels = res['cluster_labels']  # cluster (id) => label (by majority vote)
        ratios_max_votes = res['ratios_max_votes']

        lR = res['ranked_ratios'] # topn ratios by labels
        ulabels = res['ulabels']
        res_motifs = {ulabel: {} for ulabel in ulabels}

        min_length, max_length = 1, 8
        tdict_local = {1: 20, 2: 20, }  # n -> topn number of n-grams to analyze, if not specified, use default (5)
      

        # [params] settings
        params_set = []
        for ctype in ('diagnosis', 'medication', 'mixed',):  # ~ seq_ptype: (diag, med, regular)
            for policy in ('prior', 'posterior', ):  # noop => no cut => computed separately prior to this step
                # for ordering in ('partial', 'total', ): # 'total'
                params_set.append((ctype, policy))

        ordering = 'partial' if partial_order else 'total'
        for params in params_set: 
            # topn = topn_ngrams 
            ctype, cut_policy =  params[0], params[1]  # code type, cut policy

            # adjust seq_ptype to generate document (ctype)
            seq_ptype = seqparams.normalize_ptype(seq_ptype=ctype)
            D0 = {cid:[] for cid in cluster_to_docs.keys()}  # cid -> doc segments
            for cid, docs in cluster_to_docs.items(): 
                for doc in docs: 
                    D0[cid].append(diab.transform(doc, policy=cut_policy, inclusive=True, seq_ptype=seq_ptype)) # before the first diagnosis, inclusive

            # cseqxmaps = D0
            cidx = D0.keys()

            # global motifs wrt the entire docs
            # [note] topn: None => keep all counter elements
            #        merge docs from all clusters but myself => pushed to inner loop
            doc_global = merge_cluster(D0)
            n_doc_global = len(doc_global)
            print('params> ctype: %s, policy: %s, n_clusters: %d, n_doc: %d' % (ctype, cut_policy, len(cidx), n_doc_global))

            # [note] globla motifs does not depend on cluster_method
            global_motifs = eval_motif(doc_global, topn=None, ng_min=min_length, ng_max=max_length, 
                                partial_order=partial_order, 
                                    mtype='global', ctype=ctype, ptype=cut_policy,
                                        seq_ptype=seq_ptype, d2v_method=d2v_method, 
                                        save_=True, load_=True)  # n (as in n-gram) -> topn counts [(n_gram, count), ]

            tfidf_stats = {} # counts of cluster occurrences for each ngram => compute idf score
            fname_tfidf = 'tfidf-%s-%s.csv' % (ctype, identifier) # identifier: (tset_type, cluster_method, seq_ptype, d2v_method)
            # fname_ngram_cidx = 'ngram_cidx-%s-%s.csv' % (ctype, identifier) # label policy as part of the identifier? 

            if topn_clusters_only: 
                for ulabel in ulabels: # foreach (surrogate) label

                    pure_cids, ratios = zip(*lR[ulabel][:topn_clusters]) # zip is its own inverse with *
                    # print('verify> top %d pure clusters (label: %s):\n%s\n  + ratios:\n%s\n' % (len(pure_cids), ulabel, pure_cids, ratios))
                    lRD = dict(lR[ulabel])
        
                    # if topn_clusters_only set to False, then it'll iterate through all clusters
                    for cid, seqx in D0.items(): # seqx is a subset of original D_*
                        # res_motifs[ulabel][cid] = None # [output]

                        # filter 1: only top N
                        if not cid in pure_cids: continue # skip impure clusters (wrt current label)

                        # filter 2: skip if ratio too low  
                        if lRD[cid] < 0.5: 
                            print('info> label: %s is a minority in cluster id: %d => skip ...' % (ulabel, cid))
                            continue

                        clabel = clabels[cid] # cluster label based on majority votes

                        # set context motifs to global_motifs to find out how cluster motifs counts compared to global level 
                        div(message='Result: Label %s + Cluster #%d > cluster-wise statistics ... ' % (clabel, cid))
                        n_doc_cluster = len(seqx)

                        # [note] ctype: diagnosis or medication, whereas seq_ptype has wider categories: diag, med, random, regular
                        #        mtype: use CID as mtype by passing 'cid' explicitly
                        cluster_motifs = eval_cluster_motif(seqx, topn=tdict_local, ng_min=min_length, ng_max=max_length, 
                                            partial_order=partial_order, context_motifs=global_motifs, 
                                                cid=cid, label=clabel, ctype=ctype, ptype=cut_policy, 
                                                cluster_method=cluster_method, seq_ptype=seq_ptype, d2v_method=d2v_method, 
                                                    save_=True, load_=True) # n -> (n-gram -> count)

                        # [test] leave one (cluster) out test
                        # [note] topn: None => keep all counter elements
                        # div(message='Test: Leave-one-out test for cluster #%d' % cid)

                        # set focused_motifs to cluster motifs above to observe their counts in the cluster complement (i.e. all but this cluster)
                        doc_cluster_compl = merge_cluster(D0, cids=set(cidx)-set([cid]))
                        n_doc_cluster_compl = len(doc_cluster_compl)
                        assert n_doc_global == n_doc_cluster + n_doc_cluster_compl, "total: %d != cluster: %d + complement: %d" % \
                            (n_doc_global, n_doc_cluster, n_doc_cluster_compl)
                        loo_motifs = eval_cluster_motif(doc_cluster_compl,  
                                                        topn=None, ng_min=min_length, ng_max=max_length, 
                                                        partial_order=partial_order, context_motifs=global_motifs, focused_motifs=cluster_motifs,
                                                            cid='%s-complement' % cid, label=clabel, ctype=ctype, ptype=cut_policy,
                                                            cluster_method=cluster_method, seq_ptype=seq_ptype, d2v_method=d2v_method,   
                                                                save_=True, load_=True)  # n (as in n-gram) -> topn counts [(n_gram, count), ]

                        # find counts of global motifs/ngrams within this cluster
                        tfidf_stats = df_motif_stats(cluster_motifs, tfidf_stats=tfidf_stats, cid=cid)

                        # global_motifs_in_cluster = eval_given_motif(seqx, motifs=global_motifs)  # n -> (n-gram -> count)
                        # res_cluster_motifs = rel_motif_stats(global_motifs_in_cluster) # input: ngram -> count
                        # res_motifs[ulabel][cid] = res_cluster_motifs

                        # print('result> ratios:\n%s\n' % res_cluster_motifs['ratios'])
                        # print('result> fractions:\n%s\n' % res_cluster_motifs['fractions'])
            else: # iterate all clusters 
                print('status> iterating over all (%s) clusters (N=%d)' % (cluster_method, len(cidx)))
                # if topn_clusters_only set to False, then it'll iterate through all clusters

                # selective tests 
                cidx_loo = random.sample(cidx, 5)

                n_tfidf = 0
                for cid, seqx in D0.items(): # seqx is a subset of original D_*
                    
                    clabel = clabels[cid] # cluster label based on majority votes

                    # set context motifs to global_motifs to find out how cluster motifs counts compared to global level 
                    div(message='Result: Label %s + Cluster #%d > cluster-wise statistics ... ' % (clabel, cid))
                    n_doc_cluster = len(seqx)
                    cluster_motifs = eval_cluster_motif(seqx, topn=tdict_local, ng_min=min_length, ng_max=max_length, 
                                            partial_order=partial_order, context_motifs=global_motifs, 
                                                cid=cid, label=clabel, ctype=ctype, ptype=cut_policy, 
                                                cluster_method=cluster_method, seq_ptype=seq_ptype, d2v_method=d2v_method, 
                                                    save_=True, load_=True) # n -> (n-gram -> count)

                    # [test] leave one (cluster) out test
                    # [note] topn: None => keep all counter elements

                    if cid in cidx_loo: 
                        div(message='Test: Leave-one-out test for cluster #%d' % cid)

                        # set focused_motifs to cluster motifs above to observe their counts in the cluster complement (i.e. all but this cluster)
                        doc_cluster_compl = merge_cluster(D0, cids=set(cidx)-set([cid]))
                        n_doc_cluster_compl = len(doc_cluster_compl)
                        assert n_doc_global == n_doc_cluster + n_doc_cluster_compl, "total: %d != cluster: %d + complement: %d" % \
                                (n_doc_global, n_doc_cluster, n_doc_cluster_compl)
                        loo_motifs = eval_cluster_motif(doc_cluster_compl,  
                                                        topn=None, ng_min=min_length, ng_max=max_length, 
                                                        partial_order=partial_order, context_motifs=global_motifs, focused_motifs=cluster_motifs,
                                                            cid='%s-complement' % cid, label=clabel, ctype=ctype, ptype=cut_policy,
                                                            cluster_method=cluster_method, seq_ptype=seq_ptype, d2v_method=d2v_method,   
                                                                save_=True, load_=True)  # n (as in n-gram) -> topn counts [(n_gram, count), ]

                    # find counts of global motifs/ngrams within this cluster
                    tfidf_stats = df_motif_stats(cluster_motifs, tfidf_stats=tfidf_stats, cid=cid)
                    assert len(tfidf_stats) >= n_tfidf
                    n_tfidf = len(tfidf_stats)

                # keep track of cluster origins of n-grams 

            
            ### end foreach cluster motifs 

            # [I/O] save tfidf_stats 
            if len(tfidf_stats) > 0: 
                assert len(clabels) > 0 and isinstance(clabels, dict)

                fpath = os.path.join(outputdir, fname_tfidf) # fname_tdidf(ctype, )
                header = ['ngram', 'cluster_freq', 'cids', 'labels']
                wsep = ' '
                lsep = ','  # separator for list objects 
                adict = {h: [] for h in header}
                for ngr, entries in tfidf_stats.items(): 
                    ngr_count, ngr_cids = entries['count'], entries['cids']
                    ngstr = wsep.join(str(e) for e in ngr)  # k: tupled n-gram

                    adict['ngram'].append(ngstr)
                    adict['cluster_freq'].append(ngr_count)

                    labels = []
                    for cid in ngr_cids: 
                        labels.append(clabels[cid])
                    cids_str = lsep.join(str(c) for c in ngr_cids)
                    labels_str = lsep.join(str(l) for l in labels)

                    adict['cids'].append(cids_str)
                    adict['labels'].append(labels_str)
                
                df = DataFrame(adict, columns=header)
                print('io> saving tfidf stats (dim: %s) to %s' % (str(df.shape), fpath))
                df.to_csv(fpath, sep='|', index=False, header=True)

        ### end foreach document/cluster type (ctype)

        # [I/O] save cluster summary statistics (e.g. within cluster purity)
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

    else: # no labels (y) given
        raise NotImplementedError

    return

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

def eval_motif(seqx, topn=5, ng_min=1, ng_max=8, partial_order=True, **kargs):
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
    mtype = kargs.get('motif_type', kargs.get('mtype', 'global'))  # motif type: global, cluster-level (use cid) or from arbitrary sequence? 
    ctype = kargs.get('ctype', 'diagnosis') # code type: diagnosis, medication, lab, ... 
    save_motif = kargs.get('save_', True)
    load_motif = kargs.get('load_', True) 

    # [params] file 
    seq_ptype = kargs.get('seq_ptype', 'regular')
    d2v_method = kargs.get('d2v_method', 'tfidfavg')
    cluster_method = kargs.get('cluster_method', 'na')
    otype = kargs.get('otype', kargs.get('order_type', 'part' if partial_order else 'total')) # part: partial
    ptype = kargs.get('ptype', kargs.get('policy_type', 'prior'))
    print('eval_motif> input D, policy=%s | n_doc: %d, n_tokens(seqx[0]): %d' % (ptype, n_docs0, len(seqx[0])))

    # identifier = 'T%s-O%s-C%s-S%s-D2V%s' % (tset_type, order_type, cluster_method, seq_ptype, d2v_method)
    identifier = 'CMOP%s-%s-%s-%s-S%s-D2V%s' % (ctype, mtype, otype, ptype, seq_ptype, d2v_method)
    # identifier = 'C%s-M%s-S%s-D2V%s' % (ctype, mtype, seq_ptype, d2v_method)
    outputdir = os.path.join(os.getcwd(), 'data')
    load_path = os.path.join(outputdir, 'motifs-%s.pkl' % identifier)

    print('params> coordinate (seq:%s, d2v: %s) | order: %s | type (c:%s, m:%s) => input n_docs:%d' % \
        (seq_ptype, d2v_method, 'partial' if partial_order else 'total', ctype, mtype, n_docs0))

    if load_motif and os.path.exists(load_path): 
        res = pickle.load(open(load_path, 'rb'))
        if len(res) > 0:
            print('io> loaded pre-computed %s motifs from %s' % (mtype, load_path)) 
            return res

    if ng_min < 1: ng_min = 1
    if ng_max < ng_min: ng_max = ng_min

    topn_default = 100 # n_doc0  # max 
    if not isinstance(topn, dict):  # topn: n -> number of m most common n-grams
        # topn can be either None or a number
        try: 
            topn_val = int(topn)
        except: 
            topn_val = None # counter.most_common(None) returns all elements in the counter
            # raise ValueError, "invalid topn: %s" % str(topn)

        topn = {}
        for ng in range(1, ng_max+1): 
            topn[ng] = topn_val   # implicit test if topn is an integer 

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
        counter = ngrams[ng]

        n = topn.get(ng, topn_default)  # None or a number
        res[ng] = topn_counts = counter.most_common(n)

        print('verify> top %s %d-gram frequencies(%s):\n%s\n' % (n_docs0 if n is None else n, ng, mtype, topn_counts[:5]))  #  [('c', 2), ('b', 2)]
        common_ngrams = [e[0] for e in topn_counts]
        # print('verify> top %s %d-grams(%s):\n%s\n' % (n_docs0 if n is None else n, ng, mtype, common_ngrams[:5])) # ok. 

    res_df = None
    if save_motif: 
        # save both csv and pkl

        ### 1. csv
        # global path (can also save one file for each n-gram)
        fpath = os.path.join(outputdir, 'motifs-%s.csv' % identifier)
        header = ['length', 'ngrams', 'counts']
        adict = {h: [] for h in header} 
        wsep = ' ' # word separator

        for n, ngr_cnts in res.items(): 
            n_patterns = len(ngr_cnts)
            adict['length'].extend([n] * n_patterns)  # [redundancy]
            
            ngram_reprx, local_counts = [], []
            for i, (ngt, ccnt) in enumerate(ngr_cnts):  # ccnt: cluster-level count 
                # from tuple (ngt) to string (ngstr) 
                ngstr = wsep.join(str(e) for e in ngt)

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
            adict['ngrams'].extend(ngram_reprx)
            adict['counts'].extend(local_counts)
        ### end for

        df = DataFrame(adict, columns=header)
        print('verify> saving motif dataframe of dim: %s to %s' % (str(df.shape), fpath))
        df.to_csv(fpath, sep='|', index=False, header=True)

        ### 2. pickle file 
        fpath = load_path # os.path.join(outputdir, 'motifs-%s.pkl' % identifier)
        pickle.dump(res, open(fpath, "wb" ))

    ### end if save_motif


    return res # n -> counts: [(ngram, count)] // n as in ngram maps to counts which is an ordered list of (ngram, count)

def eval_cluster_motif(seqx, topn=5, ng_min=1, ng_max=8, partial_order=True, **kargs): 
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

    res = {}  # result set

    n_docs0 = len(seqx)
    mtype = kargs.get('cid', kargs.get('motif_type', kargs.get('mtype', 'cluster'))) # motif type (e.g. CID)
    # if mtype is None: mtype = kargs.get('cid_minus', 'unknown')
    ctype = kargs.get('ctype', 'diagnosis')  # code type: diagnosis, medication, lab, ... 
    otype = kargs.get('order_type', kargs.get('otype', 'part' if partial_order else 'total'))
    ptype = kargs.get('policy_type', kargs.get('ptype', 'prior'))

    save_motif = kargs.get('save_', True)
    load_motif = kargs.get('load_', True) 

    # exclusion mode? if so, don't assert ccnt < gcnt (for the counts of ngrams)
    # exclusion_mode = True if mtype.lower().find('loo') >= 0 else False

    # comparison with global motifs if given 
    global_motifs = kargs.get('global_motifs', kargs.get('context_motifs', None))
    n_global_motifs = size(global_motifs)
    if global_motifs is None: 
        return eval_motif(seqx, topn=topn, ng_min=ng_min, ng_max=ng_max, partial_order=partial_order, **kargs)

    focused_motifs = kargs.get('focused_motifs', None) # if set, only keep track of these motifs
    n_focused_motifs = size(focused_motifs)
    is_cluster_complement = kargs.get('is_complement', focused_motifs is not None)

    # [params] file 
    label = kargs.get('label', kargs.get('cluster_label', None))
    seq_ptype = kargs.get('seq_ptype', 'regular')
    d2v_method = kargs.get('d2v_method', 'tfidfavg')
    cluster_method = kargs.get('cluster_method', 'unknown')

    # [params] ID 
    identifier = 'CID%s-COP%s-%s-%s-C%s-S%s-D2V%s' % (mtype, ctype, otype, ptype, cluster_method, seq_ptype, d2v_method)  
    if label is not None: 
        identifier = 'CIDL%s-%s-COP%s-%s-%s-C%s-S%s-D2V%s' % (mtype, label, ctype, otype, ptype, cluster_method, seq_ptype, d2v_method) 
        # identifier = 'C%s-L%s-CID%s-O%s-C%s-S%s-D2V%s' % (ctype, label, mtype, otype, cluster_method, seq_ptype, d2v_method)
        
    outputdir = os.path.join(os.getcwd(), 'data')
    load_path = os.path.join(outputdir, 'motifs-%s.pkl' % identifier)

    print('params> coordinate (l:%s, seq:%s, d2v: %s) | order: %s | method: %s | type (c:%s, m:%s) => input n_docs:%d' % \
        (label, seq_ptype, d2v_method, 'partial' if partial_order else 'total', cluster_method, ctype, mtype, n_docs0))
    print('stats> size of global_motifs: %d, size of focused_motifs: %d (complement? %s)' % \
        (n_global_motifs, n_focused_motifs, n_focused_motifs > 0))

    if load_motif and os.path.exists(load_path): 
        res = pickle.load(open(load_path, 'rb'))
        if len(res) > 0:
            print('io> loaded pre-computed %s motifs from %s' % (mtype, load_path)) 
            # return res
    else: 

        if ng_min < 1: ng_min = 1
        if ng_max < ng_min: ng_max = ng_min

        topn_default = 20
        if not isinstance(topn, dict):  # topn: n -> number of m most common n-grams
            try: 
                topn_val = int(topn)
            except:
                # if None => return all elements in the counter
                # to examine a cluster complement, we do not want to "focus on" only common motifs
                topn_val = None if is_cluster_complement else topn_default
                # topn_val = topn_default 
                # raise ValueError, "invalid topn: %s" % str(topn)

            topn = {}
            for ng in range(1, ng_max+1): 
                topn[ng] = topn_val   # implicit test if topn is an integer 

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
            n = topn.get(ng, topn_default)
            res[ng] = topn_counts = counter.most_common(n)
            
            print('verify> top %s %d-gram frequencies(mtype: %s):\n%s\n' % (n_docs0 if n is None else n, ng, mtype, topn_counts[:5]))  #  [('c', 2), ('b', 2)]
            # common_ngrams = [e[0] for e in topn_counts]
            # print('verify> top %s %d-grams(%s):\n%s\n' % (n_docs0 if n is None else n, ng, mtype, common_ngrams[:5])) # ok 
    ### end if-else 
        
    if save_motif: 
        # save both csv and pkl

        ### 1. csv
        # global path (can also save one file for each n-gram)
        fpath = os.path.join(outputdir, 'motifs-%s.csv' % identifier)
        header = ['length', 'ngram', 'count', 'global_count', 'ratio']

        # count global motifs in the context of the cluster-scoped sequences
        # global_motifs_in_cluster = eval_given_motif(seqx, motifs=global_motifs)  # n -> {(n-gram, count)}, a dict of dictionaries
        adict = compare_motif(lmotifs=res, gmotifs=global_motifs, fmotifs=focused_motifs, header=header, mtype=mtype) # [input] res->lmotifs, global_motifs->gmotifs

        if size(adict) > 0: 
            try: 
                df = DataFrame(adict, columns=header)
            except Exception, e: 
                print('dump> adict:\n%s\n' % adict)
                raise ValueError, e
            print('verify> saving motif dataframe of dim: %s to %s' % (str(df.shape), fpath))
            df.to_csv(fpath, sep='|', index=False, header=True)

            # group by n-gram length and sort 

        else: 
            print('warning> coordinate (l:%s, seq:%s, d2v: %s) > No motifs found.' % (label, seq_ptype, d2v_method))

        ### 2. pickle file 
        # fpath = load_path # os.path.join(outputdir, 'motifs-%s.pkl' % identifier)
        # pickle.dump(res, open(fpath, "wb" ))

    return res 

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
    cohort_name = kargs.get('cohort', kargs.get('cohort_name', 'diabetes'))
    outputdir = os.path.join(os.getcwd(), cohort_name)
    if not os.path.exists(outputdir): os.makedirs(outputdir) # output directory

    # load data 
    # X, y, D = build_data_matrix2(**kargs) # seq_ptype, n_features, etc. 
    # 1. read_doc, seq_ptype => document type (random, regular, etc.)
    # 2. doc2vec_method, n_doc, seq_ptype => training data set
    
    read_mode = kargs.get('read_mode', 'doc') # documents/doc (one patient one sequence) or sequences/seq
    tset_type = kargs.get('tset_type', 'binary')
    standardize_ = kargs.get('std_method', 'minmax') # feature standardizing/scaling/preprocessing method
    n_sample, n_sample_small = 1000, 100
    n_clusters = kargs.get('n_clusters', 50)

    # [params] experimental settings
    seq_ptype = 'regular' # [default]
    seq_ptypes = ['regular', ] # ['random', 'regular', ]
    d2v_methods = ['tfidfavg', ]  # 'average',  'PVDM'

    # 'affinity_propogation': expensive (need subsetting)
    cluster_methods = ['kmeans', 'minibatch' ] # 'agglomerative', 'spectral', 'dbscan',

    cohort_name = kargs.get('cohort', None)
    doc_basename = 'condition_drug' if cohort_name is None else 'condition_drug-%s' % cohort_name

    # original documents 
    # [params][input] temporal docs
    ifile = kargs.get('ifile', 'condition_drug_seq-%s.dat' % cohort_name if cohort_name is not None else 'condition_drug_seq.dat')

    D0 = sa.read(load_=False, simplify_code=False, mode=read_mode, verify_=False, seq_ptype=seq_ptype, ifile=ifile)
    n_doc0 = len(D0)
    print('verify> size of original docs: %d' % n_doc0)

    # for setting in itertools.product(seq_ptypes, d2v_methods)

    ### 1. binary class dataset (tset_type <- binary)
    for seq_ptype in seq_ptypes: 
        for d2v_method in d2v_methods: 
            kargs['d2v_method'] = d2v_method

            D, ts = build_data_matrix_ts(read_mode=read_mode, seq_ptype=seq_ptype, d2v_method=d2v_method, tset_type=tset_type)
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
                                                        seq_ptype=seq_ptype, d2v_method=d2v_method, save_=True, load_=True, cohort=cohort_name)
                print('cluster> method: %s, metrics:\n%s\n' % (cluster_method, cmetrics))

                # analyze coding sequence
                # set topn_clusters_only to False in order to compute (inverse) document/cluster freq. 
                t_pathway(X=X, y=y, D=D, clusters=clabels, topn_clusters_only=False, 
                    seq_ptype=seq_ptype, d2v_method=d2v_method, cluster_method=cluster_method, cohort=cohort_name) 
            
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

def cluster_analysis(X=None, y=None, **kargs): 
    """

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
            print('evaluate> Could not sample X without replacement wrt cluster labels (dim X: %s while n_cluster: %d)' % \
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
    import scipy.spatial.distance as ssd
    # import cluster.analyzer   # [use] analyzer.summarize_cluster()
    # ssd.cosine(arr[0], [1., 2., 3.])

    # [params]
    n_clusters = kargs.get('n_clusters', 100)
    n_clusters_est = -1

    cluster_method = kargs.get('cluster_method', kargs.get('c_method', 'kmeans'))

    # [params] 
    cohort_name = kargs.get('cohort', 'diabetes')
    inputdir = datadir = os.path.join(os.getcwd(), 'data') # sys_config.read('DataExpRoot')
    outputdir = datadir = os.path.join(os.getcwd(), 'data')

    save_cluster = kargs.get('save_', kargs.get('save_cluster', True))
    plotdir = os.path.join(os.getcwd(), 'plot')

    lmap = {0: 'TD1', 1: 'TD2', 2: 'Gest', }  # [hardcode]

    # [params] identifier
    seq_ptype = kargs.get('seq_ptype', 'regular') # sequence pattern type: regular, random, diag, med, lab
    d2v_method = kargs.get('d2v_method', kargs.get('doc2vec_method', 'PVDM'))
    identifier = kargs['identifier'] if kargs.has_key('identifier') else '%s-%s' % (seq_ptype, d2v_method)
    if not identifier: identifier = '%s-%s-%s' % (cluster_method, seq_ptype, d2v_method)

    map_label_only = kargs.get('map_label_only', True) # if False, then cluster map file will keep original docs in its data field
    doc_label = False

    # [params] input data (cf: cluster.analyzer)
    # [note] training set files e.g. 'tset_%s_%s_t1t2-P%s.csv' % (doc2vec_method, n_doc, seq_ptype)
    #        see build_data_matrix_ts()
    seqx = D = kargs.get('D', None)
    if X is None or y is None: 
        if map_label_only: 
            X, y  = build_data_matrix(**kargs)
        else: 
            X, y, D = build_data_matrix2(**kargs)
            assert len(y) == len(D)

        # X, y, seqx = build_data_matrix2(**kargs)
    # labels_true = y
    print('verify> algorithm: %s > doc2vec dim: %s' % (cluster_method, str(X.shape)))

    # [params] output 
    # cluster labels 
    load_path = os.path.join(outputdir, 'cluster_ids-%s.csv' % identifier) # id: seq_ptype, d2v_method
    load_cluster = kargs.get('load_', True) and (os.path.exists(load_path) and os.path.getsize(load_path) > 0)

    ### Run Cluster Analysis ### 
    model = n_cluster_est = None   # [todo] model persistance
    if not load_cluster: 
        model, n_clusters_est = run_cluster_analysis(X, y, **kargs)
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
            model, n_clusters_est = run_cluster_analysis(X, y, **kargs)
            cluster_labels = model.labels_

        assert X.shape[0] == len(cluster_labels), "nrow of X: %d while n_cluster_ids: %d" % (X.shape[0], len(cluster_labels))

    # [note] alternatively use map_clusters2() to map cluster IDs to appropriate data point repr (including labels)
    # cluster_to_docs  = map_clusters(cluster_labels, seqx)
    print('status> completed clustering with %s' % cluster_method)

    if D is None: # label only for 'data'
        cluster_to_docs  = map_clusters(cluster_labels, y)  # y: pre-computed labels (e.g. heuristic labels)
        doc_label = True # each document is represented by its (surrogate) labels
    else: 
        cluster_to_docs  = map_clusters(cluster_labels, D)
    print('info> resulted n_cluster: %d =?= %d' % (len(cluster_to_docs), n_clusters))  # n_cluster == 2

    header = ['cluster_id', 'data', ]
    adict = {h:[] for h in header}

    # [verify]
    size_cluster = 0
    rid = random.sample(cluster_to_docs.keys(), 1)[0]  # [test]
    for cid, content in cluster_to_docs.items():
        if cid == rid: 
            print "log> cid: %s => %s" % (cid, str(cluster_to_docs[cid]))
            print "\n"

        # store 
        adict['cluster_id'].append(cid)
        adict['data'].append(content) # label, sequence, etc. 
        size_cluster += len(content)
    size_avg = size_cluster/(len(cluster_to_docs)+0.0)
    print('verify> averaged %s-cluster size: %f' % (cluster_method, size_avg))

    # [summary]
    print('status> evaluate clusters (e.g. silhouette scores) ...')
    res_metrics = evaluate_cluster(cluster_labels)
    print('status> summarize clusters (e.g. knn wrt centroids)')
    res_summary = summarize_cluster(cluster_labels)

    # save clustering result 
    if save_cluster: 
        if not doc_label:  # if only y is passed, don't save the map # '%s-%s' % (seq_ptype, d2v_method)
            fpath = os.path.join(outputdir, 'cluster_map-%s.csv' % identifier)
            df = DataFrame(adict, columns=header)
            print('output> saving cluster map (cluster labels -> cids) to %s' % fpath)
            df.to_csv(fpath, sep=',', index=False, header=True)   
        
        # save cluster IDs
        fpath = load_path  # os.path.join(outputdir, 'cluster_ids-%s.csv' % identifier) # id: seq_ptype, d2v_method    
        # sf = Series()
        header = ['cluster_id', ]
        df = DataFrame(cluster_labels, columns=header)
        print('output> saving cluster map (cluster labels -> cids) to %s' % fpath)
        df.to_csv(fpath, sep=',', index=False, header=True)      

        # save knn 
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
    if doc_label: 
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

def run_cluster_analysis(X, y=None, **kargs):  
    """
    Main routine for running cluster analysis. 


    Related
    -------
    cluster_analysis()

    """
    # from sklearn.cluster import KMeans, AgglomerativeClustering, AffinityPropagation, SpectralClustering, DBSCAN

    cluster_method = kargs.get('cluster_method', 'kmeans')
    n_clusters = kargs.get('n_clusters', 50)
    n_clusters_est = None
    print('run_cluster_analysis> requested %d clusters' % n_clusters)
    if cluster_method in ('kmeans', 'k-means', ): 
        model = KMeans(init='k-means++', n_clusters=n_clusters, n_init=10)
        model.fit(X)
    elif cluster_method in ('minibatch', 'minibatchkmeans'):
        model = MiniBatchKMeans(n_clusters=n_clusters)  # init='k-means++', n_init=3 * batch_size, batch_size=100
        model.fit(X)
    elif cluster_method.startswith('spec'):
        model = SpectralClustering(n_clusters=n_clusters, eigen_solver='arpack', affinity="nearest_neighbors") 
        model.fit(X)
    elif cluster_method.startswith('agg'):
        knn_graph = kneighbors_graph(X, 30, include_self=False)  # n_neighbors: Number of neighbors for each sample.

        # [params] AgglomerativeClustering
        # connectivity: Connectivity matrix. Defines for each sample the neighboring samples following a given structure of the data. 
        #               This can be a connectivity matrix itself or a callable that transforms the data into a connectivity matrix
        # linkage:  The linkage criterion determines which distance to use between sets of observation. 
        #           The algorithm will merge the pairs of cluster that minimize this criterion.
        #           average, complete, ward (which implies the affinity is euclidean)
        # affinity: Metric used to compute the linkage
        #           euclidean, l1, l2, manhattan, cosine, or precomputed
        #           If linkage is ward, only euclidean is accepted.
        connectivity = knn_graph # or None
        linkage = kargs.get('linkage', 'average')
        model = AgglomerativeClustering(linkage=linkage,  
                                            connectivity=connectivity,
                                            n_clusters=n_clusters)
        model.fit(X)
    elif cluster_method.startswith('aff'): # affinity propogation
        damping = kargs.get('damping', 0.9)
        preference = kargs.get('preference', -50)

        # expensive, need subsampling
        model = AffinityPropagation(damping=damping, preference=preference) 
        model.fit(X)
        
        cluster_centers_indices = model.cluster_centers_indices_
        n_clusters_est = len(cluster_centers_indices)
        print('info> method: %s (damping: %f, preference: %f) > est. n_clusters: %d' % (cluster_method, damping, preference, n_clusters)) 

    elif cluster_method.startswith('db'): # DBSCAN: density-based 
        # first estimate eps 
        n_sample_max = 500
        metric = 'euclidean'

        # [note] 
        # eps : float, optional
        #       The maximum distance between two samples for them to be considered as in the same neighborhood.
        # min_samples : int, optional
        #       The number of samples (or total weight) in a neighborhood for a point to be considered as a core point. 
        #       This includes the point itself.
        
        eps = kargs.get('eps', None)
        if eps is None:      
            X_subset = X[np.random.choice(X.shape[0], n_sample_max, replace=False)] if X.shape[0] > n_sample_max else X

            # pair-wise distances
            dmat = distance.cdist(X_subset, X_subset, metric)
            off_diag = ~np.eye(dmat.shape[0],dtype=bool)  # don't include distances to selves
            dx = dmat[off_diag]
            sim_median = np.median(dx)
            sim_min = np.min(dx)
            eps = (sim_median+sim_min)/2.

        print('verify> method: %s > eps: %f' % (cluster_method, eps))
        model = DBSCAN(eps=eps, min_samples=10, metric=metric) 
        model.fit(X)

        core_samples_mask = np.zeros_like(model.labels_, dtype=bool)
        core_samples_mask[model.core_sample_indices_] = True

        cluster_labels = model.labels_
        n_clusters_est = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    else: 
        raise NotImplementedError, "Cluster method %s is not yet supported" % cluster_method 

    return (model, n_clusters_est)
    
def config_doc2vec_model(**kargs): 
    """


    Note
    ----
    
    Parameter choices: 

    1. 100-dimensional vectors, as the 400d vectors of the paper don't seem to offer much benefit on this task; 
       similarly, frequent word subsampling seems to decrease sentiment-prediction accuracy, so it's left out

    2. cbow=0 means skip-gram which is equivalent to the paper's 'PV-DBOW' mode, 
       matched in gensim with dm=0; 
       added to that DBOW model are two DM models, one which averages context vectors (dm_mean) and 
       one which concatenates them (dm_concat, resulting in a much larger, slower, more data-hungry model)

    3. a min_count=2 saves quite a bit of model memory, discarding only words that appear in a single doc 
      (and are thus no more expressive than the unique-to-each doc vectors themselves)

    Options 
    -------
     hs: if 1 (default), hierarchical sampling will be used for model training (else set to 0).

    """
    # from gensim.models import Doc2Vec
    # import gensim.models.doc2vec
    # from collections import OrderedDict
    # import multiprocessing

    alldocs = kargs.get('sequences', kargs.get('alldocs', None))
    assert alldocs is not None

    cores = multiprocessing.cpu_count()
    assert gensim.models.doc2vec.FAST_VERSION > -1, "this will be painfully slow otherwise"

    simple_models = [
        # PV-DM w/concatenation - window=5 (both sides) approximates paper's 10-word total window size
        Doc2Vec(dm=1, dm_concat=1, size=100, window=5, negative=5, hs=0, min_count=2, workers=cores),
        # PV-DBOW 
        Doc2Vec(dm=0, size=100, negative=5, hs=0, min_count=2, workers=cores),
        # PV-DM w/average
        Doc2Vec(dm=1, dm_mean=1, size=100, window=10, negative=5, hs=0, min_count=2, workers=cores),
    ]

    # speed setup by sharing results of 1st model's vocabulary scan
    simple_models[0].build_vocab(alldocs)  # PV-DM/concat requires one special NULL word so it serves as template
    print(simple_models[0])
    for model in simple_models[1:]:
        model.reset_from(simple_models[0]) # no need to retrain
        print(model)

    models_by_name = OrderedDict((str(model), model) for model in simple_models)

    return models_by_name 

def v_mini_doc2vec():
    read_mode = 'doc'
    n_subset = 100
    doctype = 'd2v'
    descriptor = 'test'
    doc_basename = 'mini'
    basedir = sys_config.read('DataExpRoot')

    seq_ptype = kargs.get('seq_ptype', 'regular') # sequence pattern type: regular, random, diag, med, lab

    documents = seqx = sa.read(load_=True, simplify_code=False, mode=read_mode, verify_=False, seq_ptype=seq_ptype)
    assert seqx is not None and len(seqx) > 0
    n_doc = len(seqx)
    print('input> got %d documents.' % n_doc)

    # labeling 
    df_ldoc = labelDoc(documents, load_=True, seqr='full', sortby='freq', seq_ptype=seq_ptype) # sortby: order labels by frequencies
    labels = list(df_ldoc['label'].values)
    # labels = [mlabel.split(lsep)[0] for mlabel in mlabels] # take most frequent (sub-)label as the label
    print('t_doc2vec1> labels (dtype=%s): %s' % (type(labels), labels[:10]))

    lseqx = labelDoc2(sequences=seqx, labels=labels)
    
    lseqx_subset = random.sample(lseqx, n_subset)
    model = Doc2Vec(dm=1, size=20, window=8, min_count=5, workers=8)

    model.build_vocab(lseqx_subset)
    model.train(lseqx_subset)

    # for epoch in range(10):
    #     model.train(lseqx_subset)
    #     model.alpha -= 0.002  # decrease the learning rate
    #     model.min_alpha = model.alpha  # fix the learning rate, no decay    

    # Finally, we save the model
    
    ofile = '%s_%s.%s' % (doc_basename, descriptor, doctype)
    fpath = os.path.join(basedir, ofile)

    print('output> saving (test) doc2vec models (on %d docs) to %s' % (len(lseqx_subset), fpath))
    model.save(fpath)

    n_doc2 = model.docvecs.count
    print('verify> number of docs (from model): %d' % n_doc2)

    lseqx = random.sample(lseqx_subset, 10)
    for i, lseq in enumerate(lseqx): 
        tag = lseq.tags[0]   # [log] [0] label: V22.1_V74.5_V65.44 => vec (dim: (20,)):
        vec = model.docvecs[tag]  # model.docvecs[doc.tags[0]]
        print('[%d] label: %s => vec (dim: %s):\n%s\n' % (i, tag, str(vec.shape), vec)) 
        sim_docs = model.docvecs.most_similar(tag, topn=5)
        print('doc: %s ~ :\n%s\n' % (tag, sim_docs))

    return


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
    t_preclassify_weighted(load_word2vec_model=True, seq_ptype='random', n_features=100, test_model=True) 
    t_preclassify_weighted(load_word2vec_model=True, seq_ptype='regular', n_features=100, test_model=True)

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

def test2(**kargs): # PTSD 
    """
    Analyze n-grams without going through clustering. 

    Chort study: PTSD

    """
    import seqTransform as st
    import seqAnalyzer as sa

    cohort_name = 'PTSD' # diabetes
    prefix = kargs.get('prefix', kargs.get('input_dir', sys_config.read('DataExpRoot')))
    ifile = kargs.get('ifile', 'condition_drug_seq-%s.dat' % cohort_name if cohort_name is not None else 'condition_drug_seq.dat') 
    ifiles = kargs.get('ifiles', [ifile, ])

    d2v_method = kargs.get('d2v_method', 'tfidfavg')  # tfidfavg
    std_method = kargs.get('std_method', 'minmax')  # standardizing feature values
    
    # [params] auxiliary 
    cluster_method = kargs.get('cluster_method', 'unknown') # debug only

    # [params] training 
    w2v_method = kargs.get('w2v_method', 'sg')  # sg: skip-gram
    n_features = kargs.get('n_features', 100)
    window = kargs.get('window', 7)
    min_count = kargs.get('min_count', 2)

    n_cores = multiprocessing.cpu_count()
    print('info> number of cores: %d' % n_cores)
    n_workers = kargs.get('n_workers', max(n_cores-10, 15))

    # t_doc2vec1(load_doc2vec_model=True, seq_ptype='regular', n_features=100, test_model=True, cohort=cohort_name)
    # t_preclassify_weighted0(**kargs) 

    # [note] default source document files: ['condition_drug_seq.dat', ] from DataExpRoot 
    #        use 'prefix' to change source (base) directory
    #        use 'ifiles' to change the document source file set
    # print('source> reading from %s' % os.path.join(prefix, ifiles[0]))
    # seqx = sa.read(load_=True, simplify_code=False, mode='doc', verify_=False, seq_ptype=seq_ptype, ifile=ifile)

    ifile = kargs.get('ifile', 'condition_drug_seq-%s.dat' % cohort_name if cohort_name is not None else 'condition_drug_seq.dat')   
    print('input> temporal doc: %s' % ifile)
    result = sa.loadModel(n_features=n_features, window=window, min_count=min_count, n_workers=n_workers, 
                ifile=ifile, cohort=cohort_name, bypass_lookup=True, 
                read_mode='doc', load_seq=False, load_model=True, load_lookuptb=True) # attributes: sequences, lookup, model
    seqx = result['sequences']
    print('verify> loaded %d docs' % len(seqx))

    # partial_order use global 
    partial_order = True 
    ng_min, ng_max = 1, 8
    topn = 10
    topn_dict = {n:topn for n in range(ng_min, ng_max+1)}

    for partial_order in (True, False, ): 
        otype = 'partial' if partial_order else 'total'

        for ctype in ('diagnosis', 'medication', 'mixed',):  # ~ seq_ptype: (diag, med, regular)
            for ptype in ('prior', 'noop', 'posterior', ): # policy type
                # D0 = {cid:[] for cid in cluster_to_docs.keys()}
                seq_ptype = seqparams.normalize_ptype(seq_ptype=ctype)  # overwrites global value
        
                D0 = []
                for seq in seqx: 
                    D0.append(st.transform(seq, policy=ptype, inclusive=True, seq_ptype=seq_ptype, predicate=st.is_PTSD))
            
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

if __name__ == "__main__": 
    test2()
    # test_similarity()
    # t_analysis(seq_ptype='regular', n_features=100)
    # t_tdidf(seq_ptype='regular')
    # t_preclassify_weighted(load_word2vec_model=True, seq_ptype='random', n_features=100, test_model=True) 
    # t_preclassify_weighted(load_word2vec_model=True, seq_ptype='regular', n_features=100, test_model=True)

