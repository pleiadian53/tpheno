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
import string, re
import sys, os, gc 
import random, time # as a seeding mechanism for randomization

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

# local modules (assuming that PYTHONPATH has been set)
from batchpheno import icd9utils, qrymed2, utils, dfUtils  # sampling is obsolete
from config import seq_maker_config, sys_config
# from batchpheno.utils import div
from system.utils import div
from pattern import medcode as pmed

import sampler  # sampling utilities
import seqparams, vector, evaluate  # evaluate for evaluating classifier performance
# import analyzer
import vector
import seqAnalyzer as sa 
import seqUtils, plotUtils
import seqConfig

from tset import TSet  # base class is defined in seqparams

import algorithms, seqAlgo  # count n-grams, sequence-specific algorithms
import labeling

# multicore processing 
import multiprocessing

# clustering algorithms 
# from sklearn.cluster import KMeans, MiniBatchKMeans, AgglomerativeClustering, AffinityPropagation, SpectralClustering, DBSCAN
# from sklearn.cluster import KMeans

# from sklearn.cluster import AgglomerativeClustering
# from sklearn.neighbors import kneighbors_graph

# from sklearn.cluster import AffinityPropagation

# evaluation 
from sklearn import metrics
from scipy.spatial import distance
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.neighbors import NearestNeighbors  # kNN
from scipy import interp
import scipy

# training and validation 
# from sklearn.cross_validation import train_test_split

# classification 
from sklearn.multiclass import OneVsRestClassifier
from sklearn import svm

##### Set module variables ##### 
GNFeatures = vector.W2V.n_features # [note] descendent: seqparams.W2V -> vector.W2V
GWindow = vector.W2V.window
GNWorkers = vector.W2V.n_workers
GMinCount = vector.W2V.min_count


# wrapper class: training set handler 
# class tsHandler(tsHandler):  # define my own version of tsHandler
#     pass
from tsHandler import tsHandler

### end class tsHandler

def getSurrogateLabels(docs, **kargs): 
    """
    Determine class labels (cohort-specific) for the input documents (docs). 
    Note that the class label are NOT the same as the document label used for Doc2Vec. 
    If the input cohort is not recognized, this returns a uniform positive labels by default 
    i.e. each document is 'positive (1)'

    Params
    ------
    cohort
    """
    import labeling 
    return labeling.getSurrogateLabels(docs, **kargs)

def loadAugmentedDocuments(cohort, inputdir=None, label_default=None): 
    def load_augmented_docs(): 
        ret = sr.readAugmentedDocFromCSV(cohort=cohort, inputdir=inputdir, ifiles=[]) # [params] doctype (timed) 
        D, T, L = ret['sequence'], ret.get('timestamp', []), ret.get('label', [])
        
        if len(L) == 0: 
             L = [label_default] * len(D)

        return (D, L, T)
    def get_global_cohort_dir(): 
        return seqparams.getCohortGlobalDir(cohort) # sys_config.read('DataExpRoot')/<cohort>
    def is_labeled_data(lx): 
        nL = len(set(lx))
        if nL <= 1: 
            return False 
        # print('  + Detected labels (n=%d) in augmented documents.' % nL)
        return True
    def test_doc():  # <- D, L, T
        assert len(D) > 0, "No primary corpus found (nD=0)"
        assert isinstance(D, list)
        if len(T) > 0: 
            assert len(T) == len(D)
        print('  + found %d (augmented) documents' % len(D))
        print('loadAugmentedDocuments> nD=%d, labeled? %s' % (len(D), is_labeled_data(L)))
        return
    def corpus_stats(): 
        pass
    # import seqparams
    import seqReader as sr
    if inputdir is None: inputdir = get_global_cohort_dir()

    ### 2. load augmented documents
    D, L, T = load_augmented_docs()
    test_doc()

    return (D, L, T)
def loadDocuments2(cohort, inputdir=None, single_label_format=True):
    """
    Similar to loadDocuments, this routine loads documents (coding sequences), but also handles augmented data
    which typically do not have labels. 

    Output
    ------
    a dictionary with keys ['sequence', 'timestamp', 'label', ]

    Related
    -------
    loadDocuments

    Memo
    ----
    a. also see cohort.t_augment_cohort()

    """
    def load_docs(): 
        # inputdir was previously set to TDoc.prefix by default  # basedir ~ inputdir in seqReader
        ret = sr.readDocFromCSV(cohort=cohort, inputdir=inputdir, ifiles=[], complete=True) # [params] doctype (timed) 
        return ret
    def load_augmented_docs(label_default=None): 
        ret = sr.readAugmentedDocFromCSV(cohort=cohort, inputdir=inputdir, ifiles=[]) # [params] doctype (timed) 
        D2, T2, L2 = ret['sequence'], ret.get('timestamp', []), ret.get('label', [])
        
        if len(L2) == 0: 
             L2 = [label_default] * len(D2)

        return (D2, L2, T2)
    def get_global_cohort_dir(): 
        return seqparams.getCohortGlobalDir(cohort) # sys_config.read('DataExpRoot')/<cohort> 
    def test_doc(): 
        assert len(D) > 0, "No primary corpus found (nD=0)"
        assert isinstance(D, list)
        assert isinstance(D2, list)
        if len(T) > 0: 
            assert len(T) == len(D)
        if len(T2) > 0: 
            assert len(T2) == len(D2)
        print('  + Obtained %d primary (possibly labeled) documents and %d augmented documents (nTotal=%d)' % (len(nD), len(nD2), len(Dt)))
        assert len(Dt) == len(Tt)
        return
    # import seqparams
    if inputdir is None: inputdir = get_global_cohort_dir()
    
    ### 1. load regular documents
    D, L, T = loadDocuments(cohort=cohort, inputdir=inputdir, single_label_format=single_label_format)

    ### 2. load augmented documents
    D2, L2, T2 = load_augmented_docs()

    nD, nD2 = len(D), len(D2)
    Dt = Tt = Lt = None  # t: total
    if nD2 > 0: 
        print('loadDocuments2> Found augmented data (n=%d)' % nD2)
        Dt = D + D2
        Tt = T + T2 
        Lt = L + L2
    else: 
        Dt, Tt, Lt = D, T, L
    test_doc()

    return (Dt, Lt, Tt)  

def loadDocs(**kargs):
    return loadDocuments(**kargs) 
def loadDocuments(**kargs):
    """
    Load documents (and additionally, timestamps and labels if available). 

    Params
    ------
    cohort
    inputdir
    
    raw: Read processed (.csv) or raw (.dat) documents? Set to False by default
         When to set to True? e.g. reading augmented unlabeled data


    result_set: a dictionary with keys ['sequence', 'timestamp', 'label', ]
    ifiles: paths to document sources

    <defunct> 
       use_surrogate_label: applies only when no labels found in the dataframe

    Output: a 2-tuple: (D, l) where 
            D: a 2-D np.array (in which each document is a list of strings/tokens)
            l: labels in 1-D array

    Memo
    ----
    1. document source 
       used to be under tpheno/data-exp (sys_config.read('DataExpRoot'))
       but now as of 01.20.18, a cohort subdir is used to separate document sources associated with different cohort studies 

       e.g. coding sequences cohort=CKD with timestamps 

            tpheno/data-exp/CKD/condition_drug_timed_seq-CKD.csv

    """
    import docProc as dp  # document processor module (wrapper app of seqReader, seqTransformer)
    return dp.loadDocuments(**kargs) # cohort, inputdir/None, source_type/'source'

def processDocuments(**kargs):
    """
    Load and transform documents (and ensure that labeled source file exist (i.e. doctype='labeled')). 

    Params
    ------
    1. for reading source documents
        cohort
        ifiles: paths to document sources (if given, the cohort is ignored)

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
    processDocuments(cohort)

    Note
    ----
    1. No need to save a copy of the transformed documents because the derived labeled source (doctype='labeled')
       subsumes it.  

    """
    import docProc as dp  # document processor module (wrapper app of seqReader, seqTransformer)
    return dp.processDocuments(**kargs)
   
def loadDocumentsByLabel(**kargs):
    return stratifyDocuments(**kargs)
def stratifyDocuments(**kargs):
    """
    Similar to processDocument() but also allows for stratification by labels. 

    Memo
    ----
    1. CKD Data 
       {'CKD Stage 3a': 263, 'Unknown': 576, 'CKD Stage 3b': 159, 'ESRD on dialysis': 43, 'CKD G1-control': 136, 
        'CKD G1A1-control': 118, 'CKD Stage 5': 44, 'CKD Stage 4': 84, 'ESRD after transplant': 691, 
        'CKD Stage 2': 630, 'CKD Stage 1': 89}
    """
    import docProc as dp  # document processor module (wrapper app of seqReader, seqTransformer) 
    return dp.stratifyDocuments(**kargs)

def modify(docs, **kargs):  # [params] labels
    """
    Transform documents: 

    1. simply coding 
    2. preserve only needed contents (e.g. diagnostic codes only)

    Params
    ------
    simplify_code 
    filter_code

    """ 
    # import seqAlgo
    import seqTransform as st 
    return st.modify(docs, **kargs)  # output made consistent with loadDocuments? 
def parallellModify(docs, items, **kargs): 
    """
    Similar to the content filtering functionality in transform() but filter 
    both coding sequences and their corresponding timestamps in parallel. 

    Input
    -----
    docs: 2D array of documents (where each doc is a list of tokens/strings)
    items: any objects (e.g. timestamps) in parallel (and having a 1-to-1 relationship) with the input documents
           e.g. suppose items <- timestamps
                and only diagnostic codes are preserved, 
                then will fitler all positions (within each document) that point to medicinal codes (e.g. MED:12345)
                => also remove their timestamps 

    Params
    ------
    seq_ptype
    predicate

    """
    import seqTransform as st
    return st.parallellModify(docs, items, **kargs)  # output: (new_docs, new_items)
def filterDocuments(docs, **kargs): # [params] L, T, policy
    import seqTransform as st
    # return (docs2, labels2) 
    labels = kargs.get('L', [])
    timestamps = kargs.get('T', [])
    doc_filter_policy = kargs.get('policy', 'empty') # remove empty documents
    return st.filterDocuments(docs, L=labels, T=timestamps, policy=doc_filter_policy)

def transformDocuments(D, L=[], T=[], policy='empty_doc', seq_ptype='regular', predicate=None, simplify_code=False): 
    """
    Modify input documents including simplifying and filtering document content (codes). 
    Input items (e.g. timestamps) will be modified accordingly. For instance, if a subset of 
    codes are removed (say only diagnostic codes are preserved), then their timestamps are 
    removed in parallel. 

    Params
    ------
    1) modifying sequence contents: 
       seq_ptype
       predicate

    2) filtering out unwanted documents: 
       policy: policy for filtering documents 
       labels: labels will also be removed from the set if their documents are removed 

    3) simply coding (e.g. 250.01 => 250)

    """
    import seqTransform as st 
    return st.transformDocuments(docs, labels=L, items=T, policy=policy, 
        seq_ptype=seq_ptype, predicate=predicate, simplify_code=simplify_code)

def transformDocuments2(D, L=[], T=[], **kargs):  # this is not the same as seqTransform.transform()
    """
    Transform the document (as does transformDocuments() but also save a copy of the transformed document)
    """
    import seqTransform as st 
    return st.transformDocuments2(D, L, T, **kargs)

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
    Input: labeled documents (where each document, assumed to have been labeled, is a list of tokens)
    Output: d2v model (NOT document vectors, for which use getDocVec())
    """
    return vectorize(labeled_docs, **kargs) # [output] model (d2v)

def vectorize(labeled_docs, **kargs): # wrapper on top of vector.vectorize2() 
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
    # import vector

    # [params]
    cohort_name = kargs.get('cohort', 'diabetes') # 'diabetes', 'PTSD'
    # seq_compo = composition = kargs.get('composition', 'condition_drug')
    
    # [params] d2v model: use seqparams and vector
    d2v_method = vector.D2V.d2v_method
    vector.D2V.show_params()

    # [params] IO
    outputdir = basedir = kargs.get('outputdir', seqparams.getCohortDir(cohort=cohort_name))
    
    # negative: if > 0 negative sampling will be used, the int for negative specifies how many "noise words"
    #           should be drawn (usually between 5-20).
    model = vector.getDocVectorModel(labeled_docs, outputdir=outputdir) # [output] model (d2v)

    # [test]
    tag_sample = [ld.tags[0] for ld in labeled_docs][:10]
    print('verify> example tags (note: each tag can be a string or a list):\n%s\n' % tag_sample)
    # [log] ['V70.0_401.9_199.1', '746.86_426.0_V72.19', '251.2_365.44_369.00', '362.01_599.0_250.51' ... ] 

    return model

def labelize(docs, class_labels=[], label_type='doc', offset=0): # essentially a wrapper of labeling.labelize 
    """
    Label docuemnts. Note that document labels are not the same as class labels (class_labels)

    Params
    ------
    labels
    offset: minium ID (to avoid ID conflict in successive calls)

    """
    import vector
    return vector.labelDocuments(docs, class_labels=class_labels, label_type=label_type, offset=offset) # overwrite_=False

def documentToVector(D, **kargs):
    """
    Maps D to X: compute document vectors X for the input corpus D
    Derived from makeTSetCombined() by assuming that documents/corpus are available as an input. 

    Params
    ------
    a. 
    b. model ID 
       meta
       seq_ptype
       cohort

    Note
    ----
    1. For the conveninece of distinguishing models of the same d2v parameters: 
       use get_model_id() to create a specific model ID

    """
    def get_model_id(y=[]): # {'U', 'L', 'A'} + ctype + cohort
        meta = 'D'
        if tsHandler.is_configured: 
            meta = tsHandler.get_model_id(y=y)
        else: 
            # raise ValueError, "Configure tsHandler first." 
            pass
        return meta
    def get_model_dir(): 
        dir_type = 'model'
        modelPath = TSet.getPath(cohort=tsHandler.cohort, dir_type=dir_type, create_dir=True)  # ./data/<cohort>/model
        print('documentToVector> model dir: %s' % modelPath)
        return modelPath

    # import matplotlib.pyplot as plt
    from tset import TSet
    from labeling import TDocTag
    from pattern import medcode as pmed 
    from sklearn.preprocessing import LabelEncoder
    # import vector

    cohort_name = kargs.get('cohort', '?')
    # assert not cohort_name == '?' or kargs.has_key('ifiles'), "If cohort is not given, you must provide (doc) source paths."
    
    # output directory for the d2v model 
    # outputdir = TSet.getPath(cohort=cohort_name, dir_type=None)  # ./data/<cohort>/

    # [input] cohort, labels, (result_set), d2v_method, (w2v_method), (test_model)
    #         simplify_code, filter_code
    #   
    # load + transfomr + (ensure that labeled_seq exists)
    # D, L, T = processDocuments(cohort=cohort_name, seq_ptype=kargs.get('seq_ptype', 'regular'),
                # predicate=kargs.get('predicate', None), simplify_code=kargs.get('simplify_code', False))  # [params] composition
    # labels = L # documents = D; timestamps = T
    # need to save a copy if seq_ptype is specialized type (e.g. diag, med, lab)

    nDoc = len(D)
    div(message='2. Compute document embedding (params: ) ...') # [note] use W2V and D2V classes in seqparams and vector to adjust d2v parameters
    d2v_method = kargs.get('d2v_method', vector.D2V.d2v_method)
    # ratio_test = 0.2
    # print('    + train test split (test: %f%%) | cohort=%s, n_classes=%d' % (0.2*100, cohort_name, n_classes))  
    
    # cv = StratifiedKFold(n_splits=6) 
    # ts = DataFrame()
    # le = LabelEncoder()
    # nTrials = kargs.get('n_trials', 1)
    # D_augmented = kargs.get('augmented_docs', [])  # loaded from another call
    # nDocAug = len(D_augmented)

    # input: D, labels, D_augmented
    nDocTotal = nDoc # + nDocAug  # nDocTotal = nDoc + nAugmented
    for cv_id in range(nTrials): 
        # D_train, D_test, l_train, l_test = train_test_split(D, labels, test_size=ratio_test)
    
        # document labeling: labelize returns list of documents (not np.array() which is what we want here)
        D = labelize(D, label_type='doc')  # No need to use class labels for document label prefixing

        # [params] debug: cohort 
        # want documents to be in list type from here on 
        assert isinstance(D, list) 
        
        ### computing document vectors 
        X = vector.getDocVec(docs=D, d2v_method=d2v_method, 

                                outputdir=get_model_dir(), 
                                meta=tsHandler.meta_model,  # get_model_id(), # {'U', 'L', 'A'} + ctype + cohort

                                test_=kargs.get('test_model', True), 
                                load_model=kargs.get('load_model', True),
                                cohort=cohort_name)
        
        # X = X[:nDoc] # excluding augmented doc vectors 

        # feature size doubled 
        assert X.shape[0] == len(D), "X.shape[0]: %d != nDoc: %d" % (X.shape[0], nDoc)
        print('status> Model computation complete.')

    ### end foreach trial 
    
    return X 

# [todo]
def makeTSetCombined2(**kargs):
    """
    A newer design of makeTSetCombined() by separating the interfaces of: 

    a. input documents loading
    b. d2v model training 
    c. training set making 
    """
    def transform_label(l, positive=None):
        if positive is None: positive = positive_classes # go to the outer environment to find its def
        y_ = labeling.binarize(l, positive=positive) # [params] negative
        return y_ 
    def save_tset(cv_id=0, meta=None): # [params] (X_train, X_test, y_train, y_test)
        return tsHandler.save(cv_id, meta=None)
    def get_model_id(y=[]): # {'U', 'L', 'A'} + ctype + cohort
        return tsHandler.get_model_id(y=y)
    def get_model_dir(): 
        # (!!!) this could have been included in vector.D2V but do not want D2V to have to pass in 'cohort'
        dir_type = 'model'
        modelPath = TSet.getPath(cohort=cohort_name, dir_type=dir_type, create_dir=True)  # ./data/<cohort>/model
        print('makeTSetCombined> model dir: %s' % modelPath)
        return modelPath
    def get_tset_dir(): 
        cohort_name = kargs.get('cohort', '?')
        dir_type = 'combined'
        tsetPath = TSet.getPath(cohort=cohort_name, dir_type=dir_type)  # ./data/<cohort>/
        print('makeTSetCombined> training set dir: %s' % tsetPath)
        return tsetPath

    # import matplotlib.pyplot as plt
    # from tset import TSet
    from labeling import TDocTag
    from pattern import medcode as pmed 
    from sklearn.preprocessing import LabelEncoder
    # import vector

    # loadDocuments2(cohort=kargs.get())

    raise NotImplementedError, "Coming soon." 

def processDocuments2(**kargs):
    return processMCSDocuments(**kargs) 
def processMCSDocuments(**kargs): 
    """
    A wrapper of processDocument() in which a cohort-specific corpus (of MCS documents) are read from the disk, 
    transformed, and labeled. 

    Params
    ------
        inputdir: the directory that hosts MCS file(s)
           e.g. data-exp/CKD
        inputfile: MCS file
           e.g. data-exp/CKD/condition_drug_labeled_seq-CKD.csv

        min_codes: minimum number of codes in an MCS document to consider 
                   e.g. set min_codes to 10 to retain only those MCSs, the lengths of which are >= 10 bases

        segment_by_visit: set to True to segment an MCS into sessions

        last_n_visits: the number of sessions to consider

            if an MCS is segmented into sessions, each session, through d2v, has its own vector representation 
            for instance if a patient had 10 clinical visits (assuming that they had different dates), then 
            her MCS can be further broken down into 10 different smaller segments

            we then treat each segement as a separate document (or a paragraph to be more precise if we consider an 
            entire MCs as a document)

            through d2v, one can derive a feature vector for each segment/session, assuming that similar sessions 
            can be mapped to neighboring points in the vector space, reflecting their similarity in clinical properties

            given above, we can then represent an MCS document in terms of a matrix (containing a sequence of session vectors)

            X_i: [ x_s1, 
                   x_s2, ... 
                   x_s10 ]
                where x_si, i=1~10 denotes a session vector

            if we decide to only consider the last 5 sessions for predictive analytics, then last_n_visits is set to 5, giving 

            X_i = [ 
                    x_s5, 
                    x_s6, ...

                    x_s10]

                which retains only the last 5 session vectors in X_i

            This represention then can be fed into a sequence learning model such as LSTM networks 

            Note that the input X_i to the LSTM networks may need to assume the same dimension, however, the number of session
            vectors within X_i may not be the same, as it happens more often than not since the number of clinical sessions varies 
            from individual to individual. In these cases, the truncated sessions (e.g. x_s1 ~ x_s4 in the example above) 
            are zero-padded so that dimensionality of all {X_i | i=1, ... N}, i.e. entire disease cohort, will remain consistent. 


    <summmary> 
    * user-specified inputs
        inputdir 
        inputfile, ifiles 
        min_ncodes

    * subsampling: large corpus while preserving the percentage of the class labels
        <note> these are not recommended as ideally, we want the size of the corpus to be as large as possible
        max_n_docs
        max_n_docs_policy 

    * segment each document into paragraphs or smaller parts (e.g. prediagnosis sequence)
        predicate
        policy_segment

    * segmentating by visit: break down each document (entire medical history) into visit segments
        segment_by_visit
        last_n_visits (e.g. 10 to only look at the last 10 visit segments)

    Memo
    ----

    1. Look at the last 10 visits of a given document
        - s'pose each visit has max: 100 codes (boostrap)


    """
    def process_docs(): 
        ### load + transform + (ensure that labeled_seq exists)
        src_dir = kargs.get('inputdir', None)
        if src_dir is None: src_dir =  get_global_cohort_dir()

        ctype = tsHandler.ctype # kargs.get('seq_ptype', 'regular')
        
        ifile, ifiles = kargs.get('inputfile', None), kargs.get('ifiles', [])
        if ifile is not None: 
            ipath = os.path.join(src_dir, ifile)
            assert os.path.exists(ipath), "Invalid input path: %s" % ipath
            ifiles = [ifile, ]

        D, L, T = processDocuments(cohort=tsHandler.cohort, seq_ptype=ctype, 
                    inputdir=src_dir, 
                    ifiles=kargs.get('ifiles', ifiles),  # set to [] to use default
                    # meta=kargs.get('meta', None), 
                    
                    # document-wise filtering 
                    policy=kargs.get('doc_filter_policy', 'minimal_evidence'),  
                    min_ncodes=kargs.pop('min_ncodes', 10),  # retain only documents with at least n codes 

                    # padding? to string? 
                    pad_doc=False, pad_value='null', to_str=False,

                    # content modification
                    # predicate=kargs.get('predicate', None), # reserved for segment_docs()
                    simplify_code=tsHandler.is_simplified, 

                    source_type='default', 
                    create_labeled_docs=True)  # [params] composition

        # Dl = labelize(D, label_type='doc', class_labels=L)  # use class labels for document label prefixing
        print('(process_docs) nD: %d | cohort=%s, ctype=%s, labeled? %s, simplified? %s' % \
            (len(D), cohort_name, ctype, is_labeled_data(L), kargs.get('simplify_code', False)))
        return (D, L, T)
    def process_aug_docs(): # additional supporting documents
        src_dir = get_global_cohort_dir()
        ctype = tsHandler.ctype  # kargs.get('seq_ptype', 'regular')
        # D_augmented, L, T = loadAugmentedDocuments(cohort=cohort_name, inputdir=src_dir, label_default=None) # inputdir <- None => use default
        Da, L, T = processDocuments(cohort=tsHandler.cohort, seq_ptype=ctype, 
                                inputdir=src_dir, 
                                ifiles=kargs.get('ifiles', []), 
                                # meta=kargs.get('meta', None),

                                # document-wise filtering 
                                policy=kargs.get('doc_filter_policy', 'minimal_evidence'),  
                                min_ncodes=kargs.pop('min_ncodes', 10),   # retain only documents with at least n codes

                                # padding? to string? 
                                pad_doc=False, pad_value='null', to_str=False, 

                                # predicate=kargs.get('predicate', None),  # reserved for segment_docs()
                                simplify_code=tsHandler.is_simplified, 
                                source_type='augmented', create_labeled_docs=False)  # [params] composition
        assert len(Da) > 0, "No augmented data found (cohort=%s)" % cohort_name 
        # Dl = labelize(Da, label_type='augmented')   
        print('(process_aug_docs) nDAug: %d | cohort=%s, ctype=%s, labeled? %s, simplified? %s' % \
            (len(Da), tsHandler.cohort, ctype, is_labeled_data(L), tsHandler.is_simplified))
        return (Da, L, T)
    def is_labeled_data(lx): 
        # if lx is None: return False
        nL = len(np.unique(lx)) # len(set(lx))
        if nL <= 1: 
            return False 
        return True
    def verify_docs(D, min_length=10, n_test=100): # args: min_ncodes 
        # 1. minimum length: 'min_ncodes'
        minDocLength = min_length
        if minDocLength is not None: 
            nD = len(D)
            ithdoc = random.sample(range(nD), min(nD, n_test))
            for r in ithdoc: 
                assert len(D[r]) >= minDocLength, "Length(D_%d)=%d < %d" % (r, len(D[r], minDocLength))

        # 2. other tests
        # 
        return 
    def get_global_cohort_dir(): 
        return seqparams.getCohortGlobalDir(tsHandler.cohort) # sys_config.read('DataExpRoot')/<cohort>
    def save_mcs(D, T, L, index=0, docIDs=[], meta=None, sampledIDs=[]):  # save the coding sequences
        # [note] if 'max_n_docs' is given and document subsampling is applied, then we need to keep track of 
        # the document indices in their original source positions 
        return tsHandler.save_mcs(D, T, L, index=index, docIDs=docIDs, meta=meta, sampledIDs=sampledIDs)
    def load_mcs(index=None, meta=None, inputdir=None, inputfile=None): 
        return tsHandler.load_mcs(index=index, meta=meta, inputdir=inputdir, inputfile=inputfile)
    def to_str(D, sep=' '):
        return [sep.join(doc) for doc in D]  # assuming that each element is a string 
    def diagnosis_dim(D, doc_type='mcs'):  # params: nDocTotal
        # doc_type: {'mcs', 'session'}
        #      mcs: each document is a complete MCS 
        #      session: each document is just a session (of an MCS)
        print('(diagnosis_dim) Computing document statistics (each doc IS A: %s) ...' % str(doc_type).upper())
        sizeStats = doc_size_stats(D)  # use this to estimate an appropriate window size
        nDocEff = len(D)

        # [note] doc stats: mean number of tokens, median, std
        print('(doc_stats) ... nDocTotal: %d =?= nDocEff: %d | doc stats: (mean: %f, median:%f, std:%f), window: %d' % \
            (nDocTotal, nDocEff, sizeStats['mean'], sizeStats['median'], sizeStats['std'], vector.D2V.window)) # after all doc processing (segmentations, etc), nDoc may change
        if doc_type.startswith('s'): # session
            assert nDocTotal <= nDocEff, "summing total number of visits (%d) should exceed number of documents (%d)" % (nDocEff, nDocTotal)
            print('(doc_stats) ... nDocTotal: %d <? nDocEff: %d' % (nDocTotal, nDocEff))
        return
    def doc_size_stats(D):
        nD = len(D)
        acc = 0 
        ll = []
        nTokens = len(np.unique(np.hstack(D)))
        for i, d in enumerate(D):
            acc += len(d)
            ll.append(len(d))  # [todo] re-write, this can be memory consuming for large cohort
        res = {}
        res['mean'] = El = np.mean(ll)
        res['median'] = Ml = np.median(ll)
        res['std'] = stdl = np.std(ll)
        res['vocab_size'] = nTokens 
        return res
    def get_doc_stats(D, L, T, condition=''): 
        if condition: print "(condition) %s" % condition  

        labels = np.unique(L)
        n_classes = len(labels)

        # average document length (number of tokens) 
        n_tokens = [len(doc) for doc in D]
        avgL = sum(n_tokens)/(len(D)+0.0)  # compare this to avgLPerSession later
        maxL = max(n_tokens)
        minL = min(n_tokens)
        medianL = np.median(n_tokens)
        stdL = np.std(n_tokens)
        n_tokens = None; gc.collect()
        print('(doc_stats) E[n_tokens in doc]: %f ~? median: %f | max: %d, min: %d, std: %f' % \
            (avgL, medianL, maxL, minL,  stdL))
      
        # time statistics

        # label statistics
        print('(doc_stats) n_classes: %d' % n_classes)
        print('... %s' % labels)

        return
    def sample_docs(D, L, T, n, sort_index=True, random_state=53, policy='longest'):
        # if n > len(y): n = None # noop
        idx = cutils.selectPerClass(X=D, y=L, n_per_class=n, verbose=True, sort_index=sort_index, random_state=random_state, policy=policy)
        D = list(np.array(D)[idx])
        L = list(np.array(L)[idx])
        T = list(np.array(T)[idx])

        n_classes = len(np.unique(L))
        n_per_class = n
        assert len(D) <= n * n_classes, "size(D): %d, n_classes: %d, n_per_class: %d" % (len(D), n_classes, n_per_class)
        print('(sample_docs) subsampling: select only %d docs (policy=%s, n_per_class=%d, n_classes: %d)' % \
            (len(D), policy, n_per_class, n_classes))
        return (idx, D, L, T)
    def segment_docs(D, L, T, predicate=None, policy='regular', time_cutoff=None):  # <- D, L, T
        # policy: {'regular', 'two'/'halves', 'prior'/'posterior', 'complete', }
        # output: (DocIDs, D, L, T)
       
        # only consider the segment prior to time_cutoff (prior to applying eMERGE phenotyping algorithm)
        if time_cutoff is not None: 
            _, D, L, T = docProc.segmentDocumentByTime(D, L, T, timestamp=time_cutoff, policy='prior', inclusive=True)
        
        include_endpoint = kargs.get('include_endpoint', True)  # include the chunk containing diagnosis info?
        drop_docs_without_cutpoints = kargs.get('drop_nullcut', False)  # if True, do not include doc without valid cut points (diagnosis info)

        # configure file ID 
        # if predicate is None or policy is regular, then no-op
        if predicate is not None and not policy.startswith('reg'):
            # if not tsHandler.meta: tsHandler.meta = policy_segment  # other values 'A', 'U', 'D', 'L' 
            # assert tsHandler.meta and not (tsHandler.meta in ('D', 'default', )), \
                # "default user-defined file ID is not recommended since segmenting op will modify the documents (policy=%s)" % policy_segment
            print('(segment_docs) predicate: %s, policy: %s' % (predicate, policy))
        else: 
            print('(segment_docs) policy: %s' % policy)

        return docProc.segmentDocuments(D, L, T, predicate=predicate, policy=policy, 
                    inclusive=include_endpoint, drop_nullcut=drop_docs_without_cutpoints) 
    def process_label():  # L, {labeling, LabelEncoder}
        ulabels = np.unique(L)
        n_classes = ulabels.shape[0]
        print('    + class labeling ...')
        le = LabelEncoder(); le.fit(L)  # concatenate in case l_train doesn't include all the labels
        assert n_classes == le.classes_.shape[0]
        print('    + unique labels (n=%d):\n%s\n' % (n_classes, ulabels))
        lc = labeling.count(L)
        print('    + class counts: %s' % n_classes)
        print('    + labels:\n%s\n' % lc)
        return ulabels
    def tokenize(D, vocab_size=None, test_level=0):  # limit vocab size (by discarding infreq tokens) and convert tokens to integers/indices
        tLimitVocab = False if vocab_size is None else True

        Dp = D
        if tLimitVocab: 
            print('(tokenize) limiting vocab size to %d and converting input to integer representation' % vocab_size)
            Dp = docProc.tokenizeDoc(D,  
                   test_level=test_level, throw_=True, customized=False, 
                   pad_empty_doc=True, 
                   int_to_str=True,  # convert integers/indices to strings as a proper input format for the gensim doc2vec framework
               
                   # tokenizer
                   num_words=vocab_size, split=' ', oov_token=TDoc.token_unknown)
        else: 
            pass # noop
        return Dp 
    ### processMCSDocuments()
 
    # import matplotlib.pyplot as plt
    # from tset import TSet
    from tdoc import TDoc
    import docProc, labeling
    # from labeling import TDocTag
    from pattern import medcode as pmed 
    from sklearn.preprocessing import LabelEncoder
    import classifier.utils as cutils
    # import vector

    # key params: 
    #    max_visit_length
    #    min_ncodes
    #    min_n_docs 
    #    min_n_docs_policy
    #    predicate, policy_segment
    #    apply d2v? if True, prepare document input that gears towards building d2v models

    ### configure all the parameters necessary for file naming, etc. 
    cohort_name = tsHandler.cohort
    d2v_method = tsHandler.d2v
    user_file_descriptor = meta = kargs.get('meta', tsHandler.meta) 
    include_augmented = kargs.get('include_augmented', False)

    # [input] cohort, labels, (result_set), d2v_method, (w2v_method), (test_model)
    #         simplify_code, filter_code  
    div(message='processMCSDocuments> Load (MCSs) + sample + transform + labelize (required by gensim) ...')

    ### 1. load + transform + labelize   [params] min_ncodes
    # [note] or use load_mcs() to load from training data directory (e.g. sampled documents, segmented documents)
    D, L, T = process_docs(); verify_docs(D, min_length=kargs.get('min_ncodes', 10), n_test=100)  # [test]
    ulabels = process_label()
    n_classes = len(ulabels)

    ## 1a. incorporate augmented (unlabeled) documents if available (=> the so-called transfer learning)
    Da, La, Ta = [], [], []
    nDocAug = nDocTotal = 0 
    if include_augmented: 
        Da, La, Ta = process_aug_docs() # load + transform + labelize
        nDocAug = len(Da)
    if len(Da) > 0: D, L, T = D+Da, L+La, T+Ta 

    # condition: up to this point, we've loaded MCS docs from disk to (D, L, T)

    ## 1b. subsampling: large corpus while preserving the percentage of the class labels
    #     but this is usually not desirable, better practice is to use the entire corpus to get d2v and reduce training data at this level
    max_n_docs, n_docs_policy = kargs.get('max_n_docs', None), kargs.get('max_n_docs_policy', 'longest') 
    sampledIDs = []
    if max_n_docs: 
        print('(sampling) Sample a subset of documents (N: %d -> Np: %d) ...' % (len(D), max_n_docs))
        sampledIDs, D, L, T = sample_docs(D, L, T, n=max_n_docs, policy=n_docs_policy) # set None to take ALL
        kargs['save_doc'] = True # definitely have to save the new documents
    nDocTotal = nDoc = len(D)

    ## 1c. segment each document into paragraphs or smaller parts (e.g. pre-diagnosis sequence)
    # [note] DocIDs are used to associate segments to the original document (which may be broken down into several segments). 
    #        Under no-op, DocIDs are just regular positional indices.  
    print('(editing) reduce the MCS into a smaller functional segment (e.g. prediagnosis sequence) ...')
    func_segment, p_segment = kargs.get('predicate', None), kargs.get('policy_segment', 'regular')
    time_cutoff = kargs.get('time_cutoff', '2017-08-18') # only consider the records prior to applying eMERGE phenotyping algorithm
    docIDs, D, L, T = segment_docs(D, L, T, predicate=func_segment, policy=p_segment, time_cutoff=time_cutoff)  # <- policy_segment, inclusive?, include_active?
    assert not d2v_method.startswith(('bow', 'bag', )), "segment by visits does not apply in sparse models (e.g. bow)."

    # tD2V = kargs.get('apply_d2v', True) # then we need to process the document further

    # save a copy of the modified document for later reference 
    if kargs.get('save_doc') or len(sampledIDs) > 0: 
        save_mcs(D, T, L, index=0, docIDs=docIDs, meta=user_file_descriptor, sampledIDs=sampledIDs)  # D has included augmented if provided

        # save a sharable version (in which MCS bases/tokens are converted to integers)
        D_deidentified = tokenize(D, vocab_size=kargs.get('num_words', 20001), test_level=0)

        # T set to [] to be ignored because tokens in Dp no longer match time-wise with timestamps in T (due to the removal of infreq tokens)
        save_mcs(D_deidentified, T=[], L=L, index=0, docIDs=docIDs, meta=user_file_descriptor, sampledIDs=sampledIDs)
    
    # [test]
    diagnosis_dim(D, doc_type='mcs') # params: nDocTotal
    get_doc_stats(D, L=L, T=T, condition='Prior to segmenting documents into sessions')
    
    ### 2. segment by sessions
    tSegmentVisit, visitDocIDs = kargs.get('segment_by_visit', False), []  # vDocIDs: visit-modulated document IDs

    mode = 'visit'  # values: visit, uniform
    n_timesteps = kargs.get('last_n_visits', 10)  # only look at the last N visit segments 
    print('(segmenting) break down the MCS into sessions (as are defined by timestamps) ...')
    if d2v_method in ('external', ):  # document and word embedding is performed externally (esp those not using gensim's d2v model)       
        # consider segmenting by visits as well? 
        if tSegmentVisit: 

            # mode: uniform by default, all visit segments are joined together to form a unified document (plus paddings such that each 
            #       document has the same length)
            # params: 
            #    (obsolete) max_length=kargs.get('max_visit_length', 100)
            visitDocIDs, D = docProc.makeSessionDocuments(D, L, T,  
                min_length=kargs.get('min_visit_length', seqparams.D2V.window), 
                max_length=kargs.get('max_visit_length', 100), 
                max_n_visits=n_timesteps, 
                mode=kargs.get('visit_to_doc_mode', 'uniform'))
        else: 
            raise NotImplementedError

        # if kargs.get('pad_doc', False): 
        #     # set max_doc_length to None to use the max among all documents
        #     D = pad_sequences(D, value=kargs.get('pad_value', 'null'), maxlen=kargs.get('max_doc_length', None))
        #     testcases = random.sample(D, min(len(D), 10))
        #     assert len(set(len(tc) for tc in testcases)) == 1, "doc size unequal:\n%s\n" % [len(tc) for tc in testcases]

        # from list of list (of tokens) TO list of strings
        # if kargs.get('to_str', True): 
        #     for doc in random.sample(D, min(len(D), 10)): 
        #         assert instance(doc, str)

        # return (D, L, T, docIDs, visitDocIDs, sampledIDs) # D is a list of strings
    else: 
        # learning document or visit vector representation using d2v model

        # visit segments are not joined together like uniform mode such that each document is represented by a list of visit segments
        mode = kargs.get('visit_to_doc_mode', 'visit') 

        # 1d. segment by visits?  i.e. doc<i> -> [v1, v2, ... v10, ...] where v1: [c1 ,c2, c3, ...]
        #     each document becomes a list of lists, the whole corpus is a 3D structure, i.e. a list of 2D components
        #     but this 3D is subsequently "flattened" into a 2D structure => a list of visits across the boundary of documents
        #     however, there's an implict structure imposed in this 2D structure; i.e. each document is repr by N visits (e.g. N=10, last 10 visits)
        #     // and each visit is of the same length M (e.g. max_visit_length: 100)   ... 08.16.18
        #        => each visit needs not be of the same length but had better be at least >= d2v window size
        if tSegmentVisit:  # segmenting by visits => each document is repr by a sequence of visits (each comprising a set of codes)
            # need to treat each visit as a document/paragraph 
            assert not d2v_method.startswith(('bow', 'bag', )), "segment by visits does not apply in sparse models (e.g. bow)."

            # visitDocIDs: expanded document IDs e.g. [0, 0, 0, 1, 1, 2, 3, 3, 3, 3, ...] => 0th doc has 3 visits, 1th has 2 visits, etc.
            visitDocIDs, D = docProc.makeSessionDocuments(D, L, T,  
                min_length=kargs.get('min_visit_length', seqparams.D2V.window), 
                max_length=kargs.get('max_visit_length', 100), 
                max_n_visits=None,  # keep all sessions in order to learn a better d2v model 
                mode=mode)

            assert len(D) == len(visitDocIDs) and len(visitDocIDs) >= len(L), \
                      "size(D):%d, size(visitDocIDs): %d, sizes(L): %d" % (len(D), len(visitDocIDs), len(L))

    ### tokenization: remove infrequent tokens and make integer/index repr
    print('(tokenzing) remove infrequent bases/tokens and convert resulting MCS into integer representation (i.e. token indices) ...')
    D = tokenize(D, vocab_size=kargs.get('num_words', 20001), test_level=0)  # set vocab_size to None to bypass this step

    # [test]
    nDocEff = len(D)
    diagnosis_dim(D, doc_type='session' if tSegmentVisit else 'mcs') # params: nDocTotal

    print('(labelizing) label each MCS/document instance, required to be used with gensim D2V model ...')
    if tSegmentVisit: 
        Dl = labelize(D, label_type='v')  # each "document" is actually just a (bootstrapped) visit segment 
    else: 
        # labelize documents (required by gensim's d2v models)
        Dl = labelize(D, label_type='doc', class_labels=L)  # use class labels for document label prefixing

    # input: D, labels, D_augmented
    if d2v_method.startswith(('bow', 'bag', )): 
        pass # d2v model requires labeled documents but, bag-of-words and other sparse models do not 
    else: 
        D = Dl

    return (D, L, T, docIDs, visitDocIDs, sampledIDs)  # D: document-labeled (Dl) or the original (D) (e.g. in BOW mode)

def makeTSetVisitAsDocVec(**kargs): 
    return makeTSetVisit3D(**kargs)
def makeTSetVisit3D(**kargs):
    def get_model_dir(): 
        dir_type = 'model'
        modelPath = TSet.getPath(cohort=tsHandler.cohort, dir_type=dir_type, create_dir=True)  # ./data/<cohort>/model
        print('makeTSetVisit3D> model dir: %s' % modelPath)
        return modelPath
    def expand_by_visit(L, visitDocIDs): 
        Lp = []
        for vid in visitDocIDs:  # [0, 0, 1, 1, 1, 2, 2, ...]
            Lp.append(L[vid])
        return Lp
    def save_tset(X, y, cv_id=0, docIDs=[], meta=None, sparse=False, shuffle_=True, verify_=False): # [params] (X_train, X_test, y_train, y_test)

        # set verify_ to True to avoid actual saving if file exists
        return tsHandler.save(X, y, index=cv_id, docIDs=docIDs, meta=meta, sparse=sparse, shuffle_=False, verify_=verify_)
    def load_tset(cv_id=0, meta=None, sparse=False):  # cohort, seq_ptype, d2v_method (focused_labels, label_map)
        # this should conform to the protocal in makeTSetCombined()
        return tsHandler.load_tset(index=cv_id, meta=meta, sparse=sparse)
    def eval_doc_stats(): 
        pass
    def test_input_docs(Din):
        nDoc = len(Din)
        Dsub = random.sample(Din, min(nDoc, 10)) 

        lx = [len(d) for d in Dsub]
        print('test> Input document > sample lengths: %s' % lx)

        return
    def subsample(X, y, n, docIDs=[], base_factor=2): 
        from sklearn.model_selection import train_test_split  # only used for subsampling training data (when it's too big)

        Xp = yp = None
        n_training_examples = n
        if docIDs: 
            ts = TSet.toCSV(X, y=y, docIDs=docIDs, save_=False, shuffle_=False)

            if ts.shape[0] > n * base_factor: 
                ts_train, ts_test = train_test_split(ts, 
                    train_size=n_training_examples, 
                    # test_size=test_subset,   # det_test_size(), # accept test_size, ratios
                    random_state=kargs.get('random_state', int(time.time())%1000), stratify=y)
                Xp, yp = TSet.toXY(ts_train)
                docIDs = ts[TSet.index_field]

        elif X.shape[0] > n * base_factor:  # if X is big enough   
            Xp, _, yp, _ = train_test_split(X, y, 
                train_size=n_training_examples, 
                # test_size=test_subset,   # det_test_size(), # accept test_size, ratios
                random_state=kargs.get('random_state', int(time.time())%1000), stratify=y)

            # docIDs <- []
        else: 
            # noop 
            Xp, yp = X, y

        return (Xp, yp, docIDs)
    def subsample2(ts, n, base_factor=2):
        from sklearn.model_selection import train_test_split  # only used for subsampling training data (when it's too big)

        y = None # default value for stratify
        try: 
            y = ts[TSet.target_field].values 
        except: 
            pass

        tsp, _ = train_test_split(ts, 
                train_size=n, 
                # test_size=test_subset,   # det_test_size(), # accept test_size, ratios
                random_state=kargs.get('random_state', int(time.time())%1000), stratify=y)
        
        # Xp, yp = TSet.toXY(ts_train)
        # docIDs = ts[TSet.index_field] 
        return tsp
    def subsample_docs(D, n, docIDs=[]):
        
        return 

    import vector 
    # from sklearn.model_selection import train_test_split  # only used for subsampling training data (when it's too big)
    # from tset import TSet

    # tset_file_descriptor = meta = kargs.get('meta', tsHandler.meta)
    # model_file_descriptor = meta_model = kargs.get('meta', tsHandler.meta_model)

    # params: 
    #    1. load documents: 
    #          inputfile, inputdir, min_ncodes, doc_filter_policy
    #    2. segmentation policy: 
    #          predicate, policy_segment/'regular', include_endpoint/False, drop_nullcut/True
    #    3. visit segements: 
    #          max_visit_length/100, last_n_visits/10
    kargs['segment_by_visit'] = True
    kargs['last_n_visits'] = kargs.get('last_n_visits', 10)
    Din, L, T, docIDs, visitDocIDs, sampledIDs = processMCSDocuments(**kargs) # Din: labeled documents for d2v model
    nDoc, nDocEff = len(T), len(Din)

    nTrials = kargs.get('n_trials', 1)
    tset_descriptor = kargs.get('meta', tsHandler.meta)
    model_descriptor = kargs.get('meta_model', tsHandler.meta_model)
    for cv_id in range(nTrials):   # loop reserved for CV or random subsampling if necessary
        print('makeTSetVisit3D> Computing document vectors nD:%d => nDEff: %d' % (nDoc, nDocEff))
        test_input_docs(Din)
        # [note] Dl includes augmented if provided: Dl + Dal
        #        this will save a model file

        loadModel = kargs.get('load_model', True)
        y = np.array(L)

        print('... meta_model: %s' % tsHandler.meta_model)
        X = vector.getDocVec(docs=Din, d2v_method=tsHandler.d2v, 
                                outputdir=get_model_dir(),  # [params] cohort, dir_type='model' 

                                # [note] meta is a model file descriptor, which is part of the model file naming
                                meta=model_descriptor,   # user-defined file ID: get_model_id(y=L), # {'U', 'L', 'A'} + ctype + cohort
                                # labels=L,  # assess() will evaluate d2v accuracy based on the labels

                                test_=kargs.get('test_model', True), 
                                load_model=loadModel,

                                segment_by_visit=True, # need to distinguish from 'normal' document 

                                max_features=None, # only applies to d2v_method <- 'bow' (bag-of-words sparse model) 
                                cohort=tsHandler.cohort)

        # Need to combine visit vectors so that each document maps to a vector
        # policy: i) average ii) concatenate the last N visits 
        n_features, n_timesteps = X.shape[1], kargs.get('last_n_visits', 10)  # pv-dm2: vector.D2V.n_features * 2
        print('... prior to consolidateVisits | dim(X): %s, n_features: %d, n_timesteps: %d' % (str(X.shape), n_features, n_timesteps))

        # consolidate visit segments and their vectors to form a single document vector for each pateint document set 
        # [note] subsampling to reduce memory overhead (esp. when n_timestpes is large)
        n_samples = kargs.get('max_n_samples', None)
        X, y = vector.consolidateVisits(X, y=y, 
                    docIDs=visitDocIDs, last_n_visits=n_timesteps, 
                    max_n_samples=kargs.get('max_n_samples', None))  # flatten out n visit vectors by concatenating them
        assert X.shape[0] == len(y), \
            "Since visit vectors are flattened out i.e. N visit vectors => one big vector, size(X): %d ~ size(y): %d" % (X.shape[0], len(y))

        # X = X[:nDoc] # excluding augmented doc vectors 
        # condition: feature size doubled if using pv-dm2

        ### Save training set
        if kargs.get('save_', True):     
            # Output location: e.g. .../data/<cohort>/cv/
            # can also use 'tags' (in multilabel format)
                
            # need to convert X from 3D to 2D ... do this later when needing to train LSTM models
            # X = X.reshape((, X.shape[2]))   nDoc, n_timesteps, n_features
            
            # 2D to 3D
            # for i, x in enumerate(X): 
            #     X[i] = x.reshape((n_timesteps, n_features))  # nDoc, n_timesteps, n_features

            # Lp = expand_by_visit(L, visitDocIDs)  

            # If the data are too big for the disk, may want to limit its size
            # [note] subsampling is being shifted to the consolidateVisits(); otherwise, won't work with large data (pad_sequences() => MemoryError)
            if False: 
                n_samples = kargs.get('max_n_samples', None)
                if n_samples: X, y, docIDs = subsample(X, y, docIDs=docIDs, n=n_samples)

            print('makeTSetVisit3D> Saving d2v training data: dim(X):%s | N: %d=?=%d | n_features: %d, n_timesteps: %d' % \
                (str(X.shape), X.shape[0], n_samples, n_features, n_timesteps))

            # [note] set conditional_save to True to save only when the training data do not exist yet
            ts = save_tset(X, y, cv_id, docIDs=[], meta=tset_descriptor, shuffle_=False, verify_=kargs.get('conditional_save', False)) 

        print('status> Model computation complete (@nTrial=%d)' % cv_id)

    ### end foreach trial 
    return (X, y) 

def makeTSetDoc(**kargs):
    """
    Create training set directly by treating the whole MCS as a document. 
    This is essentially thes same as makeTSet() by separating document processing operations 
    from the doc2vec model. 
    """
    def get_model_dir(): 
        dir_type = 'model'
        modelPath = TSet.getPath(cohort=tsHandler.cohort, dir_type=dir_type, create_dir=True)  # ./data/<cohort>/model
        print('makeTSetVisit3D> model dir: %s' % modelPath)
        return modelPath
    def save_tset(X, y, cv_id=0, docIDs=[], meta=None, sparse=False, shuffle_=True): # [params] (X_train, X_test, y_train, y_test)
        return tsHandler.save(X, y, index=cv_id, docIDs=docIDs, meta=meta, sparse=sparse, shuffle_=False)
    def load_tset(cv_id=0, meta=None, sparse=False):  # cohort, seq_ptype, d2v_method (focused_labels, label_map)
        # this should conform to the protocal in makeTSetCombined()
        return tsHandler.load_tset(index=cv_id, meta=meta, sparse=sparse)
    def is_sparse(): 
        if tsHandler.d2v.startswith(('bow', 'bag', 'aphri')):  # aphridate
            return True 
        return False
    import vector 
    # from tset import TSet
    
    user_file_descriptor = meta = kargs.get('meta', tsHandler.meta)

    # params: 
    #    1. load documents: 
    #          inputfile, inputdir, min_ncodes, doc_filter_policy
    #    2. segmentation policy: 
    #          predicate, policy_segment/'regular', include_endpoint/False, drop_nullcut/True
    #    3. visit segements: 
    #          max_visit_length/100, last_n_visits/10
    kargs['segment_by_visit'] = False
    Din, L, T, docIDs, _, sampledIDs = processMCSDocuments(**kargs) # Din: labeled documents for d2v model

    nDoc, nDocEff = len(T), len(Din)
    nTrials = kargs.get('n_trials', 1)
    for cv_id in range(nTrials):   # loop reserved for CV or random subsampling if necessary
        print('    + computing document vectors nD:%d => nDEff: %d ...' % (nDoc, nDocEff))
        # D_train, D_test, l_train, l_test = train_test_split(D, labels, test_size=ratio_test)

        # suppose we want classify final stage of a diseases  
        # [note] Can do the following externally via binarize()
        # positive_classes = ['CKD Stage 4', 'CKD Stage 5',]
        # y_train, y_test = transform_label(l_train), transform_label(l_test)  # e.g. binarize the labels
        print('    + computing document vectors (nD:%d + nDAug:%d -> nDTotal:%d) => nDEff: %d ...' % (nDoc, nDocAug, nDocTotal, nDocEff))

        # [note] Dl includes augmented if provided: Dl + Dal
        #        this will save a model file
        y = np.array(L)
        X = vector.getDocVec(docs=Din, d2v_method=tsHandler.d2v, 
                                outputdir=get_model_dir(),  # [params] cohort, dir_type='model' 
                                meta=tsHandler.meta_model,   # user-defined file ID: get_model_id(y=L), # {'U', 'L', 'A'} + ctype + cohort
                                labels=L,  # assess() will evaluate d2v accuracy based on the labels

                                test_=kargs.get('test_model', True), 
                                load_model=kargs.get('load_model', True),

                                segment_by_visit=tSegmentVisit, # need to distinguish from 'normal' document 

                                max_features=kargs.get('max_features', None), # only applies to d2v_method <- 'bow' (bag-of-words sparse model) 
                                text_matrix_mode=kargs.get('text_matrix_mode', 'tfidf'), # only applies to sparse matrix

                                cohort=tsHandler.cohort)

        assert X.shape[0] == nDocEff and X.shape[0] == len(L)

        # save training data
        if kargs.get('save_', True):  
            # [test]
            if is_sparse(): print('makeTSetDoc> Sparse matrix (X): dim=%s, d2v=%s' % (str(X.shape), d2v_method))
            
            ts = save_tset(X, L, cv_id, docIDs=docIDs, meta=user_file_descriptor, sparse=is_sparse()) # [params] X_train, X_test, y_train, y_test
                

        print('status> Model computation complete (@nTrial=%d)' % cv_id)
        if kargs.get('save_doc', True): 
            # user_file_descriptor is tsHandler.meta by default
            tsDoc = save_mcs(D, T, L, index=cv_id, docIDs=docIDs, meta=user_file_descriptor, sampledIDs=sampledIDs)  # D has included augmented if provided
            
            # [test]
            assert X.shape[0] == tsDoc.shape[0], "Size inconsistent: size(doc): %d but size(X): %d" % (tsDoc.shape[0], X.shape[0])

    ### end foreach trial 
    return (X, y)

def makeTSetCombined(**kargs): 
    return makeTSet(**kargs)
def makeTSet(**kargs): 
    """
    Similar to makeTSet but this routine does not distinguish train and test split in d2v
    The idea is to separate CV process in evaluating classifiers (modeled over document vectors) from 
    the training of d2v models. 

    Use
    ---
    1. set dir_type to 'combined' to load the pre-computed training data
       e.g. TSet.load(cohort=cohort_name, d2v_method=d2v_method, dir_type='combined') to load

    Note
    ----
    1. For simplicity, d2v model and the training set share the same prefix directory, with the former kept in 'model'
       and the latter in {'combined', 'train', 'test', }

       s'pose prefix_dir = tpheno/seqmaker/data/<cohort>

       then, model is kept at <prefix_dir>/model  i.e. <dir_type>='model'

            e.g. tpheno/seqmaker/data/CKD/model

       while training set is kept at <prefix_dir>/<dir_type>
           

            e.g. 
               tpheno/seqmaker/data/CKD/combined  //no train-set split 
               tpheno/seqmaker/data/CKD/train     //train split 
               tpheno/seqmaker/data/CKD/test      //test split


    """
    def assemble_docs(verbose=True): # cohort_name, (seq_ptype, predicate, simplify_code?, include_augmented, inputdir)
        # note: fetch labeled and unlabeled documents separately
        # params> inputdir: path to input document source

        ### load + transfomr + (ensure that labeled_seq exists)
        ctype = tsHandler.ctype # seqparams.normalize_ctype(kargs.get('seq_ptype', 'regular'))
        D, L, T = processDocuments(cohort=tsHandler.cohort, seq_ptype=ctype,
                # predicate=kargs.get('predicate', None), # reserved for segment_docs()
                simplify_code=tsHandler.is_simplified)  # [params] create_labeled_docs
        
        D_augmented = []
        if kargs.get('include_augmented', False):
            # inputdir <- tpheno/data-exp/<cohort> by default 
            # e.g. tpheno/data-exp/CKD/condition_drug_timed_seq-CKD.csv 
            D_augmented, L2, T2 = loadAugmentedDocuments(cohort=tsHandler.cohort, inputdir=None, label_default=None) # inputdir <- None => use default
            assert len(D_augmented) > 0, "No augmented data found (cohort=%s)" % tsHandler.cohort

        # labels = L # documents = D; timestamps = T
        D = labelize(D, label_type='doc', class_labels=L)  # use class labels for document label prefixing
        if len(D_augmented) > 0: 
            D_augmented = labelize(D_augmented, label_type='augmented')
            # D = D + D_augmented
        if verbose: 
            print('.assemble_docs> Document source dir (default):\n%s\n' % get_global_cohort_dir()) # tpheno/data-exp/<cohort>
            print('  + nD=%d, nDAug=%d | cohort=%s, ctype=%s, simplified? %s' % \
                (len(D), len(D_augmented), tsHandler.cohort, ctype, tsHandler.is_simplified))
        
        return (D, D_augmented, L, T)  # the size of D may be larger 
    def process_docs(): 
        ### load + transfomr + (ensure that labeled_seq exists)
        
        src_dir = kargs.get('inputdir', None)
        if src_dir is None: src_dir =  get_global_cohort_dir()

        ctype = tsHandler.ctype # kargs.get('seq_ptype', 'regular')
        
        ifile, ifiles = kargs.get('inputfile', None), kargs.get('ifiles', [])
        if ifile is not None:  # if inputfile is given, it'll take precedence over ifiles input
            ipath = os.path.join(src_dir, ifile)
            assert os.path.exists(ipath), "Invalid input path: %s" % ipath
            ifiles = [ipath, ]  # overwrite ifiles
            print('process_docs> inputs:\n%s\n' % ifiles)

        D, L, T = processDocuments(cohort=tsHandler.cohort, seq_ptype=ctype, 
                    inputdir=src_dir, 
                    ifiles=kargs.get('ifiles', []),  # set to [] to use default
                    # meta=kargs.get('meta', None), 
                    
                    # document-wise filtering 
                    policy=kargs.get('doc_filter_policy', 'minimal_evidence'),  
                    min_ncodes=kargs.pop('min_ncodes', 10),  # retain only documents with at least n codes 

                    # content modification
                    # predicate=kargs.get('predicate', None), # reserved for segment_docs()
                    simplify_code=tsHandler.is_simplified, 

                    source_type='default', 
                    create_labeled_docs=True)  # [params] composition

        # Dl = labelize(D, label_type='doc', class_labels=L)  # use class labels for document label prefixing
        print('  + nD: %d | cohort=%s, ctype=%s, labeled? %s, simplified? %s' % \
            (len(D), cohort_name, ctype, is_labeled_data(L), kargs.get('simplify_code', False)))
        return (D, L, T)
    def process_aug_docs(): # additional supporting documents
        src_dir = get_global_cohort_dir()
        ctype = tsHandler.ctype  # kargs.get('seq_ptype', 'regular')
        # D_augmented, L, T = loadAugmentedDocuments(cohort=cohort_name, inputdir=src_dir, label_default=None) # inputdir <- None => use default
        Da, L, T = processDocuments(cohort=tsHandler.cohort, seq_ptype=ctype, 
                                inputdir=src_dir, 
                                ifiles=kargs.get('ifiles', []), 
                                # meta=kargs.get('meta', None),

                                # document-wise filtering 
                                policy=kargs.get('doc_filter_policy', 'minimal_evidence'),  
                                min_ncodes=kargs.pop('min_ncodes', 10),   # retain only documents with at least n codes

                                # predicate=kargs.get('predicate', None),  # reserved for segment_docs()
                                simplify_code=tsHandler.is_simplified, 
                                source_type='augmented', create_labeled_docs=False)  # [params] composition
        assert len(Da) > 0, "No augmented data found (cohort=%s)" % cohort_name 
        # Dl = labelize(Da, label_type='augmented')   
        print('  + nDAug: %d | cohort=%s, ctype=%s, labeled? %s, simplified? %s' % \
            (len(Da), tsHandler.cohort, ctype, is_labeled_data(L), tsHandler.is_simplified))
        return (Da, L, T)
    def corpus_stats():  # [result]
        pass
    def is_labeled_data(lx): 
        # if lx is None: return False
        nL = len(np.unique(lx)) # len(set(lx))
        if nL <= 1: 
            return False 
        return True
    def transform_label(l, positive=None):
        if positive is None: positive = positive_classes # go to the outer environment to find its def
        y_ = labeling.binarize(l, positive=positive) # [params] negative
        return y_ 
    def get_global_cohort_dir(): 
        return seqparams.getCohortGlobalDir(tsHandler.cohort) # sys_config.read('DataExpRoot')/<cohort>
    def save_tset(X, y, cv_id=0, docIDs=[], meta=None, sparse=False, shuffle_=True): # [params] (X_train, X_test, y_train, y_test)
        if kargs.get('segment_by_visit', False): shuffle_=False 

        print('(save_tset) meta (userFileID): %s=?=%s, meta_model: %s' % (meta, tsHandler.meta, tsHandler.meta_model))
        return tsHandler.save(X, y, index=cv_id, docIDs=docIDs, meta=meta, sparse=sparse, shuffle_=shuffle_)
    def load_tset(cv_id=0, meta=None, sparse=False):  # cohort, seq_ptype, d2v_method (focused_labels, label_map)
        # this should conform to the protocal in makeTSetCombined()
        return tsHandler.load_tset(index=cv_id, meta=meta, sparse=sparse)
    def save_mcs(D, T, L, index=0, docIDs=[], meta=None, sampledIDs=[]):  # save the coding sequences
        # if 'max_n_docs' is given and document subsampling is applied, then we need to keep track of 
        # the document indices in their original source positions 
        return tsHandler.save_mcs(D, T, L, index=index, docIDs=docIDs, meta=meta, sampledIDs=sampledIDs)
    def load_mcs(index=None, meta=None, inputdir=None, inputfile=None): 
        return tsHandler.load_mcs(index=index, meta=meta, inputdir=inputdir, inputfile=inputfile)

    def profile(ts): 
        return tsHandler.profile(ts)
    def get_tset_dir(): 
        return tsHandler.get_tset_dir()
    def get_tset_id(y=[]): # use L to verify if the input documents are labeled/L or unlabeled/U (default)
        return tsHandler.get_tset_id(y=y)
    def get_model_id(y=[]): # <- L    
        # ID: {'U', 'L', 'A'} + ctype + cohort
        #     unlabeld (U), labeled (L) or augmented (A) (i.e. plus unlabeled)
        meta = get_tset_id(y=y)  # property of the training data ('L', 'U', 'A')
        ctype = tsHandler.ctype # seqparams.normalize_ctype(kargs.get('seq_ptype', 'regular'))
        if ctype is not None: meta = '%s-%s' % (meta, ctype)

        # cohort_name = kargs.get('cohort', '?')
        if not cohort_name in ('?', 'n/a', None): meta = '%s-%s' % (meta, cohort_name)
        if len(meta) == 0: meta = 'generic' # if no ID info is given, then it is generic test model
        return meta
    def get_model_dir(): 
        # (!!!) this could have been included in vector.D2V but do not want D2V to have to pass in 'cohort'
        #       the directory in which model is kept
        dir_type = 'model'
        modelPath = TSet.getPath(cohort=tsHandler.cohort, dir_type=dir_type, create_dir=True)  # ./data/<cohort>/model
        print('makeTSetCombined> model dir: %s' % modelPath)
        return modelPath
    def process_label():  # L, {labeling, LabelEncoder}
        ulabels = np.unique(L)
        n_classes = ulabels.shape[0]
        print('    + class labeling ...')
        le = LabelEncoder(); le.fit(L)  # concatenate in case l_train doesn't include all the labels
        assert n_classes == le.classes_.shape[0]
        print('    + unique labels (n=%d):\n%s\n' % (n_classes, ulabels))
        lc = labeling.count(L)
        print('    + class counts (all docs, n=%d | value: %s' % (nDoc, n_classes))
        print('    + labels:\n%s\n' % lc)
        return ulabels
    def test_params(): # cohort_name, ifiles, d2v_method, outputdir
        # cohort
        assert not tsHandler.cohort == '?' or kargs.has_key('ifiles'), "If cohort is not given, you must provide (doc) source paths."

        # root dirctory path
        outputdir = TSet.getPath(cohort=tsHandler.cohort, dir_type=None)  # ./data/<cohort>/
        assert os.path.exists(outputdir), "Invalid (training set) outputdir"
    def segment_docs(D, L, T, predicate=None, policy='regular', time_cutoff=None):  # <- D, L, T
        # policy: {'regular', 'two'/'halves', 'prior'/'posterior', 'complete', }
        # output: (DocIDs, D, L, T)

        # only consider the segment prior to time_cutoff (prior to applying eMERGE phenotyping algorithm)
        if time_cutoff is not None: 
            _, D, L, T = docProc.segmentDocumentByTime(D, L, T, timestamp=time_cutoff, policy='prior', inclusive=True)
        
        include_endpoint = kargs.get('include_endpoint', True)  # include the chunk containing diagnosis info?
        drop_docs_without_cutpoints = kargs.get('drop_nullcut', False)  # if True, do not include doc without valid cut points (diagnosis info)

        # configure file ID 
        # if predicate is None or policy is regular, then no-op
        if predicate is not None and not policy.startswith('reg'):
            # assert tsHandler.meta and not (tsHandler.meta in ('D', 'default', )), \
            #     "default user-defined file ID is not recommended since segmenting op will modify the documents (policy=%s)" % policy_segment
            print('segment_docs> predicate: %s, policy: %s' % (predicate, policy))
        else: 
            print('segment_docs> policy: %s' % policy)
        return docProc.segmentDocuments(D, L, T, predicate=predicate, policy=policy, 
                    inclusive=include_endpoint, drop_nullcut=drop_docs_without_cutpoints)  
    def is_sparse(): 
        if tsHandler.d2v.startswith(('bow', 'bag', 'aphri')):  # aphridate
            return True 
        return False 
    def sample_docs(D, L, T, n, sort_index=True, random_state=53, policy='longest'):
        # if n > len(y): n = None # noop
        idx = cutils.selectPerClass(X=D, y=L, n_per_class=n, verbose=True, sort_index=sort_index, random_state=random_state, policy=policy)
        D = list(np.array(D)[idx])
        L = list(np.array(L)[idx])
        T = list(np.array(T)[idx])

        n_classes = len(np.unique(L))
        n_per_class = n
        assert len(D) <= n * n_classes, "size(D): %d, n_classes: %d, n_per_class: %d" % (len(D), n_classes, n_per_class)
        print('  + subsampling: select only %d docs (policy=%s, n_per_class=%d, n_classes: %d)' % \
            (len(D), policy, n_per_class, n_classes))
        return (idx, D, L, T)
    def subsample(X, y, n, base_factor=2): 
        from sklearn.model_selection import train_test_split  # only used for subsampling training data (when it's too big)
        
        if X.shape[0] > n_samples * base_factor:  # if X is big enough   
            Xp, _, yp, _ = train_test_split(X, y, 
                train_size=n_samples, 
                # test_size=test_subset,   # det_test_size(), # accept test_size, ratios
                random_state=kargs.get('random_state', int(time.time())%1000), stratify=y)
        return (Xp, yp)
    def subsample2(X, y, n=None, sort_index=True, random_state=53):
        """
        Subsampling operation applied after doc vec X is derived. 
        """
        if n is not None: 
            idx = cutils.samplePerClass(y, n_per_class=n, sort_index=sort_index, random_state=random_state)
            print('(subsample2) X (dim0: %s -> dim: %s)' % (X.shape[0], len(idx)))
            return (X[idx], y[idx])
        return (X, y)
    def verify_docs(D, min_length=10, n_test=100): # args: min_ncodes 
        # 1. minimum length: 'min_ncodes'
        minDocLength = min_length
        if minDocLength is not None: 
            nD = len(D)
            ithdoc = random.sample(range(nD), min(nD, n_test))
            for r in ithdoc: 
                assert len(D[r]) >= minDocLength, "Length(D_%d)=%d < %d" % (r, len(D[r], minDocLength))

        # 2. other tests
        # 
        return  
    def doc_size_stats(D):
        nD = len(D)
        acc = 0 
        ll = []
        for i, d in enumerate(D):
            acc += len(d)
            ll.append(len(d))

        res = {}
        res['mean'] = El = np.mean(ll)
        res['median'] = Ml = np.median(ll)
        res['std'] = stdl = np.std(ll)
        return res
    def diagnosis_dim(D): 
        sizeStats = doc_size_stats(D)  # use this to estimate an appropriate window size
        print('makeTSetCombined> nDocTotal: %d =?= nDocEff: %d | doc stats: (m: %f, med:%f, std:%f), window: %d' % \
            (nDocTotal, nDocEff, sizeStats['mean'], sizeStats['median'], sizeStats['std'], vector.D2V.window)) # after all doc processing (segmentations, etc), nDoc may change
        if tSegmentVisit: 
            assert nDocTotal <= nDocEff, "summing total number of visits (%d) should exceed number of documents (%d)" % (nDocEff, nDocTotal)
            print('... nDocTotal: %d <? nDocEff: %d' % (nDocTotal, nDocEff))
        return
    def expand_by_visit(L, visitDocIDs): # same visit IDs belong to the same class
        Lp = []
        for vid in visitDocIDs:  # [0, 0, 1, 1, 1, 2, 2, ...]
            Lp.append(L[vid])
        return Lp
    def tokenize(D, vocab_size=None, test_level=0):  # limit vocab size (by discarding infreq tokens) and convert tokens to integers/indices
        tLimitVocab = False if vocab_size is None else True

        Dp = D
        if tLimitVocab: 
            print('(tokenize) limiting vocab size to %d and converting input to integer representation' % vocab_size)
            Dp = docProc.tokenizeDoc(D,  
                   test_level=test_level, throw_=True, customized=False, 
                   pad_empty_doc=True, 
                   int_to_str=True,  # convert integers/indices to strings as a proper input format for the gensim doc2vec framework
               
                   # tokenizer
                   num_words=vocab_size, split=' ', oov_token=TDoc.token_unknown)
        else: 
            pass # noop
        return Dp
        
    # import matplotlib.pyplot as plt
    from tset import TSet
    from tdoc import TDoc
    from labeling import TDocTag
    from pattern import medcode as pmed 
    from sklearn.preprocessing import LabelEncoder
    import docProc
    import classifier.utils as cutils
    # import vector

    ### configure all the parameters necessary for file naming, etc. 
    # config() # configure training set handler; params> cohort, seq_ptype, d2v_method, is_augmented? simplified_code?
    cohort_name = tsHandler.cohort
    d2v_method = tsHandler.d2v

    tset_descriptor = meta = kargs.get('meta', tsHandler.meta) 
    model_descriptor = kargs.get('meta_model', tsHandler.meta_model)

    include_augmented = kargs.get('include_augmented', False)

    # [input] cohort, labels, (result_set), d2v_method, (w2v_method), (test_model)
    #         simplify_code, filter_code  
    div(message='1. Read temporal doc files ...')

    # load + transform + (segment) + labelize (also ensure that labeled_seq exists in processDocuments())
    # D, D_augmented, L, T = assemble_docs()   # return [D, L, T, D2, L2, T2]? 

    ### load + transform + labelize   [params] min_ncodes
    # [note] or use load_mcs() to load from training data directory (e.g. sampled documents, segmented documents)
    D, L, T = process_docs(); verify_docs(D, min_length=kargs.get('min_ncodes', 10), n_test=100)  # [test]
    nClasses = len(np.unique(L))

    ### incorporate augmented (unlabeled) documents if available
    Da, La, Ta = [], [], []
    nDocAug = nDocTotal = 0 
    if include_augmented: 
        Da, La, Ta = process_aug_docs() # load + transform + labelize
        nDocAug = len(Da)
    if len(Da) > 0: D, L, T = D+Da, L+La, T+Ta 

    ### subsampling #1: large corpus while preserving the percentage of the class labels
    max_n_docs, n_docs_policy = kargs.get('max_n_docs', None), kargs.get('max_n_docs_policy', 'longest') 
    sampledIDs = []
    if max_n_docs: 
        sampledIDs, D, L, T = sample_docs(D, L, T, n=max_n_docs, policy=n_docs_policy) # set None to take ALL
        kargs['save_doc'] = True # definitely have to save the new documents
        # assert len(sampledIDs) <= max_n_docs * nClasses
        # docIDs below should be mapped to their absolute location IDs in the source    
    nDocTotal = nDoc = len(D)

    ### segment each document into paragraphs or smaller parts (e.g. prediagnosis sequence)
    # [note] DocIDs are used to associate segments to the original document (which may be broken down into several segments). 
    #        Under no-op, DocIDs are just regular positional indices.  
    func_segment, p_segment = kargs.get('predicate', None), kargs.get('policy_segment', 'regular')
    time_cutoff = kargs.get('time_cutoff', '2017-08-18')
    docIDs, D, L, T = segment_docs(D, L, T, predicate=func_segment, policy=p_segment, time_cutoff=time_cutoff)  # <- policy_segment, inclusive?, include_active?

    ### segment by visits? i.e. segment by time
    tSegmentVisit, visitDocIDs = kargs.get('segment_by_visit', False), []  # vDocIDs: visit-modulated document IDs
    if tSegmentVisit:  # segmenting by visits
        # need to treat each visit as a document/paragraph 
        assert not d2v_method.startswith(('bow', 'bag', )), "segment by visits does not apply in sparse models (e.g. bow)."

        # visitDocIDs: expanded document IDs e.g. [0, 0, 0, 1, 1, 2, 3, 3, 3, 3, ...] => 0th doc has 3 visits, 1th has 2 visits, etc.
        # [note] visitDocIDs to be used later in vector.consolidateVisits
        visitDocIDs, D = docProc.makeSessionDocuments(D, L, T,  min_length=kargs.get('min_visit_length', seqparams.D2V.window))
        assert len(D) == len(visitDocIDs) and len(visitDocIDs) >= len(L), \
             "size(D):%d, size(visitDocIDs): %d, sizes(L): %d" % (len(D), len(visitDocIDs), len(L))

    ### tokenization: remove infrequent tokens and make integer/index repr
    D = tokenize(D, vocab_size=kargs.get('num_words', 20001), test_level=0)  # set vocab_size to None to bypass this step
        
    # # labelize documents (required by gensim's d2v models)
    # if tSegmentVisit: 
    #     Dl = labelize(D, label_type='v')
    # else: 
    Dl = labelize(D, label_type='doc') # other params: class_labels=L, use class labels for document label prefixing

    ### [todo] separate the above to a separate function

    ### labeling
    ulabels = process_label()
    n_classes = len(ulabels)

    div(message='2. Compute document embedding (params: ) ...') # [note] use W2V and D2V classes in seqparams and vector to adjust d2v parameters
    # ratio_test = 0.2
    # print('    + train test split (test: %f%%) | cohort=%s, n_classes=%d' % (0.2*100, cohort_name, n_classes))  

    # cv = StratifiedKFold(n_splits=6) 
    ts = DataFrame()

    # input: D, labels, D_augmented
    ts_output = None
    Din = D if d2v_method.startswith(('bow', 'bag', )) else Dl  # d2v model requires labeled documents but, bag-of-words and other sparse models do not 

    nDocEff = len(Din)
    diagnosis_dim(Din)

    nTrials = kargs.get('n_trials', 1)
    for cv_id in range(nTrials): 
        # D_train, D_test, l_train, l_test = train_test_split(D, labels, test_size=ratio_test)

        # suppose we want classify final stage of a diseases  
        # [note] Can do the following externally via binarize()
        # positive_classes = ['CKD Stage 4', 'CKD Stage 5',]
        # y_train, y_test = transform_label(l_train), transform_label(l_test)  # e.g. binarize the labels
        print('    + computing document vectors (nD:%d + nDAug:%d -> nDTotal:%d) => nDEff: %d ...' % (nDoc, nDocAug, nDocTotal, nDocEff))

        # [note] Dl includes augmented if provided: Dl + Dal
        #        this will save a model file
        y = np.array(L)
        X = vector.getDocVec(docs=Din, d2v_method=tsHandler.d2v, 
                                outputdir=get_model_dir(),  # [params] cohort, dir_type='model' 
                                meta=model_descriptor,   # user-defined file ID: get_model_id(y=L), # {'U', 'L', 'A'} + ctype + cohort
                                labels=L,  # assess() will evaluate d2v accuracy based on the labels

                                test_=kargs.get('test_model', True), 
                                load_model=kargs.get('load_model', True),

                                segment_by_visit=tSegmentVisit, # need to distinguish from 'normal' document 

                                max_features=kargs.get('max_features', None), # only applies to d2v_method <- 'bow' (bag-of-words sparse model) 
                                text_matrix_mode=kargs.get('text_matrix_mode', 'tfidf'), # only applies to sparse matrix

                                cohort=tsHandler.cohort)

        if tSegmentVisit:  # segmenting by visits
            # Need to combine visit vectors so that each document maps to a vector
            # policy: i) average ii) concategate the last 10 visits

            # if kargs.get('nnet_mode', False):  # no need, because we can just reshape X
            #     # X in 3D
            #     X = consolidateVisitsLSTM(X, y=y, docIDs=visitDocIDs, last_n_visits=10)  # 3D input for LSTM units: (n_sample, n_timesteps, n_features)
            #     assert X.shape[0] * X.shape[1] == nDocEff, "n_samples (%d) * n_timesteps (%d) <> nDocEff: %d" % \
            #         (X.shape[0], X.shape[1], nDocEff)
            n_features, n_timesteps = X.shape[1], kargs.get('last_n_visits', 10)  # pv-dm2: vector.D2V.n_features * 2
            X, y = vector.consolidateVisits(X, y=y, docIDs=visitDocIDs, 
                            last_n_visits=n_timesteps, 
                            max_n_samples=kargs.get('max_n_samples', None))
            assert X.shape[0] == len(y)

            # to LSTM 
            # X = X.reshape((nDoc, lastN, fDim))
        else: 
            assert X.shape[0] == nDocEff and X.shape[0] == len(L)

        if not is_sparse(): 
            # condition: feature size doubled if using pv-dm2

            # lookuptb = ret['symbol_chart']
            
            ### Save training set
            # e.g. .../data/PTSD/tset_nC1_IDregular-sg-tfidfavg-GPTSD.csv
            # can also use 'tags' (in multilabel format)

            if kargs.get('save_', True):     
                # Output location: e.g. .../data/<cohort>/cv/
                # can also use 'tags' (in multilabel format)
                
                if tSegmentVisit: 
                    # L is visit-expanded 
                    assert X.shape[0] == len(L)

                    # need to convert X from 3D to 2D
                    # X = X.reshape((nDocEff, X.shape[2]))
                    Lp = expand_by_visit(L, visitDocIDs)
                    
                    # can be very large!
                    assert X.shape[0] == len(Lp)

                    # n_samples = kargs.get('max_n_samples', None)
                    # if n_samples: subsample(X, y=Lp, n=n_samples)

                    # same visit IDs belong to the same class
                    ts = save_tset(X, Lp, cv_id, docIDs=visitDocIDs, meta=tset_descriptor, shuffle_=False) 
                else: 
                    # [note] subsampling is being shifted to the consolidateVisits(); otherwise, won't work with large data (pad_sequences() => MemoryError)
                    if False: 
                        n_samples = kargs.get('max_n_samples', None)
                        if n_samples: subsample(X, y=L, n=n_samples)

                    # regular training set format
                    ts = save_tset(X, L, cv_id, docIDs=[], meta=tset_descriptor) # [params] X_train, X_test, y_train, y_test
                    # if cv_id == 0: ts_output = ts  # only save the resulted training data from one trial

        else: # sparse training set
            print('makeTSet> Sparse matrix (X): dim=%s, d2v=%s' % (str(X.shape), d2v_method))
            save_tset(X, y, cv_id, docIDs=docIDs, meta=tset_descriptor, sparse=True)

        print('status> Model computation complete (@nTrial=%d)' % cv_id)
        if kargs.get('save_doc', True if not tSegmentVisit else False): 

            # user_file_descriptor is tsHandler.meta by default
            tsDoc = save_mcs(D, T, L, index=cv_id, docIDs=docIDs, meta=tset_descriptor, sampledIDs=sampledIDs)  # D has included augmented if provided
            
            # [test]
            assert X.shape[0] == tsDoc.shape[0], "Size inconsistent: size(doc): %d but size(X): %d" % (tsDoc.shape[0], X.shape[0])

    ### end foreach trial 
    return (X, y)

### end makeTSetCombined

def makeTSetCV(**kargs): 
    """
    Transform input corpus to a vector representation in which each document/sequence 
    is converted into its vector form. This routine is meant for use in an iterative 
    loop in cross validation. 

    Input
    -----
    cohort
    labels 
    d2v_method, w2v_method 
    test_model
    simplify_code, filter_code

    Output
    ------
    ./data/<cohort>

    Reference
    ---------
    1. label encoding
       http://scikit-learn.org/stable/modules/preprocessing_targets.html#preprocessing-targets
 
    Related
    -------
    makeTSet()

    """
    def load_docs(): 
        ret = kargs.get('result_set', {}) # 'sequence', 'timestamp', 'label' 
        if not ret: 
            ifiles = kargs.get('ifiles', [])
            ret = sa.readDocFromCSV(cohort=cohort_name, inputdir=docSrcDir, ifiles=ifiles, complete=True) # [params] doctype (timed)
        # seqx = ret['sequence'] # must have sequence entry
        # tseqx = ret.get('timestamp', [])  
        return ret
    def transform_label(l, positive=None):
        if positive is None: positive = positive_classes # go to the outer environment to find its def
        y_ = labeling.binarize(l, positive=positive) # [params] negative
        return y_ 
    def save_to_csv(cv_id): # [params] (X_train, X_test, y_train, y_test)
        seq_ptype = seqparams.normalize_ctype(kargs.get('seq_ptype', 'regular')) # kargs.get('seq_ptype', 'regular')
        simplify_code = kargs.get('simplify_code', False)
        suffix = None # 'simplified' if simplify_code else None

        # can also use 'tags' (in multilabel format)
        print('    + saving training data (cv=%d) ...' % cv_id)  # e.g. tpheno/seqmaker/data/CKD/train/tset-IDregular-pv-dm2-GCKD.csv
        ts_train = TSet.toCSV(X_train, y=y_train, save_=True, dir_type='cv_train', index=cv_id, 
                         d2v_method=d2v_method, cohort=cohort_name, seq_ptype=seq_ptype, suffix=suffix) # [params] labels, if not given => all positive (1)
    
        print('    + saving test data (cv=%d) ...' % cv_id) # e.g. tpheno/seqmaker/data/CKD/test/tset-IDregular-pv-dm2-GCKD.csv
        ts_test = TSet.toCSV(X_test, y=y_test, save_=True, dir_type='cv_test', index=cv_id, 
                         d2v_method=d2v_method, cohort=cohort_name, seq_ptype=seq_ptype, suffix=suffix)
        return (ts_train, ts_test)  
    def transform_docs(): 
        seq_ptype = kargs.get('seq_ptype', 'regular')
        if seq_ptype == 'regular':
            # noop 
            return D, labels
        # [params] items, policy='empty_doc', predicate=None, simplify_code=False
        predicate = kargs.get('predicate', None)
        simplify_code = kargs.get('simplify_code', False)
        D, labels, T = transformDocuments2(D, labels=labels, items=T, policy='empty_doc', seq_ptype=seq_ptype, 
            predicate=predicate, simplify_code=simplify_code, save_=True)
        # D, labels = transformDocuments(D, labels=labels, seq_ptype=seq_ptype)
        print('    + (after transform) nDoc: %d -> %d, size(D0): %d -> %d' %  (nD, len(D), nD0, len(D[0])))

        # need to save a new copy of document sources if its size becomes different (e.g. diagnostic-code-only doc is probably smaller)
        return (D, labels)  

    # import matplotlib.pyplot as plt
    from tset import TSet
    from labeling import TDocTag
    from pattern import medcode as pmed 
    from sklearn.preprocessing import LabelEncoder
    from sklearn.model_selection import StratifiedKFold
    # import vector

    cohort_name = kargs.get('cohort', '?')
    assert not cohort_name == '?' or kargs.has_key('ifiles'), "If cohort is not given, you must provide (doc) source paths."
    
    # output directory for the d2v model 
    outputdir = TSet.getPath(cohort=cohort_name, dir_type=None)  # ./data/<cohort>/<dir_type> if dir_type <- None then just ./data/<cohort>      
    div(message='1. Read temporal doc files ...')

    # D, labels = tdoc.TDoc.getDY()
    # load + transfomr + (ensure that labeled_seq exists)
    D, L, T = processDocuments(cohort=cohort_name, seq_ptype=kargs.get('seq_ptype', 'regular'),
                predicate=kargs.get('predicate', None), simplify_code=kargs.get('simplify_code', False))  # [params] composition
    nDoc = len(D)
    ulabels = np.unique(labels)
    n_classes = ulabels.shape[0]
    print('    + unique labels:\n%s\n' % ulabels)

    div(message='2. Compute document embedding (params: ) ...') # [note] use W2V and D2V classes in seqparams and vector to adjust d2v parameters
    d2v_method = kargs.get('d2v_method', vector.D2V.d2v_method)
    ratio_test = 0.2
    print('    + train test split (test: %f%%) | cohort=%s, n_classes=%d' % (0.2*100, cohort_name, n_classes))  
    
    nTrials = kargs.get('n_trials', 1)
    nFolds = kargs.get('n_folds', 5)

    assert len(D) == len(labels)
    cvIter = StratifiedKFold(labels, n_folds=nFolds, shuffle=False, random_state=None)
    
    # ts = DataFrame()
    le = LabelEncoder()

    # a. train test split
    # for ii in range(nTrials): 
    #     D_train, D_test, l_train, l_test = train_test_split(D, labels, test_size=ratio_test)

    # b. CV-based train test split
    for cv_id, (train, test) in enumerate(cvIter.split(D, labels)):
        D_train, D_test, l_train, l_test = np.array(D)[train], np.array(D)[test], np.array(labels)[train], np.array(labels)[test]
        print('   + after train-test split: type(D_train[i], D_train)=(%s, %s) example: %s' % (type(D_train[0]), type(D_train), D_train[0]))

        n_classes_train, n_classes_test = np.unique(l_train).shape[0], np.unique(l_test).shape[0]

        D_augmented = kargs.get('augmented_docs', [])

        # [test] status: ok 
        nD_train, nD_test, nD_total = len(D_train), len(D_train), len(D); r = nD_test/(nD_total+0.0)
        print('    ++ train test split (r_test: %f=?=%f) | n_classes_train: %d, n_classes_test: %d' % (r, ratio_test, n_classes_train, n_classes_test))

        ### binarize label if needed (this can be cohort dependent)
        div('2a. Label processing (e.g. binarize labels) ... ')

        print('    + 2a-1 > class labeling ...')
        le.fit(np.concatenate([l_train, l_test, ]))  # concatenate in case l_train doesn't include all the labels
        assert n_classes == le.classes_.shape[0]

        # suppose we want classify final stage of a diseases  
        # [note] Can do the following externally via binarize()
        # positive_classes = ['CKD Stage 4', 'CKD Stage 5',]
        # y_train, y_test = transform_label(l_train), transform_label(l_test)  # e.g. binarize the labels

        y_train, y_test = l_train, l_test 
        lc_train, lc_test = labeling.count(y_train), labeling.count(y_test)
        print('    + class counts (train) | value: %s' % lc_train)
        print('    + class counts (test) | value: %s' % lc_test)
    
        # document labeling
        print('    + 2a-2 > document labeling ...')
        D_train, D_test = labelize(D_train, label_type='train', class_labels=y_train), labelize(D_test, label_type='test', class_labels=y_test)
        if len(D_augmented) > 0: 
            D_augmented = labelize(D_augmented, label_type='augmented') # create document labels (NOT class labels)
            # D_train = np.concatenate((D_train, D_augmented))
            D_train = D_train + D_augmented

        # want documents to be in list type from here on 
        assert isinstance(D_train, list) and isinstance(D_test, list)
        
        # [params] debug: cohort 
        print('    + computing document vectors ...')
        X_train, X_test = vector.getDocVec2(D_train, D_test, 
                                               d2v_method=d2v_method, 
                                               outputdir=outputdir, 
                                               model_id=kargs.get('model_id', None), 
                                               test_=kargs.get('test_model', True), 
                                               load_model=kargs.get('load_model', True),
                                               cohort=cohort_name)

        # [test] a. feature size doubled 
        assert X_train.shape[0] == len(D_train), "X_train.shape[0]: %d != nD_train: %d" % (X_train.shape[0], len(D_train))
        assert X_train.shape[1] == (2 * vector.D2V.n_features), \
            "X_train: (%d by %d), n_features: %d" % (X_train.shape[0], X_train.shape[1], vector.D2V.n_features)
        assert X_test.shape[0] == len(D_test), "X_test.shape[0]: %d != nD_test: %d" % (X_test.shape[0], len(D_test))
        assert X_test.shape[1] == (2 * vector.D2V.n_features), "X_test: %d by %d" % (X_test.shape[0], X_test.shape[1])

        if kargs.get('save_', False):     
            div(message='3. Save training set')
            # Output location: e.g. .../data/<cohort>/cv/
            # can also use 'tags' (in multilabel format)
            ts_train, ts_test = save_to_csv(cv_id) # [params] X_train, X_test, y_train, y_test

        print('status> Model computation complete.')
        yield (X_train, X_test, y_train, y_test)

    ### end foreach trial 

def multiClassify(X, y, X_new=None):
    # if X_new is present, then make predictions 

    random_state = np.random.RandomState(0)
    classifier = OneVsRestClassifier(svm.SVC(kernel='linear', probability=True,
                                 random_state=random_state))
    y_score = classifier.fit(X_train, y_train).decision_function(X_test)

    y_new = None
    if X_new is not None: 
        y_new = classifier.predict(X_new)

    return y_new

def multiClassEvalTrainTestSplit(X_train, X_test, y_train, y_test, **kargs): # [params] classifier
    # Learn to predict each class against the other
    # tEvaluate = True  # if False, predict new 
    # from sklearn import svm
    random_state = np.random.RandomState(0)
    
    classifier = kargs.get('classifier', None)
    if classifier is None: 
        classifier = OneVsRestClassifier(svm.SVC(kernel='linear', probability=True,
                                 random_state=random_state))
    y_score = classifier.fit(X_train, y_train).decision_function(X_test)

    n_classes = np.unique(y_train).shape[0]
    n_classes_test = np.unique(y_test).shape[0]
    print('multiClassEaluate> n_classes in training: %d =?= test: %d' % (n_classes, n_classes_test))

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    plotROCMultiClass(fpr, tpr, roc_auc, n_classes=n_classes, 
        identifier=kargs.get('identifier', 'generic'), outputdir=kargs.get('outputdir', os.getcwd()))

    return    

def loadTSetXY(**kargs):
    """
    load training and test data into matrix form i.e. (X_train, X_test, y_train, y_test)
    """ 
    # ts_train, ts_test = loadTSetSplit(**kargs)
    
    cohort_name = kargs.get('cohort', None) 
    d2v_method = kargs.get('d2v_method', vector.D2V.d2v_method)
    X_train, y_train = TSet.getXY(cohort=cohort_name, d2v_method=d2v_method, dir_type='train') # cohort <- None will take on a default value (generic)
    X_test, y_test = TSet.getXY(cohort=cohort_name, d2v_method=d2v_method, dir_type='test')

    return (X_train, X_test, y_train, y_test)

def loadTSetSplit(**kargs): 
    """
    Load training data (resulted from applying a doc2vec model)

    Params
    ------
    dir_type: 
       {train, test}
       combined 
       {cv_train, cv_test}
    index: an integer in the file name used to distinguish data from different CV partitions. 

    w2v_method, d2v_method
    seq_ptype
    read_mode
    cohort

    """
    def check_tset(): 
        target_field = TSet.target_field
        n_classes = 1 
        for ts in [ts_train, ts_test, ]: 
            if ts is not None and not ts.empty: 
                # target_field in ts.columns # o.w. assuming that only 1 class 
                n_classes = len(ts[target_field].unique())  
                print('loadTSet> number of classes: %d' % n_classes)
            else:
                msg = 'loadTSet> Warning: No data found (cohort=ts)' % cohort_name
                if no_throw: 
                    print msg 
                else: 
                    raise ValueError, msg
        return n_classes 
    def profile(ts): 
        col_target = TSet.target_field
        all_labels = ts[col_target].unique()
        sizes = {}
        for label in all_labels: 
            sizes[label] = ts.loc[ts[col_target] == label].shape[0]
        print('profile> Found %d unique labels ...' % len(all_labels))
        for label, n in sizes.items(): 
            print('  + label=%s => N=%d' % (label, n))
        return       
    def get_tset_id(ts=None): # L <-
        return _TSetHandler.get_tset_id(ts)

    # import vector
    from tset import TSet  

    # w2v_method = kargs.get('w2v_method', vector.D2V.w2v_method) 
    d2v_method = kargs.get('d2v_method', vector.D2V.d2v_method)

    cohort_name = kargs['cohort'] # System.cohort
    seq_ptype = kargs.get('seq_ptype', 'regular') # naming
    no_throw = kargs.get('no_throw', True)

    # sequence content: mixed, diag only? med only? 
    # [note] training data by default are derived from the coding sequences consisting of diag and med codes
    # seq_ptype = 'regular' # seqparams.normalize_ctype(**kargs) # sequence pattern type: regular, random, diag, med, lab

    ts_train = TSet.load(cohort=cohort_name, d2v_method=d2v_method, seq_ptype=seq_ptype,
                            dir_type='train', index=0) # [params] index
    ts_test = TSet.load(cohort=cohort_name, d2v_method=d2v_method, seq_ptype=seq_ptype,
                            dir_type='test', index=0)
    n_classes = check_tset()

    # subsetting 
    # subset_idx = kargs.get('subset_idx', None) # seqparams.arg(['idx', 'subset_idx'], None, **kargs)
    # if subset_idx is not None: 
    #     assert hasattr(subset_idx, '__iter__')
    #     print('loadTSet> Only return a subset (n=%d) out of total %d' % (len(subset_idx), ts.shape[0]))
    #     return ts.loc[ts[TSet.index_field].isin(subset_idx)]
    
    return (ts_train, ts_test) # dataframe or None (if no throw)

def sampleTSet(**kargs):
    """
    Load training set in (X, y) format + subsampling. 

    **kargs
    -------
    precomputed_subset: if True, load the precomputed subset sampled earlier from disk

    Memo
    ----
    1. In parallel, pathAnalyzer has a sampleDSet(**kargs) method. 

    """
    from tset import TSet
    from sampler import sampling

    ### precomputed
    if kargs.get('precomputed_subset', True): 
        print('loadTSet> Re-use precomputed sample.')
        ts = tsHandler.load_tset(index=0, meta='S', sparse=False) 
        if ts is not None and not ts.empty: 
            Xs, ys = TSet.toXY(ts)
            return (Xs, ys) 

    ### load + subsample the desire subset
    ts = loadTSet(**kargs)  # load the entired training data (computed from d2v)
    
    ### ts -> (X, y)
    X, y = TSet.toXY(ts)
    tsHandler.profile2(X, y)
    assert X is not None

    ratio = kargs.get('ratio', 0.5)  
    tSaveTSetSubset = kargs.get('save_tset', True) # save sample subset by default (reuse)

    Xs, ys = X, y
    if ratio < 1.0: 
        n0 = X.shape[0]
        ridx = sampling.splitDataPerClass(y, ratios=[ratio, ])

        Xs, ys = X[ridx[0]], y[ridx[0]]   # training; X_test, y_test is included within (X, y)

        ridx = None
        n1 = Xs.shape[0]
        print('sampleTSet> sample size %d => %d | r=%f' % (n0, n1, ratio))

        if tSaveTSetSubset: 
            # need to save docIDs in order for pathway analysis (pathAnalyzer) to be consistent
            tsHandler.save_tset(Xs, ys, index=0, docIDs=ridx[0], meta='S', sparse=False, shuffle_=True)

    return (Xs, ys)

def loadTSetCombined(**kargs): # used only for backward compatibility; combined: combining data splits
    return loadTSet(**kargs)  
def loadTSet(**kargs): 
    """
    Load training data. Similar to loadTSetSplit() but assume no separation between train and test splits. 

    Params
    ------
    dir_type: 
       {train, test}
       combined 
       {cv_train, cv_test}
    index: an integer in the file name used to distinguish data from different CV partitions. 

    w2v_method, d2v_method
    seq_ptype
    read_mode
    cohort

    Related
    -------
    1. see _TSetHandler

    """
    def config(): # use sysConfig() instead
        tsHandler.config(cohort=kargs.get('cohort', 'CKD'), 
            seq_ptype=kargs.get('seq_ptype', 'regular'), 
            d2v_method=kargs.get('d2v_method', vector.D2V.d2v_method), 
            is_augmented=kargs.get('is_augmented', False), 

            meta=kargs.get('meta', None), 
            simplify_code=kargs.get('simplify_code', False), dir_type='combined')
        return 
    def check_tset(ts): 
        target_field = TSet.target_field
        n_classes = 1 
        no_throw = kargs.get('no_throw', True)
        if ts is not None and not ts.empty: 
            # target_field in ts.columns # o.w. assuming that only 1 class 
            n_classes = len(ts[target_field].unique())  
            print('loadTSet> number of classes: %d' % n_classes)
        else:
            msg = 'loadTSet> Warning: No data found (cohort=%s)' % kargs.get('cohort', 'CKD')
            if no_throw: 
                print msg 
            else: 
                raise ValueError, msg
        return n_classes      
    def get_tset_id(ts=None): # L <-
        return tsHandler.get_tset_id(ts) 
    def profile(ts): 
        return tsHandler.profile(ts)   
    def get_tset_dir(): 
        return tsHandler.get_tset_dir()
    def customize_tset(ts): 
        focused_classes = kargs.get('focused_labels', None) # None: N/A
        if focused_classes is not None: # only preserve these classes and everything else becomes "Control"
            # e.g. In CKD data, one may only want to focus on predicting ‘Stage 1’, ‘Stage 2’, ‘Stage 3a’, ... 'Stage 5'
            ts = focus(ts, labels=focused_classes, other_label='Control')
        lmap = kargs.get('label_map', seqparams.System.label_map)
        if lmap: # relabel so that classes only retain the main CKD stages
            print('(customize_tset) Modifying training data ...\n> Prior to re-labeling ...')
            profile(ts)
            ts = merge(ts, lmap)
            print('... After re-labeling ...')
            profile(ts)
        return ts
    def remove_classes(ts, labels=[], other_label='Control'):
        exclude_set = labels if len(labels)>0 else [other_label, ]
        col_target = TSet.target_field
        N0 = ts.shape[0]
        ts = ts.loc[~ts[col_target].isin(exclude_set)]    # isin([other_label, ])
        print('... remove labels: %s > size(ts): %d -> %d' % (labels, N0, ts.shape[0]))         
        return ts
    def subsample(ts, n=None, random_state=53):
        # maxNPerClass = kargs.get('n_per_class', None)
        n_per_class = n
        if n_per_class: # 0 or None => noop
            ts = sampling.sampleDataframe(ts, col=TSet.target_field, n_per_class=n_per_class, random_state=random_state)
            n_classes_prime = check_tset(ts)
            print('... after subsampling, size(ts)=%d, n_classes=%d (same?)' % (ts.shape[0], n_classes_prime))
        else: 
            # noop 
            pass 
        return ts

    # import vector
    from tset import TSet  
    from sampler import sampling
    # from sklearn.preprocessing import StandardScaler, MaxAbsScaler # feature scaling for sparse repr. 

    # w2v_method = kargs.get('w2v_method', vector.D2V.w2v_method) 
    
    # parameters for tset identifier
    # config()
    # tsHandler.dir_type = 'combined'
    # sequence content: mixed, diag only? med only? 
    # [note] training data by default are derived from the coding sequences consisting of diag and med codes
    # seq_ptype = 'regular' # seqparams.normalize_ctype(**kargs) # sequence pattern type: regular, random, diag, med, lab

    # [params] loading training data
    #     index: used to distinguish among multiple training and test sets
    #     suffix: served as secondary ID (e.g. training data made from sequences of different contents)
    #     meta: user-defined file ID (default='D')
    userFileID = kargs.get('meta', tsHandler.meta)
    print('loadTest> meta (userFileID): %s =?= %s (default), meta_model: %s | prior to tsHandler.load' % (userFileID, tsHandler.meta, tsHandler.meta_model))
    ts = tsHandler.load(index=kargs.get('index', 0), meta=userFileID) 
    
    ### scaling is unnecessary for d2v
    if kargs.get('scale_', False): 
        pass
    #     # scaler = StandardScaler(with_mean=False)
    #     scaler = MaxAbsScaler()
    #     X = scaler.fit_transform(X)

    ### modify classes
    ts = customize_tset(ts)  # <- label_map, focused_labels
    n_classes0 = check_tset(ts)

    # drop explicit control data in multiclass
    if kargs.get('drop_ctrl', False): # drop control group (e.g. label: 'Control')
        assert n_classes0 > 2, "Not a multiclass domain (n_classes=%d)" % n_classes0
        ts = remove_classes(ts, labels=['Control', ])

    maxNPerClass = kargs.get('n_per_class', None) 
    if maxNPerClass is not None: 
        assert maxNPerClass > 1
        # prior to samplig, need to convert to canonical labels first 
        ts = subsample(ts, n=maxNPerClass)

    return ts # dataframe or None (if no throw)

def loadLCSFeatureTSet(**kargs): 
    def profile(X, y): 
        n_classes = len(np.unique(y))
        no_throw = kargs.get('no_throw', True)

        print('loadLCSFeatureTSet> X (dim: %s), y (dim: %d)' % (str(X.shape), len(y)))
        print("      + number of store values (X): %d" % X.nnz)
        print('      + number of classes: %d' % n_classes)

        # assert X.shape[0] == len(y)
        return 
    def customize_tset(X, y): 
        focused_classes = kargs.get('focused_labels', None) # None: N/A
        if focused_classes is not None: # only preserve these classes and everything else becomes "Control"
            # e.g. In CKD data, one may only want to focus on predicting ‘Stage 1’, ‘Stage 2’, ‘Stage 3a’, ... 'Stage 5'
            y = focusLabels(y, labels=focused_classes, other_label='Control')  # output in np.ndarray

        lmap = kargs.get('label_map', seqparams.System.label_map)
        if lmap: 
            profile(X, y)
            y = mergeLabels(y, lmap=lmap)
            print('> After re-labeling ...')
            profile(X, y)
        return (X, y)

    import pathAnalyzer as pa 

    X, y = pa.loadLCSFeatureTSet(**kargs)

    return (X, y)

def loadSparseTSet(**kargs): 
    def profile(X, y): 
        n_classes = len(np.unique(y))
        no_throw = kargs.get('no_throw', True)

        print('loadSparseTSet> X (dim: %s), y (dim: %d)' % (str(X.shape), len(y)))
        print("                + number of store values (X): %d" % X.nnz)
        print('                + number of classes: %d' % n_classes)

        # assert X.shape[0] == len(y)
        return    
    def get_tset_id(ts=None): # L <-
        return tsHandler.get_tset_id(ts)  
    def customize_tset(X, y): 
        focused_classes = kargs.get('focused_labels', None) # None: N/A
        if focused_classes is not None: # only preserve these classes and everything else becomes "Control"
            # e.g. In CKD data, one may only want to focus on predicting ‘Stage 1’, ‘Stage 2’, ‘Stage 3a’, ... 'Stage 5'
            y = focusLabels(y, labels=focused_classes, other_label='Control')  # output in np.ndarray

        lmap = kargs.get('label_map', seqparams.System.label_map)
        if lmap: 
            profile(X, y)
            y = mergeLabels(y, lmap=lmap)
            print('> After re-labeling ...')
            profile(X, y)
        return (X, y)
    def remove_classes(X, y, labels=[], other_label='Control'):
        exclude_set = labels if len(labels)>0 else [other_label, ]
        
        N0 = len(y)
        ys = Series(y)
        cond_pos = ~ys.isin(exclude_set)

        idx = ys.loc[cond_pos].index.values
        y = ys.loc[cond_pos].values 
        X = X[idx]

        print('... remove labels: %s > size(ts): %d -> %d' % (labels, N0, X.shape[0]))         
        return (X, y)
    def get_tset_dir(): 
        return tsHandler.get_tset_dir()
    def subsample(X, y, n=None, sort_index=False, random_state=53):
        """
        Subsampling operation applied after doc vec X is derived. 
        """
        if n is not None: 
            idx = cutils.samplePerClass(y, n_per_class=n, sort_index=sort_index, random_state=random_state)
            print('  + X (dim0: %s -> dim: %s)' % (X.shape[0], len(idx)))
            return (X[idx], y[idx])
        return (X, y)

    from sklearn.preprocessing import StandardScaler, MaxAbsScaler # feature scaling for sparse repr.
    import classifier.utils as cutils

    # [note] use TSet.loadSparse0(...) to include docIDs with a return value (X, y, docIDs)
    #        example: tset-IDregular-bow-regular-GCKD.npz
    X, y = tsHandler.loadSparse(index=kargs.get('index', None), meta=kargs.get('meta', None)) 
    
    ### scaling X
    if kargs.get('scale_', False): 
        # scaler = StandardScaler(with_mean=False)
        scaler = MaxAbsScaler()
        X = scaler.fit_transform(X)

    ### modify classes
    if X is not None and y is not None: 
        X, y = customize_tset(X, y)

    # drop explicit control data in multiclass
    n_classes0 = len(np.unique(y)) 
    if kargs.get('drop_ctrl', False): # drop control group (e.g. label: 'Control')
        assert n_classes0 > 2, "Not a multiclass domain (n_classes=%d)" % n_classes0
        X, y = remove_classes(X, y, labels=['Control', ])

    ### subsampling 
    maxNPerClass = kargs.get('n_per_class', None)
    if maxNPerClass:
        assert maxNPerClass > 1
        y = np.array(y)
        X, y = subsample(X, y, n=maxNPerClass)

    return (X, y)

def loadTSetCV(**kargs):
    """
    Load training data. 

    Params
    ------
    dir_type: 
       {train, test}
       combined 
       {cv_train, cv_test}
    index: an integer in the file name used to distinguish data from different CV partitions. 

    w2v_method, d2v_method
    seq_ptype
    read_mode
    cohort

    """
    def check_tset(): 
        target_field = TSet.target_field
        n_classes = 1 
        for ts in [ts_train, ts_test, ]: 
            if ts is not None and not ts.empty: 
                # target_field in ts.columns # o.w. assuming that only 1 class 
                n_classes = len(ts[target_field].unique())  
                print('loadTSet> number of classes: %d' % n_classes)
            else:
                msg = 'loadTSet> Warning: No data found (cohort=ts)' % cohort_name
                if no_throw: 
                    print msg 
                else: 
                    raise ValueError, msg
        return n_classes 

    # import vector
    from tset import TSet  

    w2v_method = kargs.get('w2v_method', vector.D2V.w2v_method) 
    d2v_method = kargs.get('d2v_method', vector.D2V.d2v_method)

    cohort_name = kargs['cohort'] # System.cohort
    seq_ptype = kargs.get('seq_ptype', 'regular')  # content type affects training set, should be included in naming
    no_throw = kargs.get('no_throw', True)
    n_folds = kargs.get('n_folds', 5)

    # sequence content: mixed, diag only? med only? 
    # [note] training data by default are derived from the coding sequences consisting of diag and med codes
    # seq_ptype = 'regular' # seqparams.normalize_ctype(**kargs) # sequence pattern type: regular, random, diag, med, lab
    
    for cv_id in range(n_folds): 
        ts_train = TSet.load(cohort=cohort_name, d2v_method=d2v_method, seq_ptype=seq_ptype,
                                dir_type='cv_train', index=cv_id) # [params] index
        ts_test = TSet.load(cohort=cohort_name, d2v_method=d2v_method, seq_ptype=seq_ptype,
                                dir_type='cv_test', index=cv_id)
        n_classes = check_tset()
        yield (ts_train, ts_test) # dataframe or None (if no throw)

def multiClassEvaluateSparse(**kargs):
    def resolve_classifier(X, y): # [params] classifier, classifier_name
        ### method 1: externally specify a classifier (model parameters configured)
        ###        2: specfiy the name of the classifier (but have to be identifiable by modelSelect module)
        classifier = kargs.get('classifier', None) # condition: if a classifier is passed, assume that hyperparams are tuned
        clf_name = kargs.get('classifier_name', None) 
        if clf_name: 
            # choose by name > model selection
            try: 
                print('status> chooose estimator + model selection ...')
                classifier = ms.selectOptimalEstimator(clf_name, X, y, is_multiclass=True, max_n_samples=5000) # select classifier by name + model selection
            except Exception, e: 
                print('status> Error:\n%s\nstatus> unresolved classifier: %s' % (e, clf_name))
                raise ValueError, "Classifier %s is not available" % clf_name
        else:  # in multi-class case, perhaps this is more preferable
            assert classifier is not None
            clf_name = get_clf_name()
            print('status> classifier:\n%s\n' % clf_name)

            # condition: assuming that the input classifier supports multiclass 
            param_grid = kargs.get('param_grid', None)
            if param_grid is not None: # model selection 
                n_folds = kargs.get('n_folds', 5)
                metric = kargs.get('scoring', 'roc_auc')
                classifier = ms.selectModel(X, y, estimator=classifier, param_grid=param_grid, n_folds=n_folds, scoring=metric) 
            else: 
                print('status> assuming that the input classifier already has optimial hyperparam setting ...')
                # noop

        assert classifier is not None, "Could not choose classifier."
        return (classifier, clf_name)  # classifier with hyperparams optimized ~ (X, y)
    def get_clf_name(): # can be called before or after model selection (<- select_classifier)
        classifier = kargs.get('classifier', None)
        if classifier is not None: 
            try: 
                name = classifier.__name__
            except: 
                print('info> infer classifier name from class name ...')
                # name = str(estimator).split('(')[0]
                name = classifier.__class__.__name__
        else: 
            name = kargs.get('classifier_name', None) 
            assert name is not None
        return name
    def save_tset(X, y, cv_id=0, docIDs=[], meta=None): # [params] (X_train, X_test, y_train, y_test)
        ### tsHandler has to be configured first, use tsHandler.config()
        return tsHandler.save(X, y, index=cv_id, docIDs=docIDs, meta=meta)
    def load_tset(cv_id=0, meta=None):  # cohort, seq_ptype, d2v_method (focused_labels, label_map)
        ### tsHandler has to be configured first, use tsHandler.config()
        # this should conform to the protocal in makeTSetCombined()

        if kargs.has_key('X') and kargs.has_key('y'): 
            X, y = kargs['X'], kargs['y']
            assert isinstance(y, np.ndarray), "type(y): %s" % type(y)
            print('multiClassEvaluateSparse> X (dim: %s)' % str(X.shape))
        else: 
            # this is highly unlikely 
            print('multiClassEvaluateSparse> Warning: this is unlikely, csv format would be unnecessarily large for a sparse training set!')
            ts = kargs.get('ts', None)
            if ts is None: 
                print('  + automatically loading training set (cohort=%s, d2v=%s, ctype=%s)' % \
                         (tsHandler.cohort, tsHandler.d2v, tsHandler.ctype))
                # config()
                ts = tsHandler.load(cv_id, meta=meta)  # opt: focus_classes, label_map
                ts = customize_tset(ts)  
                assert ts is not None and not ts.empty, "multiClassEvaluate> Null training set."
                profile(ts)
            # allow loadTSet-like operations to take care of this
            X, y = TSet.toXY(ts)

        # show the training set profile (class label vs size)
        tsHandler.profile2(X, y)
        return (X, y)
    def validate_labels(): # targets, y
        assert set(targets).issubset(np.unique(y)), "targets contain unknown labels:\n%s\n" % targets
        return
    def make_file_id(): # [classifier_name, cohort, d2v_method, seq_ptype, suffix]
        identifier = kargs.get('identifier', None)
        if identifier is None: 
            cohort_name = tsHandler.cohort
            d2v_method = tsHandler.d2v  # vector.D2V.d2v_method
            ctype = tsHandler.ctype
            identifier = seqparams.makeID(params=[get_clf_name(), cohort_name, d2v_method, ctype, 
                            kargs.get('meta', tsHandler.meta)])  # null characters and None will not be included
        return identifier
    def roc_cv(X, y, classifier, fpath=None, target_labels=[]):    
        identifier = make_file_id()
        outputdir = Graphic.getPath(cohort=tsHandler.cohort, dir_type='plot', create_dir=True)
        # evaluation
        # ms.runCVROC(X, y, classifier=classifier, fpath=fpath)   # remember to call plt.clf() for each call (o.w. plots overlap)
        res = ms.runCVROCMulticlass(X, y, classifier=classifier, prefix=outputdir, identifier=identifier, target_labels=target_labels)
        return res

    import modelSelect as ms
    # from tset import TSet  # base class is defined in seqparams
    from seqparams import Graphic
    from tset import TSet

    random_state = np.random.RandomState(0)
    X, y = load_tset(meta=kargs.get('meta', None))  # precedence (X, y), ts, auto 

    # condition: classifier needs to support predict_proba() method
    classifier, classifier_name = resolve_classifier(X, y)  # [params] classifier (assumed tuned) | classifier_name (=> model selection), param_grid
    
    # roc_per_class: To prevent the figure from being clobbered, select only, say 3, classes to present in the ROC curve
    targets = kargs.get('roc_per_class', ['CKD Stage 1', 'CKD Stage 3', 'CKD Stage 5', ])  # assuming that CKD Stage 3a, 3b are merged
    validate_labels()
    res = roc_cv(X, y, classifier=classifier, target_labels=targets) 

    return res # resturn max performance and min performance; key: 'min': label -> auc, 'max': label -> auc

def modelEvaluateBatch(X, y, **kargs):
    """
    Wrapper of modelSelect.modelEvaluateBatch() with customized input directory and file name. 

    """
    def resolve_model(): 
        classifier = None
        for opt in ['model', 'classifier', ]: 
            if kargs.has_key(opt):
                classifier = kargs[opt]
                break 
        assert classifier is not None, "No classifier provided."
        return classifier 
    def get_clf_name(): # can be called before or after model selection (<- select_classifier)
        classifier = resolve_model()
        if classifier is not None: 
            try: 
                name = classifier.__name__
            except: 
                print('info> infer classifier name from class name ...')
                # name = str(estimator).split('(')[0]
                name = classifier.__class__.__name__
        else: 
            name = kargs.get('classifier_name', None) 
            assert name is not None
        return name
    def make_file_id(): # [classifier_name, cohort, d2v_method, seq_ptype, meta]
        identifier = kargs.get('identifier', None)
        if identifier is None: 
            cohort_name = tsHandler.cohort
            d2v_method = tsHandler.d2v  # vector.D2V.d2v_method
            ctype = tsHandler.ctype
            identifier = seqparams.makeID(params=[get_clf_name(), cohort_name, d2v_method, ctype, 
                            kargs.get('meta', tsHandler.meta)])  # null characters and None will not be included
        return identifier
    def get_target_labels(): # targets, y
        # To prevent the figure from being clobbered, select only, say 3, classes to present in the ROC curve
        targets = kargs.get('roc_per_class', ['CKD Stage 1', 'CKD Stage 3', 'CKD Stage 5', ])  # assuming that CKD Stage 3a, 3b are merged
        assert set(targets).issubset(np.unique(y)), "targets contain unknown labels:\n%s\n" % targets
        return targets

    import modelSelect as ms
    from seqparams import Graphic
    from tset import TSet

    random_state = np.random.RandomState(0)
    userFileID = make_file_id()  # kargs: meta 
    outputdir = Graphic.getPath(cohort=tsHandler.cohort, dir_type='plot', create_dir=True)

    targets = get_target_labels() # kargs: roc_per_class
    tCV = True if kargs.get('evaluation', 'cv').startswith('c') else False  # 'cv', 'split'  

    model = resolve_model()  # assuming model selection is completed
    res = ms.modelEvaluateBatch(X, y, 
            classifier=model, 
            n_folds=kargs.get('n_folds', 5), 
            n_trials=kargs.get('n_trials', 5), 
            use_cv=tCV, 
            
            # ROC: specifiy classes on the plot 
            plot_selected_classes=False, # set to True to avoid cluttering; if False, target_labels will be ignored 
            target_labels=targets, 

            # plots
            outputdir=outputdir, 
            identifier=userFileID) # use meta or identifier to distinguish different classificaiton tasks; set to None to use default

    return res
 
def multiClassEvaluate(**kargs): 
    """

    Input
    -----
    ts: training set (optional)
    classifier: 

    classifier_name: 
    
    cohort
    seq_ptype 
    d2v_method

    

    Memo
    ----
    1. cohort=CKD 
       classes: ['CKD G1-control' 'CKD G1A1-control' 'CKD Stage 1' 'CKD Stage 2'
        'CKD Stage 3a' 'CKD Stage 3b' 'CKD Stage 4' 'CKD Stage 5'
        'ESRD after transplant' 'ESRD on dialysis' 'Unknown']


    Evaluation methods for multiclass: 
    
    cohen_kappa_score(y1, y2[, labels, weights, …]) Cohen’s kappa: a statistic that measures inter-annotator agreement.
    confusion_matrix(y_true, y_pred[, labels, …])   Compute confusion matrix to evaluate the accuracy of a classification
    hinge_loss(y_true, pred_decision[, labels, …])  Average hinge loss (non-regularized)
    matthews_corrcoef(y_true, y_pred[, …])  Compute the Matthews correlation coefficient (MCC)

    """
    def to_numeric(labels, to_binary_rep=False):
        le = LabelEncoder()
        le.fit(labels)  # concatenate in case l_train doesn't include all the labels
        assert le.classes_.shape[0] == n_classes
        
        if to_binary_rep: 
            return label_binarize(le.transform(labels), classes=range(n_classes))

        return le.transform(labels)  # transform to numerical levels 
    def save_tset(X, y, cv_id=0, docIDs=[], meta=None): # [params] (X_train, X_test, y_train, y_test)
        ### tsHandler has to be configured first, use tsHandler.config()
        return tsHandler.save(X, y, index=cv_id, docIDs=docIDs, meta=meta)
    def load_tset(cv_id=0, meta=None):  # cohort, seq_ptype, d2v_method (focused_labels, label_map)
        ### tsHandler has to be configured first, use tsHandler.config()
        # this should conform to the protocal in makeTSetCombined()

        if kargs.has_key('X') and kargs.has_key('y'): 
            X, y = kargs['X'], kargs['y']
        else: 
            ts = kargs.get('ts', None)
            if ts is None: 
                print('  + automatically loading training set (cohort=%s, d2v=%s, ctype=%s)' % \
                         (tsHandler.cohort, tsHandler.d2v, tsHandler.ctype))
                # config()
                ts = tsHandler.load(cv_id, meta=meta)  # opt: focus_classes, label_map
                ts = customize_tset(ts)  # <- focused_labels, label_map
                assert ts is not None and not ts.empty, "multiClassEvaluate> Null training set."
                profile(ts)
            # allow loadTSet-like operations to take care of this
            X, y = TSet.toXY(ts)
        tsHandler.profile2(X, y)
        return (X, y)
    def profile(ts): 
        return tsHandler.profile(ts)   
    def get_tset_dir(): 
        return tsHandler.get_tset_dir()
    def customize_tset(ts): 
        focused_classes = kargs.get('focused_labels', None) # None: N/A
        if focused_classes is not None: # only preserve these classes and everything else becomes "Control"
            # e.g. In CKD data, one may only want to focus on predicting ‘Stage 1’, ‘Stage 2’, ‘Stage 3a’, ... 'Stage 5'
            ts = focus(ts, labels=focused_classes, other_label='Control')
        lmap = kargs.get('label_map', None)
        if lmap: # relabel so that classes only retain the main CKD stages
            print('  + before re-labeling ...')
            profile(ts)
            ts = merge(ts, lmap)
            print('  + after re-labeling ...')
            profile(ts)
        return ts
    def run_nested_cv(X, y, prefix='nested_cv', fpath=None): # input: (X, y)
        # [params] estimator=None, param_grid=None, n_trials=None, scoring='roc_auc'
        if fpath is None:    
            # prefix = 'nested_cv'
            # [todo] Graphic.getFullPath(cohort=cohort_name, d2v_method=d2v_method, seq_ptype=seq_ptype, dir_type='plot', ext_plot='tif')
            fname = Graphic.getName(prefix=prefix, identifier=make_file_id(), ext_plot='tif')
            outputdir = Graphic.getPath(cohort=cohort_name, dir_type='plot', create_dir=True)
            fpath = os.path.join(outputdir, fname)
        non_nested_scores, nested_scores, score_difference = ms.runNestedCV(X, y, nfold_inner=5, nfold_outer=5, estimator_name=estimator_name)
        ms.plotComparison(non_nested_scores, nested_scores, score_difference, fpath=fpath)
        return
    def get_clf_name(): # can be called before or after model selection (<- select_classifier)
        classifier = kargs.get('classifier', None)
        if classifier is not None: 
            try: 
                name = classifier.__name__
            except: 
                print('info> infer classifier name from class name ...')
                # name = str(estimator).split('(')[0]
                name = classifier.__class__.__name__
        else: 
            name = kargs.get('classifier_name', None) 
            assert name is not None
        return name
    def make_file_id(): # [classifier_name, cohort, d2v_method, seq_ptype, meta]
        identifier = kargs.get('identifier', None)
        if identifier is None: 
            cohort_name = tsHandler.cohort
            d2v_method = tsHandler.d2v  # vector.D2V.d2v_method
            ctype = tsHandler.ctype
            identifier = seqparams.makeID(params=[get_clf_name(), cohort_name, d2v_method, ctype, 
                            kargs.get('meta', tsHandler.meta)])  # null characters and None will not be included
        return identifier
    def roc_cv(X, y, classifier, fpath=None, target_labels=[], n_folds=5, n_trials=1):    
        identifier = make_file_id()
        outputdir = Graphic.getPath(cohort=tsHandler.cohort, dir_type='plot', create_dir=True)
        # classifier = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)

        # [note] can use nested CV but too expensive
        # evaluation
        # ms.runCVROC(X, y, classifier=classifier, fpath=fpath)   # remember to call plt.clf() for each call (o.w. plots overlap)

        # other params: plot_selected_classes/False
        res = ms.runCVROCMulticlass(X, y, classifier=classifier, 
            outputdir=outputdir, identifier=identifier, target_labels=target_labels, 
            n_folds=n_folds, 

            general_evaluation=True, 
            plot_selected_classes=False) # set to True to avoid cluttering; if False, target_labels will be ignored 
        return res # resturn max performance and min performance; key: 'min': label -> auc, 'max': label -> auc
    def roc_train_test_split(X, y, classifier, fpath=None, target_labels=[], ratios=[0.7, ]): # useful for deep learning model, which takes much longer to train
        identifier = make_file_id()
        outputdir = Graphic.getPath(cohort=tsHandler.cohort, dir_type='plot', create_dir=True)

        # other params: plot_selected_classes/False
        res = ms.runROCMulticlass(X, y, classifier=classifier, outputdir=outputdir, 
            identifier=identifier, target_labels=target_labels, ratios=ratios)
        return res 
    def resolve_classifier(X, y, n_folds=5, metric='roc_auc'): # [params] classifier, classifier_name
        ### method 1: externally specify a classifier (model parameters configured)
        ###        2: specfiy the name of the classifier (but have to be identifiable by modelSelect module)
        classifier = kargs.get('classifier', None) # condition: if a classifier is passed, assume that hyperparams are tuned
        clf_name = kargs.get('classifier_name', None) 
        if clf_name: # preferable set of classifiers predefined
            # choose by name > model selection
            try: 
                print('status> chooose estimator + model selection ...')
                classifier = ms.selectOptimalEstimator(clf_name, X, y, is_multiclass=True, max_n_samples=5000) # select classifier by name + model selection
            except Exception, e: 
                print('status> Error:\n%s\nstatus> unresolved classifier: %s' % (e, clf_name))
                raise ValueError, "Classifier %s is not available" % clf_name
        else:  # in multi-class case, perhaps this is more preferable
            assert classifier is not None
            clf_name = get_clf_name()
            print('status> classifier:\n%s\n' % clf_name)

            # condition: assuming that the input classifier supports multiclass 
            param_grid = kargs.get('param_grid', None)
            if param_grid: # model selection 
                classifier = ms.selectModel(X, y, estimator=classifier, param_grid=param_grid, n_folds=n_folds, scoring=metric) 
            else: 
                print('status> assuming that the input classifier already has optimial hyperparam setting ...')
                # noop

        assert classifier is not None, "Could not choose classifier."
        return (classifier, clf_name)  # classifier with hyperparams optimized ~ (X, y)
    def relabel(ts):  # ad-hoc
        # CKD stage classes
        lmap = {}
        lmap['Control'] = ['CKD G1-control', 'CKD G1A1-control', 'Unknown', ]  # ... control data 
        lmap['CKD Stage 5'] = ['ESRD after transplant', 'ESRD on dialysis', ]
        lmap['CKD Stage 3'] = ['CKD Stage 3a', 'CKD Stage 3b', ]
        ts = merge(ts, lmap)
        return ts 
    def validate_labels(): # targets, y
        assert set(targets).issubset(np.unique(y)), "targets contain unknown labels:\n%s\n" % targets
        return 

    from sklearn.metrics import roc_curve, auc
    # from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import label_binarize
    # from sklearn.multiclass import OneVsRestClassifier
    from scipy import interp

    # multiclass evaluation
    from sklearn.metrics import cohen_kappa_score
    from sklearn.metrics import confusion_matrix
    # from sklearn import svm

    import modelSelect as ms
    # from tset import TSet  # base class is defined in seqparams
    from seqparams import Graphic
    from tset import TSet
    from sklearn.preprocessing import LabelEncoder
    
    # from sklearn.neural_network import MLPClassifier
    # from sklearn import svm
    random_state = np.random.RandomState(0)
    # config() # use sysConfig() to configure training set related params; tsHandler.dir_type = 'combined'
    
    # focused_classes? Set to None to include all
    X, y = load_tset(meta=kargs.get('meta', None))  # precedence (X, y), ts, auto 

    # only use a subset of features (assuming that feature selection had been applied)
    projection = kargs.get('support', None)  # indices/positions of active features 
    if projection is not None: 
        dim0 = X.shape[0]
        X = X[:, projection]
        print('multiClassEvaluate> Select only %d features (from %d)' % (len(projection), dim0))

    # condition: classifier needs to support predict_proba() method
    # use: provide param_grid for model selection
    classifier, classifier_name = resolve_classifier(X, y)  # [params] classifier (assumed tuned) | classifier_name (=> model selection), param_grid

    # roc_per_class: To prevent the figure from being clobbered, select only, say 3, classes to present in the ROC curve
    targets = kargs.get('roc_per_class', ['CKD Stage 1', 'CKD Stage 3', 'CKD Stage 5', ])  # assuming that CKD Stage 3a, 3b are merged
    validate_labels()

    policy_evaluation = kargs.get('evaluation', 'cv')  # 'cv', 'split'
    if policy_evaluation.startswith('c'): 

        # other params: plot_selected_classes/False
        res = roc_cv(X, y, classifier=classifier, 
            target_labels=targets, n_folds=kargs.get('n_folds', 5))  # target_labels takes effect only when plot_selected_classes set to True
    else:   # policy_evaluate <- 'split'
        res = roc_train_test_split(X, y, classifier=classifier, 
            target_labels=targets, ratios=kargs.get('ratios', [0.7, ]))  
    # plotROCMultiClass(fpr, tpr, roc_auc, n_classes=n_classes, 
    #     identifier=kargs.get('identifier', 'generic'), outputdir=kargs.get('outputdir', os.getcwd()))

    return res

def plotROCMultiClass(fpr, tpr, roc_auc, n_classes):
    """

    Reference
    ---------
    1. http://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html

    """
    # import plotUtils
    from sklearn.metrics import roc_curve, auc

    # Compute macro-average ROC curve and ROC area

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    plt.figure()
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)

    plt.plot(fpr["macro"], tpr["macro"],
                label='macro-average ROC curve (area = {0:0.2f})'
                ''.format(roc_auc["macro"]),
                color='navy', linestyle=':', linewidth=4)

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                 ''.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Some extension of Receiver operating characteristic to multi-class')
    plt.legend(loc="lower right")
    # plt.show() 

    plotUtils.render_fig(plt, **kargs) 

    return

def binarize(ts, positive_classes, **kargs):
    from tset import TSet  # [todo] use seqmaker.tset
    # from labeling import TDocTag

    positive = positive_classes
    if not hasattr(positive, '__iter__'): 
        positive = [positive, ]

    N = ts.shape[0]
    col_target = TSet.target_field
    labels = ts[col_target].unique()
    assert len(set(positive)-set(labels))==0, "Some of the positive classes %s are not part of the label set:\n%s\n" % (positive, str(labels))

    # renaming the labels 
    #    df.loc[selection criteria, columns I want] = value
    cond_pos = ts[col_target].isin(positive)

    tsp = ts.loc[cond_pos]
    tsn = ts.loc[~cond_pos]
    assert N == tsp.shape[0] + tsn.shape[0]

    tsp.loc[:, col_target] = 1
    tsn.loc[:, col_target] = 0 

    # ts = pd.concat([tsp, tsn]).sort_index()
    print('binarize> n_pos: %d, n_neg: %d' % (tsp.shape[0], tsn.shape[0]))

    return pd.concat([tsp, tsn]).sort_index()  # recover original indexing
def mergeLabels(L, lmap={}):
    import tsHandler
    return tsHandler.mergeLabels(L, lmap)
def merge(ts, lmap={}): 
    """
    
    lmap: label map: representative label -> equivalent labels

    Example 
    -------
    1. CKD cohort 
       Given 11 labels: 

       [‘G1-control’, ‘G1A1-control’,  ‘Stage 1’, ‘Stage 2’, ‘Stage 3a’, ‘Stage 3b’, 
     ’Stage 4’, ‘Stage 5’, ‘ESRD after transplant’, ‘ESRD on dialysis’, ‘Unknown’]

       merge the 2 ESRD-related classes to 'Stage 5'

       then specify

       lmap['Stage 5'] = ['Stage 5', 'ESRD after transplant', ‘ESRD on dialysis’] 

       Note that 'Stage 5' itself needs not be in the list


       Example map
       lmap['Control'] = [‘G1-control’, ‘G1A1-control’, 'Unknown']  ... control data 
       lmap['Stage 5'] = [‘ESRD after transplant’, ‘ESRD on dialysis’]
       
       classes that map to themselves do not need to be specified

    Related
    -------
    binarize() 
    focus()
    """
    from tset import TSet  # [todo] use seqmaker.tset

    if not lmap: return ts

    # N = ts.shape[0]
    col_target = TSet.target_field
    labelSet = ts[col_target].unique()
    print('  + (before) unique labels:\n%s\n' % labelSet)
    for label, eq_labels in lmap.items(): 
        print('  + %s <- %s' % (label, eq_labels))
        cond_pos = ts[col_target].isin(eq_labels)
        ts.loc[cond_pos, col_target] = label  # rename to designated label 

    labelSet2 = ts[col_target].unique()   
    print('merge> n_labels: %d -> %d' % (len(labelSet), len(labelSet2)))
    print('  + (after) unique labels:\n%s\n' % labelSet2)

    return ts

def focusLabels(L, labels, other_label='Control'):
    """
    Out of all the possible labels in L, focus only on the labels in 'labels'

    """
    return tsHandler.focusLabels(L, labels, other_label=other_label)
def focus(ts, labels, other_label='Control'): 
    """
    Consolidate multiclass labels into smaller categories. 
    Focuse only on the target labels and classify all of the other labels
    using as others.  

    Example: CKD has 11 classes, some of which are not stages. 
             Want to only focus on stage-related labels (Stage 1 - Stage 5), and 
             leave all the other labels, including unknown, to the other category

    Related
    -------
    binarize() 
    merge()


    Memo
    ----
    1. CKD dataset 
       [‘G1-control’, ‘G1A1-control’,  ‘Stage 1’, ‘Stage 2’, ‘Stage 3a’, ‘Stage 3b’, 
     ’Stage 4’, ‘Stage 5’, ‘ESRD after transplant’, ‘ESRD on dialysis’, ‘Unknown’]
    
    """
    return tsHandler.focus(ts, labels, other_label=other_label)
    
def biClassEvaluate(**kargs): 
    """

    Memo
    ----
    1. ['CKD G1-control' 'CKD G1A1-control' 'CKD Stage 1' 'CKD Stage 2'
        'CKD Stage 3a' 'CKD Stage 3b' 'CKD Stage 4' 'CKD Stage 5'
        'ESRD after transplant' 'ESRD on dialysis' 'Unknown']

    2. detect late stage 
       binarize> n_pos: 126, n_neg: 2697

    3. detect early stage 
          
    """
    def transform_label(l, positive=None):
        if positive is None: positive = positive_classes # go to the outer environment to find its def
        y_ = labeling.binarize(l, positive=positive) # [params] negative
        return y_ 
    def detect_late_stage(): # create training data (X, y) for detecting early stages (as per user definitions)
        positive = ['CKD Stage 4', 'CKD Stage 5',]
        ts_bin = binarize(ts, positive_classes=positive)
        X, y = TSet.toXY(ts_bin)
        return (X, y)
    def detect_early_stage(): 
        positive = ['CKD Stage 1', 'CKD Stage 2',]
        ts_bin = binarize(ts, positive_classes=positive)
        X, y = TSet.toXY(ts_bin)
        return (X, y)
    def run_nested_cv(X, y, prefix='nested_cv', fpath=None): # input: (X, y)
        # [params] estimator=None, param_grid=None, n_trials=None, scoring='roc_auc'
        if fpath is None: 
            # prefix = 'nested_cv'
            # [todo] Graphic.getFullPath(cohort=cohort_name, d2v_method=d2v_method, seq_ptype=seq_ptype, dir_type='plot', ext_plot='tif')
            fname = Graphic.getName(prefix=prefix, cohort=cohort_name, d2v_method=d2v_method, seq_ptype=seq_ptype, ext_plot='tif')
            outputdir = Graphic.getPath(cohort=cohort_name, dir_type='plot', create_dir=True)
            fpath = os.path.join(outputdir, fname)
        non_nested_scores, nested_scores, score_difference = ms.runNestedCV(X, y, nfold_inner=5, nfold_outer=5, estimator_name=estimator_name)
        ms.plotComparison(non_nested_scores, nested_scores, score_difference, fpath=fpath)
        return
    def get_clf_name(): # can be called before or after model selection (<- select_classifier)
        classifier = kargs.get('classifier', None)
        if classifier is not None: 
            try: 
                name = classifier.__name__
            except: 
                print('info> infer classifier name from class name ...')
                # name = str(estimator).split('(')[0]
                name = classifier.__class__.__name__
        else: 
            name = kargs.get('classifier_name', None) 
            assert name is not None
        return name
    def roc_cv(X, y, classifier, prefix='roc', fpath=None): 
        if fpath is None: 
            # prefix = 'roc' # graph identifier
            # [todo] Graphic.getFullPath(cohort=cohort_name, d2v_method=d2v_method, seq_ptype=seq_ptype, dir_type='plot', ext_plot='tif')
            
            # prefix = '%s-%s' % (prefix, get_clf_name()) # include classifier name in the prefix
            fname = Graphic.getName(prefix=prefix, cohort=cohort_name, d2v_method=d2v_method, seq_ptype=seq_ptype, ext_plot='tif')
            outputdir = Graphic.getPath(cohort=cohort_name, dir_type='plot', create_dir=True)
            fpath = os.path.join(outputdir, fname)  
        
        # classifier = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)

        # [note] can use nested CV but too expensive
        # evaluation
        ms.runCVROC(X, y, classifier=classifier, fpath=fpath)   # remember to call plt.clf() for each call (o.w. plots overlap)
        return 
    def select_classifier(X, y): # [params] classifier, classifier_name
        classifier = kargs.get('classifier', None) # condition: if a classifier is passed, assume that hyperparams are tuned
        if classifier is None: 
            # choose by name > model selection
            clf_name = kargs.get('classifier_name', None) 
            if clf_name is not None: 
                try: 
                    print('status> chooose estimator + model selection ...')
                    classifier = ms.selectOptimalEstimator(clf_name, X, y, max_n_samples=5000) # select classifier by name + model selection
                except Exception, e: 
                    print('status> Error:\n%s\nstatus> unresolved classifier: %s' % (e, clf_name))
            else: 
                # don't use default, too confusing
                # random_state = np.random.RandomState(0)
                # classifier = svm.SVC(kernel='linear', probability=True,
                #                  random_state=random_state)
                raise ValueError, "No classifier or classifier name given."
        else: 
            print('status> classifier given (assumed tuned):\n%s\n' % classifier)
            param_grid = kargs.get('param_grid', None)
            if param_grid is not None: # model selection 
                n_folds = kargs.get('n_folds', 5)
                metric = kargs.get('scoring', 'roc_auc')
                classifier = ms.selectModel(X, y, estimator=classifier, param_grid=param_grid, n_folds=n_folds, scoring=metric) 
            else: 
                print('status> assuming that the input classifier already has optimial hyperparam setting ...')
                # noop

        assert classifier is not None, "Could not choose classifier."
        return classifier

    import modelSelect as ms
    from tset import TSet  # base class is defined in seqparams
    from seqparams import Graphic
    from sklearn.neural_network import MLPClassifier
    from sklearn import svm
    
    cohort_name = kargs.get('cohort', 'CKD')
    seq_ptype = kargs.get('seq_ptype', 'regular')
    d2v_method = vector.D2V.d2v_method
    include_augmented = kargs.get('include_augmented', False)

    # load combined training data (no distinction between train test split on the d2v level)
    # [params] dir_type='combined', index=0
    # [dir] # ./data/<cohort>/combined
    ts = loadTSet(n_per_class=kargs.get('n_per_class', None))  

    div(message='1. Detect late stage CKD ...')
    X, y = detect_late_stage()  # [params] ts 
    classifier = select_classifier(X, y)  # [params] classifier (assumed tuned) | classifier_name (=> model selection)
    classifier_name = get_clf_name()  # [note] can be called before select_classifier()

    # run_nested_cv(X, y) # (X, y)    # nested CV + plot differences in nested and non-nested
    roc_cv(X, y, classifier=classifier, prefix='roc-late_stage-%s' % classifier_name)

    div(message='1. Detect early stage CKD ...')
    X, y = detect_early_stage()
    classifier = select_classifier(X, y)  # [params] classifier (assumed tuned) | classifier_name (=> model selection)
    # classifier_name = get_clf_name()  # [note] can be called before select_classifier()
    roc_cv(X, y, classifier=classifier, prefix='roc-early_stage-%s' % classifier_name)
    
    # input from ts_bin = binarize(ts, positive_class='', )
    # evaluate.binary_classify(ts_bin, seq_ptype=seq_ptype, d2v_method=d2v_method)    
    return

def t_classify_sparse(**kargs):

    import evaluate
    from tset import TSet
    from sklearn.preprocessing import LabelEncoder

    from sklearn.neural_network import MLPClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import SVC
    from sklearn.gaussian_process import GaussianProcessClassifier
    from sklearn.gaussian_process.kernels import RBF
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.multiclass import OneVsRestClassifier

    ### code start here 

    return 

def t_classify(**kargs):
    """

    Memo
    ----
    1. Example training sets: 
        a. trained with labeled data only (cohort=CKD)
           tpheno/seqmaker/data/CKD/combined/tset-n0-IDregular-pv-dm2-U-GCKD.csv
        b. labeled + augmented data
           tpheno/seqmaker/data/CKD/combined/tset-n0-IDregular-pv-dm2-A-GCKD.csv

    2. training set params: 
        ts = TSet.load(cohort=tsHandler.cohort, 
                       d2v_method=tsHandler.d2v, 
                       seq_ptype=tsHandler.ctype, 
                       suffix=tsHandler.meta, 
                       index=index,  # CV index (if no CV, then 0 by default)
                       dir_type=tsHandler.dir_type) # [params] index
    """
    def config(): 
        tsHandler.config(cohort=kargs.get('cohort', 'CKD'), 
            seq_ptype=seqparams.normalize_ctype(kargs.get('seq_ptype', 'regular')), 
            d2v_method=kargs.get('d2v_method', vector.D2V.d2v_method), 
            simplify_code=kargs.get('simplify_code', False), is_augmented=kargs.get('is_augmented', False))
        return 
    def validate_classes(no_throw=True): 
        n_classes = np.unique(y_train).shape[0]
        n_classes_test = np.unique(y_test).shape[0]
        print('t_classify> n_classes: %d =?= n_classes_test: %d' % (n_classes, n_classes_test))
        return n_classes
    def transform_label(l, positive=None):
        if positive is None: positive = positive_classes # go to the outer environment to find its def
        y_ = labeling.binarize(l, positive=positive) # [params] negative
        return y_ 
    def relabel(): 
        # CKD stage classes
        lmap = {}
        lmap['Control'] = ['CKD G1-control', 'CKD G1A1-control', 'Unknown', ]  # ... control data 
        lmap['CKD Stage 5'] = ['ESRD after transplant', 'ESRD on dialysis', ]
        lmap['CKD Stage 3'] = ['CKD Stage 3a', 'CKD Stage 3b', ]
        print('  + Relabeling the data set according to the following map:\n%s\n' % lmap)
        return lmap
    def subsample(X, y, n=None, sort_index=False, random_state=53):
        """
        Subsampling operation applied after doc vec X is derived. 
        """
        if n is not None: 
            idx = cutils.samplePerClass(y, n_per_class=n, sort_index=sort_index, random_state=random_state)
            print('  + X (dim0: %s -> dim: %s)' % (X.shape[0], len(idx)))
            return (X[idx], y[idx])
        return (X, y)
    def subsample2(ts, n=None, sort_index=True, random_state=53):
        if n is not None: # 0 or None => noop
            ts = cutils.sampleDataframe(ts, col=TSet.target_field, n_per_class=n, random_state=random_state)
            n_classes_prime = check_tset(ts)
            print('  + after subsampling, size(ts)=%d, n_classes=%d (same?)' % (ts.shape[0], n_classes_prime))
        else: 
            # noop 
            pass 
        return ts
    def check_tset(ts): 
        target_field = TSet.target_field
        n_classes = 1 
        no_throw = kargs.get('no_throw', True)
        if ts is not None and not ts.empty: 
            # target_field in ts.columns # o.w. assuming that only 1 class 
            n_classes = len(ts[target_field].unique())  
            print('t_classify> number of classes: %d' % n_classes)
        else:
            msg = 't_classify> Warning: No data found (cohort=%s)' % kargs.get('cohort', 'CKD')
            if no_throw: 
                print msg 
            else: 
                raise ValueError, msg
    def summary(ts=None, X=None, y=None): 
        # experimental_settings
        msg = "... cohort=%s, ctype=%s, d2v=%s, meta=%s\n" % \
                (tsHandler.cohort, tsHandler.ctype, tsHandler.d2v_method, tsHandler.meta)
        msg += "  + is_simplified? %s, ... \n" % tsHandler.is_simplified
        msg += '  + classification mode: %s\n' % mode
        msg += '  + classifiers:\n%s\n' % clf_list
        msg += '  + training set type:%s\n' % ts_dtype

        nrow = n_classes = -1
        if ts is not None: 
            nrow = ts.shape[0]
            n_classes = len(ts[TSet.target_field].unique())
        else: 
            assert X is not None
            nrow = X.shape[0]
            n_classes = len(np.unique(y)) if y is not None else 1
        msg += '  + training set dim:%d\n' % nrow 
        msg += '  + n classes:%d\n' % n_classes

        print msg 
        return 
    def choose_classifier(name='random_forest'):
        param_grid = {}
        if name.startswith(('rand', 'rf')):  # n=389K
            # max_features: The number of features to consider when looking for the best split; sqrt by default
            # sample_leaf_options = [1,5,10,50,100,200,500] for tuning minimum sample leave (>= 50 usu better)
            clf = RandomForestClassifier(n_jobs=10, random_state=53, 
                    n_estimators=100,   # default: 10
                    min_samples_split=250, min_samples_leaf=50)  # n_estimators=500
        elif name.startswith('log'):
            clf = LogisticRegression(class_weight='balanced', solver='saga', penalty='l2')
        elif name.startswith( ('stoc', 'sgd')):  
            clf = SGDClassifier(loss='log', penalty='elasticnet', max_iter=1500, tol=1e-3, n_jobs=10) # l1_ratio: elastic net mixing param: 0.15 by default
        elif name.startswith(('grad', 'gb')):  # gradient boost tree
            # min_samples_split=250, min_samples_leaf=50 
            # max_leaf_nodes: If None then unlimited number of leaf nodes.
            # subsample: fraction of samples to be used for fitting the individual base learners
            #            Choosing subsample < 1.0 leads to a reduction of variance and an increase in bias
            clf = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=53, 
                min_samples_split=250, min_samples_leaf=50, max_depth=8,  # prevent overfitting
                max_features = 'sqrt', # Its a general thumb-rule to start with square root.
                subsample=0.85)
            param_grid = {'n_estimators': [50, 100, 150, 200, 500, ], 
                          'min_samples_split': [100, 200, 250, 300, ], 
                          'min_samples_leaf': [100, 50, 25], }

        else: 
            raise ValueError, "Unrecognized classifier: %s" % name
        return (clf, param_grid)
    def experimental_settings(): 
        print('t_classify> tset type %s, maxNPerClass: %s, drop control? %s' % \
            (kargs.get('tset_dtype', 'dense'), kargs.get('n_per_class', None), kargs.get('drop_ctrl', False)))

        print('  + cohort: %s, ctype: %s' % (tsHandler.cohort, tsHandler.ctype))
        print('  + d2v: %s, params: ' % (tsHandler.d2v, ))
        print('  + userFileID: %s' % kargs.get('meta', tsHandler.meta))

        try: 
            print('  + using classifier: %s' % str(clf_list[0]))
        except: 
            print('t_classify> classifier has not been determined yet.')

        return
    def estimate_performance(res): 
        from seqUtils import format_list
        missing = []
        # res: a dictionary with keys: {min, max, micro, macro, loss, accuracy, auc}
        classifier_type = 'classical'
        tset_type = 'sparse' if tsHandler.is_sparse() else 'dense'

        print('result> performance | tset type: %s, classifer type: %s' % (tset_type, classifier_type))
        minLabel, minScore = res['min']  # min auc score among all classes
        maxLabel, maxScore = res['max']  # max auc score among all classes

        if res.has_key('min_err'): 
            print('result> min(label: %s, score: %f), err: %s' % (minLabel, minScore, res['min_err']))
        if res.has_key('max_err'): 
            print('        max(label: %s, score: %f), err: %s' % (maxLabel, maxScore, res['max_err']))

        print('result> other performance metrics ...')
        missing = []
        for metric in ('micro', 'macro', 'loss', 'acc', 'auc_roc', 'precision', ):  # precision: micro-averaged precision score 
            if res.has_key(metric): 
                if res.has_key('%s_err' % metric):
                    print('    + metric=%s => %f (err: %s)' % (metric, res[metric], res['%s_err' % metric])) 
                else: 
                    print('    + metric=%s => %f' % (metric, res[metric])) 
            else: 
                missing.append(metric)
        
        if missing: print('... missing metrics: %s' % format_list(missing))
        # todo: consider labels
        return (minScore, maxScore)

    import evaluate
    from sampler import sampling
    import classifier.utils as cutils
    from tset import TSet
    import modelSelect as ms

    from sklearn.preprocessing import LabelEncoder
    from sklearn.neural_network import MLPClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import SVC
    from sklearn.gaussian_process import GaussianProcessClassifier
    from sklearn.gaussian_process.kernels import RBF
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.multiclass import OneVsRestClassifier
    from sklearn.linear_model import LogisticRegression, Lasso, LassoCV, SGDClassifier
    
    from sklearn.preprocessing import StandardScaler, MaxAbsScaler # feature scaling for sparse repr. 

    # [params]
    #    set is_augmented to True to use augmented training set
    # tsHandler.config(cohort='CKD', seq_ptype='regular', is_augmented=True) # 'regular' # 'diag', 'regular'
    mode = kargs.get('mode', 'multiclass')  # values: 'binary', 'multiclass'
    param_grid = {}

    if mode.startswith('bin'): # binary class (need to specifiy positive classes to focus on)
        return t_binary_classify(**kargs)    

    ### training document vectors 
    # t_model(corhot='CKD', seq_ptype=seq_ptype, load_model=True)

    ### classification 
    clf_list = []

    # choose classifiers
    # random forest: n_estimator = 100, oob_score = TRUE, n_jobs = -1,random_state =50, max_features = "auto", min_samples_leaf = 50

    # 1. logistic 
    # clf = LogisticRegression(class_weight='balanced', solver='saga', penalty='l2')

    # 2. random forest 
    # clf = RandomForestClassifier(n_jobs=5, random_state=53)  # n_estimators=500, 

    # 3. SGD classifier
    #    when tol is not None, max_iter:1000 by default
    # clf = SGDClassifier(loss='log', penalty='elasticnet', max_iter=1500, tol=1e-3) # l1_ratio: elastic net mixing param: 0.15 by default
    clf, param_grid = choose_classifier(name=kargs.pop('clf_name', 'rf')) # rf: random forest
    clf_list.append( (clf, param_grid) )  # supports multiclass 

    # gradient boosting: min_samples_split=2
    # param_grid = {'n_estimators':[500, 100], 'min_samples_split':[2, 5], 'max_features': [None, 2, 5], 'max_leaf_nodes': [None, 4]}
    # clf_list.append( GradientBoostingClassifier(n_estimators=500, learning_rate=0.1, random_state=53, max_features=2, max_leaf_nodes=4) )
    # clf = GradientBoostingClassifier(n_estimators=1000, learning_rate=0.1, random_state=53, max_features=2, max_leaf_nodes=4)

    # random_state = np.random.RandomState(0)
    # clf = svm.SVC(kernel='linear', probability=True, random_state=random_state)

    focusedLabels = None # None to include ALL  # focus(), merge()
    classesOnROC = ['CKD Stage 1', 'CKD Stage 3', 'CKD Stage 5', ] # assuming that CKD Stage 3a and 3b are merged into CKD Stage 3

    # can provide customized training set (ts); if None, then ts is automatically determined by tsHandler.load()
    # ts(cohort, d2v, ctype, meta('A', 'D', ), index)
    ts_dtype = kargs.get('tset_dtype', 'dense')
    maxNPerClass = kargs.get('n_per_class', None)

    # train-validation-test split ratios and in that order 
    # [note] test split ratio is usually omitted here (included implicitly in the train split)
    maxSampleRatios = kargs.get('ratios', [])  
    userFileID = kargs.get('meta', tsHandler.meta) 
    tDropControlGroup = kargs.get('drop_ctrl', False)
    
    # model selection, also for speedup at pathway analysis 
    tSaveTSetSubset = False  # subsample at loadTSet() instead
    tUsePrecomputedSubset = tSaveTSetSubset

    nTrials = kargs.get('n_trials', 5)
    experimental_settings()
    if ts_dtype.startswith('d'): 

        # [note] set n_per_class to None => subsampling now is delegated to makeTSetCombined() 
        # load, (scale), modify, subsample
        ts = loadTSet(label_map=seqparams.System.label_map, 
               n_per_class=None,   # set an upper limit for class-wise sample sizes
               ratios=maxSampleRatios,  # sampling according to class proportions
               drop_ctrl=tDropControlGroup, 
               meta=kargs.get('meta', tsHandler.meta)) # all parameters should have been configured
        print('t_classify> load training set of dim: %s' % str(ts.shape))

        ### subsampling 
        # if maxNPerClass: 
        #     ts = subsample2(ts, n=maxNPerClass)

        ### ts -> (X, y)
        X, y = TSet.toXY(ts)
        tsHandler.profile2(X, y)

        ### model selection
        #   [note] 1. subsetting for model selection 
        #          2. also subsetting a smaller sample for efficiency

        X_train = y_train = X_val = y_val = X_test = y_test = None  # set aside a small subset of (X, y) for model selection
        if maxSampleRatios: 
            n0 = X.shape[0]
            ridx = sampling.splitDataPerClass(y, ratios=maxSampleRatios)

            # validation data: prepare a separate validation set for model selection if possible
            if len(maxSampleRatios) >= 2: 
                X_val, y_val = X[ridx[1]], y[ridx[1]] 

            # training data
            if tUsePrecomputedSubset: 
                print('t_classify> Re-use precomputed sample.')
                ts = tsHandler.load_tset(index=0, meta='S', sparse=False) 
                X_train, y_train = TSet.toXY(ts)
            if X_train is None and y_train is None: 
                X_train, y_train = X[ridx[0]], y[ridx[0]]   # training; X_test, y_test is included within (X, y)

            ridx = None
            n1 = X_train.shape[0]
            print('t_classify> sample size %d (r=%s)=> %d' % (n0, maxSampleRatios, n1))

            if X_val is not None: 
                n2 = X_val.shape[0]
                print('... validation set size %d' % n2)

            if tSaveTSetSubset: 
                tsHandler.save_tset(X_train, y_train, index=0, docIDs=ridx[0], meta='S', sparse=False, shuffle_=True)

        scoring_metric = 'neg_log_loss'
        for i, (clf, param_grid) in enumerate(clf_list): 
            if (X_val is not None and y_val is not None) and len(param_grid) > 0:  # if not, just assume clf has been optimally configured
                print('... model selection on classifier: %s | metric: %s' % (clf, scoring_metric))
                clf = ms.selectModel(X_val, y_val, estimator=clf, param_grid=param_grid, n_folds=5, scoring=scoring_metric)
            clf_list[i] = clf  

        # precedence: classifier_name -> classifier
        summary(ts=ts)
        result_set = []

        X, y = X_train, y_train
        for clf in clf_list: 
            res = modelEvaluateBatch(X, y, 
                    classifier=clf, 

                    n_trials=nTrials, 
                    roc_per_class=classesOnROC,
                    # label_map=seqparams.System.label_map, # use sysConfig to specify
                    meta=userFileID, identifier=None)  # use meta as one of the parameters to determine file ID
            # res = multiClassEvaluate(X=X, y=y, 
            #         classifier=clf, 
            #         # classifier_name='l1_logistic',   # if not None, classifier_name takes precedence over classifier
            #         focused_labels=focusedLabels, 
            #         roc_per_class=classesOnROC,
            #         param_grid=None, # if None, then we assume that input classifier has been optimized for its hyperparams 
            #         label_map=seqparams.System.label_map, # use sysConfig to specify
            #         meta=userFileID, identifier=None) # use meta or identifier to distinguish different classificaiton tasks; set to None to use default
            result_set.append(res)
    else: 
        X, y = kargs.get('X', None), kargs.get('y', None)
        if X is None or y is None: 
            # load from the disk; operations: load, (scale), modify, subsample
            X, y = loadSparseTSet(scale_=kargs.get('scale_', True), 
                                    label_map=seqparams.System.label_map, 
                                    drop_ctrl=tDropControlGroup, 
                                    n_per_class=maxNPerClass)  # subsampling
        else: 
            y = mergeLabels(y, lmap=seqparams.System.label_map) # y: np.ndarray
            # subsampling 
            if maxNPerClass:
                y = np.array(y)
                X, y = subsample(X, y, n=maxNPerClass)

            if kargs.get('scale_', True): 
                # scaler = StandardScaler(with_mean=False)
                scaler = MaxAbsScaler()
                X = scaler.fit_transform(X)

        assert X is not None and y is not None

        # [test]
        # clf_list = [LogisticRegression(class_weight='balanced', solver='saga', penalty='l1'), ]
        summary(X=X, y=y)
        result_set = []
        for clf in clf_list: 
            res = modelEvaluateBatch(X, y, 
                    classifier=clf, 

                    n_trials=nTrials, 
                    roc_per_class=classesOnROC,
                    # label_map=seqparams.System.label_map, # use sysConfig to specify
                    meta=userFileID, identifier=None)

            # res = multiClassEvaluateSparse(X=X, y=y, 
            #         classifier=clf, 
            #         # classifier_name='l1_logistic',   # if not None, classifier_name takes precedence over classifier
            #         focused_labels=focusedLabels, 
            #         roc_per_class=classesOnROC,
            #         param_grid=param_grid, 
            #         label_map=seqparams.System.label_map, # use sysConfig to specify
            #         meta=userFileID, identifier=None) # use meta or identifier to distinguish different classificaiton tasks; set to None to use default
            result_set.append(res)

    minscore, maxscore = estimate_performance(result_set[0])
    # [todo] loop through cartesian product of seq_ptype and d2v_method? 

    return (minscore, maxscore)

def t_deep_classify(**kargs):
    """

    Memo
    ----
    1. Example training sets: 
        a. trained with labeled data only (cohort=CKD)
           tpheno/seqmaker/data/CKD/combined/tset-n0-IDregular-pv-dm2-U-GCKD.csv
        b. labeled + augmented data
           tpheno/seqmaker/data/CKD/combined/tset-n0-IDregular-pv-dm2-A-GCKD.csv

    2. training set params: 
        ts = TSet.load(cohort=tsHandler.cohort, 
                       d2v_method=tsHandler.d2v, 
                       seq_ptype=tsHandler.ctype, 
                       suffix=tsHandler.meta, 
                       index=index,  # CV index (if no CV, then 0 by default)
                       dir_type=tsHandler.dir_type) # [params] index

    *3. Import dnn_utils for example networks. 
    """
    def validate_classes(no_throw=True): 
        n_classes = np.unique(y_train).shape[0]
        n_classes_test = np.unique(y_test).shape[0]
        print('t_classify> n_classes: %d =?= n_classes_test: %d' % (n_classes, n_classes_test))
        return n_classes
    def subsample(X, y, n=None, sort_index=False, random_state=53):
        """
        Subsampling operation applied after doc vec X is derived. 
        """
        if n is not None: 
            idx = cutils.samplePerClass(y, n_per_class=n, sort_index=sort_index, random_state=random_state)
            print('  + X (dim0: %s -> dim: %s)' % (X.shape[0], len(idx)))
            return (X[idx], y[idx])
        return (X, y)
    def check_tset(ts): 
        target_field = TSet.target_field
        n_classes = 1 
        no_throw = kargs.get('no_throw', True)
        if ts is not None and not ts.empty: 
            # target_field in ts.columns # o.w. assuming that only 1 class 
            n_classes = len(ts[target_field].unique())  
            print('t_classify> number of classes: %d' % n_classes)
        else:
            msg = 't_classify> Warning: No data found (cohort=%s)' % kargs.get('cohort', 'CKD')
            if no_throw: 
                print msg 
            else: 
                raise ValueError, msg
    def summary(ts=None, X=None, y=None): 
        # experimental_settings
        msg = "... cohort=%s, ctype=%s, d2v=%s, meta=%s\n" % \
                (tsHandler.cohort, tsHandler.ctype, tsHandler.d2v_method, tsHandler.meta)
        msg += "  + is_simplified? %s, ... \n" % tsHandler.is_simplified
        msg += '  + classification mode: %s\n' % mode
        msg += '  + classifiers:\n%s\n' % clf_list
        msg += '  + training set type:%s\n' % ts_dtype

        nrow = n_classes = -1
        if ts is not None: 
            nrow = ts.shape[0]
            n_classes = len(ts[TSet.target_field].unique())
        else: 
            assert X is not None
            nrow = X.shape[0]
            n_classes = len(np.unique(y)) if y is not None else 1
        msg += '  + training set dim:%d\n' % nrow 
        msg += '  + n classes:%d\n' % n_classes

        print msg 
        return 
    def experimental_settings(): 
        print('\n ... Experimental Settings ...')
        print('\n   + tset type %s, maxNPerClass: %s, drop control? %s' % \
            (kargs.get('tset_dtype', 'dense'), kargs.get('n_per_class', 'ALL'), kargs.get('drop_ctrl', False)))

        print('  + cohort: %s, ctype: %s\n' % (tsHandler.cohort, tsHandler.ctype))
        print('  + D2V: %s, params> window: %s, n_features: %s' % (tsHandler.d2v, vector.D2V.window, vector.D2V.n_features))
        print('       + n_iter: %d, min_count: %d\n' % (vector.D2V.n_iter, vector.D2V.min_count))
        print('  + userFileID: %s' % kargs.get('meta', tsHandler.meta))

        try: 
            print('\n... data: ')
            print('  + n_timesteps: %d, n_features: %d' % ())
            print('  + reshaped X: %s | n_classes=%d' % (str(X.shape), n_classes))
        except: 
            pass
        try: 
            print('\n... params (model selection): ')
            print('  + epochs: %d, batch_size: %d' % NNS.epochs_ms, NNS.batch_size_ms)
            print('\n... params (after model selection)')
            print('  + epochs: %d, batch_size: %d' % NNS.epochs, NNS.batch_size)
        except: 
            pass
        return
    def estimate_performance(res): 
        from seqUtils import format_list
        missing = []
        # res: a dictionary with keys: {min, max, micro, macro, loss, accuracy, auc}
 
        classifier_type = 'nns'
        tset_type = 'sparse' if tsHandler.is_sparse() else 'dense'
        print('result> performance | tset type: %s, classifer type: %s' % (tset_type, classifier_type))

        minLabel, minScore = res['min']  # min auc score among all classes
        maxLabel, maxScore = res['max']  # max auc score among all classes

        if res.has_key('min_err'): 
            print('result> min(label: %s, score: %f), err: %s' % (minLabel, minScore, res['min_err']))
        if res.has_key('max_err'): 
            print('        max(label: %s, score: %f), err: %s' % (maxLabel, maxScore, res['max_err']))

        print('result> other performance metrics ...')
        missing = []
        for metric in ('micro', 'macro', 'loss', 'acc', 'auc_roc', ): 
            if res.has_key(metric): 
                if res.has_key('%s_err' % metric):
                    print('    + metric=%s => %f (err: %s)' % (metric, res[metric], res['%s_err' % metric])) 
                else: 
                    print('    + metric=%s => %f' % (metric, res[metric])) 
            else: 
                missing.append(metric)
        
        if missing: print('... missing metrics: %s' % format_list(missing))
        # todo: consider labels
        return (minScore, maxScore)
    def load_tset(cv_id=0, meta=None):  # cohort, seq_ptype, d2v_method (focused_labels, label_map)
        ### tsHandler has to be configured first, use tsHandler.config()
        # this should conform to the protocal in makeTSetCombined()

        if kargs.has_key('X') and kargs.has_key('y'): 
            X, y = kargs['X'], kargs['y']
        else: 
            ts = kargs.get('ts', None)
            if ts is None: 
                print('  + automatically loading training set (cohort=%s, d2v=%s, ctype=%s)' % \
                         (tsHandler.cohort, tsHandler.d2v, tsHandler.ctype))
                # config()
                ts = tsHandler.load(cv_id, meta=meta)  # opt: focus_classes, label_map
                ts = customize_tset(ts)  # <- focused_labels, label_map
                assert ts is not None and not ts.empty, "multiClassEvaluate> Null training set."
                profile(ts)
            # allow loadTSet-like operations to take care of this
            X, y = TSet.toXY(ts)
        tsHandler.profile2(X, y)
        return (X, y)
    def profile(ts): 
        return tsHandler.profile(ts)   
    def get_tset_dir(): 
        return tsHandler.get_tset_dir()
    def customize_tset(ts): 
        focused_classes = kargs.get('focused_labels', None) # None: N/A
        if focused_classes is not None: # only preserve these classes and everything else becomes "Control"
            # e.g. In CKD data, one may only want to focus on predicting ‘Stage 1’, ‘Stage 2’, ‘Stage 3a’, ... 'Stage 5'
            ts = focus(ts, labels=focused_classes, other_label='Control')
        lmap = kargs.get('label_map', None)
        if lmap: # relabel so that classes only retain the main CKD stages
            print('  + before re-labeling ...')
            profile(ts)
            ts = merge(ts, lmap)
            print('  + after re-labeling ...')
            profile(ts)
        return ts
    def describe_classifier(clf):
        myName = '?'
        try: 
            myName = clf.__class__.__name__ 
        except: 
            print('warning> Could not access __class__.__name__ ...')
            myName = str(clf)
        print('... classifier name: %s' % myName)
        return
    def create_model(n_units=100): # closure: n_timeteps, n_features, n_classes
        # def
        model = Sequential()
        model.add(LSTM(n_units, input_shape=(n_timesteps, n_features)))
        model.add(Dropout(0.2))
        model.add(Dense(n_classes, activation='softmax'))
        # Compile model
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy', ])
        return model
    def baseline_model(n_units=100): # closure: n_timeteps, n_features, n_classes
        # create model
        model = Sequential()
        model.add(Dense(n_units, input_dim=n_features, activation='relu'))
        model.add(Dense(n_classes, activation='softmax'))
        
        # Compile model
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model
    def reshape3D(X): 
        # n_samples = kargs.get('n_samples', X.shape[0])
        # n_timesteps = kargs.get('last_n_visits', 10)
        # n_features = kargs.get('n_features', vector.D2V.n_features * 2 if tsHandler.d2v in ('pv-dm2', ) else vector.D2V.n_features)    
        print('  + reshape(X) for LSTM: n_samples: %d, nt: %d, nf: %d' % (n_samples, n_timesteps, n_features))

        return X.reshape((n_samples, n_timesteps, n_features))
    def save(model): # save NNs model 
        identifier = kargs.get('identifier', None) 
        if identifier is None: identifier = make_file_id()
        NNS.save(model, identifier=identifier) # params: model_name
        return
    def load(): # load NNs model
        identifier = kargs.get('identifier', None) 
        if identifier is None: identifier = make_file_id()
        return NNS.load(identifier=identifier) 
    def rank_performance(scores, gaps, param, metrics_ordered=['loss', 'acc', 'auc_roc', ]): # <- score_map, grid_scores
        
        # score_map = {0:'loss', 1:'accuracy', 2:'auc_roc'}  
        # [update] grid_scores
        for si, metric in enumerate(metrics_ordered):
            setting = {'n_units': param['n_units'], 'dropout_rate': param['dropout_rate'], }
            pmetric = (setting, scores[si], gaps[si])  # [params] modify the desire performance measures here
            grid_scores[metric].append(pmetric)

        return grid_scores
    def rank_model(target_metric, metrics_ordered=['loss', 'acc', 'auc_roc', ], score_pos=1):  # given grid_scores, rank them 
        # [todo]
        if metrics_ordered is None: metrics_ordered = NNS.metrics_ordered

        rank_min = [0, ]  # the indices for which the higher the rank, the smaller the score
        rank_max = [1, 2]   # model.metrics_names[1]
        print('result> performance scores ...\n')

        opt = opt_score = opt_gap = None
        
        # [design]
        max_loss = 0.1
        min_acc, min_auc = 0.9, 0.9

        topN = 5 # keep track of how many times a particular setting was chosen among top N
        popularModels = collections.Counter()
        for si, metric in enumerate(metrics_ordered):
            
            # A. sort by performance score
            # grid_scores[si] = sorted(grid_scores[si], key=lambda x:x[score_pos], reverse=False if metric.startswith('loss') else True)
            # print('verify> full ranking:\n%s\n' % grid_scores[si])

            # B. sort in terms of gaps
            candidates = sorted(grid_scores[metric], key=lambda x:x[score_pos+1], reverse=False) # always the smaller the better
            n_models = len(candidates)
            # print('verify> full ranking (n_models=%d):\n%s\n' % (len(candidates), candidates))
            
            # policy: rank according to gap (between training perfomrance and validation performance, the smaller the better)
            #    subject to: acc >= 0.9, auc >= 0.9 
            candidates2 = []
            for candidate in candidates: 
                score = candidate[score_pos]
                if metric.startswith('loss'): 
                    if score <= max_loss: candidates2.append(candidate)
                elif metric == 'acc': 
                    if score >= min_acc: candidates2.append(candidate)
                elif metric == 'auc_roc':
                    if score >= min_auc: candidates2.append(candidate)
            grid_scores[metric] = candidates2

            # most popular (e.g. top 5) across different metrics? 
            configs = []
            for pmetric in grid_scores[metric][:topN]:  # pmetric: (setting, scores[si], gaps[si])
                setting = pmetric[0]
                assert len(setting) >= 2
                e = tuple([(k, v) for k, v in setting.items()]) 
                configs.append(e)
            popularModels.update(configs)     

            best_scores = grid_scores[metric][0]
            print('... performance ranking (n_models:%d -> %d):\n%s\n' % (n_models, len(candidates2), candidates2))
            print('... under metric (%s), best score: %f, gap: %f' % (metric, best_scores[score_pos], best_scores[score_pos+1]))
            print('... model config: %s\n' % best_scores[0])
            
            if target_metric == si or target_metric.startswith(metric):
                opt = best_scores[0]  # a dictionary
                
        topNFinal = 10
        print('result> popular %d model (out of %d metric-neutral options with topN=%d) ...' % (topNFinal, len(popularModels), topN))
        for setting, n_selected in popularModels.most_common(topNFinal):
            print('  + (n_selected=%d) model: %s' % (n_selected, setting))

        print('result> best configuration:\n%s\n' % opt)   # {'n_units': 200, 'dropout_rate': 0.5}
        return opt
    
    # import evaluate, seqparams, vector
    from sampler import sampling
    import classifier.utils as cutils
    # from tset import TSet
    # from sklearn.linear_model import LogisticRegression, Lasso, LassoCV, SGDClassifier
    from sklearn.preprocessing import StandardScaler, MaxAbsScaler # feature scaling for sparse repr. 

    from keras.models import Sequential
    from keras.layers import Dense
    from keras.layers import LSTM
    from keras.layers import Dropout
    from keras.wrappers.scikit_learn import KerasClassifier  ###
    from keras.utils import np_utils
    from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold, GridSearchCV
    from sklearn.preprocessing import LabelEncoder
    from sklearn.pipeline import Pipeline
    
    import dnn_utils as dnn
    from dnn_utils import NNS 
    import seqDNN
    from seqDNN import LSTMSpec

    from sklearn.model_selection import ParameterGrid
    from functools import partial

    # from dnn_utils import

    # [params]
    #    set is_augmented to True to use augmented training set
    # tsHandler.config(cohort='CKD', seq_ptype='regular', is_augmented=True) # 'regular' # 'diag', 'regular'
    mode = kargs.get('mode', 'multiclass')  # values: 'binary', 'multiclass'
    param_grid = None

    if mode.startswith('bin'): # binary class (need to specifiy positive classes to focus on)
        return t_binary_classify(**kargs)    

    ### training document vectors 
    # t_model(corhot='CKD', seq_ptype=seq_ptype, load_model=True)
    n_timesteps = kargs.get('last_n_visits', 10)
    n_features = kargs.get('n_features', vector.D2V.n_features * 2 if tsHandler.d2v in ('pv-dm2', ) else vector.D2V.n_features) 

    ### classification 
    clf_list = []

    focusedLabels = None # None to include ALL  # focus(), merge()
    classesOnROC = ['CKD Stage 1', 'CKD Stage 3', 'CKD Stage 5', ] # assuming that CKD Stage 3a and 3b are merged into CKD Stage 3

    # can provide customized training set (ts); if None, then ts is automatically determined by tsHandler.load()
    # ts(cohort, d2v, ctype, meta('A', 'D', ), index)
    ts_dtype = kargs.get('tset_dtype', 'dense')
    maxNPerClass = kargs.get('n_per_class', None)
    
    # [0.9, 0.1, ] => 10% goes to model seletion, 90% can be used for training 
    #                 BUT if 90% is still too large, can futher subsampling the training data
    maxSampleRatios = kargs.get('ratios', [])  # e.g. [0.9, 0.1, ] => 10% goes to model seletion
    userFileID = kargs.get('meta', tsHandler.meta) 
    tDropControlGroup = kargs.get('drop_ctrl', False)

    minscore = maxscore = -1
    grid_score = []

    tModelSelection = kargs.get('model_selection', True)
    if ts_dtype.startswith('d'): 

        # [note] set n_per_class to None => subsampling now is delegated to makeTSetCombined() 
        # load, (scale), modify, subsample
        ts = loadTSet(label_map=seqparams.System.label_map, 
               n_per_class=maxNPerClass, 
               drop_ctrl=tDropControlGroup, 
               meta=kargs.get('meta', tsHandler.meta)) # all parameters should have been configured
        summary(ts=ts)
        print('deep_classify> dim(ts): %s | n_timesteps: %d, n_features: %d' % (str(ts.shape), n_timesteps, n_features))
       
        X, y = TSet.toXY(ts)
        ts = None; gc.collect()

        n_samples = kargs.get('n_samples', X.shape[0])
        n_classes = kargs.get('n_classes', len(np.unique(y)))
        print('  + dim(X <- ts): %s' % str(X.shape))

        # to 3D in order to use LSTM
        #   [n_samples, n_timesteps, n_features]
        X = reshape3D(X)
        print('deep_classify> reshaped X: %s | n_classes:%d' % (str(X.shape), n_classes))
        assert X.shape[0] == len(y)

        ### sample subset (for model selection)
        # example use: 
        #   set ratios <- [0.9, 0.1, ]  # a train-validation-test split 
        #       90% goes to the training set, 10% goes to the validation set
        #         test split is implicitly defined in the train split 
        X_train = y_train = X_val = y_val = X_test = y_test = None  # set aside a small subset of (X, y) for model selection
        if maxSampleRatios: 
            n0 = X.shape[0]
            ridx = sampling.splitDataPerClass(y, ratios=maxSampleRatios)

            # prepare a separate validation set for model selection if possible
            if len(maxSampleRatios) >= 2: 
                X_val, y_val = X[ridx[1]], y[ridx[1]] 

            X, y = X[ridx[0]], y[ridx[0]]   # training; X_test, y_test is included within (X, y)

            ridx = None
            n1 = X.shape[0]
            print('... total sample size %d (r=%s)=> subset: %d' % (n0, maxSampleRatios, n1))

            if X_val is not None: 
                n2 = X_val.shape[0]
                print('... validation set size %d' % n2)
        else: 
            # => no separate validation set, use (X, y) itself for model selection 
            pass

        ### Alternatively, ... 
        # X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=maxSampleRatios[1], random_state=0, stratify=y)

        # A. Define a fixed model: define + compile
        # model = choose_classifier(n_timesteps=n_timesteps, n_features=n_features, n_classes=n_classes, 
        #     n_units=50, dropout_rate=0.2,  # params for model definition
        #     epochs=200, batch_size=32)  # params for model.fit()
        # clf_list.append(model)
        targetMetric = kargs.get('target_metric', 'loss') # model selection and policy for saving the model
        optimizer = kargs.get('optimizer', 'SGD')  # values: 'adam', 'SGD', 
        if not tModelSelection: 
            # model = load()
            model = dnn.make_lstm(n_units=n_features, n_timesteps=n_timesteps, n_features=n_features, 
                n_classes=n_classes, dropout_rate=0.2, optimizer=optimizer) # compiled model
            clf_list.append(model)
            # fit (+evaluate)
        else: 
            if X_val is None:
                print('deep_classify> Warning: No separate validation set provided!')
                X_val, y_val = X, y

            # B. Model selection
            param_grid = {'n_units': [50, 100, 200, 300, 500, 1000, ], 'dropout_rate': [0.2, 0.3, 0.4, 0.5, 0.6, ]}
            
            NNS.epochs_ms = kargs.get('epochs_ms', 30)  # previously: 100
            NNS.batch_size_ms = kargs.get('batch_size_ms', 32)
            NNS.patience_ms = kargs.get('patience_ms', 20)
        
            test_ratio = 0.3
            score_map = {0:'loss', 1:'accuracy', 2:'auc'}
            score_index = 1  # score_index: 0/loss, 1/accuracy, 2/auc
        
            # Run model selection
            grid_scores = {metric:[] for metric in NNS.metrics_ordered}  # ['loss', 'acc', 'auc_roc', ]

            experimental_settings()
            for param in list(ParameterGrid(param_grid)): 
                print('\nmodel_selection> trying %s ...\n' % param) 

                # use KerasClassifier Wrapper
                # model = makeClassifier(X, y, build_fn=build_fn_lstm, 
                #             n_layers=n_layers, n_units=n_units, dropout_rate=dropout_rate, # params for architecture 
                #             callbacks=None)  # callbacks: None to use default 

                model = dnn.make_lstm(n_units=param['n_units'], n_timesteps=n_timesteps, n_features=n_features, 
                                n_classes=n_classes, dropout_rate=param['dropout_rate'], save_init_weights=False, 
                                optimizer=optimizer)

                # params: random_state
                # scores and gaps are in the following order: ['loss','acc', 'auc_roc']
                scores, gaps = dnn.modelEvaluate0(X_val, y_val, model=model, focused_labels=None, roc_per_class=classesOnROC, 
                                    meta=userFileID, identifier=None, 
                                    test_size=test_ratio,  # only applies when using train_test_split
                                    validation_split=test_ratio, 
                                    patience=NNS.patience_ms, 
                                    epochs=NNS.epochs_ms, batch_size=NNS.batch_size_ms)  # score_index: 0/loss, 1/accuracy, 2/auc
            
                # grid_scores[i]: (setting, scores[si], gaps[si])
                rank_performance(scores, gaps, param=param)  # modify: grid_scores; other params: metrics_ordered
                
                # score = scores[score_index]
                # grid_score.append((param['n_units'], param['dropout_rate'], score))

            # choose metric given by 'score_index' as the final measure
            opt = rank_model(target_metric=targetMetric)  # rank hyperparams and their scores
            n_units, r_dropout = opt['n_units'], opt['dropout_rate']
 
            # choose_classifier(build_fn=partial(make_lstm, ...))
            print('... optimal setting > n_units: %d, dropout: %f | nF: %d, nTSteps: %d, nC: %d' % \
                (n_units, r_dropout, n_features, n_timesteps, n_classes))
            model = dnn.make_lstm(n_units=n_units, n_timesteps=n_timesteps, n_features=n_features, 
                            n_classes=n_classes, dropout_rate=r_dropout, optimizer=optimizer) # compiled model
            clf_list.append(model)

        ### model defined and weights trained 

        # control the total number of training data 
        N = X.shape[0]
        train_ratio = kargs.get('ratio', 0.7)
        train_subset = min(int(N*train_ratio), tsHandler.N_train_max)  # max: 10000
        test_subset = min(N-train_subset, tsHandler.N_test_max)  # max: 5000 
        if tsHandler.isBig(X):
            print('info> Big training data (n=%d) ...' % N)
            print('      n_train: %d, n_test: %d' % (train_subset, test_subset))
            assert train_subset <= tsHandler.N_train_max

        result_set = []
        nTrials = kargs.get('n_trials', 1)
        for clf in clf_list:  # for clf, grid_score in ... 
            describe_classifier(clf)

            ### a. train test split
            # other params: outputfile, outputdir
            # res = dnn.modelEvaluate(X, y, model=model, focused_labels=None, roc_per_class=classesOnROC, 
            #         meta=userFileID, identifier=None, 
            #         patience=NNS.patience, epochs=NNS.epochs, save_model=True)

            ### a2. multiple trails 
            # seqDNN.lstmEvaluateBatch()
            # NNS.patience = kargs.get('patience', 50)   
            # NNS.epochs = kargs.get('epochs', 500) 
            # NNS.batch_size = kargs.get('batch_size', 32)
            # NNS.summary()
            # res = dnn.modelEvaluateBatch(X, y, model, focused_labels=None, roc_per_class=classesOnROC, 
            #         meta=userFileID, identifier=None, 
            #         patience=NNS.patience, epochs=NNS.epochs, batch_size=NNS.batch_size, 
            #         save_model=True, n_trials=nTrials, 

            #         ratio=0.7,  # used for small data (<= 10K), not applicable when N > 10K
            #         ratio_validation=0.3, # within train split, use this fraction of data to validate the model e.g. 10K * 0.3 = 3K 
            #         train_subset=train_subset, # only use this fraction of data to train model (which includes validation)
            #         test_subset=test_subset, # only use this portion of data to test model 

            #         target_metric=targetMetric, init_weights=NNS.init_weights)

            ### a3.
            spec = LSTMSpec(n_units=n_features, n_timesteps=n_timesteps, n_features=n_features) 
            spec.n_layers = 2
            spec.optimizer = optimizer  
            spec.patience = kargs.get('patience', 300) 
            spec.epochs = kargs.get('epochs', 1000) 
            spec.batch_size = kargs.get('batch_size', 16)
            spec.r_validation_split = 0.3
            # spec.params_summary()

            # [note] very often the data (X, y) can be very large
            # 
            # res = seqDNN.lstmEvaluateChunks(X, y)
            res = seqDNN.lstmEvaluateBatch(X, y, 
                    focused_labels=None, roc_per_class=classesOnROC,  # used in ROC plot
                    meta=userFileID, identifier=None, 
                    lstm_spec=spec, 

                    save_model=True, n_trials=nTrials, 

                    ratio=0.7,  # used for small data (<= 10K), not applicable when N > 10K
                    ratio_validation=0.3, # within train split, use this fraction of data to validate the model e.g. 10K * 0.3 = 3K 
                    train_subset=train_subset, # only use this fraction of data to train model (which includes validation)
                    test_subset=test_subset, # only use this portion of data to test model 

                    target_metric=targetMetric)

            # ## b. CV
            # res = multiClassEvaluate(X=X, y=y, 
            #         classifier=clf, 
            #         focused_labels=None, 
            #         roc_per_class=classesOnROC,
            #         param_grid=None, 
                  
            #         evaluation='split', ratios=[0.7, ],  # non-CV evaluation (e.g. train test split)

            #         label_map=seqparams.System.label_map, # use sysConfig to specify
            #         meta=userFileID, identifier=None, # use meta or identifier to distinguish different classificaiton tasks; set to None to use default
            #         ) 
            result_set.append(res)
        minscore, maxscore = estimate_performance(result_set[0])
    else: 
        raise ValueError, "LSTM mode does not support sparse training set."

    
    # [todo] loop through cartesian product of seq_ptype and d2v_method? 

    return (minscore, maxscore)

def t_deep_classify2(**kargs):
    """
    Similar to t_deep_classify() but handles large training dataset by loading
    partial training set at a time and incrementally train the NNs. 


    Memo
    ----
    1. Example training sets: 
        a. trained with labeled data only (cohort=CKD)
           tpheno/seqmaker/data/CKD/combined/tset-n0-IDregular-pv-dm2-U-GCKD.csv
        b. labeled + augmented data
           tpheno/seqmaker/data/CKD/combined/tset-n0-IDregular-pv-dm2-A-GCKD.csv

        c. training data with visit sessions
           tpheno/seqmaker/data/CKD/combined/tset-n0-IDregular-pv-dm2-regular-visit-tLb50-mLen10-GCKD.csv

    2. training set params: 
        ts = TSet.load(cohort=tsHandler.cohort, 
                       d2v_method=tsHandler.d2v, 
                       seq_ptype=tsHandler.ctype, 
                       suffix=tsHandler.meta, 
                       index=index,  # CV index (if no CV, then 0 by default)
                       dir_type=tsHandler.dir_type) # [params] index

    *3. Import dnn_utils for example networks. 
    """
    def validate_classes(no_throw=True): 
        n_classes = np.unique(y_train).shape[0]
        n_classes_test = np.unique(y_test).shape[0]
        print('t_classify> n_classes: %d =?= n_classes_test: %d' % (n_classes, n_classes_test))
        return n_classes
    def subsample(X, y, n=None, sort_index=False, random_state=53):
        """
        Subsampling operation applied after doc vec X is derived. 
        """
        if n is not None: 
            idx = cutils.samplePerClass(y, n_per_class=n, sort_index=sort_index, random_state=random_state)
            print('  + X (dim0: %s -> dim: %s)' % (X.shape[0], len(idx)))
            return (X[idx], y[idx])
        return (X, y)
    def check_tset(ts): 
        target_field = TSet.target_field
        n_classes = 1 
        no_throw = kargs.get('no_throw', True)
        if ts is not None and not ts.empty: 
            # target_field in ts.columns # o.w. assuming that only 1 class 
            n_classes = len(ts[target_field].unique())  
            print('t_classify> number of classes: %d' % n_classes)
        else:
            msg = 't_classify> Warning: No data found (cohort=%s)' % kargs.get('cohort', 'CKD')
            if no_throw: 
                print msg 
            else: 
                raise ValueError, msg
    def summary(ts=None, X=None, y=None): 
        # experimental_settings
        msg = "... cohort=%s, ctype=%s, d2v=%s, meta=%s\n" % \
                (tsHandler.cohort, tsHandler.ctype, tsHandler.d2v_method, tsHandler.meta)
        msg += "  + is_simplified? %s, ... \n" % tsHandler.is_simplified
        msg += '  + classification mode: %s\n' % mode
        msg += '  + classifiers:\n%s\n' % clf_list
        msg += '  + training set type:%s\n' % ts_dtype

        nrow = n_classes = -1
        if ts is not None: 
            nrow = ts.shape[0]
            n_classes = len(ts[TSet.target_field].unique())
        else: 
            assert X is not None
            nrow = X.shape[0]
            n_classes = len(np.unique(y)) if y is not None else 1
        msg += '  + training set dim:%d\n' % nrow 
        msg += '  + n classes:%d\n' % n_classes

        print msg 
        return 
    def experimental_settings(): 
        print('\n ... Experimental Settings ...')
        print('\n   + tset type %s, maxNPerClass: %s, drop control? %s' % \
            (kargs.get('tset_dtype', 'dense'), kargs.get('n_per_class', 'ALL'), kargs.get('drop_ctrl', False)))

        print('  + cohort: %s, ctype: %s\n' % (tsHandler.cohort, tsHandler.ctype))
        print('  + D2V: %s, params> window: %s, n_features: %s' % (tsHandler.d2v, vector.D2V.window, vector.D2V.n_features))
        print('       + n_iter: %d, min_count: %d\n' % (vector.D2V.n_iter, vector.D2V.min_count))
        print('  + userFileID: %s' % kargs.get('meta', tsHandler.meta))

        try: 
            print('\n... data: ')
            print('  + n_timesteps: %d, n_features: %d' % ())
            print('  + reshaped X: %s | n_classes=%d' % (str(X.shape), n_classes))
        except: 
            pass
        try: 
            print('\n... params (model selection): ')
            print('  + epochs: %d, batch_size: %d' % NNS.epochs_ms, NNS.batch_size_ms)
            print('\n... params (after model selection)')
            print('  + epochs: %d, batch_size: %d' % NNS.epochs, NNS.batch_size)
        except: 
            pass
        return
    def estimate_performance(res): 
        from seqUtils import format_list
        missing = []
        # res: a dictionary with keys: {min, max, micro, macro, loss, accuracy, auc}
 
        classifier_type = 'nns'
        tset_type = 'sparse' if tsHandler.is_sparse() else 'dense'
        print('result> performance | tset type: %s, classifer type: %s' % (tset_type, classifier_type))

        minLabel, minScore = res['min']  # min auc score among all classes
        maxLabel, maxScore = res['max']  # max auc score among all classes

        if res.has_key('min_err'): 
            print('result> min(label: %s, score: %f), err: %s' % (minLabel, minScore, res['min_err']))
        if res.has_key('max_err'): 
            print('        max(label: %s, score: %f), err: %s' % (maxLabel, maxScore, res['max_err']))

        print('result> other performance metrics ...')
        missing = []
        for metric in ('micro', 'macro', 'loss', 'acc', 'auc_roc', ): 
            if res.has_key(metric): 
                if res.has_key('%s_err' % metric):
                    print('    + metric=%s => %f (err: %s)' % (metric, res[metric], res['%s_err' % metric])) 
                else: 
                    print('    + metric=%s => %f' % (metric, res[metric])) 
            else: 
                missing.append(metric)
        
        if missing: print('... missing metrics: %s' % format_list(missing))
        # todo: consider labels
        return (minScore, maxScore)
    def profile(ts): 
        return tsHandler.profile(ts)   
    def get_tset_dir(): 
        return tsHandler.get_tset_dir()
    def customize_tset(ts): 
        focused_classes = kargs.get('focused_labels', None) # None: N/A
        if focused_classes is not None: # only preserve these classes and everything else becomes "Control"
            # e.g. In CKD data, one may only want to focus on predicting ‘Stage 1’, ‘Stage 2’, ‘Stage 3a’, ... 'Stage 5'
            ts = focus(ts, labels=focused_classes, other_label='Control')
        lmap = kargs.get('label_map', None)
        if lmap: # relabel so that classes only retain the main CKD stages
            print('  + before re-labeling ...')
            profile(ts)
            ts = merge(ts, lmap)
            print('  + after re-labeling ...')
            profile(ts)
        return ts
    def describe_classifier(clf):
        myName = '?'
        try: 
            myName = clf.__class__.__name__ 
        except: 
            print('warning> Could not access __class__.__name__ ...')
            myName = str(clf)
        print('... classifier name: %s' % myName)
        return
    def reshape3D(X): 
        # n_samples = kargs.get('n_samples', X.shape[0])
        # n_timesteps = kargs.get('last_n_visits', 10)
        # n_features = kargs.get('n_features', vector.D2V.n_features * 2 if tsHandler.d2v in ('pv-dm2', ) else vector.D2V.n_features)    
        print('  + reshape(X) for LSTM: n_samples: %d, nt: %d, nf: %d' % (n_samples, n_timesteps, n_features))

        return X.reshape((n_samples, n_timesteps, n_features))
    def save(model):
        identifier = kargs.get('identifier', None) 
        if identifier is None: identifier = make_file_id()
        NNS.save(model, identifier=identifier) # params: model_name
        return
    def load(): 
        identifier = kargs.get('identifier', None) 
        if identifier is None: identifier = make_file_id()
        return NNS.load(identifier=identifier) 
    def rank_performance(scores, gaps, param, metrics_ordered=['loss', 'acc', 'auc_roc', ]): # <- score_map, grid_scores
        
        # score_map = {0:'loss', 1:'accuracy', 2:'auc_roc'}  
        # [update] grid_scores
        for si, metric in enumerate(metrics_ordered):
            setting = {'n_units': param['n_units'], 'dropout_rate': param['dropout_rate'], }
            pmetric = (setting, scores[si], gaps[si])  # [params] modify the desire performance measures here
            grid_scores[metric].append(pmetric)

        return grid_scores
    def rank_model(target_metric, metrics_ordered=['loss', 'acc', 'auc_roc', ], score_pos=1):  # given grid_scores, rank them 
        # [todo]
        if metrics_ordered is None: metrics_ordered = NNS.metrics_ordered

        rank_min = [0, ]  # the indices for which the higher the rank, the smaller the score
        rank_max = [1, 2]   # model.metrics_names[1]
        print('result> performance scores ...\n')

        opt = opt_score = opt_gap = None
        
        # [design]
        max_loss = 0.1
        min_acc, min_auc = 0.9, 0.9

        topN = 5 # keep track of how many times a particular setting was chosen among top N
        popularModels = collections.Counter()
        for si, metric in enumerate(metrics_ordered):
            
            # A. sort by performance score
            # grid_scores[si] = sorted(grid_scores[si], key=lambda x:x[score_pos], reverse=False if metric.startswith('loss') else True)
            # print('verify> full ranking:\n%s\n' % grid_scores[si])

            # B. sort in terms of gaps
            candidates = sorted(grid_scores[metric], key=lambda x:x[score_pos+1], reverse=False) # always the smaller the better
            n_models = len(candidates)
            # print('verify> full ranking (n_models=%d):\n%s\n' % (len(candidates), candidates))
            
            # policy: rank according to gap (between training perfomrance and validation performance, the smaller the better)
            #    subject to: acc >= 0.9, auc >= 0.9 
            candidates2 = []
            for candidate in candidates: 
                score = candidate[score_pos]
                if metric.startswith('loss'): 
                    if score <= max_loss: candidates2.append(candidate)
                elif metric == 'acc': 
                    if score >= min_acc: candidates2.append(candidate)
                elif metric == 'auc_roc':
                    if score >= min_auc: candidates2.append(candidate)
            grid_scores[metric] = candidates2

            # most popular (e.g. top 5) across different metrics? 
            configs = []
            for pmetric in grid_scores[metric][:topN]:  # pmetric: (setting, scores[si], gaps[si])
                setting = pmetric[0]
                assert len(setting) >= 2
                e = tuple([(k, v) for k, v in setting.items()]) 
                configs.append(e)
            popularModels.update(configs)     

            best_scores = grid_scores[metric][0]
            print('... performance ranking (n_models:%d -> %d):\n%s\n' % (n_models, len(candidates2), candidates2))
            print('... under metric (%s), best score: %f, gap: %f' % (metric, best_scores[score_pos], best_scores[score_pos+1]))
            print('... model config: %s\n' % best_scores[0])
            
            if target_metric == si or target_metric.startswith(metric):
                opt = best_scores[0]  # a dictionary
                
        topNFinal = 10
        print('result> popular %d model (out of %d metric-neutral options with topN=%d) ...' % (topNFinal, len(popularModels), topN))
        for setting, n_selected in popularModels.most_common(topNFinal):
            print('  + (n_selected=%d) model: %s' % (n_selected, setting))

        print('result> best configuration:\n%s\n' % opt)   # {'n_units': 200, 'dropout_rate': 0.5}
        return opt
    
    # import evaluate, seqparams, vector
    from sampler import sampling
    import classifier.utils as cutils
    from sklearn.preprocessing import StandardScaler, MaxAbsScaler # feature scaling for sparse repr. 

    from keras.models import Sequential
    from keras.layers import Dense
    from keras.layers import LSTM
    from keras.layers import Dropout
    from keras.wrappers.scikit_learn import KerasClassifier  ###
    from keras.utils import np_utils
    from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold, GridSearchCV
    from sklearn.preprocessing import LabelEncoder
    from sklearn.pipeline import Pipeline
    
    import dnn_utils as dnn
    from dnn_utils import NNS 
    import seqDNN
    from seqDNN import LSTMSpec

    from sklearn.model_selection import ParameterGrid
    from functools import partial

    # from dnn_utils import

    # [params]
    #    set is_augmented to True to use augmented training set
    # tsHandler.config(cohort='CKD', seq_ptype='regular', is_augmented=True) # 'regular' # 'diag', 'regular'
    mode = 'multiclass'  # values: 'binary', 'multiclass'
    param_grid = None

    ### training document vectors 
    # t_model(corhot='CKD', seq_ptype=seq_ptype, load_model=True)
    n_timesteps = kargs.get('last_n_visits', 10)
    n_features = kargs.get('n_features', 200) 

    ### classification 
    model_list = []

    # clf = choose_classifier(name=kargs.pop('clf_name', 'lstm'), \
    #     epochs=kargs.get('epochs', 500), batch_size=kargs.get('batch_size', n_timesteps)) # rf: random forest
    # clf_list.append(clf)  # supports multiclass 

    focusedLabels = None # None to include ALL  # focus(), merge()
    classesOnROC = ['CKD Stage 1', 'CKD Stage 3', 'CKD Stage 5', ] # assuming that CKD Stage 3a and 3b are merged into CKD Stage 3

    # can provide customized training set (ts); if None, then ts is automatically determined by tsHandler.load()
    # ts(cohort, d2v, ctype, meta('A', 'D', ), index)
    maxNPerClass = kargs.get('n_per_class', None)
    
    # [0.9, 0.1, ] => 10% goes to model seletion, 90% can be used for training 
    #                 BUT if 90% is still too large, can futher subsampling the training data
    maxSampleRatios = kargs.get('ratios', [])  # e.g. [0.9, 0.1, ] => 10% goes to model seletion
    userFileID = kargs.get('meta', tsHandler.meta) 
    tDropControlGroup = kargs.get('drop_ctrl', False)

    # tModelSelection = False
    
    # [note] set n_per_class to None => subsampling now is delegated to makeTSetCombined() 
    # load, (scale), modify, subsample

    targetMetric = kargs.get('target_metric', 'loss') # model selection and policy for saving the model
    optimizer = kargs.get('optimizer', 'SGD')  # values: 'adam', 'SGD',
    chunkSize = kargs.get('chunksize', 1000)
    testSize = kargs.get('test_size', 3000)
    n_classes = kargs.get('n_classes', 5+1)

    # A. Define a fixed model: define + compile 
    # if not tModelSelection: 
    #     model = dnn.make_lstm(n_layers=2, n_units=n_features, n_timesteps=n_timesteps, n_features=n_features, 
    #             n_classes=n_classes, dropout_rate=0.2, optimizer=optimizer) # compiled model
    #     describe_classifier(model)
    # else: 
    #     raise NotImplementedError, "For simplicity, this subroutine assumes that model (hyper)paramters have been optimized."

    nTrials = kargs.get('n_trials', 1)
    minscore = maxscore = -1
    grid_score = []

    # specify the LSTM model
    spec = LSTMSpec(n_units=n_features, n_timesteps=n_timesteps, n_features=n_features) 
    spec.n_layers = 2
    spec.optimizer = optimizer  
    spec.patience = kargs.get('patience', 300) 
    spec.epochs = kargs.get('epochs', 1000) 
    spec.batch_size = kargs.get('batch_size', 32)
    spec.r_validation_split = 0.3

    # load training set parts by parts
    # for ts in loadTSetChunk(label_map=seqparams.System.label_map, 
    #            n_per_class=maxNPerClass, 
    #            drop_ctrl=tDropControlGroup) # all parameters should have been configured
    #     # do something 
    #     passs

    # [log] example path: <prefix>/tpheno/seqmaker/data/CKD/combined/tset-n0-IDregular-pv-dm2-regular-visit-tLb100-mLen10-GCKD.csv
    fpath = tsHandler.load_from(index=kargs.get('index', 0), meta=userFileID, verify_=True)
    res = seqDNN.lstmEvaluateChunk(fpath, 
     
                    # plot params               
                    # focused_labels=None, 
                    # roc_per_class=classesOnROC,  # used in ROC plot

                    # training set params
                    label_map=seqparams.System.label_map, 
                    drop_ctrl=tDropControlGroup, 
     
                    # file ID 
                    meta=userFileID, 
                    identifier=None, 

                    # lstm network params 
                    lstm_spec=spec, 
                    save_model=True, 
                    
                    # can define model here and pass it in
                    # model=model,  

                    # model training params
                    n_trials=nTrials, 
                    ratio=0.7,  # used for small data (<= 10K), not applicable when N > 10K
                    ratio_validation=0.3, # within train split, use this fraction of data to validate the model e.g. 10K * 0.3 = 3K 
                    target_metric=targetMetric)

    minscore, maxscore = estimate_performance(res)

    # [todo] loop through cartesian product of seq_ptype and d2v_method? 

    return (minscore, maxscore)


def t_lstm_many2one():
    def build(n_timesteps, n_features, n_units=10, n_classes=5):  
        model = Sequential()
        model.add(LSTM(n_units, input_shape=(n_timesteps, n_features)))
        model.add(Dropout(0.2))
        model.add(Dense(n_classes, activation='softmax'))
        
        # Compile model
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy', ])

        # estimator = KerasClassifier(build_fn=model, epochs=200, batch_size=10, verbose=0)
        # return estimator
        return model
    def create_model(n_timesteps=3, n_features=5, n_units=10, n_classes=3): 
        # def
        model = Sequential()
        model.add(LSTM(n_units, input_shape=(n_timesteps, n_features)))
        model.add(Dropout(0.2))
        model.add(Dense(n_classes, activation='softmax'))
        # Compile model
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy', ])
        return model
    def baseline_model(n_classes=5):
        # create model
        model = Sequential()
        model.add(Dense(8, input_dim=4, activation='relu'))
        model.add(Dense(n_classes, activation='softmax'))
        
        # Compile model
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model


    import numpy
    from keras.models import Sequential
    from keras.layers import Dense
    from keras.layers import LSTM
    from keras.layers import Dropout
    from keras.wrappers.scikit_learn import KerasClassifier  ###
    from keras.utils import np_utils

    from sklearn.model_selection import cross_val_score
    from sklearn.model_selection import StratifiedKFold
    from sklearn.model_selection import KFold
    from sklearn.model_selection import GridSearchCV

    from sklearn.preprocessing import LabelEncoder
    from sklearn.pipeline import Pipeline

    # 1, 2, 3 => 1 
    # 2, 1, 3 = > 2
    # 3, 2, 1 => 3 
    X = visits = np.array([ [[1, 1, 2, 1, 1], [2, 2, 1, 3, 2], [3, 1, 2, 3, 3]],
                          [[2, 2, 2, 2, 1], [1, 1, 3, 2, 1], [3, 3, 3, 2, 3]], 
                          [[3, 2, 1, 3, 3], [2, 2, 2, 3, 1], [1, 2, 1, 1, 3]], 
                          [[3, 2, 3, 3, 3], [2, 2, 2, 3, 2], [1, 2, 1, 1, 1]], 
                          [[2, 2, 1, 2, 1], [1, 1, 1, 2, 1], [3, 2, 3, 3, 3]], 
                          [[1, 2, 1, 1, 1], [2, 2, 2, 3, 2], [3, 2, 1, 3, 3]], 
                          [[1, 2, 1, 2, 1], [2, 2, 3, 1, 2], [3, 2, 1, 3, 3]], 
                          ], dtype='float32')
    # 3 t-step, 5 features
    # y = np.array(['a', 'b','c', ])
    y = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 1], [0, 1, 0], [1, 0, 0], [1, 0, 0], ])
    yw = np.array([1, 2, 3, 3, 2, 1, 1, ])

    # ytest = [1, 2, 3, 3, 2]
    Xtest = np.array([ [[1, 1, 3, 1, 2], [2, 2, 2, 3, 2], [3, 2, 2, 3, 3]],
                          [[2, 2, 2, 1, 1], [1, 1, 3, 2, 1], [3, 1, 3, 2, 3]], 
                          [[3, 3, 1, 3, 3], [2, 2, 2, 3, 2], [1, 2, 1, 1, 1]], 
                          [[3, 3, 2, 3, 3], [2, 3, 2, 2, 2], [1, 1, 1, 1, 1]], 
                          [[2, 1, 2, 2, 1], [1, 1, 2, 1, 1], [3, 3, 3, 2, 3]],
                        ], dtype='float32')    

    nt, nf = X.shape[1], X.shape[2]
    print('> n_timesteps: %d, n_features: %d' % (nt, nf))
    
    tWrapper = False
    # a. define model & train the model directly 
    if not tWrapper: 
        model = build(n_timesteps=nt, n_features=nf, n_units=15, n_classes=3)
        model.fit(X, y, epochs=100, batch_size=10)   # binary encoded labels
 
        # make predictions
        ypred = model.predict(Xtest)
        yl = [np.argmax(y)+1 for y in ypred]

        print('> predictions:\n%s\n ~ \n%s\n' % (ypred, yl))
    else: 
        # b. sklearn wrapper 
        model = KerasClassifier(build_fn=create_model, epochs=200, batch_size=10, verbose=2)
        model.fit(X, yw) # numeric labels 

        # make predictions
        ypred = model.predict(Xtest)
        # yl = [np.argmax(y)+1 for y in ypred]

        print('> wrapper predictions:\n%s\n' % ypred)
                 
        # evaluate using 10-fold cross validation
        seed = 53
        kfold = StratifiedKFold(n_splits=2, shuffle=True, random_state=seed)
        results = cross_val_score(model, X, yw, cv=kfold)
        print(results.mean())
    
    return 

def t_lcs_classify(**kargs):
    raise NotImplementedError, "Defined in pathAnalyzer" 

def t_process_docs(**kargs):
    def get_global_cohort_dir(): 
        return seqparams.getCohortGlobalDir(cohort_name) # sys_config.read('DataExpRoot')/<cohort>
    def test_docs(n=5): # <- D + (src_type, ctype, cohort_name)
        assert len(D) == len(T)
        print('  + source type <- %s, ctype <- %s  | cohort=%s' % (src_type, ctype, cohort_name))
        print('\n  ... sampling documents (n=%d) ...\n' % n)
        Dsub = random.sample(D, min(n, len(D)))
        for i, doc in enumerate(Dsub): 
            print('  + [%d] %s' % (i, ' '.join(doc)))
        return

    import sys

    cohort_name = kargs.get('cohort', 'CKD')

    ### load unlabeled document associated with a cohort
    
    # method 1
    # D_augmented, L, T = loadAugmentedDocuments(cohort=cohort_name, inputdir=None, label_default=None)

    # method 2: load + transform 
    ctype='regular'; src_dir = get_global_cohort_dir() # source_type='augmented'

    for ctype in ['regular', 'diag', 'med', ]: 
        for src_type in ['s', 'a', 'r' ]:   # source_type: {'s'/'source' (default),  'a'/'augmented', 'r'/'raw', } 
            D, L, T = processDocuments(cohort=cohort_name, seq_ptype=ctype, inputdir=src_dir, 
                                        predicate=kargs.get('predicate', None), 
                                        simplify_code=kargs.get('simplify_code', False), 
                                        source_type=src_type, create_labeled_docs=False)  # [params] composition

            test_docs()

    sys.exit(0)

    ### stratify cohorts
    stratified = stratifyDocuments(cohort=cohort_name, seq_ptype=kargs.get('seq_ptype', 'regular'),
                     predicate=kargs.get('predicate', None), simplify_code=kargs.get('simplify_code', False))  # [params] composition
    labels = stratified.keys() # documents = D; timestamps = T
    n_labels = len(set(labels))

    nD = nT = 0
    for label, entry in stratified.items():
        Di = entry['sequence']
        Ti = entry['timestamp']
        Li = entry['label']
        nD += len(Di)
        nT += len(Ti)

    assert nD == nT, "nD=%d <> nT=%d" % (nD, nT)
    print('t_process_docs> nD=%d, nT=%d, n_labels=%d' % (nD, nT, n_labels))
   
    return 

def compute_d2v_model(**kargs):
    """
    A template/demo function that illustrates how d2v-based training set is built. 



    Settings
    --------
    cohort: {'CKD', 'PTSD', 'diabetes', }
    seq_ptype: {'regular', 'diag', }

    Memo
    ----
    (*) where are training data saved? 
        makeTSetCombied -> 
            ./data/CKD/combined/tset-n0-IDregular-pv-dm2-GCKD.csv

    (*) naming of the training set depends on: 
        d2v_method
        cohort_name
        index

    Output
    ------
    1. training data (seq_type='diag', cohort='CKD', d2v_method='pv-dm2')

       ./tpheno/seqmaker/data/CKD/combined/tset-n0-IDregular-pv-dm2-GCKD.csv
       ./tpheno/seqmaker/data/CKD/combined/tset-n0-IDdiag-pv-dm2-GCKD.csv

    """
    def verify_segment_op():  
        predicate = kargs.get('predicate', None)
        # tval = False
        msg = 'n/a' 
        if predicate is not None: 
            policy_segment=kargs.get('policy_segment', 'regular')
            msg = '[predicate: %s, policy_segment: %s]' % (predicate.__name__, policy_segment)
            # tval = True 
        return msg       
 
    # use sysConfig() to configure the following parameters
    # cohort_name = kargs.get('cohort', tsHandler.cohort) # sysparams.cohort=cohort_name # system parameters are shared across modules
    # d2v_method = kargs.get('d2v_method', tsHandler.d2v)
    seq_ptype = seqparams.normalize_ctype(tsHandler.ctype) # e.g. 'regular'
    
    tset_descriptor = meta = kargs.get('meta', tsHandler.meta)
    model_descriptor = meta_model = kargs.get('meta_model', tsHandler.meta_model) # [note] use meta_model but NOT meta (used for training set files)

    div(message='t_model> cohort=%s, d2v_method=%s, ctype=%s, descriptor (model)=%s, descriptor (tset)=%s, segment? %s' % \
        (tsHandler.cohort, tsHandler.d2v, seq_ptype, model_descriptor, tset_descriptor, verify_segment_op()))

    # visit-oriented architecture
    tSegmentVisit = kargs.get('segment_by_visit', True)

    # [options]
    # makeTSet: train test split (randomized subsampling, n_trials is usu. 1)
    # makeTSetCombined: no separation of train and test data on the d2v level 
    # makeTSetCV: 
    # makeTSetVisit: 

    print('t_model> predicate: %s' % kargs.get('predicate', None))
    print('         + max_features: %s, max_n_docs: %s' % (kargs.get('max_features', 'unbounded'), kargs.get('max_n_docs', 'unbounded')))
    if tSegmentVisit: 
        n_timesteps = kargs.get('last_n_visits', 10)
        max_visit_length = kargs.get('max_visit_length', 100)
        min_visit_length = kargs.get('min_visit_length', seqparams.D2V.window)
        print('t_model> min_visit_length: %d, last_n_visits (n_timesteps): %d' % (min_visit_length, n_timesteps))

        # choose the appropriate subroutine
        # makeTSetVisit = makeTSetVisit2D if tsHandler.d2v.startswith(('ext', )) else makeTSetVisit3D
        X, y = makeTSetVisit3D(
                               ## 1. paramters for processMCSDocuments (i.e. load+transform) 
                               
                               # user-specified input
                               inputfile=kargs.get('inputfile', None), 
                               inputdir=kargs.get('inputdir', None),   # None to let system determine the path (~ global cohort)
 
                               # document set subsampling
                               max_n_docs=kargs.get('max_n_docs', None), # max number of documents used to build d2v model (usu for debug only)
                               max_n_docs_policy=kargs.get('max_n_docs_policy', 'longest'), # only relevant when docs are sampled

                               # document filtering (e.g. by length and other criteria)
                               min_ncodes=kargs.get('min_ncodes', 10),  # process_docs()

                               # document editing (e.g. retain only prediagnosis segments)
                               time_cutoff=kargs.get('time_cutoff', '2017-08-18'), 
                               predicate=kargs.get('predicate', None), 
                               policy_segment=kargs.get('policy_segment', 'regular'), 
                               include_endpoint=kargs.get('include_endpoint', False), 

                               # [note] set to True to drop documents without cutopints? 
                               #        reminder: cutpoints are typically the first occurrence of any of the diagnosis code that 
                               #                  defines a disease (e.g. CKD is linked to a few diagnosis codes)
                               #        so in cases where there's no diagnostic code that exist in the MCS to define prior or post segment 
                               #        then we may not want these to be included in our model. To exclude these documents, 
                               #        set drop_nullcut to True; to include documents without cutpoints anyway, set it to False
                               drop_nullcut=kargs.get('drop_nullcut', False), 

                               ## 2. parameters for D2V computations 
                               # max_visit_length=max_visit_length,  # this is not used now, i.e. no upper bound 
                               min_visit_length=min_visit_length, # the minimum length of a session (at least this number of bases should be there in a session document/paragraph)
                               last_n_visits=n_timesteps,  # only consider the very last N sessions (specified via n_timestamps)
                               max_n_samples=kargs.get('max_n_samples', None),  # keep only at most this number of training instances
                               conditional_save=kargs.get('conditional_save', False),  # set to True to save only when the training data do not already exist

                               meta=tsHandler.meta, # user-defined file ID (d2v model, training set, derived mcs file)
                               meta_model=tsHandler.meta_model, 

                               ## 3. parameters for I/O and testing
                               test_model=kargs.get('test_model', True), 
                               load_model=kargs.get('load_model', True))

        print('t_model> dim(X): %s, dim(y): %s' % (str(X.shape), str(y.shape)))
    else: 
        min_visit_length = kargs.get('min_visit_length', seqparams.D2V.window)

        # each patient is represented by the entire MCS 
        X, y = makeTSet(    
                            ## 1. paramters for document processing (i.e. load+transform) 

                            # user-specified input
                            inputfile=kargs.get('inputfile', None), 
                            inputdir=kargs.get('inputdir', None),   # None to let system determine the path (~ global cohort)

                            # sparse representation (i.e. bag-of-word model)
                            max_features=kargs.get('max_features', None), # only applies to d2v_method: 'bow' (i.e. bag of words)
                            text_matrix_mode=kargs.get('text_matrix_mode', 'tfidf'),  # only applies to d2v_method = 'bow'  

                            # document set subsampling
                            max_n_docs=kargs.get('max_n_docs', None), # max number of documents used to build d2v model (usu for debug only)
                            max_n_docs_policy=kargs.get('max_n_docs_policy', 'longest'), 

                            # document filtering (e.g. by length and other criteria)
                            min_ncodes=kargs.get('min_ncodes', 10),  # process_docs()

                            # document editing (e.g. retain only prediagnosis segments)
                            time_cutoff=kargs.get('time_cutoff', '2017-08-18'), 
                            predicate=kargs.get('predicate', None), 
                            policy_segment=kargs.get('policy_segment', 'regular'), 
                            include_endpoint=kargs.get('include_endpoint', False), 
                            drop_nullcut=kargs.get('drop_nullcut', False), # drop documents without cutopints? 

                            segment_by_visit=False, 

                            ## 2. parameters for D2V computations
                            # max_visit_length=kargs.get('max_visit_length', 100), # this is not used now, i.e. no upper bound 
                            min_visit_length=min_visit_length,
                            last_n_visits=kargs.get('last_n_visits', 10), 
                            max_n_samples=kargs.get('max_n_samples', None),  # keep only at most this number of training instances
                            # nnet_mode=kargs.get('nnet_mode', False), 

                            meta=tsHandler.meta, # user-defined file ID (d2v model, training set, derived mcs file)
                            meta_model=tsHandler.meta_model, 
                            # model_id=kargs.get('model_id', seq_ptype), # distinguish models based on sequence contents

                            include_augmented=kargs.get('include_augmented', False), 
                            test_model=kargs.get('test_model', True), 
                            load_model=kargs.get('load_model', False)) # [note] this shouldn't revert back and call load_tset() again 

    print('t_model> training set dimension: %s' % str(X.shape))
    # to load training set use the following ...
    # ts = loadTSet(cohort=cohort_name, d2v_method=vector.D2V.d2v_method)  # ./data/<cohort>/combined

    return (X, y)  # X: dense or sparse matrix of the training set; y: class labels

def t_binary_classify0(**kargs): # template for binary classifer
    import modelSelect as ms
    from tset import TSet  # base class is defined in seqparams
    
    cohort_name = kargs.get('cohort', seqparams.System.cohort)
    seq_ptype = kargs.get('seq_ptype', 'regular')
    estimator_name = kargs.get('estimator_name', 'linear_svm')

    # load combined training data (no distinction between train test split on the d2v level)
    # [params] dir_type='combined', index=0
    # [dir] # ./data/<cohort>/combined
    ts = loadTSet(cohort=cohort_name, d2v_method=vector.D2V.d2v_method, seq_ptype=seq_ptype)  
    X, y = TSet.toXY(ts)

    # [params] estimator=None, param_grid=None, n_trials=None, scoring='roc_auc'
    ms.runNestedCV(X, y, nfold_inner=5, nfold_outer=5, estimator_name=estimator_name)

    return
def t_classify_binary(**kargs): 
    from sklearn.neural_network import MLPClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import SVC
    from sklearn.gaussian_process import GaussianProcessClassifier
    from sklearn.gaussian_process.kernels import RBF
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.multiclass import OneVsRestClassifier

    seq_ptype = 'diag' # 'regular' # 'diag', 'regular'

    ### training document vectors 
    cohort_name = kargs.get('cohort', seqparams.System.cohort)
    compute_d2v_model(corhot=cohort_name, seq_ptype=seq_ptype, load_model=True)

    ### classification 
    # t_classify(**kargs)

    # binary classifier
    clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
    biClassEvaluate(corhot=cohort_name, seq_ptype=seq_ptype, classifier=clf)

    clf = GradientBoostingClassifier(learning_rate=0.1) # need to be wrapped in OneVsRestClassifier for multiclass
    biClassEvaluate(corhot=cohort_name, seq_ptype=seq_ptype, classifier=clf)

    clf = RandomForestClassifier(n_estimators=20)  # supports multiclass 
    biClassEvaluate(corhot=cohort_name, seq_ptype=seq_ptype, classifier=clf)

    # + model selection
    for name in ['linear_svm', 'rbf_svm', 'l2_logistic', ]:
        biClassEvaluate(corhot=cohort_name, seq_ptype=seq_ptype, classifier_name=name)  # slow

    return 

def t_label(**kargs):
    def load_tset():  # cohort, seq_ptype, d2v_method (focused_labels)
        ts = kargs.get('ts', None) # external training set
        if ts is None: 
            cohort_name = kargs.get('cohort', 'CKD')
            seq_ptype = kargs.get('seq_ptype', 'regular')
            d2v_method = vector.D2V.d2v_method
            focused_classes = kargs.get('focused_labels', None) # None: N/A
            # load combined training data (no distinction between train test split on the d2v level)
            # [params] dir_type='combined', index=0
            # [dir] # ./data/<cohort>/combined

            # ts = TSet.load(cohort=cohort_name, d2v_method=d2v_method, seq_ptype=seq_ptype, 
            #         dir_type='combined', index=0) # [params] index
            ts = loadTSet(cohort=cohort_name, d2v_method=d2v_method, seq_ptype=seq_ptype)
            if focused_classes is not None: # only preserve these classes and everything else becomes "Control"
                # e.g. In CKD data, one may only want to focus on predicting ‘Stage 1’, ‘Stage 2’, ‘Stage 3a’, ... 'Stage 5'
                ts = focus(ts, labels=focused_classes, other_label='Control')

        assert ts is not None and not ts.empty  
        return ts
    def test_group(ts): 
        col_target = TSet.target_field
        # cond_pos = ts[col_target].isin(eq_labels)
        n_esrd_trans = ts.loc[ts[col_target] == 'ESRD after transplant'].shape[0]
        n_esrd_dia = ts.loc[ts[col_target] == 'ESRD on dialysis'].shape[0]
        n_stage5 = ts.loc[ts[col_target] == 'CKD Stage 5'].shape[0]
        
        print('  + N(ESRD after transplant): %d' % n_esrd_trans)
        print('  + N(ESRD on dialysis): %d' % n_esrd_dia)
        print('  + N(stage 5): %d' % n_stage5)
        # print('  + N(stage 5 total): %d' % )

        n_others = ts.loc[ts[col_target] == 'Control'].shape[0]  # should be 0 before re-labeling
        print('  + N(Control): %d' % n_others)
        print('  ....... ')
        profile(ts)
        return
    def profile(ts): 
        col_target = TSet.target_field
        all_labels = ts[col_target].unique()
        sizes = {}
        for label in all_labels: 
            sizes[label] = ts.loc[ts[col_target] == label].shape[0]
        print('profile> Found %d unique labels ...' % len(all_labels))
        for label, n in sizes.items(): 
            print('  + label=%s => N=%d' % (label, n))
        return
    def relabel(): 
        # CKD stage classes
        lmap = {}
        lmap['Control'] = ['CKD G1-control', 'CKD G1A1-control', 'Unknown', ]  # ... control data 
        lmap['CKD Stage 5'] = ['ESRD after transplant', 'ESRD on dialysis', ]
        lmap['CKD Stage 3'] = ['CKD Stage 3a', 'CKD Stage 3b', ]
        return lmap

    from tset import TSet 
    ts = load_tset()
    N0 = ts.shape[0]

    test_group()

    # CKD stage classes
    lmap = relabel()

    ts = merge(ts, lmap)
    N = ts.shape[0]
    print('t_label> N: %d =?= %d' % (N0, N))
    test_group(ts)
    
    return  

def sysConfig(cohort, **kargs):
    """
    Configure system-wide paramters applicable to all modules that import seqparams.

    Params
    ------
    """
    def relabel():  # ad-hoc
        # CKD stage classes
        lmap = {}
        lmap['Control'] = ['CKD G1-control', 'CKD G1A1-control', 'Unknown', ]  # ... control data 
        lmap['CKD Stage 5'] = ['ESRD after transplant', 'ESRD on dialysis', ]
        lmap['CKD Stage 3'] = ['CKD Stage 3a', 'CKD Stage 3b', ]
        return lmap

    from seqparams import System as sysparams 
    sysparams.cohort = cohort # system parameters are shared across modules 
    sysparams.label_map = relabel()

    # configure d2v params
    if kargs.has_key('window'): 
        seqparams.D2V.window = wsize = kargs.pop('window', 5); assert vector.D2V.window == wsize
    if kargs.has_key('n_features'):
        seqparams.D2V.n_features = n_features = kargs.pop('n_features', 100)  # default: 100 but with pv-dm2 => end up getting 200-D vectors 
        assert vector.D2V.n_features == n_features
    seqparams.D2V.n_iter = n_iter = kargs.pop('n_iter', 50)  # default: 100 but with pv-dm2 => end up getting 200-D vectors

    # training set parameters 
    d2v_method = kargs.get('d2v_method', vector.D2V.d2v_method)
    if d2v_method is None: d2v_method = vector.D2V.d2v_method

    # params: cohort, seq_ptype/'regular', d2v_method/None, simplify_code/False, is_augmented/False, dir_type/'combined', meta/None, meta_model/None
    
    print('sysConfig> File descriptors | meta: %s, meta_model: %s' % (kargs.get('meta', None), kargs.get('meta_model', None)))
    tsHandler.config(cohort=cohort, d2v_method=d2v_method, 
                    seq_ptype=kargs.get('seq_ptype', 'regular'),

                    simplify_code=kargs.get('simplify_code', False), 
                    is_augmented=False, 

                    meta=kargs.get('meta', None), 
                    meta_model=kargs.get('meta_model', None))  # is_augmented/False, is_simplified/False, dir_type/'combined'
    return 

def runWorkFlow(**kargs): 
    """

    Settings
    ---------
    seqClassify (base)
       use deep learning methods (e.g. LSTM) to classify concatenated visit vectors (instead of classical classifiers such as GBT)

       tSegmentVisit := True 
       tNNet := True

       drop_ctrl := False  ... retain control group

    Memo
    ----
    1. example configuration 
       nPerClass = None     # use all 
       nDPerClass = 3000

       max_visit_length = 50  # max number of tokens/codes in a visit segment
       n_timesteps = 10 
       
       => <prefix>/data/CKD/model/Pf100w5i50-regular-D3000-visit-M50t10.{dm, dbow}

    """
    def includeEndPoint(): 
        ret = {'meta': 'prior_inclusive', 'include_endpoint': True}
        return
    def is_default_visit(maxlen=100, nt=10, d2v='pv-dm2'): 
        if not d2v in ('pv-dm2', ): return False
        return maxlen == 100 and nt == 10
    def hasTSet(): 
        return X is not None and X.shape[0] > 0
    def doClassify(): 
        return tClassify and hasTSet()
    def policyDeepModel(): 
        if tsHandler.is_sparse():
            return False 
        return kargs.get('use_deep_model', True)
    def determine_n_features(): 

        # params
        nf = kargs.get('n_features', 200)
        d2v = kargs.get('d2v_method', vector.D2V.d2v_method) 

        if d2v in ('pv-dm2', ): 
            nf = nf/2   # pv-dm half, and pv-dbow another half
        return nf
    def update_descriptor(base, *args, **kargs):
        bp = base
        fsep = '-'

        params = set(bp.split(fsep))
        # print('... %s' % params)
        for arg in args: 
            if not arg in params: 
                bp = '%s-%s' % (bp, arg)
            params.add(arg)
        
        for param, value in kargs.items(): 
            assert param is not None and len(param) > 0, "Invalid parameter: %s" % param
            if value is None and (not param in params): 
                bp = '%s-%s' % (bp, param)
                params.add(param)
                # print('... + %s' % params)
            else: 
                v = '%s%s' % (param, value)
                if v in params:  # don't include the same descriptor
                    pass
                else: 
                    bp = '%s-%s' % (bp, v)
                    params.add(v)
                    # print('... + %s' % params)
        return bp
    def select_data_source(cohort='CKD', source_type='full'):
        # example source type: {'dev', } for development (smaller cohort), {'full', 'regular'} for the complete cohort 
        assert isinstance(cohort, str)

        fpath = seqparams.getCohortGlobalDir(cohort=cohort)
        if source_type.startswith(('dev', )): 
            fpath = seqparams.getCohortGlobalDir(cohort='%s0' % cohort) # e.g. CKD0
        else: 
            raise ValueError, "Unknown source type: %s" % source_type
        return fpath

    from pattern import ckd
    import timing  # track execution time
    # import seqConfig

    # import seqConfig as sq
    # params for runWorkFlow 
    #     - MCS document 
    # 
    #        cohort 
    #        policy_segment: 'regular', 'prior', 'posterior'
    #    
    #     - d2v model 
    #       use_session_vector: True/False 
    #       window
    #       n_iter? 
    # 
    #       n_features 
    #    
    #       n_timestamps: number of timestamps/sessions to keep for predictions via sequence learning model (e.g. LSTM)
    #                     [note] only relevant when use_session_vector is True
    #      
    #    

    cohort = kargs.get('cohort', 'CKD')
    policy_segment = kargs.get('policy_segment', 'regular')  # policy values: {'regular'/noop, 'two', 'prior', 'posterior', 'complete', 'visit', }
    
    # model visit-specific documents (in which each visit -> document vector i.e. visit vector)
    tNNet = policyDeepModel() # neural net mode

    # if tNNet is True, we probably want this to be True as well ...
    # With the classicial classifier, segmenting by visits is not recommended because there isn't a good mechanism for combining 
    # session vectors besides concatenating them, which leads to long, high-D vectors. 
    tSegmentVisit = True if tNNet else False 

    # predicate = ckd.isCase    # default predicate function for segmenting documents
    predicate = None if policy_segment.startswith('reg') else ckd.isCaseCCS   # same as ckd.isCase plus minor error correction

    # C5K: 5K samples per class 
    # suffix_examples = {'C5K', }

    nPerClass = None # None: use all; 5000
    nDPerClass = None  
    d2v = kargs.get('d2v_method', vector.D2V.d2v_method) # None/default: i.e. use 'pv-dm2' (pv-dm + pv-dbow)

    # meta serves as the file ID for d2v model, training set, etc. 
    # params: policy_segment, nPerClass + (max_visit_length, n_timesteps)

    # vector.D2V.n_features * 2 if d2v in (None, 'pv-dm2', ) else vector.D2V.n_features
    n_features = determine_n_features() # params: d2v_method, n_features
    n_timesteps = kargs.get('n_timesteps', 20)  # only keep track of the last k visits => n_timesteps: k
    
    # d2v model parameters
    window = kargs.get('window', 10)
    n_iter = kargs.get('n_iter', 20)

    min_visit_length = window
    max_visit_length = 100    # default 100

    # introducing file descriptors: meta & meta_model 
    meta = '%s' % policy_segment  # tset size, include_endpoint? drop_nullcut?   e.g. C5K
    if nPerClass is not None: meta = '%s-C%d' % (meta, nPerClass) # max number of training instances per class
    if nDPerClass is not None: meta = '%s-D%d' % (meta, nDPerClass) # max number of documents per class

    # [note] what influences the file descriptor: meta? 
    #        i)  MCS segmenting policy (regular, prior, posterior)
    #        ii) use of document vector or session vectors
    #        ii) cohort: regular or test (small test set)
    meta_model = meta  # for model file, no need to train every time
    if tSegmentVisit: 
        meta_model = update_descriptor(meta_model, 'visit')

        if is_default_visit(maxlen=max_visit_length, nt=n_timesteps): # by default: max_visit_length=100, last_n_visits=10 
            meta = update_descriptor(meta, 'visit')
        else: 
            # specify parameters 
            # meta = '%s-visit-M%dt%d' % (meta, max_visit_length, n_timesteps) 
            meta = update_descriptor(meta, 'visit', mLen=min_visit_length, tLb=n_timesteps)  # tbk: lookback n timestamps

    # [note] what influences the file descriptor: meta_model? 
    #        i) cohort (regular or small)
    #        ii) vocab size
    vocabSize = 20001  # set to None to bypass this step
    tLimitVocab = False if vocabSize is None else True
    if tLimitVocab: 
        # [note] or use seqConfig.updateDescriptor(...)
        meta_model = update_descriptor(meta_model, 'vocab%d' % vocabSize) # use Tokenizer to limit vocab size -> convert to integer repr

    include_endpoint, drop_nullcut, drop_ctrl = False, False, False  # only relevant when policy_segment is not 'regular'
    if policy_segment.startswith(('pri', 'post')): 
        drop_nullcut=True  # sequences where predicate does not apply (e.g. lack of decisive diagnosis) are nullcuts
        drop_ctrl=True   # need to drop control group because there may not exist data points for say in pre-diagnostic sequences

    ### document processing 
    # a. user-specified inputs 
    tUseDevData = kargs.get('use_dev', False)
    
    inputfile = kargs.get('inputfile', None) # 'condition_drug_labeled_seq-CKD.csv'
    inputdir = kargs.get('inputdir', None) # seqparams.getCohortGlobalDir(cohort='CKD0') # small cohort (n ~ 2.8K)
    if tUseDevData: inputdir = select_data_source(cohort=cohort, source_type='dev')

    secondary_id = 'dev'
    if inputfile is not None: 
        ipath = os.path.join(inputdir, inputfile)
        assert os.path.exists(ipath), "Invaid user input path:\n%s\n" % ipath
        meta = update_descriptor(meta, secondary_id); meta_model = update_descriptor(meta_model, secondary_id)
        print('(dev) meta: %s, meta_model: %s' % (meta, meta_model))
        n_features = 50
    # t_process_docs(cohort='CKD')

    ### configure system 
    sysConfig(cohort=cohort, d2v_method=d2v, seq_ptype='regular',
        n_features=n_features, window=window, n_iter=n_iter, 
        meta=meta, meta_model=meta_model)  # [params] d2v_method, seq_ptype

    ### Modeling starts from here ### 
    tMakeTset, tClassify = kargs.get('make_tset', True), kargs.get('do_classify', True)
    tLoadD2VModel = kargs.get('load_d2v_model', True)
    tConditionalSaveD2V = False    # save the model only when the training data do not already exist

    n_samples = 0; X = y = None
    if tMakeTset: 
        ## a. full documents
        if tSegmentVisit: 

            # [note] the model output gets its file descriptor from tsHandler.meta_model
            X, y = compute_d2v_model(
                        min_ncodes=10, 
                        predicate=predicate, policy_segment=policy_segment, 

                        inputfile=inputfile, inputdir=inputdir, # user inputs; set to None to use default (for CKD experiments)

                        max_n_docs=nDPerClass,  # load only a max of this number of documents for each class
                        max_n_docs_policy='longest', 
                        
                        num_words=vocabSize, # only consider this vocab size and transform documents into integer representation
                        segment_by_visit=True, 
                        
                        # max_visit_length=max_visit_length, 
                        min_visit_length=min_visit_length, 
                        last_n_visits=n_timesteps,  # only keep track of the last 10 visits => n_timesteps: 10
                        max_n_samples=tsHandler.N_train_max * 2,  # only save this many training instances
                        conditional_save=tConditionalSaveD2V, 

                        load_model=tLoadD2VModel, test_model=False, 
                        include_augmented=False)
        else: 
            X, y = compute_d2v_model(
                        min_ncodes=10, 
                        predicate=predicate, policy_segment=policy_segment, 

                        inputfile=inputfile, inputdir=inputdir, # user inputs

                        num_words=vocabSize,  # only consider this vocab size and transform documents into integer representation
                        segment_by_visit=False,
                        max_n_docs=nDPerClass, 
                        max_n_docs_policy='longest', 

                        max_n_samples=tsHandler.N_train_max * 2, 

                        load_model=tLoadD2VModel, test_model=False, 
                        include_augmented=False)  # use the default d2v method defined in vector module
        
        # n_features here is NOT necessarily identical to vector.D2V.n_features
        n_samples, n_features = X.shape[0], X.shape[1]
    
        ## b. bag of words
        # compute_d2v_model(
        #            min_ncodes=10, 
        #            load_model=False, test_model=False, 
        #            max_features=10000, 
        #            include_augmented=False)  # bag-of-words


        ## c. pre-diagnostic, post-diagnostic segments 
        # [note] set max n_features to None to include ALL features
        # X, y = compute_d2v_model(
        #                min_ncodes=10, 
        #                predicate=predicate, policy_segment=policy_segment, 

        #                # max_n_docs=None,  # 5000

        #                include_endpoint=include_endpoint,
        #                drop_nullcut=drop_nullcut,  
        #                load_model=False, test_model=False, 
        #                include_augmented=False)  # use the default d2v method defined in vector module

        # large CKD cohort | X (dim: (389350, 2000)), y: (n_classes: 10)
        print('... t_model completed. X (dim: %s), y: (n_classes: %d)' % (str(X.shape), len(np.unique(y)) if y is not None else 1))
    
    m, M = 0.0, 1.0
    sample_ratios = [0.9, 0.1, ] # 90% for training (which is to be subsetted further if still too big), 10% for model selection
    if tClassify: 
        # binary classification 
        # t_binary_classify(**kargs)
        classifier = kargs.get('classifier', 'gradientboost')

        # multiclass classification 
        print('test> meta: %s, meta_model: %s' % (tsHandler.meta, tsHandler.meta_model))
        if tsHandler.is_sparse(): # check tsHandler.d2v

            # params: 
            #    nPerClass: use the entire sample by setting this to None 
            m, M = t_classify(X=X, y=y, mode='multiclass', n_per_class=nPerClass, tset_dtype='sparse', drop_ctrl=drop_ctrl,
                clf_name=classifier) 
            # t_classify(mode='multiclass', n_per_class=nPerClass, tset_dtype='sparse', drop_ctrl=drop_ctrl,
            #     clf_name='gradientboost') 
        else: 
            # load training set from the saved file
            X=y=None; gc.collect()  # get training set from disk
            deep_classifier = kargs.get('deep_classifier', 'lstm')

            if tNNet: 
                # import dnn_utils for example networks
                # sample_ratios = [0.1, 0.05, ]  # training/test, validation ... only use a fraction of the total sample size
                epochs_ms, batch_size_ms = 100, 32
                epochs, batch_size = kargs.get('epochs', 3000), kargs.get('batch_size', 32)
                n_lookback = n_timesteps

                # [note] use t_deep_classify(...) for regular-sized data (< 10G); t_deep_classify2() for big data (>= 10G)
                m, M = t_deep_classify2(mode='multiclass', n_per_class=nPerClass, tset_dtype='dense', 
                           drop_ctrl=drop_ctrl, 
                           last_n_visits=n_lookback, 

                           clf_name=deep_classifier,

                           # this is only used for specifying the fraction of data for model selection
                           ratios=[],  # set to empty list to bypass model selection; # e.g. 10% for training, 5% for validation, the rest ignored
                           ratio = 0.7, # used to specify the ratio of data used for training when N is small (< 10K)
                           
                           model_selection=False, # only applied to t_deep_classify()
                           epochs_ms=epochs_ms, batch_size_ms=batch_size_ms,   # other params: patience_ms,
                           patience_ms=100,

                           n_trials=1, 
                           patience=300, 
                           epochs=epochs, batch_size=batch_size)
            else: 
                m, M = t_classify(mode='multiclass', n_per_class=nPerClass, tset_dtype='dense', drop_ctrl=drop_ctrl, 
                    ratios=sample_ratios,  # ... 10% for training, 5% for validation, the rest ignored
                    clf_name='gradientboost') # don't use 'classifier_name'

    ### re-labeing 
    # t_label(**kargs)

    return (m, M)  # min and max performance scores (e.g. AUC)

def test(**kargs): 


    return

def runQuery(**kargs):
    """
    Answer research questions to be covered in the paper or to quench your ultimate thirst of curiosity. 

    """
    processMCSDocuments()

    return

def runWorkFlowBatch(**kargs):
    """
    A wrapper function that executes runWorkFlow() multiple times. 

    Memo
    ----


    """
    def estimate_performance(scores, low=0.05, high=0.95, msg='n/a'): 
        res = sampling.ci4(scores, low=low, high=high)
        mean = res['mean']
        median = res['median']
        std_err = res['se']
        ci_low, ci_high = res['ci_low'], res['ci_high']

        print('(estimate_performance) score (%s): [%f, %f] | mean=%f, median=%f, SE=%f' % \
            (msg, ci_low, ci_high, mean, median, std_err))
        # todo: consider labels
        return (ci_low, ci_high)

    from sampler import sampling

    nTrials = 20
    nGBTrees = 500
    lowers, uppers = [], []
    for i in range(nTrials):
        nt = i+1
        div(message='Beginning Trial #%d (n_est=%d)' % (nt, nGBTrees), symbol='%')
        m, M = runWorkFlow(load_model=True, n_estimators=nGBTrees)
        lowers.append(m); uppers.append(M)
        div(message='End Trial #%d' % nt, symbol='%')

    estimate_performance(lowers, msg='min AUC')
    estimate_performance(uppers, msg='max AUC')
    return

if __name__ == "__main__": 
    
    runWorkFlow(cohort='CKD', policy_segment='regular', 

        # data source & training set
        use_dev=True,  # if True, use a dev set (i.e. smaller cohort such as CKD0)
        make_tset=True, 

        # d2v model
        load_d2v_model=True, 

        # classifier switch
        do_classify=True, 
        use_deep_model=False,

        # classifier 
        classifier='gradientboost', # relevant only when use_deep_model is False
        deep_classifier='lstm', # relevant only when use_deep_model is True

        # NNs, sequence models
        use_session_vector=False, 
        n_features=200, n_timesteps=20, epochs=3000, batch_size=32) # example workflow (for running experiments) are specified in this subroutine

    # runWorkFlowBatch() # [note] usually applied to classificaiton

    # test() # test suite is specified here  


