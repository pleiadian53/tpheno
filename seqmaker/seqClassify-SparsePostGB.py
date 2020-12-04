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

# local modules (assuming that PYTHONPATH has been set)
from batchpheno import icd9utils, qrymed2, utils, dfUtils  # sampling is obsolete
from config import seq_maker_config, sys_config
# from batchpheno.utils import div
from system.utils import div
from pattern import medcode as pmed

import sampler  # sampling utilities
import seqparams
import analyzer
import vector
import seqAnalyzer as sa 
import seqUtils, plotUtils
from tset import TSet  # base class is defined in seqparams

import evaluate  # classification
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
class tsHandler(object):  # <- seqparams, tset.TSet

    # default parameter values
    is_augmented = False 
    cohort = 'CKD'
    seq_ptype = ctype = 'regular'
    d2v = d2v_method = 'pv-dm2'  # pv-dm, bow
    is_simplified = False
    meta = 'D'
    dir_type = 'combined'

    # class state
    is_configured = False

    @staticmethod
    def init(**kargs):   
        return tsHandler.config(**kargs)
    @staticmethod
    def config(cohort, seq_ptype='regular', d2v_method=None, simplify_code=False, 
            is_augmented=False, dir_type='combined', meta=None):
        if d2v_method is None: d2v_method = vector.D2V.d2v_method 
        user_file_descriptor = meta
        tsHandler.cohort = cohort
        tsHandler.ctype = tsHandler.seq_ptype = seqparams.normalize_ctype(seq_ptype)
        tsHandler.d2v = d2v_method 
        tsHandler.is_simplified = simplify_code
        tsHandler.is_augmented = is_augmented
        tsHandler.dir_type = dir_type
        tsHandler.meta = tsHandler.get_tset_id() if meta is None else meta # use training set ID as default
        print('config> d2v: %s, user descriptor (model, tset, mcs): %s' % (d2v_method, user_file_descriptor))
        print('        cohort: %s, ctype: %s' % (tsHandler.cohort, tsHandler.ctype))
        print('            + augmented? %s, simplified? %s dir_type=%s' % \
            (tsHandler.is_augmented, tsHandler.is_simplified, tsHandler.dir_type))
        tsHandler.is_configured = True
        return
    @staticmethod
    def is_configured(): 
        return tsHandler.is_configured
    @staticmethod
    def is_sparse(): 
        return True if tsHandler.d2v.startswith(('bow', 'bag', 'aph', )) else False
    @staticmethod
    def load_tset(index=0, meta=None, sparse=False):
        tSparse = sparse # kargs.get('sparse', False)
        if tSparse: 
            return tsHandler.loadSparse(index=index, meta=meta)   # (X, y)
        return tsHandler.load(index=index, meta=meta) 
    @staticmethod
    def load(index=0, meta=None):  # cohort, seq_ptype, d2v_method (focused_labels, label_map)
        # this should conform to the protocal in makeTSetCombined()

        # if sparse: 
        #     X, y = tsHandler.loadSparse(index=index, meta=meta)
        #     return 

        # ts = TSet.load(cohort=cohort_name, d2v_method=d2v_method, seq_ptype=seq_ptype, 
        #         dir_type='combined', index=0) # [params] index
        ts = TSet.load(cohort=tsHandler.cohort, 
                       d2v_method=tsHandler.d2v, 
                       seq_ptype=tsHandler.ctype, 
                       index=index,  # CV index (if no CV, then 0 by default)
                       dir_type=tsHandler.dir_type, 
                       suffix=tsHandler.meta if not meta else meta) 
        assert ts is not None and not ts.empty  
        tsHandler.profile(ts)
        return ts
    @staticmethod
    def loadSparse(index=0, meta=None):
        # load sparse matrix
        X, y = TSet.loadSparse(cohort=tsHandler.cohort, 
                            d2v_method=tsHandler.d2v, 
                            seq_ptype=tsHandler.ctype, 
                            index=index,  # CV index (if no CV, then 0 by default)
                            dir_type=tsHandler.dir_type, 
                            suffix=tsHandler.meta if not meta else meta)  
        if X is not None and y is not None: assert X.shape[0] == len(y)
        return (X, y)
    @staticmethod
    def save_tset(**kargs):
        return tsHandler.save(**kargs) 
    @staticmethod
    def save(X, y, index=0, docIDs=[], meta=None, sparse=False): # [params] (X_train, X_test, y_train, y_test)
        # can also use 'tags' (in multilabel format)
        print('tsHandler> save document vectors (cv=%d), sparse? %s ...' % (index, sparse))  # e.g. tpheno/seqmaker/data/CKD/train/tset-IDregular-pv-dm2-GCKD.csv
        # suffix=kargs.get('tset_id', kargs.get('seq_ptype', None))
        ts = None
        if not sparse: 
            ts = TSet.toCSV(X, y=y, save_=True, index=index,   
                               docIDs=docIDs,     # if a doc is segmented, then it's useful to know from which documents each segment came from
                               outputdir=tsHandler.get_tset_dir(),  # depends on cohort, dir_type
                            
                               d2v_method=tsHandler.d2v, 
                               cohort=tsHandler.cohort, 
                               seq_ptype=tsHandler.ctype, 
                               suffix=tsHandler.meta if not meta else meta) # [params] labels, if not given => all positive (1)
        else: 
            TSet.saveSparseMatrix(X, y=y,
                    docIDs=docIDs, 
                    outputdir=tsHandler.get_tset_dir(), 
                    d2v_method=tsHandler.d2v, 
                               cohort=tsHandler.cohort, 
                               seq_ptype=tsHandler.ctype, 
                               suffix=tsHandler.meta if not meta else meta)
            # saves two files .npz (X) and .csv (y)
        return ts # or None
    @staticmethod
    def saveSparse(): 
        pass 

    @staticmethod
    def load_mcs(index=0, meta=None, inputdir=None, ifiles=[]): # use processDocuments()
        """
        Each training set should come with a document source file, which is derived from the source file. 

        Params
        ------
        meta: user-defined file ID

        Example
        -------
        1.  a. source file is derived directly from the OHASI DB; e.g. 

                tpheno/CKD/condition_drug_labeled_seq-CKD.csv
                    pattern: tpheno/<cohort>/condition_drug_labeled_seq-<cohort>.csv

            b.  derived mcs file is derived from the source (upon segmenting, simplifying, slicing operations, etc) and 
                it is from the derived mcs file that a d2v-model-based training set file is generated. 

                dervied mcs file is usually kept under a separate directory from the source is prefixed by 'mcs'; e.g. 

                    tpheno/seqmaker/data/CKD/combined/mcs-n0-IDregular-pv-dm2-D-GCKD.csv
                    
        """
        ### load + transfomr + (ensure that labeled_seq exists)
        
        # this is the MCS file from which the training set is made (not the source or orginal)
        src_dir = tsHandler.get_tset_dir() if inputdir is None else inputdir  
        assert os.path.exists(src_dir), "Invalid input dir: %s" % src_dir
        ctype = tsHandler.ctype # kargs.get('seq_ptype', 'regular')
        

        D, L, T = processDocuments(cohort=tsHandler.cohort, seq_ptype=ctype, 
                    inputdir=src_dir, 
                    ifiles=kargs.get('ifiles', []),
                    meta=meta, 
                    
                    # document-wise filtering 
                    policy=kargs.get('doc_filter_policy', 'minimal_evidence'),  
                    min_ncodes=0,  # retain only documents with at least n codes

                    # content modification
                    # predicate=kargs.get('predicate', None), # reserved for segment_docs()
                    simplify_code=tsHandler.is_simplified, 

                    source_type='default', 
                    create_labeled_docs=False)  # [params] composition
        print('load_mcs> nD: %d | cohort=%s, ctype=%s, labeled? %s, simplified? %s' % \
            (len(D), tsHandler.cohort, ctype, tsHandler.is_labeled_data(L), tsHandler.is_simplified))
        return (D, L, T)
    @staticmethod
    def save_mcs(D, T=[], L=[], index=0, docIDs=[], meta=None):
        """
        
        Params
        ------
        meta: user-defined file ID

        """
        from tdoc import TDoc
        # can also use 'tags' (in multilabel format)
        print('tsHandler> save transformed documents parallel to tset (cv=%d) ...' % index)  # e.g. tpheno/seqmaker/data/CKD/train/tset-IDregular-pv-dm2-GCKD.csv
        # suffix=kargs.get('tset_id', kargs.get('seq_ptype', None))
        
        # this should basically share the same file naming parameters as TSet.toCSV()
        tsDoc = TDoc.toCSV(D, T=T, L=L, save_=True, index=index,   # index: k in k-fold CV or a trial index
                            docIDs=docIDs,   # if a doc is segmented, then it's useful to know from which documents each segment came from
                            outputdir=tsHandler.get_tset_dir(),  # depends on cohort, dir_type
                            
                            d2v_method=tsHandler.d2v, 
                            cohort=tsHandler.cohort, 
                            seq_ptype=tsHandler.ctype, 
                            suffix=tsHandler.meta if not meta else meta) # [params] labels, if not given => all positive (1)
        return tsDoc # or None
    @staticmethod
    def profile(ts): 
        from tset import TSet
        col_target = TSet.target_field
        Lv = ts[col_target].unique()
        sizes = {}

        # sizes = collections.Counter(ts[col_target].values)
        for label in Lv: 
            sizes[label] = ts.loc[ts[col_target] == label].shape[0]
        print('tsHandler.profile> Found %d unique labels ...' % len(Lv))
        for label, n in sizes.items(): 
            print('  + label=%s => N=%d' % (label, n))
        return  
    @staticmethod
    def profile2(X, y): 
        sizes = collections.Counter(y)
        for label, n in sizes.items(): 
            print('  + label=%s => N=%d' % (label, n))
        return 
    @staticmethod
    def is_labeled_data(lx): 
        # if lx is None: return False
        nL = len(set(lx))
        if nL <= 1: 
            return False 
        return True
    @staticmethod     # [design] this can be a friend function or TSet's class method
    def get_tset_id(ts=None, y=[]): # (ts, y, include_augmetned)
        """

        Memo
        ----
        1. A typical training set file looks like: 
           tset-n0-IDregular-pv-dm2-A-GCKD.csv

           full path: tpheno/seqmaker/data/<cohort>/combined/tset-n0-IDregular-pv-dm2-A-GCKD.csv
        """
        meta = 'D'  # default (not distinguishing labeled (L) or unlabeled (U))
        nL = 0
        if ts is not None:  
            col_target = TSet.target_field
            nL = len(ts[col_target].unique())
            meta = 'U' if nL <= 1 else 'L' # labeled or unlabeled?
        elif y: 
            nL = len(set(y))
            meta = 'U' if nL <= 1 else 'L' # labeled or unlabeled?
        if tsHandler.is_augmented: meta = 'A'  # augmented data included?
        # if kargs.get('simplify_code', False): meta = '%s%s' % (meta, 'S')
        return meta 
    @staticmethod
    def get_tset_dir(): 
        from tset import TSet
        tsetPath = TSet.getPath(cohort=tsHandler.cohort, dir_type=tsHandler.dir_type)  # ./data/<cohort>/
        print('tsHandler> training set dir: %s' % tsetPath)
        return tsetPath 

    @staticmethod
    def get_model_id(y=[]): # <- L
        """
        Depends on training set property (get_tset_id()), which depends on labels (y or L)

        Memo
        ----
        1. A typical model file looks like: 
           

           Note that the suffix naming follows the convention of the ID field of the training set file name
                tset-n0-IDregular-pv-dm2-A-GCKD.csv => ID string: regular-pv-dm2-A-GCKD
        """    
        # add to model file to distinguish different d2v models in addition to its parameter-specific identifier
        # ID: {'U', 'L', 'A'} + ctype + cohort
        #     unlabeld (U), labeled (L) or augmented (A) (i.e. plus unlabeled)
        ctype = tsHandler.ctype # seqparams.normalize_ctype(kargs.get('seq_ptype', 'regular'))
        meta = '%s-%s' % (ctype, tsHandler.d2v)
        
        tset_type = tsHandler.get_tset_id(y=y)  # property of the training data ('D'/default, 'L'/labeled, 'U'/unlabeled, 'A'/augmetned)
        meta = '%s-%s' % (meta, tset_type)
        
        # cohort_name = kargs.get('cohort', '?')
        if not tsHandler.cohort in ('?', 'n/a', None): meta = '%s-G%s' % (meta, tsHandler.cohort)
        if len(meta) == 0: meta = 'generic' # if no ID info is given, then it is generic test model
        return meta

    @staticmethod
    def get_model_dir(): 
        # (!!!) this could have been included in vector.D2V but do not want D2V to have to pass in 'cohort'
        #       the directory in which model is kept
        dir_type = 'model'
        modelPath = TSet.getPath(cohort=tsHandler.cohort, dir_type=dir_type, create_dir=True)  # ./data/<cohort>/model
        print('tsHandler> model dir: %s' % modelPath)
        return modelPath

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

def makeTSet(**kargs): 
    """
    Make training set data (based on w2v and d2v repr) for classification and clusterig
    
    Input
    -----
    labels: user-provided labels (when coding sequence dataframe does not have labels)
    cohort 

    Params
    ------
    a. paramters that distinguish training data 

    d2v_method
    cohort
    seq_ptype

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

    Reference
    ---------
    1. label encoding
       http://scikit-learn.org/stable/modules/preprocessing_targets.html#preprocessing-targets

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
    def save_to_csv(cv_id=None): # [params] (X_train, X_test, y_train, y_test)
        seq_ptype = kargs.get('seq_ptype', 'regular')
        simplify_code = kargs.get('simplify_code', False)

        suffix = None # 'simplified' if simplify_code else None
        # can also use 'tags' (in multilabel format)
        print('    + save training data (cv=%d) ...' % cv_id)  # e.g. tpheno/seqmaker/data/CKD/train/tset-IDregular-pv-dm2-GCKD.csv
        ts_train = TSet.toCSV(X_train, y=y_train, save_=True, dir_type='train', index=cv_id, 
                         d2v_method=d2v_method, cohort=cohort_name, seq_ptype=seq_ptype, suffix=suffix) # [params] labels, if not given => all positive (1)
    
        print('    + save test data (cv=%d) ...' % cv_id) # e.g. tpheno/seqmaker/data/CKD/test/tset-IDregular-pv-dm2-GCKD.csv
        ts_test = TSet.toCSV(X_test, y=y_test, save_=True, dir_type='test', index=cv_id, 
                         d2v_method=d2v_method, cohort=cohort_name, seq_ptype=seq_ptype, suffix=suffix)
        return (ts_train, ts_test)  


    # import matplotlib.pyplot as plt
    from tset import TSet
    from labeling import TDocTag
    from pattern import medcode as pmed 
    from sklearn.preprocessing import LabelEncoder
    # import vector

    cohort_name = kargs.get('cohort', '?')
    assert not cohort_name == '?' or kargs.has_key('ifiles'), "If cohort is not given, you must provide (doc) source paths."
    
    # output directory for the d2v model 
    outputdir = TSet.getPath(cohort=cohort_name, dir_type=None)  # ./data/<cohort>/

    # [input] cohort, labels, (result_set), d2v_method, (w2v_method), (test_model)
    #         simplify_code, filter_code
    #         
    div(message='1. Read temporal doc files ...')

    ### load + transform + create labeled data
    # load params: cohort, seq_ptype, use_surrogate
    # transform params: predicate, simplify_code
    D, L, T = processDocuments(cohort=cohort_name, seq_ptype=kargs.get('seq_ptype', 'regular'),
                    predicate=kargs.get('predicate', None), simplify_code=kargs.get('simplify_code', False))  # [params] composition

    labels = L  # documents = D; timestamps = T
    assert len(labels) > 0, "processDocuments should guarantee at least a default labeling"

    nDoc = len(D)
    ulabels = np.unique(labels)
    n_classes = ulabels.shape[0]
    print('    + unique labels:\n%s\n' % ulabels)

    div(message='2. Compute document embedding (params: ) ...') # [note] use W2V and D2V classes in seqparams and vector to adjust d2v parameters
    d2v_method = kargs.get('d2v_method', vector.D2V.d2v_method)
    ratio_test = 0.2
    print('    + train test split (test: %f%%) | cohort=%s, n_classes=%d' % (0.2*100, cohort_name, n_classes))  
    
    # cv = StratifiedKFold(n_splits=6) 
    ts = DataFrame()
    le = LabelEncoder()
    nTrials = kargs.get('n_trials', 1)
    ts_train = ts_test = None  # [output]
    for cv_id in range(nTrials): 
        D_train, D_test, l_train, l_test = train_test_split(D, labels, test_size=ratio_test)
        print('   + after train-test split: type(D_train[i], D_train)=(%s, %s) example: %s' % (type(D_train[0]), type(D_train), D_train[0]))

        n_classes_train, n_classes_test = np.unique(l_train).shape[0], np.unique(l_test).shape[0]

        D_augmented = kargs.get('augmented_docs', [])

        # [test] status: ok 
        nD_train, nD_test, nD_total = len(D_train), len(D_train), len(D); r = nD_test/(nD_total+0.0)
        print('    ++ train test split (r_test: %f=?=%f) | n_classes_train: %d, n_classes_test: %d' % (r, ratio_test, n_classes_train, n_classes_test))

        ### binarize label if needed (this can be cohort dependent)
        div('2a. Label processing (e.g. binarize labels) ... ')

        print('    + class labeling ...')
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
    
        # document labeling: labelize returns list of documents (not np.array() which is what we want here)
        print('    + document labeling ...')
        D_train, D_test = labelize(D_train, label_type='train', class_labels=y_train), labelize(D_test, label_type='test', class_labels=y_test)
        if len(D_augmented) > 0: 
    	    D_augmented = labelize(D_augmented, label_type='augmented')
            # D_train = np.concatenate((D_train, D_augmented))
            D_train = D_train + D_augmented

        # [params] debug: cohort 
        # want documents to be in list type from here on 
        assert isinstance(D_train, list) and isinstance(D_test, list)
        
        print('    + computing document vectors ...')
        X_train, X_test = vector.getDocVec2(D_train, D_test, 
        	                                   d2v_method=d2v_method, 
        	                                   outputdir=outputdir, 
                                               model_id=kargs.get('model_id', None), 
        	                                   test_=kargs.get('test_model', True), 
        	                                   load_model=kargs.get('load_model', True),
        	                                   cohort=cohort_name)

        # feature size doubled 
        assert X_train.shape[0] == len(D_train), "X_train.shape[0]: %d != nD_train: %d" % (X_train.shape[0], len(D_train))
        assert X_train.shape[1] == (2 * vector.D2V.n_features), \
            "X_train: (%d by %d), n_features: %d" % (X_train.shape[0], X_train.shape[1], vector.D2V.n_features)
        assert X_test.shape[0] == len(D_test), "X_test.shape[0]: %d != nD_test: %d" % (X_test.shape[0], len(D_test))
        assert X_test.shape[1] == (2 * vector.D2V.n_features), "X_test: %d by %d" % (X_test.shape[0], X_test.shape[1])
  
        # [note] computing model separately is not correct (because then the doc vec in test data cannot leverage training data)
        # X_train = getDocVec(D_train, d2v_method=d2v_method, label_type='train',
        #                 outputdir=outputdir,
        #                 test_=kargs.get('test_model', True), 
        #                 load_model=kargs.get('load_model', True), 
        #                 cohort=cohort_name) # [params] w2v_method, outputdir, outputfile
        # X_test =  getDocVec(D_test, d2v_method=d2v_method, label_type='test',
        #                 outputdir=outputdir,
        #                 test_=kargs.get('test_model', True), 
        #                 load_model=kargs.get('load_model', True), 
        #                 cohort=cohort_name) # [params] w2v_method, outputdir, outputfile
        
        # save one train-test split

        # lookuptb = ret['symbol_chart']
        print('status> Model computation complete.')
        if kargs.get('save_', True): 
            div(message='3. Save training set')
            # e.g. .../data/PTSD/tset_nC1_IDregular-sg-tfidfavg-GPTSD.csv
            # can also use 'tags' (in multilabel format)

            ts_train, ts_test = save_to_csv(cv_id=ci_id)

        # yield (X_train, X_test, y_train, y_test)
        # if cv_id == 0: 
        #     ts = pd.concat([ts_train, ts_test], ignore_index=True) 
    ### end foreach trial 
    return (ts_train, ts_test)  # save one of the split

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
        # (!!!) this could have been included in vector.D2V but do not want D2V to have to pass in 'cohort'
        #       the directory in which model is kept
        if tsHandler.is_configured: 
            model_path = tsHandler.get_model_dir()
        else: 
            dir_type = 'model'
            model_path = TSet.getPath(cohort=cohort_name, dir_type=dir_type, create_dir=True)  # ./data/<cohort>/model
        print('get_model_dir> model dir: %s' % model_path)
        return model_path

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
    
        ### document labeling: labelize returns list of documents (not np.array() which is what we want here)
        D = labelize(D, label_type='doc')  # No need to use class labels for document label prefixing

        # [params] debug: cohort 
        # want documents to be in list type from here on 
        assert isinstance(D, list) 
        
        ### computing document vectors 
        X = vector.getDocVec(docs=D, d2v_method=d2v_method, 
                                outputdir=outputdir, 
                                meta=get_model_id(), # {'U', 'L', 'A'} + ctype + cohort
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
    from tset import TSet
    from labeling import TDocTag
    from pattern import medcode as pmed 
    from sklearn.preprocessing import LabelEncoder
    # import vector

    # loadDocuments2(cohort=kargs.get())

    raise NotImplementedError, "Coming soon." 

def makeTSetCombined(**kargs): # same as makeTSet but does not distinguish train and test split in d2v
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
    def config(): # configure training set related parameters
        # if tsHandler.is_configured: 
        #     # noop
        #     return
        tsHandler.config(cohort=kargs.get('cohort', 'CKD'), 
            seq_ptype=seqparams.normalize_ctype(kargs.get('seq_ptype', 'regular')), 
            d2v_method=kargs.get('d2v_method', vector.D2V.d2v_method), 
            simplify_code=kargs.get('simplify_code', False), is_augmented=kargs.get('is_augmented', False), 
            dir_type='combined')
        return 
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
        src_dir = get_global_cohort_dir()
        ctype = tsHandler.ctype # kargs.get('seq_ptype', 'regular')
        D, L, T = processDocuments(cohort=tsHandler.cohort, seq_ptype=ctype, 
                    inputdir=src_dir, 
                    ifiles=kargs.get('ifiles', []),
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
    def save_tset(X, y, cv_id=0, docIDs=[], meta=None, sparse=False): # [params] (X_train, X_test, y_train, y_test)
        return tsHandler.save(X, y, index=cv_id, docIDs=docIDs, meta=meta, sparse=sparse)
    def load_tset(cv_id=0, meta=None, sparse=False):  # cohort, seq_ptype, d2v_method (focused_labels, label_map)
        # this should conform to the protocal in makeTSetCombined()
        return tsHandler.load_tset(index=cv_id, meta=meta, sparse=sparse)
    def save_mcs(D, T, L, index=0, docIDs=[], meta=None):  # save the coding sequences
        return tsHandler.save_mcs(D, T, L, index=index, docIDs=docIDs, meta=meta)
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
    def segment_docs(predicate=None, policy='regular'):  # <- D, L, T
        # policy: {'regular', 'two'/'halves', 'prior'/'posterior', 'complete', }
        # output: (DocIDs, D, L, T)
        
        include_endpoint = kargs.get('include_endpoint', True)  # include the chunk containing diagnosis info?
        drop_docs_without_cutpoints = kargs.get('drop_nullcut', False)  # if True, do not include doc without valid cut points (diagnosis info)

        # configure file ID 
        # if predicate is None or policy is regular, then no-op
        if predicate is not None and not policy.startswith('reg'):
            if not tsHandler.meta: tsHandler.meta = policy_segment  # other values 'A', 'U', 'D', 'L' 
            assert tsHandler.meta and not (tsHandler.meta in ('D', 'default', )), \
                "default user-defined file ID is not recommended since segmenting op will modify the documents (policy=%s)" % policy_segment
            print('segment_docs> predicate: %s, policy: %s' % (predicate, policy))
        else: 
            print('segment_docs> policy: %s' % policy)
        return docProc.segmentDocuments(D, L, T, predicate=predicate, policy=policy, 
                    inclusive=include_endpoint, drop_nullcut=drop_docs_without_cutpoints) 
    def is_sparse(): 
        if tsHandler.d2v.startswith(('bow', 'bag', 'aphri')):  # aphridate
            return True 
        return False 
    def sample_docs(D, L, T, n, sort_index=True, random_state=53):
        # if n > len(y): n = None # noop
        idx = cutils.sampleByLabels(L, n_samples=n, verbose=True, sort_index=sort_index, random_state=random_state)
        D = list(np.array(D)[idx])
        L = list(np.array(L)[idx])
        T = list(np.array(T)[idx])
        assert len(D) == n 
        print('  + subsampling: select only %d docs.' % len(D))
        return (D, L, T)
    def subsample(X, y, n=None, sort_index=True, random_state=53):
        """
        Subsampling operation applied after doc vec X is derived. 
        """
        if n is not None: 
            idx = cutils.samplePerClass(y, n_per_class=n, sort_index=sort_index, random_state=random_state)
            print('  + X (dim0: %s -> dim: %s)' % (X.shape[0], len(idx)))
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

    # import matplotlib.pyplot as plt
    from tset import TSet
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
    user_file_descriptor = meta = kargs.get('meta', tsHandler.meta) 
    include_augmented = kargs.get('include_augmented', False)

    # [input] cohort, labels, (result_set), d2v_method, (w2v_method), (test_model)
    #         simplify_code, filter_code  
    div(message='1. Read temporal doc files ...')

    # load + transform + (segment) + labelize (also ensure that labeled_seq exists in processDocuments())
    # D, D_augmented, L, T = assemble_docs()   # return [D, L, T, D2, L2, T2]? 

    # load + transform + labelize   [params] min_ncodes
    D, L, T = process_docs(); verify_docs(D, min_length=kargs.get('min_ncodes', 10), n_test=100)  # [test]

    # incorporate augmented (unlabeled) documents if available
    Da, La, Ta = [], [], []
    nDoc = len(D)
    nDocAug = nDocTotal = 0
    if include_augmented: 
        Da, La, Ta = process_aug_docs() # load + transform + labelize
        nDocAug = len(Da)
    if len(Da) > 0: D, L, T = D+Da, L+La, T+Ta 
    nDocTotal = nDoc+nDocAug

    # subsampling #1: large corpus while preserving the percentage of the class labels
    max_n_docs = kargs.get('max_n_docs', None)
    if max_n_docs: 
        D, L, T = sample_docs(D, L, T, n=max_n_docs) # set None to take ALL
        kargs['save_doc'] = True # definitely have to save the new documents
 
    # segment each document into paragraphs or smaller parts (e.g. prediagnosis sequence)
    # [note] DocIDs are used to associate segments to the original document (which may be broken down into several segments). 
    #        Under no-op, DocIDs are just regular positional indices.  
    func_segment, p_segment = kargs.get('predicate', None), kargs.get('policy_segment', 'regular')
    docIDs, D, L, T = segment_docs(predicate=func_segment, policy=p_segment)  # <- policy_segment, inclusive?, include_active?

    # labelize documents (required by gensim's d2v models)
    Dl = labelize(D, label_type='doc', class_labels=L)  # use class labels for document label prefixing

    ### [todo] separate the above to a separate function

    ### labeling
    ulabels = process_label()
    n_classes = len(ulabels)

    div(message='2. Compute document embedding (params: ) ...') # [note] use W2V and D2V classes in seqparams and vector to adjust d2v parameters
    # ratio_test = 0.2
    # print('    + train test split (test: %f%%) | cohort=%s, n_classes=%d' % (0.2*100, cohort_name, n_classes))  

    # cv = StratifiedKFold(n_splits=6) 
    ts = DataFrame()
    nTrials = kargs.get('n_trials', 1)

    # input: D, labels, D_augmented
    ts_output = None
    Din = D if d2v_method.startswith(('bow', 'bag', )) else Dl  # d2v model requires labeled documents but, bag-of-words and other sparse models do not 
    nDocEff = len(Din)
    print('makeTSetCombined> nDocTotal: %d =?= nDocEff: %d' % (nDocTotal, nDocEff)) # after all doc processing (segmentations, etc), nDoc may change
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
                                meta=user_file_descriptor,   # user-defined file ID: get_model_id(y=L), # {'U', 'L', 'A'} + ctype + cohort
                                labels=L,  # assess() will evaluate d2v accuracy based on the labels

                                test_=kargs.get('test_model', True), 
                                load_model=kargs.get('load_model', True),

                                max_features=kargs.get('max_features', None), # only applies to d2v_method <- 'bow' (bag-of-words sparse model) 
                                cohort=tsHandler.cohort)

        assert X.shape[0] == nDocEff and X.shape[0] == len(L)
        if not is_sparse(): 
            X = X[:nDoc] # excluding augmented doc vectors 
            # condition: feature size doubled if using pv-dm2

            # lookuptb = ret['symbol_chart']
            
            ### Save training set
            # e.g. .../data/PTSD/tset_nC1_IDregular-sg-tfidfavg-GPTSD.csv
            # can also use 'tags' (in multilabel format)

            if kargs.get('save_', True):     
                # Output location: e.g. .../data/<cohort>/cv/
                # can also use 'tags' (in multilabel format)

                ts = save_tset(X, L, cv_id, docIDs=docIDs, meta=user_file_descriptor) # [params] X_train, X_test, y_train, y_test
                # if cv_id == 0: ts_output = ts  # only save the resulted training data from one trial

        else: 
            print('makeTSetCombined> Sparse matrix (X): dim=%s, d2v=%s' % (str(X.shape), d2v_method))
            save_tset(X, y, cv_id, docIDs=docIDs, meta=user_file_descriptor, sparse=True)

        print('status> Model computation complete (@nTrial=%d)' % cv_id)
        if kargs.get('save_doc', True): 
            tsDoc = save_mcs(D, T, L, index=cv_id, docIDs=docIDs, meta=user_file_descriptor)  # D has included augmented if provided
            
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
    # ts_train, ts_test = loadTSet(**kargs)
    
    cohort_name = kargs.get('cohort', None) 
    d2v_method = kargs.get('d2v_method', vector.D2V.d2v_method)
    X_train, y_train = TSet.getXY(cohort=cohort_name, d2v_method=d2v_method, dir_type='train') # cohort <- None will take on a default value (generic)
    X_test, y_test = TSet.getXY(cohort=cohort_name, d2v_method=d2v_method, dir_type='test')

    return (X_train, X_test, y_train, y_test)

def loadTSet(**kargs): 
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

# def loadTSet(cohort, d2v_method=None, ctype='regular', user_file_descriptor='D', index=0):
#     if d2v_method is None: d2v_method = vector.D2V.d2v_method 
#     return loadTSetCombined(cohort, d2v_method=d2v_method, seq_ptype=ctype, meta=user_file_descriptor, index=index)
def loadTSetCombined(**kargs): 
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
    def modify_tset(ts): 
        focused_classes = kargs.get('focused_labels', None) # None: N/A
        if focused_classes is not None: # only preserve these classes and everything else becomes "Others"
            # e.g. In CKD data, one may only want to focus on predicting ‘Stage 1’, ‘Stage 2’, ‘Stage 3a’, ... 'Stage 5'
            ts = focus(ts, labels=focused_classes, other_label='Others')
        lmap = kargs.get('label_map', seqparams.System.label_map)
        if lmap: # relabel so that classes only retain the main CKD stages
            print('loadTSetCombined> Modifying training data ...\n> Prior to re-labeling ...')
            profile(ts)
            ts = merge(ts, lmap)
            print('> After re-labeling ...')
            profile(ts)
        return ts
    def remove_classes(ts, labels=[], other_label='Others'):
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
            print('  + after subsampling, size(ts)=%d, n_classes=%d (same?)' % (ts.shape[0], n_classes_prime))
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
    ts = tsHandler.load(index=kargs.get('index', 0), meta=kargs.get('meta', None)) 
    
    ### scaling is unnecessary for d2v
    if kargs.get('scale_', False): 
        pass
    #     # scaler = StandardScaler(with_mean=False)
    #     scaler = MaxAbsScaler()
    #     X = scaler.fit_transform(X)

    ### modify classes
    ts = modify_tset(ts)  # <- label_map, focused_labels
    n_classes0 = check_tset(ts)

    # drop explicit control data in multiclass
    if kargs.get('drop_ctrl', False): # drop control group (e.g. label: 'Others')
        assert n_classes0 > 2, "Not a multiclass domain (n_classes=%d)" % n_classes0
        ts = remove_classes(ts, labels=['Others', ])

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
    def modify_tset(X, y): 
        focused_classes = kargs.get('focused_labels', None) # None: N/A
        if focused_classes is not None: # only preserve these classes and everything else becomes "Others"
            # e.g. In CKD data, one may only want to focus on predicting ‘Stage 1’, ‘Stage 2’, ‘Stage 3a’, ... 'Stage 5'
            y = focusLabels(y, labels=focused_classes, other_label='Others')  # output in np.ndarray

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
    def modify_tset(X, y): 
        focused_classes = kargs.get('focused_labels', None) # None: N/A
        if focused_classes is not None: # only preserve these classes and everything else becomes "Others"
            # e.g. In CKD data, one may only want to focus on predicting ‘Stage 1’, ‘Stage 2’, ‘Stage 3a’, ... 'Stage 5'
            y = focusLabels(y, labels=focused_classes, other_label='Others')  # output in np.ndarray

        lmap = kargs.get('label_map', seqparams.System.label_map)
        if lmap: 
            profile(X, y)
            y = mergeLabels(y, lmap=lmap)
            print('> After re-labeling ...')
            profile(X, y)
        return (X, y)
    def remove_classes(X, y, labels=[], other_label='Others'):
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
    X, y = tsHandler.loadSparse(index=kargs.get('index', None), meta=kargs.get('meta', None)) 
    
    ### scaling X
    if kargs.get('scale_', False): 
        # scaler = StandardScaler(with_mean=False)
        scaler = MaxAbsScaler()
        X = scaler.fit_transform(X)

    ### modify classes
    if X is not None and y is not None: 
        X, y = modify_tset(X, y)

    # drop explicit control data in multiclass
    n_classes0 = len(np.unique(y)) 
    if kargs.get('drop_ctrl', False): # drop control group (e.g. label: 'Others')
        assert n_classes0 > 2, "Not a multiclass domain (n_classes=%d)" % n_classes0
        X, y = remove_classes(X, y, labels=['Others', ])

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
                ts = modify_tset(ts)  
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
        ms.runCVROCMulticlass(X, y, classifier=classifier, prefix=outputdir, identifier=identifier, target_labels=target_labels)
        return  

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
    roc_cv(X, y, classifier=classifier, target_labels=targets) 

    return 

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
    
    ### tsHandler routines
    def config():
        if tsHandler.is_configured: return 
        tsHandler.config(cohort=kargs.get('cohort', 'CKD'), 
            seq_ptype=seqparams.normalize_ctype(kargs.get('seq_ptype', 'regular')), 
            d2v_method=kargs.get('d2v_method', vector.D2V.d2v_method), 
            simplify_code=kargs.get('simplify_code', False), is_augmented=kargs.get('is_augmented', False))
        return
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
                ts = modify_tset(ts)  # <- focused_labels, label_map
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

    def modify_tset(ts): 
        focused_classes = kargs.get('focused_labels', None) # None: N/A
        if focused_classes is not None: # only preserve these classes and everything else becomes "Others"
            # e.g. In CKD data, one may only want to focus on predicting ‘Stage 1’, ‘Stage 2’, ‘Stage 3a’, ... 'Stage 5'
            ts = focus(ts, labels=focused_classes, other_label='Others')
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
    def roc_cv(X, y, classifier, fpath=None, target_labels=[]):    
        identifier = make_file_id()
        outputdir = Graphic.getPath(cohort=tsHandler.cohort, dir_type='plot', create_dir=True)
        # classifier = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)

        # [note] can use nested CV but too expensive
        # evaluation
        # ms.runCVROC(X, y, classifier=classifier, fpath=fpath)   # remember to call plt.clf() for each call (o.w. plots overlap)
        ms.runCVROCMulticlass(X, y, classifier=classifier, prefix=outputdir, identifier=identifier, target_labels=target_labels)
        return 
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
    def relabel(ts):  # ad-hoc
        # CKD stage classes
        lmap = {}
        lmap['Others'] = ['CKD G1-control', 'CKD G1A1-control', 'Unknown', ]  # ... control data 
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
    classifier, classifier_name = resolve_classifier(X, y)  # [params] classifier (assumed tuned) | classifier_name (=> model selection), param_grid

    # roc_per_class: To prevent the figure from being clobbered, select only, say 3, classes to present in the ROC curve
    targets = kargs.get('roc_per_class', ['CKD Stage 1', 'CKD Stage 3', 'CKD Stage 5', ])  # assuming that CKD Stage 3a, 3b are merged
    validate_labels()
    roc_cv(X, y, classifier=classifier, target_labels=targets)  
    
    # plotROCMultiClass(fpr, tpr, roc_auc, n_classes=n_classes, 
    #     identifier=kargs.get('identifier', 'generic'), outputdir=kargs.get('outputdir', os.getcwd()))

    return 

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
    if not lmap: lmap = System.label_map

    Ls = Series(L)
    labelSet = Ls.unique()
    print('  + (before) unique labels:\n%s\n' % labelSet)
    for label, eq_labels in lmap.items(): 
        print('  + %s <- %s' % (label, eq_labels))
        cond_pos = Ls.isin(eq_labels)
        Ls.loc[cond_pos] = label  # rename to designated label 

    labelSet2 = Ls.unique()   
    print('mergeLabels> n_labels: %d -> %d' % (len(labelSet), len(labelSet2)))
    print('  + (after) unique labels:\n%s\n' % labelSet2)

    return np.array(Ls)
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
       lmap['Others'] = [‘G1-control’, ‘G1A1-control’, 'Unknown']  ... control data 
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

def focusLabels(L, labels, other_label='Others'):
    """
    Out of all the possible labels in L, focus only on the labels in 'labels'

    """
    if not hasattr(labels, '__iter__'): 
        labels = [labels, ]

    N = len(L)
    Ls = Series(L)
    all_labels = Ls.unique()
    assert len(set(labels)-set(all_labels))==0, "Some of the focused classes %s are not part of the label set:\n%s\n" % (labels, str(labels))

    cond_pos = Ls.isin(labels)
    lsp = Ls.loc[cond_pos]
    lsn = Ls.loc[~cond_pos] # other classes
    
    print('focus> n_focus: %d, n_other: %d' % (lsp.shape[0], lsn.shape[0]))
    lsn.loc[:] = other_label

    return pd.concat([lsp, lsn]).sort_index().values # recover original indexing; ignore_index <- False by default
def focus(ts, labels, other_label='Others'): 
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
    from tset import TSet  # [todo] use seqmaker.tset

    if not hasattr(labels, '__iter__'): 
        labels = [labels, ]

    N = ts.shape[0]
    col_target = TSet.target_field
    all_labels = ts[col_target].unique()
    assert len(set(labels)-set(all_labels))==0, "Some of the focused classes %s are not part of the label set:\n%s\n" % (labels, str(labels))

    cond_pos = ts[col_target].isin(labels)
    tsp = ts.loc[cond_pos]
    tsn = ts.loc[~cond_pos] # other classes
    
    print('focus> n_focus: %d, n_other: %d' % (tsp.shape[0], tsn.shape[0]))
    tsn.loc[:, col_target] = other_label

    return pd.concat([tsp, tsn]).sort_index()  # recover original indexing
    
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
    ts = loadTSetCombined(n_per_class=kargs.get('n_per_class', None))  

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
    def load_tset(fpath, index=0):
        ts = None
        if os.path.exists(fpath) and os.path.getsize(fpath) > 0: 
            ts = pd.read_csv(fpath, sep=TSet.sep, header=0, index_col=False, error_bad_lines=True)
        else: 
            msg = 't_classify> Warning: training set (cohort=%s) does not exist in:\n%s\n' % (cohort, fpath)
            raise ValueError, msg
            # ts = tsHandler.load(index)
        assert ts is not None and not ts.empty, "t_classify> Null training set."
        return ts
    def relabel(): 
        # CKD stage classes
        lmap = {}
        lmap['Others'] = ['CKD G1-control', 'CKD G1A1-control', 'Unknown', ]  # ... control data 
        lmap['CKD Stage 5'] = ['ESRD after transplant', 'ESRD on dialysis', ]
        lmap['CKD Stage 3'] = ['CKD Stage 3a', 'CKD Stage 3b', ]
        print('  + Relabeling the data set according to the following map:\n%s\n' % lmap)
        return lmap
    def sample_docs(D, L, T, n, sort_index=True, random_state=53):
        # if n > len(y): n = None # noop
        idx = cutils.sampleByLabels(L, n_samples=n, verbose=True, sort_index=sort_index, random_state=random_state)
        D = list(np.array(D)[idx])
        L = list(np.array(L)[idx])
        T = list(np.array(T)[idx])
        assert len(D) == n 
        print('  + subsampling: select only %d docs.' % len(D))
        return (D, L, T)
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
        if name.startswith(('rand', 'rf')):  # n=389K
            # max_features: The number of features to consider when looking for the best split; sqrt by default
            # sample_leaf_options = [1,5,10,50,100,200,500] for tuning minimum sample leave (>= 50 usu better)
            clf = RandomForestClassifier(n_jobs=12, 
                    # random_state=53, 
                    n_estimators=100,   # default: 10
                    min_samples_split=250, min_samples_leaf=50)  # n_estimators=500
        elif name.startswith('log'):
            clf = LogisticRegression(class_weight='balanced', solver='saga', penalty='l2')
        elif name == 'sgd':  
            clf = SGDClassifier(loss='log', penalty='elasticnet', max_iter=1500, tol=1e-3) # l1_ratio: elastic net mixing param: 0.15 by default
        elif name.startswith( ('grad', 'gb') ):  # gradient boost tree
            # min_samples_split=250, min_samples_leaf=50 
            # max_leaf_nodes: If None then unlimited number of leaf nodes.
            # subsample: fraction of samples to be used for fitting the individual base learners
            #            Choosing subsample < 1.0 leads to a reduction of variance and an increase in bias
            nTrees = kargs.get('n_estimators', 500)
            clf = GradientBoostingClassifier(n_estimators=nTrees, learning_rate=0.1, 
                # random_state=53, 
                min_samples_split=250, min_samples_leaf=50, max_depth=8,  # prevent overfitting
                max_features = 'sqrt', # Its a general thumb-rule to start with square root.
                subsample=0.82)
        else: 
            raise ValueError, "Unrecognized classifier: %s" % name
        return clf
    def experimental_settings(): 
        print('t_classify> tset type %s, maxNPerClass: %s, drop control? %s' % \
            (kargs.get('tset_dtype', 'dense'), kargs.get('n_per_class', None), kargs.get('drop_ctrl', False)))

        print('  + cohort: %s, ctype: %s' % (tsHandler.cohort, tsHandler.ctype))
        print('  + d2v: %s, params: ' % (tsHandler.d2v, ))
        print('  + userFileID: %s' % kargs.get('meta', tsHandler.meta))
        return

    import evaluate
    import classifier.utils as cutils
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
    from sklearn.linear_model import LogisticRegression, Lasso, LassoCV, SGDClassifier
    
    from sklearn.preprocessing import StandardScaler, MaxAbsScaler # feature scaling for sparse repr. 

    # [params]
    #    set is_augmented to True to use augmented training set
    # tsHandler.config(cohort='CKD', seq_ptype='regular', is_augmented=True) # 'regular' # 'diag', 'regular'
    mode = kargs.get('mode', 'multiclass')  # values: 'binary', 'multiclass'
    param_grid = None

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
    clf = choose_classifier(name=kargs.pop('clf_name', 'rf')) # rf: random forest
    clf_list.append(clf)  # supports multiclass 

    # gradient boosting: min_samples_split=2
    # param_grid = {'n_estimators':[500, 100], 'min_samples_split':[2, 5], 'max_features': [None, 2, 5], 'max_leaf_nodes': [None, 4]}
    # clf_list.append( GradientBoostingClassifier(n_estimators=500, learning_rate=0.1, random_state=53, max_features=2, max_leaf_nodes=4) )
    # clf = GradientBoostingClassifier(n_estimators=1000, learning_rate=0.1, random_state=53, max_features=2, max_leaf_nodes=4)

    # random_state = np.random.RandomState(0)
    # clf = svm.SVC(kernel='linear', probability=True, random_state=random_state)

    focusedLabels = None # None to include ALL  # focus(), merge()
    classesOnROC = ['CKD Stage 1', 'CKD Stage 3', 'CKD Stage 5', ] # assuming that CKD Stage 3a and 3b are merged into CKD Stage 3

    # can provide customized training set (ts); if None, then ts is automatically determined by tsHanlder.load()
    # ts(cohort, d2v, ctype, meta('A', 'D', ), index)
    ts_dtype = kargs.get('tset_dtype', 'dense')
    maxNPerClass = kargs.get('n_per_class', None)
    userFileID = kargs.get('meta', tsHandler.meta) 
    tDropControlGroup = kargs.get('drop_ctrl', False)

    experimental_settings()
    if ts_dtype.startswith('d'): 

        # [note] set n_per_class to None => subsampling now is delegated to makeTSetCombined() 
        # load, (scale), modify, subsample
        ts = loadTSetCombined(label_map=seqparams.System.label_map, 
               n_per_class=maxNPerClass, 
               drop_ctrl=tDropControlGroup) # all parameters should have been configured
        print('t_classify> load training set of dim: %s' % str(ts.shape))

        # subsampling 
        # if maxNPerClass: 
        #     ts = subsample2(ts, n=maxNPerClass)

        # precedence: classifier_name -> classifier
        summary(ts=ts)
        for clf in clf_list: 
            multiClassEvaluate(ts=ts, cohort=tsHandler.cohort, seq_ptype=tsHandler.ctype, # not needed 
                classifier=clf, 
                # classifier_name='l1_logistic',   # if not None, classifier_name takes precedence over classifier
                focused_labels=focusedLabels, 
                roc_per_class=classesOnROC,
                param_grid=param_grid, 
                label_map=seqparams.System.label_map, # use sysConfig to specify
                meta=userFileID, identifier=None) # use meta or identifier to distinguish different classificaiton tasks; set to None to use default
    else: 
        X, y = kargs.get('X', None), kargs.get('y', None)
        if X is None or y is None: 
            # load, (scale), modify, subsample
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
        for clf in clf_list: 
            multiClassEvaluateSparse(X=X, y=y, cohort=tsHandler.cohort, seq_ptype=tsHandler.ctype, # not needed 
                classifier=clf, 
                # classifier_name='l1_logistic',   # if not None, classifier_name takes precedence over classifier
                focused_labels=focusedLabels, 
                roc_per_class=classesOnROC,
                param_grid=param_grid, 
                label_map=seqparams.System.label_map, # use sysConfig to specify
                meta=userFileID, identifier=None) # use meta or identifier to distinguish different classificaiton tasks; set to None to use default

    # [todo] loop through cartesian product of seq_ptype and d2v_method? 

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

def t_model(**kargs):
    """


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
    cohort_name = kargs.get('cohort', tsHandler.cohort) # sysparams.cohort=cohort_name # system parameters are shared across modules
    d2v_method = kargs.get('d2v_method', tsHandler.d2v)
    seq_ptype = seqparams.normalize_ctype(kargs.get('seq_ptype', tsHandler.ctype)) # 'regular'
    user_file_descriptor = kargs.get('meta', tsHandler.meta)
    div(message='t_model> cohort=%s, d2v_method=%s, ctype=%s, descriptor=%s, segment? %s' % \
        (cohort_name, d2v_method, seq_ptype, user_file_descriptor, verify_segment_op()))


    # [options]
    # makeTSet: train test split (randomized subsampling, n_trials is usu. 1)
    # makeTSetCombined: no separation of train and test data on the d2v level 
    # makeTSetCV: 
    print('t_model> predicate: %s' % kargs.get('predicate', None))
    print('         + max_features: %s, max_n_docs: %s' % (kargs.get('max_features', 'unbounded'), kargs.get('max_n_docs', 'unbounded')))
    X, y = makeTSetCombined(d2v_method=d2v_method, 
                            seq_ptype=seq_ptype, 
                            cohort=cohort_name, 

                            max_features=kargs.get('max_features', None), # only applies to d2v_method <- 'bow'
                            max_n_docs=kargs.get('max_n_docs', None), # max number of documents used to build d2v model (usu for debug only)

                            # document selection 
                            min_ncodes=kargs.get('min_ncodes', 10),  # process_docs()

                            # segmenting input MCSs/documents
                            predicate=kargs.get('predicate', None), 
                            policy_segment=kargs.get('policy_segment', 'regular'), 
                            include_endpoint=kargs.get('include_endpoint', False), 
                            drop_nullcut=kargs.get('drop_nullcut', False), # drop documents without cutopints? 

                            meta=user_file_descriptor, # user-defined file ID (d2v model, training set, derived mcs file)
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
    t_model(corhot=cohort_name, seq_ptype=seq_ptype, load_model=True)

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
            if focused_classes is not None: # only preserve these classes and everything else becomes "Others"
                # e.g. In CKD data, one may only want to focus on predicting ‘Stage 1’, ‘Stage 2’, ‘Stage 3a’, ... 'Stage 5'
                ts = focus(ts, labels=focused_classes, other_label='Others')

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

        n_others = ts.loc[ts[col_target] == 'Others'].shape[0]  # should be 0 before re-labeling
        print('  + N(Others): %d' % n_others)
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
        lmap['Others'] = ['CKD G1-control', 'CKD G1A1-control', 'Unknown', ]  # ... control data 
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

def sysConfig(cohort, d2v_method=None, seq_ptype='regular', meta='tbd', **kargs):
    """
    Configure system-wide paramters applicable to all modules that import seqparams.

    Params
    ------
    """
    def relabel():  # ad-hoc
        # CKD stage classes
        lmap = {}
        lmap['Others'] = ['CKD G1-control', 'CKD G1A1-control', 'Unknown', ]  # ... control data 
        lmap['CKD Stage 5'] = ['ESRD after transplant', 'ESRD on dialysis', ]
        lmap['CKD Stage 3'] = ['CKD Stage 3a', 'CKD Stage 3b', ]
        return lmap

    from seqparams import System as sysparams 
    sysparams.cohort = cohort # system parameters are shared across modules 
    sysparams.label_map = relabel()

    # training set parameters 
    if d2v_method is None: d2v_method = vector.D2V.d2v_method
    user_file_descriptor = meta
    tsHandler.config(cohort=sysparams.cohort, d2v_method=d2v_method, 
                    seq_ptype=kargs.get('seq_ptype', 'regular'),
                    meta=user_file_descriptor, **kargs)  # is_augmented/False, is_simplified/False, dir_type/'combined'
    return 

def test(**kargs): 
    def includeEndPoint(): 
        ret = {'meta': 'prior_inclusive', 'include_endpoint': True}
        return

    from pattern import ckd
    # import seqConfig as sq

    cohort = 'CKD'
    policy_segment = 'posterior'  # policy values: {'regular'/noop, 'two', 'prior', 'posterior', 'complete', }
    # predicate = ckd.isCase    # default predicate function for segmenting documents
    predicate = ckd.isCaseCCS   # same as ckd.isCase plus minor error correction

    # C5K: 5K samples per class 
    # suffix_examples = {'C5K', }
    meta = '%s' % policy_segment  # tset size, include_endpoint? drop_nullcut?   e.g. C5K
    nPerClass = None # None: use all; 5000
    d2v = 'bow' # None: default   # sparse: {'bow', 'bag'}

    include_endpoint, drop_nullcut, drop_ctrl = False, False, True  # only relevant when policy_segment is not 'regular'
    if policy_segment.startswith(('pri', 'post')): 
        drop_nullcut=True  # sequences where predicate does not apply (e.g. lack of decisive diagnosis) are nullcuts
        drop_ctrl=True   # need to drop control group because there may not exist data points
    sysConfig(cohort=cohort, meta=meta, d2v_method=d2v)  # [params] d2v_method, seq_ptype

    ### document processing 
    # t_process_docs(cohort='CKD')

    ### training document vectors 
    
    # a. full documents
    # t_model(min_ncodes=10, 
    #            load_model=True, test_model=False, 
    #            include_augmented=False)  # use the default d2v method defined in vector module
    
    # b. bag of words
    # t_model(min_ncodes=10, 
    #            load_model=False, test_model=False, 
    #            max_features=10000, 
    #            include_augmented=False)  # bag-of-words


    # c. pre-diagnostic, post-diagnostic segments 
    # [note] set max n_features to None to include ALL features
    # X, y = t_model(min_ncodes=10, 
    #                predicate=predicate, policy_segment=policy_segment, 

    #                # max_n_docs=None,  # 5000

    #                include_endpoint=include_endpoint,
    #                drop_nullcut=drop_nullcut,  
    #                load_model=True, test_model=False, 
    #                include_augmented=False)  # use the default d2v method defined in vector module
    # print('... t_model completed. X (dim: %s), y: (n_classes: %d)' % (str(X.shape), len(np.unique(y)) if y is not None else 1))
    # binary classification 
    # t_binary_classify(**kargs)

    nTrees = kargs.pop('n_estimators', 500)

    # multiclass classification 
    print('test> meta: %s' % tsHandler.meta)
    if tsHandler.is_sparse(): # check tsHandler.d2v
        # t_classify(X=X, y=y, mode='multiclass', n_per_class=nPerClass, tset_dtype='sparse', drop_ctrl=drop_ctrl,
        #     clf_name='sgd') 
        # example tset: tset-IDregular-bow-regular-GCKD.npz
        t_classify(mode='multiclass', n_per_class=nPerClass, tset_dtype='sparse', drop_ctrl=drop_ctrl, 
            clf_name='gbt', n_estimators=nTrees) 
    else: 
        # load training set from the saved file
        t_classify(mode='multiclass', n_per_class=None, tset_dtype='dense', drop_ctrl=drop_ctrl, 
            clf_name='gbt', n_estimators=nTrees)

    ### re-labeing 
    # t_label(**kargs)

    return

def test_batch(**kargs):
    nTrials = 20
    nGBTrees = 500
    for i in range(nTrials):
        nt = i+1
        div(message='Beginning Trial #%d (n_est=%d)' % (nt, nGBTrees), symbol='%')
        test(load_model=True, n_estimators=nGBTrees)
        div(message='End Trial #%d' % nt, symbol='%')
    return

if __name__ == "__main__": 
    test_batch()