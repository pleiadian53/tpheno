# encoding: utf-8

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

# local modules (assuming that PYTHONPATH has been set)
from batchpheno import icd9utils, qrymed2, utils, dfUtils  # sampling is obsolete
from config import seq_maker_config, sys_config
# from batchpheno.utils import div
from system.utils import div
from pattern import medcode as pmed

import sampler  # sampling utilities
import seqparams
import vector
import seqUtils, plotUtils
# from tset import TSet  # base class is defined in seqparams

import algorithms, seqAlgo  # count n-grams, sequence-specific algorithms
import labeling

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

def loadDocuments(cohort, inputdir=None, source_type='source', **kargs):
    """
    Load documents (and additionally, timestamps and labels if available). 

    Params
    ------
    cohort
    basedir
    
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

    Use
    ---
    D, L, T = loadDocuments(cohort, inputdir, source_type
                                    use_surrogate=False, 
                                    single_label_format=True)  

    """
    def load_docs():  # source_type: {'s'/'source' (default), 'l'/labeld,  'a'/'augmented', 'r'/'raw', }
        ret = kargs.get('result_set', {}) # 'sequence', 'timestamp', 'label' 
        sourceType = kargs.get('source_type', 's')  # {a', 'r', 'l', 's'}
        if not ret: 
            if sourceType.startswith('a'): # augmented 
                ret = load_augmented_docs() 
            if sourceType.startswith('l'):
                ret = load_labeled_docs() 
            elif sourceType.startswith('r'): # raw 
                ret = load_raw_docs()
            else:  # everything else is default
                ifiles = kargs.get('ifiles', [])
                ret = sr.readDocFromCSV(cohort=cohort, inputdir=inputdir, ifiles=ifiles, complete=True) # [params] doctype (timed)
        return ret
    def load_augmented_docs(label_default=None):  
        ret = sr.readAugmentedDocFromCSV(cohort=cohort, inputdir=inputdir, ifiles=[]) # [params] doctype (timed) 
        return ret
    def load_raw_docs(): # load documents from .dat file
        ret = kargs.get('result_set', {}) # 'sequence', 'timestamp', 'label' 
        if not ret:   # this is assuming that document sources exist in .csv format 
            ifiles = kargs.get('ifiles', [])
            dx, tx = sr.readTimedDocPerPatient(cohort=cohort, inputdir=inputdir, ifiles=ifiles)
            assert len(dx) == len(tx), "The size between documents and timestamps is not consistent."
            ret['sequence'] = dx 
            ret['timestamp'] = tx 
        return ret
    def load_labeled_docs():  # 
        ret = kargs.get('result_set', {}) # 'sequence', 'timestamp', 'label' 
        
        # [note] this function should be agnostic to seq_ptype
        # fpath = TDoc.getPath(cohort=kargs['cohort'], seq_ptype=kargs.get('seq_ptype', 'regular'), 
        #     doctype='labeled', ext='csv', basedir=prefix) 
        # assert os.path.exists(fpath), "labeled docuemnt source should have been created previously (e.g. by processDocuments)"
        # df_src = pd.read_csv(fpath, sep='|', header=0, index_col=False, error_bad_lines=True) # ['sequence', 'timestamp', 'label']
        if not ret: 
            ifiles = kargs.get('ifiles', [])
            ret = sr.readDocFromCSV(cohort=cohort, inputdir=inputdir, ifiles=ifiles, complete=True, 
                            stratify=True, label_name='label') # [params] doctype (timed)

            # this 'ret' is indexed by labels
        return ret
    def is_labeled_data(lx): 
        nL = len(set(lx))
        if nL <= 1: 
            return False 
        # print('  + Detected labels (n=%d) in augmented documents.' % nL)
        return True
    def get_global_cohort_dir(): 
        return seqparams.getCohortGlobalDir(cohort) # sys_config.read('DataExpRoot')/<cohort> 
    def parse_row(df, col='sequence', sep=','): # [note] assuming that visit separators ';' were removed
        seqx = [] # a list of (lists of tokens)  
        
        isStr = False
        for i, row in enumerate(df[col].values):
            if isinstance(row, str): 
                isStr = True
            if i >= 3: break
            
        if isStr:  # sequence, timestamp
            for row in df[col].values:   # [note] label is not string
                tokens = row.split(sep)
                seqx.append(tokens)
        else: 
            # integer, etc. 
            seqx = list(df[col].values)
        return seqx
    def show_stats(): 
        print('seqReaderApp.load> nD: %d, nT: %d, nL: %d' % (len(D), len(T), len(L)))
        if is_labeled_data(L): 
            n_classes = len(set(L))  # seqparams.arg(['n_classes', ], default=1, **kargs) 
            print('  + Stats: n_docs: %d, n_classes:%d | cohort: %s' % (len(D), n_classes, cohort))
        else: 
            print('  + Stats: n_docs: %d, n_classes:? (no labeling info) | cohort: %s' % (len(D), cohort)) 
        return
    # import matplotlib.pyplot as plt
    # from sklearn.feature_extraction.text import TfidfVectorizer
    # from pattern import diabetes as diab 
    from seqparams import TSet, TDoc
    from labeling import TDocTag
    # import seqAnalyzer as sa 
    import seqReader as sr
    # import vector

    # [input] cohort, labels, (result_set), d2v_method, (w2v_method), (test_model)
    #         simplify_code, filter_code    
    if inputdir is None: inputdir = get_global_cohort_dir()

    # [params] cohort   # [todo] use classes to configure parameters, less unwieldy
    # seq_ptype = kargs.get('seq_ptype', 'regular')
    tSingleLabelFormat = kargs.get('single_label_format', True)  # single label format ['l1', 'l2', ...] instead of multilabel format [['l1', 'l2', ...], [], [], ...]
    # tSurrogateLabels = kargs.get('use_surrogate', False) # use surrogate/nosiy labels if the dataframe itself doesn't not carry labeling information

    # [params]
    # read_mode = seqparams.TDoc.read_mode  # assign 'doc' (instead of 'seq') to form per-patient sequences
    # docSrcDir = kargs.get('basedir', TDoc.prefix) # sys_config.read('DataExpRoot')  # document source directory

    ### load model
    # 1. read | params: cohort, inputdir, doctype, labels, n_classes, simplify_code
    #         | assuming structured sequencing files (.csv) have been generated
    div(message='1. Read temporal doc files ...')

    # [note] csv header: ['sequence', 'timestamp', 'label'], 'label' may be missing
    # [params] if 'complete' is set, will search the more complete .csv file first (labeled > timed > doc)
    
    # if result set (sequencing data is provided, then don't read and parse from scratch)
    ret = load_docs()  # params: source_type
    D, T, L = ret['sequence'], ret.get('timestamp', []), ret.get('label', [])
    nD, nT, nL = len(D), len(T), len(L) 

    ### determine labels and n_classes 
    ### [note] document labeling is now automatic that depends on label_type (train, test, validation, unlabeled, etc.)

    hasTimes = True if len(T) > 0 else False
    hasLabels = True if len(L) > 0 else False   # caveat of is_labeled_data(L): L at this point is a list of lists
    # [condition] len(seqx) == len(tseqx) == len(labels) if all available

    if hasTimes: assert len(D) == len(T), "Size inconsistency between (D: %d) and (T: %d)" % (len(D), len(T))

    # labels is not the same as tags (a list of lists)
    if hasLabels and tSingleLabelFormat: 
        # use the first label as the label by default (pos=0)
        L = TDocTag.toSingleLabel(L, pos=0) # to single label format (for single label classification)
        print('    + labels (converted to single-label format, n=%d): %s' % (len(np.unique(L)), np.unique(L)))
        assert len(D) == len(L), "Size inconsistency between (D: %d) and (L: %d)" % (len(D), len(L))

        # [condition] n_classes determined
    show_stats()
    return (D, L, T)

def makeLabeledDocuments(cohort, inputdir=None, **kargs):
    """
    From document source (generated via the cohort module) to labeled_seq .csv file. 
    """
    def load_docs(): # <- cohort, (inputdir, source_type, include_augmeted?, single_label_format?)
        # include_augmeted = kargs.get('include_augmeted', False) # include unlabeled data for semi-supervised learning? 
        tSingleLabel = kargs.get('single_label_format', True)
        D, L, T = loadDocuments(cohort=cohort, inputdir=inputdir,
                                    source_type='default',  
                                    use_surrogate=False, 
                                    single_label_format=tSingleLabel)  # [params] composition
        return (D, L, T)
    def get_global_cohort_dir(): 
        return seqparams.getCohortGlobalDir(cohort) # sys_config.read('DataExpRoot')/<cohort> 
    def make_labeled_docs(): # [params] cohort, doctype, seq_ptype, doc_basename, ext + basedir
        # also see seqReader.verifyLabeledSeqFile()
        assert len(L) > 0, "coding sequences are not labeled (at least use default all-positive labeling)." 
        seq_ptype = kargs.get('seq_ptype', 'regular')
        fpath = TDoc.getPath(cohort=cohort, seq_ptype=seq_ptype, doctype='labeled', ext='csv', basedir=inputdir)  # usually there is one file per cohort  
        print('  + saving labeled .csv file to:\n%s\n' % fpath)
        if kargs.get('save_', True) and not os.path.exists(fpath): 
            # path params: cohort=get_cohort(), seq_ptype=seq_ptype, doctype=doctype, ext='csv', baesdir=basedir
            
            # labels = L 
            # if not labels: 
            #     # create dummy labels [todo]
            #     labels = [1] * len(D)
            assert len(T) > 0, "timestamps were not found in this source (cohort=%s)" % cohort
            assert len(D) == len(T), "inconsistent sizes between sequences and timestamps"

            sr.readDocToCSV(sequences=D, timestamps=T, labels=L, cohort=cohort, seq_ptype=seq_ptype, outputdir=inputdir)
            # condition: a labeled source documents is generated 
        return 

    import seqReader as sr
    if inputdir is None: inputdir = get_global_cohort_dir()    

    D, L, T = load_docs()
    make_labeled_docs()  # D, L, T

    return 

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
    def load_docs(): # <- cohort_name, (inputdir, source_type, include_augmeted?, single_label_format?)
        # include_augmeted = kargs.get('include_augmeted', False) # include unlabeled data for semi-supervised learning? 
        tSingleLabel = kargs.get('single_label_format', True)
        inputdir = kargs.get('inputdir', get_global_cohort_dir())
        sourceType = kargs.get('source_type', 's') # source_type: {'s'/'source' (default),  'a'/'augmented', 'r'/'raw', }
        D, L, T = loadDocuments(cohort=cohort_name, inputdir=inputdir,
                                    source_type=sourceType,  
                                    use_surrogate=False, 
                                    single_label_format=tSingleLabel)  # [params] composition
        return (D, L, T)
    def get_global_cohort_dir(): 
        return seqparams.getCohortGlobalDir(cohort_name) # sys_config.read('DataExpRoot')/<cohort> 
    def transform_docs(D, L, T): # params: seq_ptype, predicate, simplify_code
        seq_ptype = kargs.get('seq_ptype', 'regular')
        # [params] items, policy='empty_doc', predicate=None, simplify_code=False
        predicate = kargs.get('predicate', None)
        simplify_code = kargs.get('simplify_code', False)

        nD, nD0 = len(D), len(D[0])
        # this modifies D 
        D2, L2, T2 = transformDocuments2(D, L=L, T=T, policy='empty_doc', seq_ptype=seq_ptype, 
            predicate=predicate, simplify_code=simplify_code, save_=False) # save only if doesn't exist 
        # D, labels = transformDocuments(D, L=labels, seq_ptype=seq_ptype)

        print('    + (after transform) nDoc: %d -> %d, size(D0): %d -> %d' %  (nD, len(D2), nD0, len(D2[0])))
        print('    + (after transform) nD: %d, nT: %d, nL: %d' % (len(D2), len(T2), len(L2)))

        return (D2, L2, T2)
    def do_splice(): 
        if not kargs.has_key('splice_policy'): return False
        if kargs['splice_policy'].startswith(('noop', 'reg', 'compl', )):  # effective operations 
            return False 
        return True  # prior, posterior [todo] 'in-between'; e.g. from first occurence of X to last occurence of X
    def splice_docs(D, T):   # use: only consider LCS labels in the pre- or post- diagnostic sequences
        # inplace operation by default
        if not do_splice(): return (D, T)  # noop
        print('  + splicing sequences in D and T | policy=%s, predicate=%s, cutpoint=%s, inclusive? %s' % \
            (kargs.get('splice_policy', 'noop'), kargs.get('splice_predicate', None), kargs.get('cutpoint', None), kargs.get('inclusive', True)))
        nD0 = len(D)
        D, T = st.spliceDocuments(D, T=T, 
                        policy=kargs.get('splice_policy', 'noop'), 
                        cohort=cohort_name, 
                        predicate=kargs.get('splice_predicate', None), # infer predicate from cohort if possible
                        cutpoint=kargs.get('cutpoint', None), 
                        n_active=1, 
                        inclusive=kargs.get('inclusive', True))
        assert len(D) == len(T)
        assert len(D) == nD0, "size of document set should not be different after splicing nD: %d -> %d" % (len(D), nD0) 
        return (D, T)
    def make_labeled_docs(): # [params] cohort, doctype, seq_ptype, doc_basename, ext + basedir
        # also see seqReader.verifyLabeledSeqFile()
        assert len(L) > 0, "coding sequences are not labeled (at least use default all-positive labeling)." 

        # if kargs.get('include_augmeted', False): 
        #     # no-op
        #     print('  + No labeled .csv file is created for now if include_augmeted is set to True')
        #     return 
        inputdir = kargs.get('inputdir', get_global_cohort_dir())
        seq_ptype = kargs.get('seq_ptype', 'regular')
        fpath = TDoc.getPath(cohort=cohort_name, seq_ptype=seq_ptype, doctype='labeled', ext='csv', basedir=inputdir)  # usually there is one file per cohort  
        if kargs.get('save_', True) and not os.path.exists(fpath): 
            # path params: cohort=get_cohort(), seq_ptype=seq_ptype, doctype=doctype, ext='csv', baesdir=basedir
            
            # labels = L 
            # if not labels: 
            #     # create dummy labels [todo]
            #     labels = [1] * len(D)
            assert len(T) > 0, "timestamps were not found in this source (cohort=%s)" % cohort_name
            assert len(D) == len(T), "inconsistent sizes between sequences and timestamps"

            sr.readDocToCSV(sequences=D, timestamps=T, labels=L, cohort=cohort_name, seq_ptype=seq_ptype, outputdir=prefix)
            # condition: a labeled source documents is generated 
        return 
    def get_cohort(): 
        try: 
            return kargs['cohort']
        except: 
            pass 
        raise ValueError, "cohort info is mandatory."

    import seqReader as sr
    import seqTransform as st
    from seqparams import TDoc  # or use tdoc.TDoc 

    cohort_name = get_cohort()
    
    # verify the existence of labele_seq file 
    # sr.verifyLabeledSeqFile(corhot=cohort_name, seq_ptype=, ext='csv', **kargs): # [params] cohort, doctype, seq_ptype, doc_basename, ext + basedir
    D, L, T = load_docs()  # [params] inputdir, composition

    # labels = L  # documents = D; timestamps = T
    if len(L) == 0: L = kargs.get('labels', [1] * len(D)) # if no labeling found => user-provided label set 

    # ensure that labeled_seq file exists
    if kargs.get('create_labeled_docs', False): make_labeled_docs()  # include labeling to the sequence source file
    assert len(D) == len(L) == len(T), "Size mismatch: nD=%d, nL=%d, nT=%d" % (len(D), len(L), len(T))

    ### document transfomration
    print('processDocuments> Begin document transformation operations (e.g. simplify, diag-only, etc)')

    # [params] items, policy='empty_doc', predicate=None, simplify_code=False
    D, L, T = transform_docs(D, L, T) 

    # prior, posterior only? note that labeled sequence source above contain the entire sequence, not the spliced version
    D, T = splice_docs(D, T)  # params: splice_policy; this should not change L

    # condition: 
    # a. documents transformed ~ seq_type
    # b. labeled source document (doctype='labeled') is created => the triple: (D, L, T) is guaranteed 

    return (D, L, T)

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
    def get_global_cohort_dir(): 
        return seqparams.getCohortGlobalDir(cohort_name) # sys_config.read('DataExpRoot')/<cohort> 
    def load_labeled_docs():  # <- seq_ptype
        ret = kargs.get('result_set', {}) # 'sequence', 'timestamp', 'label' 
        # prefix = inputdir
        # [note]  
        if not ret: 

            # [note] this is the path to ctype-dependent labeled documents but this is not necessary 
            # for now, just load the full documents and then extract the desired portion on the fly.  
            # fpath = TDoc.getPath(cohort=cohort_name, seq_ptype=seq_ptype, 
            #                 doctype='labeled', ext='csv', basedir=prefix) 

            ifiles = kargs.get('ifiles', [])
            ret = sr.readDocFromCSV(cohort=cohort_name, inputdir=inputdir, ifiles=ifiles, doctype='labeled', 
                            stratify=True, label_name=kargs.get('label_name', 'label')) # [params] label_name: {'label', 'label_lcs'}
            assert len(ret) > 0, "No data found given cohort=%s, doctype=%s, prefix=%s" % (cohort_name, doctype, inputdir)
        return ret # 'ret' is a nested dictionary indexed by labels
    def transform_docs(D, L, T): # params: seq_ptype, predicate, simplify_code
        # seq_ptype = kargs.get('seq_ptype', 'regular')
        # [params] items, policy='empty_doc', predicate=None, simplify_code=False
        predicate = kargs.get('predicate', None)
        simplify_code = kargs.get('simplify_code', False)

        nD, nD0 = len(D), len(D[0])
        # this modifies D 
        D2, L2, T2 = st.transformDocuments2(D, L=L, T=T, policy='empty_doc', seq_ptype=seq_ptype, 
            predicate=predicate, simplify_code=simplify_code, save_=False) # save only if doesn't exist 
        # D, labels = transformDocuments(D, L=labels, seq_ptype=seq_ptype)

        print('    + (after transform) nDoc: %d -> %d, size(D0): %d -> %d' %  (nD, len(D2), nD0, len(D2[0])))
        print('    + (after transform) nD: %d, nT: %d, nL: %d' % (len(D2), len(T2), len(L2)))

        return (D2, L2, T2)
    def do_splice(): 
        if not kargs.has_key('splice_policy'): return False
        if kargs['splice_policy'].startswith(('noop', 'reg', 'compl', )):  # effective operations 
            return False 
        return True  # prior, posterior [todo] 'in-between'; e.g. from first occurence of X to last occurence of X
    def do_transform(): 
        if not seq_ptype.startswith('reg'): return True
        if do_splice(): return True
        if kargs.get('simplify_code', False): return True
        return True
    def splice_docs(D, T):   # use: only consider LCS labels in the pre- or post- diagnostic sequences
        # inplace operation by default
        if not do_splice(): return (D, T)  # noop
        print('  + splicing sequences in D and T | policy=%s, predicate=%s, cutpoint=%s, inclusive? %s' % \
            (kargs.get('splice_policy', 'noop'), kargs.get('splice_predicate', None), kargs.get('cutpoint', None), kargs.get('inclusive', True)))
        nD0 = len(D)
        D, T = st.spliceDocuments(D, T=T, 
                        policy=kargs.get('splice_policy', 'noop'),  # {'noop', 'prior', 'posterior'}
                        cohort=cohort_name,  # help determine predicate if not provided
                        predicate=kargs.get('splice_predicate', None), # infer predicate from cohort if possible
                        cutpoint=kargs.get('cutpoint', None), 
                        n_active=1, 
                        inclusive=kargs.get('inclusive', True))
        assert len(D) == len(T)
        assert len(D) == nD0, "size of document set should not be different after splicing nD: %d -> %d" % (len(D), nD0) 
        return (D, T)
    def test_stratum(label, data): 
        D, T, L = data['sequence'], data.get('timestamp', []), data.get('label', [])
        nD, nT, nL = len(D), len(T), len(L)
        print('  + label: %s | nD: %d, nT: %d' % (label, nD, nT))
        n_classes = len(set(L))
        assert len(D) == len(T), "Size inconsistency between (D: %d) and (T: %d)" % (len(D), len(T))
        assert n_classes == 1, "labeled data were not stratified properly n_classes=%d in the same stratum." % n_classes
        # data distribution by stratum 
        return
    def profile(): 
        sizes = {}
        for label, entry in stratified.items(): 
            sizes[label]=len(entry['sequence'])
            test_stratum(label, data=entry)
        print('profile> data size distribution:\n%s\n' % sizes)
        return
    def get_cohort(): 
        try: 
            return kargs['cohort']
        except: 
            pass 
        raise ValueError, "cohort info is mandatory."

    from seqparams import TSet, TDoc
    from labeling import TDocTag
    import seqReader as sr
    import seqTransform as st

    # single label format ['l1', 'l2', ...] instead of multilabel format [['l1', 'l2', ...], [], [], ...]
    inputdir = kargs.get('inputdir', get_global_cohort_dir())
    cohort_name = get_cohort()
    seq_ptype = seqparams.normalize_ctype(kargs.get('seq_ptype', 'regular'))
    tSingleLabelFormat = kargs.get('single_label_format', True) 
    tTransformSequence = do_transform() 

    units = load_labeled_docs() # if not available, should run makeLabeledDocuments(cohort) first
    print('stratifyDocuments> Found %d entries/labels' % (len(units)))
    stratified = {l: {} for l in units.keys()}
    for label, entry in units.items(): 
        D, T, L = entry['sequence'], entry.get('timestamp', []), entry.get('label', [])
        # nD, nT, nL = len(D), len(T), len(L) 
        # test_stratum(label, data=entry)  # -> profile()
        
        # labels is not the same as tags (a list of lists)
        if tSingleLabelFormat: 
            # use the first label as the label by default (pos=0)
            L = TDocTag.toSingleLabel(L, pos=0) # to single label format (for single label classification)
            print('    + labels (converted to single-label format, n=%d): %s' % (len(np.unique(L)), np.unique(L)))
            assert len(D) == len(L), "Size inconsistency between (D: %d) and (L: %d)" % (len(D), len(L))
        
        ### tansform data 
        if tTransformSequence: 
            # [params] items, policy='empty_doc', predicate=None, simplify_code=False
            D, L, T = transform_docs(D, L, T) 
            # prior, posterior only? note that labeled sequence source above contain the entire sequence, not the spliced version
            D, T = splice_docs(D, T)  # params: splice_policy; this should not change L

        stratified[label]['sequence'] = D
        stratified[label]['timestamp'] = T
        stratified[label]['label'] = L
    profile() # test_stratum()

    return stratified 

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

    # sys.exit(0)

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

def test(**kargs): 
    """

    Memo
    ----
    1. also see seqClassify
    """

    ### document processing 
    t_process_docs(**kargs)

    return 

if __name__ == "__main__": 
    test()

