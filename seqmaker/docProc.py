# encoding: utf-8

#############################################################################################
#
#  An app of seqReader, seqTransformer for pre-processing coding sequence documents
#
#
#  Memo
#  ----
#  1. Early incarnation: seqReaderApp
#


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

from tdoc import TDoc

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


def loadTrainingDocuments(cohort, seq_ptype='regular', d2v_method=None, suffix=None, index=None, dir_type='combined'): 
    import vector
    # from tdoc import TDoc
    if d2v_method is None: d2v_method = vector.D2V.d2v_method
    return TDoc.load(**kargs)

def sampleDSet(cohort, inputdir=None, source_type='source', **kargs):     
    raise NotImplementedError

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
        sourceType = kargs.get('source_type', 's')  # {'a', 'r', 'l', 's'}
        user_file_descriptor = kargs.get('meta', None)
        if not ret: 
            if sourceType.startswith('a'): # augmented 
                ret = load_augmented_docs() 
            if sourceType.startswith('l'):  # labeled and stratified
                ret = load_labeled_docs() 
            elif sourceType.startswith('r'): # raw (.dat)
                ret = load_raw_docs()
            else:  # everything else is default; suggested keywords: {'source', 'default'}
                ifiles = kargs.get('ifiles', [])
                print('loadDocuments> input files: %s' % ifiles)

                # other params: doctype (timed)
                ret = sr.readDocFromCSV(cohort=cohort, inputdir=inputdir, ifiles=ifiles, complete=True, meta=user_file_descriptor)
        return ret
    def load_augmented_docs(label_default=None):  
        user_file_descriptor = kargs.get('meta', None)
        # [todo] ifiles
        ret = sr.readAugmentedDocFromCSV(cohort=cohort, inputdir=inputdir, ifiles=[], meta=user_file_descriptor) # [params] doctype (timed) 
        return ret
    def load_raw_docs(): # load documents from .dat file
        ret = kargs.get('result_set', {}) # 'sequence', 'timestamp', 'label' 
        user_file_descriptor = kargs.get('meta', None)
        if not ret:   # this is assuming that document sources exist in .csv format 
            ifiles = kargs.get('ifiles', [])
            dx, tx = sr.readTimedDocPerPatient(cohort=cohort, inputdir=inputdir, ifiles=ifiles, meta=user_file_descriptor)
            assert len(dx) == len(tx), "The size between documents and timestamps is not consistent."
            ret['sequence'] = dx 
            ret['timestamp'] = tx 
        return ret
    def load_labeled_docs():  # 
        ret = kargs.get('result_set', {}) # 'sequence', 'timestamp', 'label' 
        user_file_descriptor = kargs.get('meta', None)  # user-defined file ID
        # [note] this function should be agnostic to seq_ptype
        # fpath = TDoc.getPath(cohort=kargs['cohort'], seq_ptype=kargs.get('seq_ptype', 'regular'), 
        #     doctype='labeled', ext='csv', basedir=prefix) 
        # assert os.path.exists(fpath), "labeled docuemnt source should have been created previously (e.g. by processDocuments)"
        # df_src = pd.read_csv(fpath, sep='|', header=0, index_col=False, error_bad_lines=True) # ['sequence', 'timestamp', 'label']
        if not ret: 
            ifiles = kargs.get('ifiles', [])
            ret = sr.readDocFromCSV(cohort=cohort, inputdir=inputdir, ifiles=ifiles, complete=True, 
                            meta=user_file_descriptor, # user-defined file ID
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

    # P = ret.get('person_id', [])

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
        user_file_descriptor = kargs.get('meta', None)
        D, L, T = loadDocuments(cohort=cohort, inputdir=inputdir,
                                    source_type='default',  
                                    meta=user_file_descriptor, 
                                    use_surrogate=False, 
                                    single_label_format=tSingleLabel)  # [params] composition
        return (D, L, T)
    def get_global_cohort_dir(): 
        return seqparams.getCohortGlobalDir(cohort) # sys_config.read('DataExpRoot')/<cohort> 
    def make_labeled_docs(): # [params] cohort, doctype, seq_ptype, doc_basename, ext + basedir
        # also see seqReader.verifyLabeledSeqFile()
        assert len(L) > 0, "coding sequences are not labeled (at least use default all-positive labeling)." 
        
        seq_ptype = kargs.get('seq_ptype', 'regular')
        user_file_descriptor = kargs.get('meta', None)

        # usually there is one file per cohort
        fpath = TDoc.getPath(cohort=cohort, seq_ptype=seq_ptype, doctype='labeled', ext='csv', basedir=inputdir, meta=user_file_descriptor)    
        print('  + saving labeled .csv file to:\n%s\n' % fpath)
        if kargs.get('save_', True) and not os.path.exists(fpath): 
            # path params: cohort=get_cohort(), seq_ptype=seq_ptype, doctype=doctype, ext='csv', baesdir=basedir
            
            # labels = L 
            # if not labels: 
            #     # create dummy labels [todo]
            #     labels = [1] * len(D)
            assert len(T) > 0, "timestamps were not found in this source (cohort=%s)" % cohort
            assert len(D) == len(T), "inconsistent sizes between sequences and timestamps"

            sr.readDocToCSV(sequences=D, timestamps=T, labels=L, cohort=cohort, seq_ptype=seq_ptype, 
                outputdir=inputdir, meta=user_file_descriptor)
            # condition: a labeled source documents is generated 
        return 

    import seqReader as sr
    if inputdir is None: inputdir = get_global_cohort_dir()    

    D, L, T = load_docs()
    make_labeled_docs()  # D, L, T

    return 

def segmentDocumentByTime(D, L, T, timestamp, **kargs): 
    import seqTransform as st 
    return st.segmentDocumentByTime(D, L, T, timestamp, **kargs) # output: (docIDs, D, L, T)

def segmentDocuments(D, L, T, predicate, policy='regular', inclusive=False, drop_nullcut=False, segment_by_visit=False):
    """

    Params
    ------
    drop_nullcut: if True, include only documents with active cutpionts (i.e. with cohort-specifiying diagnosis codes define in pattern module)


    **kargs
      policy: regular/noop 
              two/halves 
              prior, posterior 
              complete

    """ 
    import seqTransform as st 

    # docIDs = []
    # if segment_by_visit: # visit 
    #     docToVisit = segmentByVisits(D, T) # kargs: max_visit_length
    #     docIDs = sorted(docToVisit.keys())  # ascending order

    #     # D: [ [v1, v2, ... v10], [v1, v2, v3] ... ] v1: [c1 ,c2, c3]
    #     D = [docToVisit[docid] for docid in docIDs] # each document is a list of visits (a list), in which each visit comprises a list of tokens
    #     # return (docIDs, D, L, T)   # return visit-combined document set or the docToVisit map?
     
    # segment document {regular, prior, posterior}
    if predicate is None or policy.startswith('reg'): 
        print('segmentDocuments> Noop.')
        return (range(0, len(D)), D, L, T)  # (DocIDs, sequences, labels, timestamps)
    else: 
        assert hasattr(predicate, '__call__'), "Invalid predicate: %s" % str(predicate)
        print('segmentDocuments> nD=%d, policy=%s, include endpoint? %s' % (len(D), policy, inclusive))

    # [output] (docIDs, Dp, Lp, Tp)
    return st.segmentDocuments2(D, L, T, predicate, policy=policy, inclusive=inclusive, drop_nullcut=drop_nullcut)

# or just do 
# from sampler.sampling import sample_wr
def sample_wr(population, k):
    "Chooses k random elements (with replacement) from a population"
    n = len(population)
    _random, _int = random.random, int  # speed hack 
    result = [None] * k
    for i in range(k):
        j = _int(_random() * n)
        result[i] = population[j]
    return result

def standardizeByVisits(D, L, T, **kargs):
    return makeSessionDocuments(D, T, **kargs) 
def makeSessionDocuments(D, L, T, **kargs):
    """
    segmentByVisits() + visitDocuments() + padding each doc to equal length

    mode: i) uniform_length => each document is made equal length 
             e.g. say we represent each document using a max of 10 visits 
                  then we find the max length among all documents with 10 visits 
                  use that length the max length for the document set, pad those with shorter lengths

    """ 
    # note that boostrapping used in segmentByVisits is used to preserve the distribution of the codes 
    # but the optimal max_length will depend on the variability and sample size. 
    min_length, max_length = kargs.get('min_visit_length', seqparams.D2V.window), kargs.get('max_visit_length', 100)

    # [params] 
    docToVisit = segmentByVisits(D, T, L=L, max_visit_length=max_length, min_visit_length=min_length) # kargs: max_visit_length
    # docIDs = sorted(docToVisit.keys())  # ascending order

    # D: [ [v1, v2, ... v10], [v1, v2, v3] ... ] v1: [c1 ,c2, c3]

    # kargs: L, n_per_class
    # mode: if 'visit' then each visit is a document 
    #       if 'uniform' then all visits are combined into a joint document but a max visits specified by last_n_visits
    #           => uniform stands for uniform length: each document is made equal length 
    #       last_n_visits is only effective in mode: uniform 
    #           => limit document length (otherwise, may create too many 'paddings)
    # [note] each document is a list of visits (a list), in which each visit comprises a list of tokens
    mode = kargs.get('mode', 'visit')  # values: {'visit', 'uniform'}
    if mode.startswith('v'): 
        docIDs, D = visitToDocument(V=docToVisit, mode=mode) # each document is a list of visits (a list), in which each visit comprises a list of tokens
    else:
        docIDs, D = visitToDocument(V=docToVisit, last_n_visits=kargs.get('last_n_visits', None), mode=mode) 
    return (docIDs, D) 
def segmentByVisits(D, T, **kargs):  # seqTransform
    """
    Segment documents into visit sequences. 

    Input
    -----
    D: expected to be the whole document set
    max_visit_length

    D: [[ ... d1 ... ], []] 

       [ [d1: [v1, v2, v3], 
          d2: [v1, v2]
          ...

          dn: [] ]

    L: 
    T: 

    
    kargs
    -----

    Memo
    ----
    1. small CKD cohort
            + avgL: 3.401748, max number of tokens in a visit: 71
            + avgV: 90.685169, max number of visits in a doc:   1082
                # 20 visits
    2. large CKD cohort
            + avgL: 3.056786, max n_tokens_in_visit: 104, min: 1, std: 4.183465
            + avgV: 38.118913, max n_visits_in_doc: 2019, min: 1, std: 61.838292

    """
    def test_input(): 
        L = kargs.get('L', []) # labels may not be necessary here
        assert len(D) == len(T)
        if len(L) > 0: assert len(D) == len(L)
        return 
    def get_stats(docToVisit): 
        nLGrand = 0  # total length across all documents
        nVGrand = 0  # grand total number of visits across all documents
        maxL = -1 # the maximum length of a visit (number of codes) across all visist and all documents
        maxV = -1 # maximum number of visits across all documents
        maxVDocID = -1

        nTokensInVisit, nVisitsInDoc = [], []  # memory
        for docid, vseq  in docToVisit.items(): # e.g. [[v1, v2], [v3, v4, v5], [] ]
            nVDoc = len(vseq)  # number of visits for this document
            nVisitsInDoc.append(nVDoc)

            if maxV < nVDoc: 
                maxV = nVDoc; maxVDocID = docid
            for v in vseq: # foreach visit
                nVTok = len(v)  # number of tokens in each visit; e.g. v1: [c1, c2, c3]
                nTokensInVisit.append(nVTok)

                if maxL < nVTok: maxL = nVTok
                nLGrand += nVTok  # to get total length of visits 
            nVGrand += nVDoc

        avgL = nLGrand/(nVGrand+0.0) # averag length (number of tokens) per visit
        avgV = nVGrand/(len(docToVisit)+0.0)  # average number of visits per document

        # [test]
        assert maxL == max(nTokensInVisit), "maxL=%d <> max(nTokensInVisit)=%d" % (maxL, max(nTokensInVisit))
        print('  + E[n_tokens in session]: %f | max n_tokens_in_session: %d, min: %d, std: %f' % \
            (avgL, maxL, min(nTokensInVisit),  np.std(nTokensInVisit)))
        print('  + E[n_sessions]: %f | max n_sessions_in_doc: %d, min: %d, std: %f' % \
            (avgV, maxV, min(nVisitsInDoc), np.std(nVisitsInDoc)))

        # [test]
        n_udates = len(set(T[maxVDocID]))
        assert maxV == n_udates, "max n_visits in a doc: %d but got only %d uniq dates (from docID=%d)" % (maxV, n_udates, maxVDocID)

        return (avgL, maxL, avgV, maxV)
    # from sampler.sampling import sample_wr
    # docIDs = kargs.get('docIDs', [])
    # if not docIDs: docIDs = range(len(D))

    test_input()
    docToVisit = {}
    nD = len(D)
    test_idx = set(random.sample(range(nD), min(nD, 20)))
    for i, doc in enumerate(D): # 'i' is not necessary identical to document ID
        docid, tdoc = i, T[i]  # docIDs[i]; if D is not the whole document set then 'i' is not necessary identical to document ID
        nW = len(doc)

        docToVisit[docid] = []
        # bypass empty documents
        if nW == 0: 
            print('Warning: %d-th document (docID=%d) is empty!' % (i, docid))
            continue

        indices = [0, ]  # indicate i-th visit
        for j in range(1, nW):
            t = tdoc[j]  # 
            t0 = tdoc[j-1]

            id_prev = id_cur = indices[j-1]
            if t != t0: # tdoc was already sorted
                id_cur += 1
            indices.append(id_cur)
        id_max = indices[-1]
        n_visits = id_max + 1

        # indices e.g. [0, 0, 1, 1, 1, 2] contains i-sequence representing sequence of ith-visits

        # knowing n_visits, now we can init docToVisit[docId]
        for ith in range(n_visits):
            docToVisit[docid].append([])

        for j in range(nW):
            ithv = indices[j] # ith visit
            docToVisit[docid][ithv].append(doc[j]) # j-th token assigned to ith-visit
        
        # [test]
        if i in test_idx: 
            n_timesteps = len(set(tdoc))
            assert n_timesteps == n_visits, "unique timesteps: %d but got %d visits!" % (n_timesteps, n_visits)
            for ith in range(n_visits):
                assert len(docToVisit[docid][ith]) > 0

            lastn = 150
            print('  + doc(last %d chars):\n%s\n' % (lastn, doc[-lastn:]))
            print('  + vdoc')
            for lasti, v in enumerate(docToVisit[docid][-10:]): # list only last 10 visits
                print('    + %s' % v)

    avgL, maxL, avgV, maxV = get_stats(docToVisit)

    minVLen = kargs.get('min_visit_length', seqparams.D2V.window) 
    maxVLen = kargs.get('max_visit_length', 100)
    maxVLen = min(maxL, maxVLen)

    # boostrapping to ensure that each visit segment has the same length 
    # this is to faciliate the d2v model training; now each visit segment is considered as a document
    n_short_session = 0
    for docid, V in docToVisit.items(): 
        for i, visit in enumerate(V): 
            if len(visit) < minVLen: 
                V[i] = sample_wr(visit, k=minVLen)
                n_short_session += 1
            # if docid < 10: 
            #     print('segmentByVisits>\n   + before: %s' % visit)
            #     print('                     + after: %s' % V[i])
    print('info> Found %d short sessions (< %d tokens)' % (n_short_session, minVLen))
    return docToVisit  # docId -> visists [[c1, c2], [c3, c4, c5], [c6, c7], ... [c124]]

def visitToDocument(V, docIDs=[], **kargs):
    """
    Combine visits from across documents. 

    Input
    -----
    docIDs: provide docIDs if only a subset of V is needed (e.g. sampled subset of each class label)
    V: visit document, a map from document ID to its associated lists of visits (i.e. docToVisit from segmentByVisits)

    kargs 
    -----
    L: class labels (used when a subset of each class label is desired)
    T:  
    n_per_class: 

    mode: i) uniform_length: each document is made equal length 
             e.g. say we represent each document using a max of 10 visits (usually the most recent 10)
                  then we find the max length among all documents with 10 visits 
                  use that length the max length for the document set, pad those with shorter lengths
          ii) visit_document: concatenate all visit segments from all documents together into a flatten structure 
                  in which the document set essentially comprises of visit semgents, i.e. each visit segment IS a 
                  document

    Memo
    ----
    1. pad_sequences in Keras does not support strings 
       
       https://stackoverflow.com/questions/46323296/keras-pad-sequences-throwing-invalid-literal-for-int-with-base-10

    """
    def test_visit_doc(): 
        testIDs = random.sample(range(len(V)), min(10, len(V)))
        for i in testIDs: 
            doc = V[i]
            assert hasattr(doc, '__iter__'), "each doc is a list of visits and each visit is a list of tokens."
            for visit in doc: 
                assert hasattr(visit, '__iter__'), "each visit is a list of tokens."
                assert isinstance(random.choice(visit), str)
        if len(L) > 0: 
            assert len(V) == len(L)
        if len(T) > 0: 
            assert len(V) == len(T)
        return
    def subsample(V, L, n=1000, sort_index=False, random_state=53): 
        nD = len(V)
        # n = kargs.get('n_per_class', 5000)

        # L must be in the order of document IDs obtained from the source 
        docidx = cutils.samplePerClass(L, n_per_class=n, sort_index=sort_index, random_state=random_state)

        Vp = {}
        for i, docid in enumerate(docidx): 
            # doc = V[docid]
            Vp[docid] = V[docid]

        # V = list(np.array(V)[idx])
        print("  + size(V):%d -> size(V_sampled):%d" % (nD, len(Vp)))
        return Vp
    def to_str(D, sep=' '):
        return [sep.join(doc) for doc in D]  # assuming that each element is a string
    def test_docs(Dt):
        testdocs = random.sample(Dt, min(10, len(Dt)))
        for i, doc in enumerate(testdocs): 
            print('[%d] %s' % (i, doc))
        print('\n')
        return 
    
    import classifier.utils as cutils
    from keras.preprocessing.sequence import pad_sequences

    nD = len(V)
    L, T = kargs.get('L', []), kargs.get('T', [])
    test_visit_doc()

    if len(L) > 0:  # randomly select at most N documents per class
        V = subsample(V, L, n=kargs.get('n_per_class', 5000))

    Dv, idv = [], [] 
    mode = kargs.get('mode', 'visit_document')  # values: visit_document, uniform_length
    maxLastN = kargs.get('last_n_visits', None)   # only look at the last N visits.  

    if mode.startswith('visit'): 
        if not docIDs: docIDs = sorted(V.keys())  # ascending order
        for i, docid in enumerate(docIDs): # o.w. assuming that docIDs have been sorted in desired order

            # [design] only look at the last N visit? V[docid][-maxLastN:]  
            #          No, because we want to train d2v model with as many data as possible
            doc = V[docid]  # V[docid][-maxLastN:] 
            for j, visit in enumerate(doc): 
                Dv.append(visit)
                idv.append(docid)

        # [log] ~90 visits per doc in small CKD cohort (n=2360)
        print("visitToDocment> size(V):%d -> size(Dv):%d (E[nVperDoc]=%f)" % (nD, len(Dv), len(Dv)/(nD+0.0)))
    elif mode.startswith('uni'):  # uniform_length  
        maxL = 0
        if not docIDs: docIDs = sorted(V.keys())  # ascending order

        if maxLastN is None: # default, include ALL visits
            for i, docid in enumerate(docIDs): # o.w. assuming that docIDs have been sorted in desired order
                newDoc = list(np.hstack([visit for visit in V[docid]]))   # concatenate ALL visits 
                if len(newDoc) > maxL: maxL = len(newDoc)
                Dv.append(newDoc)
        else: # include at most N visits 
            for i, docid in enumerate(docIDs): # o.w. assuming that docIDs have been sorted in desired order
                newDoc = list(np.hstack([visit for visit in V[docid][-maxLastN:]]))   # concatenate all visits 
                if len(newDoc) > maxL: maxL = len(newDoc)
                Dv.append(newDoc)          
   
        test_docs(Dv)
        print('visitDocuments> Beginning padding operations (to_str -> tokenize -> reconstruct) ...')
        # pad sequence
        #    1. pad valud has to be in float type
        #    2. operations: convert Dv to a list of strings -> tokenize each list and convert tokens to integers (i.e. word indices) 
        #                       -> test if can be reconstructed 
        Dv = padSequences(Dv, maxlen=None, value=0, tostr_=True, tokenize_=True, test_level=1)  # params: padding='post', 'unknown'

        # [Q] back to strings for d2v? 
        # for i, doc in enumerate(Dv): 
        #     Dv[i] = [str(e) for e in doc]

        idv = docIDs
        assert len(idv) == len(Dv), "In 'uniform' mode, the final documents should remain the same size as before: %d but got %d" % \
            (len(docIDs), len(Dv))
    else: 
        raise NotImplementedError

    # [output]
    # each visit/session is a document by itself
    # a. visit mode
    #    idv: document ID (the same ID may repeat n times if a document has n sessions)
    #    Dv: session paragraphs 
    # b. uniform mode
    #    idv: 
    #    Dv: 
    return (idv, Dv)  

def padSequences(sequences=[], maxlen=None, value=0.0, dtype='int32', tostr_=True, tokenize_=True, test_level=0): 
    """

    Input
    -----
    maxlen: None by default, then the max length of all sequences will be the longest of them 

    Memo
    ----
    1. pad_sequences in Keras does not support strings 
       
       https://stackoverflow.com/questions/46323296/keras-pad-sequences-throwing-invalid-literal-for-int-with-base-10



    """
    def to_str(D, sep=' ', throw_=False):
        try: 
            D = [sep.join(doc) for doc in D]  # assuming that each element is a string
        except: 
            msg = 'padSequences> Warning: Some documents may not contain string-typed tokens!'
            if throw_: 
                raise ValueError, msg 
            else: 
                print msg
            D = [sep.join(str(e) for e in doc) for doc in D]
        return D

    from keras.preprocessing.sequence import pad_sequences

    # example sequences
    if not sequences: 
        sequences = [
             [1, 2, 3, 4],
                [1, 2, 3],
                      [1]
        ]

    # MCSs consist of strings, we need to tokenize them first so that tokens are converted to integers
    if tokenize_:
        if tostr_: sequences = to_str(sequences)
        sequences = docToInteger(sequences, policy='exact', test_level=test_level)  # tokenize + transform to integers
 
    # # pad sequence
    # if maxlen is None: 
    #     # then we need to find out the max length among all documents 
    #     maxlen = -1 
    #     for seq in sequences: 
    #         if len(seq) > maxlen: maxlen = len(seq)
    Dp = pad_sequences(sequences, maxlen=maxlen, value=value, dtype=dtype)  
    # print(Dp)
    return Dp

def tokenizeDoc(D, test_level=0, throw_=True, **kargs):
    """
    Tokenize documents and retain only frequent words
    """
    def to_str(D, sep=' ', throw_=False):
        print('to_str> converting list of lists to list of strings ...')
        try: 
            D = [sep.join(doc) for doc in D]  # assuming that each element is a string
        except: 
            msg = 'tokenizeDoc> Warning: Some documents may not contain string-typed tokens!'
            if throw_: 
                raise ValueError, msg 
            else: 
                print msg
            D = [sep.join(str(e) for e in doc) for doc in D]
        return D
    def test_subset():
        nd = len(D)
        nt = 10
        if nd > nt: 
            rp = random.sample(range(0, len(D)-1), min(nt, len(D)))  # pick at least 10 indices as test cases
        else: 
            rp = range(0, nd)

        return rp
    def test_prior_length(D, rp, sep=' '): 
        # assert not isinstance(D[0], str)
        # print('(test) prior size(D)=%d' % len(D))
        lengths = []
        for r in rp: 
            d = D[r].split(code_sep) if isinstance(D[r], str) else D[r]
            lengths.append(len(d))
            # print('(test) %s' % d) 
        return lengths
    def test_posterior_length(D, rp, length_prior): 
        assert not isinstance(D[0], str)
        # print('(test) posterior size(D)=%d' % len(D))
        for i, r in enumerate(rp): 
            # msg = "Inconsistent lengths before & after tokenize op: %d vs %d" % (length_prior[r], len(D[r]))
            if untokenized is None: 
                pass
            else: 
                assert length_prior[i] == len(D[r]), "Inconsistent lengths before & after tokenize op: %d vs %d" % (length_prior[r], len(D[r]))
    def count_tokens(D, n=100): 
        counter = collections.Counter()
        for d in D: 
            counter.update(d)

        print('(count_tokens) topn:\n%s\n---#\n' % counter.most_common(n))
        return len(counter)
    def int_to_str(D, prefix='c'): 
        # [note] inplace operation
        for i, d in enumerate(D): 
            D[i] = map(str, d)  # add a prefix? 
        return D  # redundant

    # MCSs consist of strings, we need to tokenize them first so that tokens are converted to integers
    if len(D) == 0: return D

    code_sep = kargs.get('split', ' ')

    # [test]
    rp = test_subset(); lengths = test_prior_length(D, rp, sep=code_sep)

    # Tokenizer takes a list of strings as inputs
    tostr_ = False if isinstance(D[0], str) else True
    if tostr_: D = to_str(D, sep=code_sep) # Tokenizer takes a list of strings as inputs

    # input: each doc in D must be a string
    vocab_size = kargs.get('num_words', 20001)
    # filters = kargs.get('filters', '!"#$%&()*+,/;<=>?@[\]^`{|}~') # '.', ':', '_' '-' should be consiidered as part of the codes/tokens
    untokenized = kargs.get('oov_token', TDoc.token_unknown)
    D = docToInteger(D, policy='exact', 

           # params for Tokenizer: tokenize + transform to integers
           num_words=vocab_size, split=code_sep, oov_token=untokenized, 

           customized=kargs.get('customized', False),  # use customized tokenizer instead of Keras library version? 
           pad_empty_doc=kargs.get('pad_empty_doc', True),  # if True, empty documents are represeted by a minimum of one chosen token
           test_level=test_level)  

    # [test]
    # test_posterior_length(D, rp, lengths)
    count_tokens(D)

    # convert integers to strings (but retain the numerical values)? 
    # [note] if not, then got TypeError: unsupported operand type(s) for +: 'int' and 'str'
    int_to_str(D)

    return D

def docToInteger(D, policy='exact', **kargs): 
    """
    Create integer encoded documents. 

    Input
    -----
    policy: 
       one_hot: maps input documents to integers using  one-hot encoding 
                Note that the name suggests that it will create a one-hot encoding of the document but this is NOT the case.
                Instead, the function is a wrapper for the hashing_trick() function. 

                The use of a hash function means that there may be collisions and not all words will be assigned unique integer values.

                The size of the vocabulary defines the hashing space from which words are hashed. 
                Ideally, this should be larger than the vocabulary by some percentage (perhaps 25%) 
                to minimize the number of collisions. 

       exact: use 


    Reference
    ---------
    1. one-hot encoding: One-hot encodes a text into a list of word indexes of size n.
       https://keras.io/preprocessing/text/#one_hot

    Memo
    ----
    1. The reconstructed sequence (see test_reconstruct) may not be exactly the same. 

    """
    def test_input(): # check if 'D' consists of a list of strings 
        testcases = random.sample(D, min(len(D), 10))
        for doc in testcases: 
            assert isinstance(doc, str), "doucment is not of type string (type=%s):\n%s\n" % (type(doc), doc[:100])

    # from keras.preprocessing.text import one_hot
    # from tdoc import TDoc

    test_input() # input documents must be a list of strings 
    testLevel = kargs.get('test_level', 1)
    tCostomized = kargs.get('customized', True)

    # Dint = None
    assert isinstance(D[0], str), "Each input document must be a string > D[0]:\n%s\n" % D[0]
    if policy.startswith('exact'):  # maps to integers exactly as they are (no implication of hashing collisions)

        # kargs: num_words, split, other augments that keras.preprocessing.text.Tokenizer takes
        # t = tokenize(D, num_words=kargs.get('num_words', None), split=kargs.get('split', ' ')) 

        # based on word frequency, only the most common 'num_words' words will be kept; the other tokens are discarded, which means some 
        # coding sequences may end up being much shorter if not empty! 
        # D = t.texts_to_sequences(D)
        D = texts_to_sequences(D, untokenized=TDoc.token_unknown, 
                num_words=kargs.get('num_words', None), split=kargs.get('split', ' '),
                oov_token=kargs.get('oov_token', None),

                customized=kargs.get('customized', False), 
                pad_empty_doc=kargs.get('pad_empty_doc', True), 
                test_level=testLevel)

    # elif policy.startswith('one'):
    #     pass 
    else: 
        raise NotImplementedError

    return D

def texts_to_sequences(D, tokenizer=None, untokenized=None, **kargs):
    """
    An alternative version of Kera's texts_to_sequences, which will discard less frequently tokens. 

    Input
    -----
    t: tokenizer that expects attributes: word_index

    """
    def to_str(d, sep=' ', throw_=False):
        print('(to_str) converting list of lists to list of strings ...')
        dp = ''
        try: 
            dp = sep.join(d)  # assuming that each element is a string
        except: 
            msg = '(to_str) Warning: input doc may not contain string-typed tokens!'
            if throw_: 
                raise ValueError, msg 
            else: 
                print msg
            dp = sep.join(str(e) for e in d)
        return dp
    def lists_to_strings(D, sep=' ', throw_=False):
        print('(lists_to_strings) Converting list of lists to list of strings ...')
        try: 
            D = [sep.join(doc) for doc in D]  # assuming that each element is a string
        except: 
            msg = '(lists_to_strings) Warning: Some documents may not contain string-typed tokens!'
            if throw_: 
                raise ValueError, msg 
            else: 
                print msg
            D = [sep.join(str(e) for e in doc) for doc in D]
        return D

    # from tdoc import TDoc

    # check input data type
    s = random.randint(0, len(D)-1)
    tListOfStr = True if isinstance(D[s], str) else False

    if untokenized is None: untokenized = TDoc.token_unknown

    # condition: D is a list of strings  
    sep = kargs.get('split', ' ')
    vocab_size = kargs.get('num_words', None)
    oov_token = kargs.get('oov_token', untokenized)
    if tokenizer is None: 
        print('(texts_to_sequences) num_words: %d, oov_token: %s' % (vocab_size, oov_token))

        if not tListOfStr: D = lists_to_strings(D, sep=sep, throw_=False)

        # condition: input doc in D must be a string
        tokenizer = tokenize(D, num_words=vocab_size, split=sep, oov_token=oov_token) 

    # retain length of document (padding infrequent tokens)? 
    tCostomized = kargs.get('customized', True)

    Dp = []
    word_index = {}
    punctuations = kargs.get('filters', '!"#$%&()*+,/;<=>?@[\]^`{|}~') # '.', ':', '_' '-' should be consiidered as part of the codes/tokens

    if not tCostomized: # use Keras' library 
        Dp = tokenizer.texts_to_sequences(D)
        word_index = tokenizer.word_index
    else: 
        
        if tListOfStr: # input D is a list of string => convert each string to a list of tokens
            Dp = [d.translate(None, punctuations).split(sep) for d in D]
            # Dp = [d.split(sep) for d in D]
            print('(texts_to_sequences) Dp[%d]:\n%s\n' % (s, Dp[s]))
        else: 
            Dp = D 
        # condition: D is a list of list of tokens

    
        # if int_: 
        #     untokenized = -1
        counter = collections.Counter(tokenizer.word_index)
        word_index = dict(counter.most_common(vocab_size))

        # [hardcode]
        untokenized_id = len(word_index)+1 # the infrequent tokens not being indexed
        print('texts_to_sequences> size(word_index): %d, untokenized_id=%d' % (len(word_index), untokenized_id))
        
        # assuming that D is already a list of lists of tokens
        for i, doc in enumerate(Dp): 
            # print('(texts_to_sequences) doc> %s' % doc)
            Dp[i] = [word_index.get(tok, untokenized_id) for tok in doc]
  
        # [test]
        # for i, doc in enumerate(D): 
        #     doc2 = []
        #     for tok in doc: 
        #         print('(test) tok: %s doc: %s' % (tok, doc))
        #         try: 
        #             doc2.append(t.work_index[tok])
        #         except: 
        #             doc2.append(untokenized)
        #     D[i] = doc2

        #     # [design] keep integer representation?
        #     if i < 10: 
        #         print('... example: %s' % D[i])

    if kargs.get('pad_empty_doc', True): 
        for i, d in enumerate(Dp): 
            if not d: 
                Dp[i] = [oov_token, ]

    # [test] reconstruct the original sequence/input using the tokenized portion of the input
    if kargs.get('test_level', 0) > 0: 
        rp = random.sample(range(len(D)), min(len(D), 20)) # select test cases 
        index_word = dict(map(reversed, word_index.items()))  # reverse_word_map

        assert isinstance(D[rp[0]], str), "document format expected a string but got a list of tokens?\n%s\n" % rp[0]
        for i, r in enumerate(rp): 
             # standardize cases in order to compare 
            Ds = D[r].translate(None, punctuations).upper() if tListOfStr else to_str(D[r], sep=sep, throw_=False)
            Dr = sep.join(index_word.get(e, untokenized).upper() for e in Dp[r]) # ad-hoc reconstruction rule
            
            print("\n(reconstruction) test #%d:\n ... D[j]:\n%s\n ... D_reconstructed:\n%s\n" % (r, Ds, Dr))

            # [test] 
            # assert len(Ds) > 0, "Source empty!"
            if len(Ds) == 0: continue
            if not Ds == Dr: 
                SDs = Ds.split(sep)
                SDr = Dr.split(sep)
                n_diff = len(set(SDs)-set(SDr))# number of tokens represented
                ratio = 1- n_diff/(len(SDs)+0.0)
                print(" ... miss %d out of %d (acc: %f)\n---\n" % (n_diff, len(SDs), ratio)) # number of different bases/tokens

    return Dp

def intDocToDoc(D, tokenizer, lowercase_tokens=['unknown', ], sep=' '):
    index_word = dict(map(reversed, tokenizer.word_index.items()))  # reverse_word_map
    for i, doc in enumerate(D): 
        D[i] = sep.join(index_word[e].upper() for e in doc)
    return D

def tokenize(docs=[], **kargs): 
    """

    Memo
    ----
    1. tokenized output attributes 
       word_counts: A dictionary of words and their counts.
       word_docs: A dictionary of words and how many documents each appeared in.
       word_index: A dictionary of words and their uniquely assigned integers.
       document_count:An integer count of the total number of documents that were used to fit the Tokenizer.

    2. num_words: the maximum number of words to keep, based on word frequency. Only the most common num_words words will be kept.
       filters: a string where each element is a character that will be filtered from the texts. 
                The default is all punctuation, plus tabs and line breaks, minus the ' character.

       lower: boolean. Whether to convert the texts to lowercase.
       split: str. Separator for word splitting.
       char_level: if True, every character will be treated as a token.
       oov_token: if given, it will be added to word_index and used to replace out-of-vocabulary words during text_to_sequence calls

    3. Tokenizer class was written under the assumption that you'll be using it with an Embedding layer, 
    where the Embedding layer expects input_dim to be vocab size + 1. The Tokenizer reserves the 0 index for masking 
    (even though the default for the Embedding layer is mask_zero=False...), so you are only actually tokenizing 
    the top num_words - 1 words. This is convenient if you are using the Embedding layer, 
    because then you can use the same number for num_words and input_dim.

    
    4. inner working 

    In [2]: texts = ['a a a', 'b b', 'c']
    In [3]: tokenizer = Tokenizer(num_words=2)
    In [4]: tokenizer.fit_on_texts(texts)
    In [5]: tokenizer.word_index
    Out[5]: {'a': 1, 'b': 2, 'c': 3}
    In [6]: tokenizer.texts_to_sequences(texts)
    Out[6]: [[1, 1, 1], [], []]


    """
    from keras.preprocessing.text import Tokenizer
    from keras.preprocessing.text import text_to_word_sequence
    
    # example documents
    if not docs: 
        docs = ['Well done!',
                'Enjoy life!!',
                'Great effort!!!',
                'Balance between science and spirituality',
                'Pleiades has many systems where intelligent lives exist.',
                'The universe is teeming with life!',
                'Interdimensional beings reside on the same plane as we do',  
                'x y z u', 
                '120.7, 250.0 250.11 toxic scam', ]

    # create the tokenizer
    # filter default (original): filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~'
    # filter default (project): '!"#$%&()*+,/;<=>?@[\]^`{|}~'
    kargs['filters'] = kargs.get('filters', '!"#$%&()*+,/;<=>?@[\]^`{|}~') # '.', ':', '_' '-' should be consiidered as part of the codes/tokens
    
    # 0 is a reserved index that won't be assigned to any word.  disable by reserve_zero=False, but not working in current release 
    kargs['lower']=True
    print('Tokenizer> vocab_size: %d, oov_token: %s' % (kargs['num_words'], kargs.get('oov_token', None)))
    if kargs.has_key('oov_token'): kargs.pop('oov_token')  # [note] this attribute doesn't work 
    t = Tokenizer(**kargs)  

    # fit the tokenizer on the documents
    # print('(input) docs[0]:\n%s\n' % docs[0])
    t.fit_on_texts(docs)

    # summarize what was learned
    # print(t.word_counts)
    # print(t.document_count)
    print(t.word_index)  # mapping between token to integer
    # print(t.word_docs)

    return t

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
    def load_docs(n_docs=None, policy=None): # <- cohort_name, (inputdir, source_type, include_augmeted?, single_label_format?)
        # include_augmeted = kargs.get('include_augmeted', False) # include unlabeled data for semi-supervised learning? 
        tSingleLabel = kargs.get('single_label_format', True)
        inputdir = kargs.get('inputdir', get_global_cohort_dir())
        sourceType = kargs.get('source_type', 's') # source_type: {'s'/'source' (default),  'a'/'augmented', 'r'/'raw', }
        inputfiles = kargs.get('ifiles', [])
        user_file_descriptor = kargs.get('meta', None)
        D, L, T = loadDocuments(cohort=cohort_name, inputdir=inputdir,
                                    ifiles=inputfiles, 
                                    source_type=sourceType, 
                                    
                                    # partial load ... [note] easier to process it after documents are loaded
                                    # max_n_docs=n_docs, 
                                    # max_n_docs_policy=policy, 

                                    meta=user_file_descriptor, 
                                    use_surrogate=False, 
                                    single_label_format=tSingleLabel)  # [params] composition
        print('(load_docs) nD: %d, nT: %d, nL: %d' % (len(D), len(T), len(L)))
        
        # [test]
        get_doc_stats(D, L=L, T=T, condition='After loading raw corpus directly from source')

        return (D, L, T)
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

    def get_global_cohort_dir(): 
        return seqparams.getCohortGlobalDir(cohort_name) # sys_config.read('DataExpRoot')/<cohort> 
    def transform_docs(D, L, T): # params: seq_ptype, predicate, simplify_code
        seq_ptype = kargs.get('seq_ptype', 'regular')
        # [params] items, policy='empty_doc', predicate=None, simplify_code=False
        predicate = kargs.get('predicate', None)
        simplify_code = kargs.get('simplify_code', False)

        nD, nD0 = len(D), len(D[0])
        # this modifies D 

        minDocLength = kargs.pop('min_ncodes', 1)
        D2, L2, T2 = transformDocuments2(D, L=L, T=T, 

            # document-wise fitler
            policy=kargs.get('doc_filter_policy', 'minimal_evidence'),  # which triaggers the use of min_ncodes
            min_ncodes=minDocLength, # only used when policy: minimual_evidence

            # content-filter
            seq_ptype=seq_ptype, 
            predicate=predicate, 
            simplify_code=simplify_code, 

            save_=False) # if True, save only if doesn't exist 
        # D, labels = transformDocuments(D, L=labels, seq_ptype=seq_ptype)

        # the same document 
        print('(transform_docs) after transform | nDoc: %d -> %d, size(D0): %d -> %d' %  (nD, len(D2), nD0, len(D2[0])))
       
        # if nD0 != len(D2[0]): 
        #     # [note] this would be a meaningless comparison because 0th position refers to a different document
        #     # print('... original:    %s' % D[0][:100])  
        #     print('... transformed: %s' % D2[0][:100])
        
        print('(transform_docs) after transform nD: %d, nT: %d, nL: %d | min_ncodes=%d' % (len(D2), len(T2), len(L2), minDocLength))

        return (D2, L2, T2)
    def do_slice(): 
        if not kargs.has_key('slice_policy'): return False
        if kargs['slice_policy'].startswith(('noop', 'reg', 'compl', )):  # effective operations 
            return False 
        return True  # prior, posterior [todo] 'in-between'; e.g. from first occurence of X to last occurence of X
    def slice_docs(D, T):   # use: only consider LCS labels in the pre- or post- diagnostic sequences
        # inplace operation by default
        if not do_slice(): return (D, T)  # noop
        print('(slice_docs) splicing sequences in D and T | policy=%s, predicate=%s, cutpoint=%s, inclusive? %s' % \
            (kargs.get('slice_policy', 'noop'), kargs.get('predicate', None), kargs.get('cutpoint', None), kargs.get('inclusive', True)))
        nD0 = len(D)
        D, T = st.sliceDocuments(D, T=T, 
                        policy=kargs.get('slice_policy', 'noop'), 
                        cohort=cohort_name, 
                        predicate=kargs.get('predicate', None), # infer predicate from cohort if possible
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
        basedir = kargs.get('inputdir', get_global_cohort_dir())
        seq_ptype = kargs.get('seq_ptype', 'regular')
        user_file_descriptor = kargs.get('meta', None)

        # usually there is one file per cohort  
        fpath = TDoc.getPath(cohort=cohort_name, seq_ptype=seq_ptype, doctype='labeled', ext='csv', basedir=basedir, meta=user_file_descriptor)  
        if kargs.get('save_', True) and not os.path.exists(fpath): 
            # path params: cohort=get_cohort(), seq_ptype=seq_ptype, doctype=doctype, ext='csv', baesdir=basedir
            
            # labels = L 
            # if not labels: 
            #     # create dummy labels [todo]
            #     labels = [1] * len(D)
            assert len(T) > 0, "timestamps were not found in this source (cohort=%s)" % cohort_name
            assert len(D) == len(T), "inconsistent sizes between sequences and timestamps"

            sr.readDocToCSV(sequences=D, timestamps=T, labels=L, cohort=cohort_name, seq_ptype=seq_ptype, 
                outputdir=basedir, meta=user_file_descriptor)
            # condition: a labeled source documents is generated 
        return 
    def get_cohort(): 
        try: 
            return kargs['cohort']
        except: 
            pass 
        raise ValueError, "cohort info is mandatory."
    def remove_diag_prefix(D): 
        # newer format for diagnosic codes in odhsi DB admits prefixes such as I9, I10
        for i, doc in enumerate(D):  # D: a list of list of tokens 
            # D[i] = map(removeDiagPrefix(x, test_=True), doc) # 
            D[i] = map(remove_prefix, doc)  # inplace
        return D
    def remove_prefix(x, regex='^I(10|9):'):
        return re.sub(regex, '', x)
    def to_str(D, sep=' '):
        return [sep.join(doc) for doc in D]  # assuming that each element is a string 

    import seqReader as sr
    import seqTransform as st
    from seqparams import TDoc  # or use tdoc.TDoc 

    cohort_name = get_cohort()
    
    # verify the existence of labele_seq file 
    # sr.verifyLabeledSeqFile(corhot=cohort_name, seq_ptype=, ext='csv', **kargs): # [params] cohort, doctype, seq_ptype, doc_basename, ext + basedir
    
    # max_n_docs, max_n_docs_policy = kargs.get('max_n_docs', None), kargs.get('max_n_docs_policy', 'sorted')
    D, L, T = load_docs()  # [params] inputdir, composition
    nD0 = len(D)

    # labels = L  # documents = D; timestamps = T
    if len(L) == 0: L = kargs.get('labels', [1] * len(D)) # if no labeling found => user-provided label set 

    # ensure that labeled_seq file exists
    if kargs.get('create_labeled_docs', False): make_labeled_docs()  # include labeling to the sequence source file
    assert len(D) == len(L) == len(T), "Size mismatch: nD=%d, nL=%d, nT=%d" % (len(D), len(L), len(T))

    ### document transfomration
    print('processDocuments> Begin document transformation operations (e.g. simplify, diag-only, etc)')

    # prior, posterior only? note that labeled sequence source above contain the entire sequence, not the sliced version
    # [note] this is usually handled by segment_docs in makeTSetCombined() now
    D, T = slice_docs(D, T)  # params: slice_policy; this should not change L
    assert len(D) == len(L), "Slice operation should preserve corpus size: %d -> %d" % (nD0, len(D))

    # [params] items, policy='empty_doc', predicate=None, simplify_code=False
    D, L, T = transform_docs(D, L, T)  # <- doc_filter_policy, min_ncodes 

    # condition: 
    # a. documents transformed ~ seq_type
    # b. labeled source document (doctype='labeled') is created => the triple: (D, L, T) is guaranteed 

    # [todo] reduce token set
    
    if kargs.get('pad_doc', False): 
        # set max_doc_length to None to use the max among all documents
        D = pad_sequences(D, value=kargs.get('pad_value', 'null'), maxlen=kargs.get('max_doc_length', None))
        testcases = random.sample(D, min(len(D), 10))
        assert len(set(len(tc) for tc in testcases)) == 1, "doc size unequal:\n%s\n" % [len(tc) for tc in testcases]

    if kargs.get('remove_cond_prefix', True):
        st.removePrefix(D, regex='^I(10|9):', inplace=True)

    # [note] no-op for now
    # if kargs.get('remove_med_prefix', False): 
    #     st.removePrefix(D, regex='^(MED|MULTUM|NDC):', inplace=True)

    # convert D from list of list (of tokens) to a list of strings
    if kargs.get('to_str', False):  # [note] string format is eariser to work with Keras, Tensorflow, and other deep learning libraries. 
        D = to_str(D)  # use ' ' by default

    return (D, L, T)

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

def processDocuments2(**kargs):
    return prepareDocumentInput(**kargs) 
def prepareDocumentInput(**kargs): # same as makeTSet but does not distinguish train and test split in d2v
    """
    Similar to makeTSetCombined each document in the document set (D) is broken down into a sequence 
    of visits, each of which consist of codes associated with a single timestamp. 

    Use document embedding method (e.g. paragraph vector) to convert each visit into a vector. 

    Use
    ---
    1. Look at the last 10 visits of a given document
        - s'pose each visit has max: 100 codes (boostrap)

    """
    raise NotImplementedError, "Prototype available only in module seqClasify only for now."

def stratifyDocuments(**kargs):
    """
    Similar to processDocument() but also allows for stratification by labels. 

    Memo
    ----
    1. CKD Data 
       {'CKD Stage 3a': 263, 'Unknown': 576, 'CKD Stage 3b': 159, 'ESRD on dialysis': 43, 'CKD G1-control': 136, 
        'CKD G1A1-control': 118, 'CKD Stage 5': 44, 'CKD Stage 4': 84, 'ESRD after transplant': 691, 
        'CKD Stage 2': 630, 'CKD Stage 1': 89}

    2. Relabeling example: 

        lmap['Others'] = ['CKD G1-control', 'CKD G1A1-control', 'Unknown', ]  # ... control data 
        lmap['CKD Stage 5'] = ['ESRD after transplant', 'ESRD on dialysis', ]
    """
    def load_labeled_docs():  # <- seq_ptype, (label_name, label_map)
        ret = kargs.get('result_set', {}) # 'sequence', 'timestamp', 'label' 
        # prefix = inputdir
        # [note]
        src_dir = kargs.get('inputdir', None)
        if src_dir is None: src_dir =  get_global_cohort_dir()

        ifile, ifiles = kargs.get('inputfile', None), kargs.get('ifiles', [])
        if ifile is not None: 
            ipath = os.path.join(src_dir, ifile)
            assert os.path.exists(ipath), "Invalid input path: %s" % ipath
            ifiles = [ifile, ]

        if not ret: 
            # [note] 'fpath' is the path to ctype-dependent labeled documents but this is not necessary 
            #        for now, just load the full documents and then extract the desired portion on the fly.  
            # fpath = TDoc.getPath(cohort=cohort_name, seq_ptype=seq_ptype, 
            #                 doctype='labeled', ext='csv', basedir=prefix) 
            ret = sr.readDocFromCSV(cohort=cohort_name, inputdir=src_dir, ifiles=ifiles, doctype='labeled', 
                            stratify=True, 
                            label_name=kargs.get('label_name', 'label'), 
                            label_map=kargs.get('label_map', {})) # [params] label_name: {'label', 'label_lcs'}
            assert len(ret) > 0, "No labeled data found given cohort=%s, doctype=%s, prefix=%s" % (cohort_name, doctype, inputdir)
        
        return ret # 'ret' is a nested dictionary indexed by labels
    def transform_docs(D, L, T): # params: seq_ptype, predicate, simplify_code
        seq_ptype = kargs.get('seq_ptype', 'regular')
        # [params] items, policy='empty_doc', predicate=None, simplify_code=False
        predicate = kargs.get('predicate', None)
        simplify_code = kargs.get('simplify_code', False)

        nD, nD0 = len(D), len(D[0])
        # this modifies D 

        minDocLength = kargs.pop('min_ncodes', 10)
        print('(transform_docs) Condition: ctype=%s, predicate=%s, simplified? %s' % (seq_ptype, predicate, simplify_code))

        D2, L2, T2 = transformDocuments2(D, L=L, T=T, 

            # document-wise fitler
            policy=kargs.get('doc_filter_policy', 'minimal_evidence'),  # which triaggers the use of min_ncodes
            min_ncodes=minDocLength, # only used when policy: minimual_evidence

            # content-filter
            seq_ptype=seq_ptype, 

            # [note] is it diagnosis code, is it medication? this is always specified via seq_ptype, no need to use predicate
            #        predicate used for slicing, segmenting is deferred to slice_docs(), segment_docs()
            predicate=None,  # predicate is used to filter unwanted codes by preserving only those upon which the predicate evaluates to True
            simplify_code=simplify_code, 

            save_=False) # if True, save only if doesn't exist 
        # D, labels = transformDocuments(D, L=labels, seq_ptype=seq_ptype)

        print('(transform_docs) after transform | nDoc: %d -> %d' %  (nD, len(D2)))
        
        # if nD0 != len(D2[0]): 
        # 
        #     # print('... original:    %s' % D[0][:100])
        #     print('... transformed: %s' % D2[0][:100])
        
        print('(transform_docs) after transform nD: %d, nT: %d, nL: %d | min_ncodes=%d' % (len(D2), len(T2), len(L2), minDocLength))

        return (D2, L2, T2)
    def do_slice(): 
        if not kargs.has_key('slice_policy'): return False
        if kargs['slice_policy'].startswith(('noop', 'reg', 'compl', )):  # effective operations 
            return False 
        return True  # prior, posterior [todo] 'in-between'; e.g. from first occurence of X to last occurence of X
    def do_transform(): 
        if not seq_ptype.startswith('reg'): return True
        if do_slice(): return True
        if kargs.get('simplify_code', False): return True
        return True
    def slice_docs(D, T):   # use: only consider LCS labels in the pre- or post- diagnostic sequences
        # inplace operation by default
        if not do_slice(): return (D, T)  # noop
        print('  + splicing sequences in D and T | policy=%s, predicate=%s, cutpoint=%s, inclusive? %s' % \
            (kargs.get('slice_policy', 'noop'), kargs.get('predicate', None), kargs.get('cutpoint', None), kargs.get('inclusive', True)))
        nD0 = len(D)
        D, T = st.sliceDocuments(D, T=T, 
                        policy=kargs.get('slice_policy', 'noop'),  # {'noop', 'prior', 'posterior'}
                        cohort=cohort_name,  # help determine predicate if not provided
                        predicate=kargs.get('predicate', None), # infer predicate from cohort if possible
                        cutpoint=kargs.get('cutpoint', None), 
                        n_active=1, 
                        inclusive=kargs.get('inclusive', True))
        assert len(D) == len(T)
        assert len(D) == nD0, "size of document set should not be different after splicing nD: %d -> %d" % (len(D), nD0) 
        return (D, T)
    def segment_docs(D, L, T):  # <- D, L, T
        ### An improved version of slice_docs() above 
 
        predicate = kargs.get('predicate', None)
        policy_segment =  kargs.get('policy_segment', 'regular')

        # policy: {'regular', 'two'/'halves', 'prior'/'posterior', 'complete', }
        # output: (DocIDs, D, L, T)

        # only consider the segment prior to time_cutoff (prior to applying eMERGE phenotyping algorithm)
        time_cutoff = kargs.get('time_cutoff', '2017-08-18') # only consider the records prior to applying eMERGE phenotyping algorithm
        if time_cutoff is not None: 
            _, D, L, T = segmentDocumentByTime(D, L, T, timestamp=time_cutoff, policy='prior', inclusive=True)
        
        include_endpoint = kargs.get('inclusive', True)  # include the chunk containing diagnosis info?
        drop_docs_without_cutpoints = kargs.get('drop_nullcut', True)  # if True, do not include doc without valid cut points (diagnosis info)

        # configure file ID 
        # if predicate is None or policy is regular, then no-op
        if predicate is not None and not policy_segment.startswith('reg'):
            # assert tsHandler.meta and not (tsHandler.meta in ('D', 'default', )), \
            #     "default user-defined file ID is not recommended since segmenting op will modify the documents (policy=%s)" % policy_segment
            print('segment_docs> predicate: %s, policy: %s' % (predicate, policy_segment))
        else: 
            print('segment_docs> policy: %s' % policy_segment)

        # [output] 4-tuple: (docIDs, D, L, T)
        return segmentDocuments(D, L, T, predicate=predicate, policy=policy_segment, 
                    inclusive=include_endpoint, drop_nullcut=drop_docs_without_cutpoints) 
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
    def get_global_cohort_dir(): 
        return seqparams.getCohortGlobalDir(get_cohort()) # sys_config.read('DataExpRoot')/<cohort> 
    def canonicalize_lmap(lmap): # lmap
        adict = {}  
        for label, eq_labels in lmap.items():  
            for eql in eq_labels: 
                adict[eql] = label   # maps eql to label (e.g. 'ESRD after transplant' to 'CKD Stage 5')
        return adict 
    def relabel(): # lmap, units: label -> entry: {sequence, timestamp} 
        pass
    def remove_diag_prefix(D): 
        # newer format for diagnosic codes in odhsi DB admits prefixes such as I9, I10
        for i, doc in enumerate(D):  # D: a list of list of tokens 
            # D[i] = map(removeDiagPrefix(x, test_=True), doc) # 
            D[i] = map(remove_prefix, doc)  # inplace
        return D
    def remove_prefix(x, regex='^I(10|9):'):
        return re.sub(regex, '', x)
    def test_diff(D, D2, n=10, maxtok=100): 
        nD = len(D)

        N = sum(1 for d in D if len(d) > 0)
        N2 = sum(1 for d2 in D2 if len(d2) > 0)
        r = N2/(N+0.0)
        print('(test_diff) ratio of documents containing disease codes: %f | N=%d, N2=%d' % (r, N, N2))

        n_tests = n
        for i, d2 in enumerate(D2): 
            if len(d2) > 0 and (D[i] != d2): 
                print('... original:    %s' % D[i][:maxtok])  
                print('... transformed: %s' % d2[:maxtok])  
        return

    from seqparams import TSet, TDoc
    from labeling import TDocTag
    import seqReader as sr
    import seqTransform as st

    # single label format ['l1', 'l2', ...] instead of multilabel format [['l1', 'l2', ...], [], [], ...]
    cohort_name = get_cohort()
    inputdir = kargs.get('inputdir', get_global_cohort_dir())
    seq_ptype = seqparams.normalize_ctype(kargs.get('seq_ptype', 'regular'))
    tSingleLabelFormat = kargs.get('single_label_format', True) 
    
    tRemoveCondPrefix = kargs.get('remove_cond_prefix', True)  # new format of diag code in odhsi DB admits prefixes (e.g. I9, I10)
    tRemoveMedPrefix = kargs.get('remove_med_prefix', False)

    tTransformSequence = do_transform() 
    lmap = kargs.get('label_map', {})  # rename labels e.g. ()

    units = load_labeled_docs() # if not available, should run makeLabeledDocuments(cohort) first 
    print('stratifyDocuments> Found %d entries/labels' % (len(units)))
    stratified = {l: {} for l in units.keys()}
    for label, entry in units.items(): 

        # the per-stratum 3-tuple: (D, T, L) representing MCS documents, timestamps and labels respectively
        D, T, L = entry['sequence'], entry.get('timestamp', []), entry.get('label', [])
        print('(stratifyDocuments) nD: %d, nT: %d, nL: %d' % (len(D), len(T), len(L)))
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
            D0, L0, T0 = transform_docs(D, L, T) 
            # prior, posterior only? note that labeled sequence source above contain the entire sequence, not the sliced version
            
            # D, T = slice_docs(D, T)  # params: slice_policy; this should not change L
            docIDs, D, L, T = segment_docs(D0, L0, T0)
            test_diff(D0, D, n=10, maxtok=100)  

        if len(lmap) > 0: relabel() # params: units

        if tRemoveCondPrefix: 
            st.removePrefix(D, regex='^I(10|9):', inplace=True)

        stratified[label]['sequence'] = D
        stratified[label]['timestamp'] = T
        stratified[label]['label'] = L

    # relabels (e.g. merging labels) => this is already done at load_labeled_docs()
    # if len(lmap) > 0: 
    #     print('stratifyDocuments> Found label map with %d entries ...' % len(lmap))
    #     lmapc = canonicalize_lmap(lmap)

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

def cutByTimeToDiagnosis(D, T, is_case, min_mention=1, drop_empty=False):
    """
    Take the subsequence of D[i] up to and include the first n times 
    that isCase routine is evaluated to True. 

    Memo
    ----
    isCase: e.g. pattern.ckd.isCase(code) that uses diagnostic code as a proxy to 
            indicate a case for a given disease
    """ 
    def analyze_doc(): # D, D2
        print('cutByTimeToDiagnosis> len(D): %d -> %d' % (len(D), len(T)))
        Elen0 = sum(len(d) for d in D)/(len(D)+0.0)
        Elen= sum(len(d) for d in D2)/(len(D2)+0.0)
        print('  + expected length: %f -> %f' % (Elen0, Elen))

        # test
        assert len(T2) == len(D2)
        return

    assert hasattr(is_case, '__call__'), "is_case has to be a callable predicate"
    D2, T2 = [], []
    for i, seq in enumerate(D):
        n, end_index = 0, 0
        for j, code in enumerate(seq): 
            if is_case(code): n += 1 
            if n >= min_mention: 
                end_index = j

        seq2 = seq[:end_index]
        if not drop_empty or len(seq2) > 0: 
            D2.append(seq2)
            T2.append(T[i][:end_index])

    analyze_doc()
    return (D2, T2)

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

def t_docs(**kargs): 
    def process_docs(inputfile, inputdir, ifiles=[]): 
        ### load + transfomr + (ensure that labeled_seq exists)
        src_dir = inputdir
        if src_dir is None: src_dir =  get_global_cohort_dir()
        ctype = tsHandler.ctype # kargs.get('seq_ptype', 'regular')
        
        ifile = inputfile
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
        print('  + nD: %d | cohort=%s, ctype=%s, labeled? %s, simplified? %s' % \
            (len(D), tsHandler.cohort, ctype, len(set(L))>1, kargs.get('simplify_code', False)))
        return (D, L, T)
    def get_global_cohort_dir(): 
        return seqparams.getCohortGlobalDir(tsHandler.cohort) # sys_config.read('DataExpRoot')/<cohort>
    def countTokens(D, n_top=10): 
        counter = collections.Counter()
        for d in D: 
            counter.update(d)

        print('info> topn:\n%s\n' % counter.most_common(n_top))
        return len(counter)

    import collections
    import seqparams
    from seqConfig import tsHandler
    from pattern import ckd

    ### tunable parameters
    tsHandler.cohort =  kargs.get('cohort', 'CKD')
    d2v_method = 'pv-dm2' # tsHandler.d2v
    tUseTestSet = False
    tSegmentByVisit = False
    policy_segment = kargs.get('policy_segment', 'regular')
    predicate = None if policy_segment.startswith('reg') else ckd.isCaseCCS

    user_file_descriptor = meta = kargs.get('meta', tsHandler.meta) 
    include_augmented = kargs.get('include_augmented', False)
    cohort_name = '%s0' % tsHandler.cohort if tUseTestSet else tsHandler.cohort

    ### document input
    inputfile = kargs.get('inputfile', 'condition_drug_labeled_seq-CKD.csv')
    inputdir = kargs.get('inputdir', seqparams.getCohortGlobalDir(cohort=cohort_name)) # small cohort (n ~ 2.8K)

    D, L, T = process_docs(inputfile, inputdir)
    nD0 = len(D)  

    nTokens = countTokens(D, n_top=10)
    print('... n_uniq_tokens: %d' % nTokens)

    include_endpoint, drop_nullcut, drop_ctrl = True, False, False  # only relevant when policy_segment is not 'regular'
    if policy_segment.startswith(('pri', 'post')): 
        drop_nullcut=True  # sequences where predicate does not apply (e.g. lack of decisive diagnosis) are nullcuts
        drop_ctrl=True   # need to drop control group because there may not exist data points for say in pre-diagnostic sequences

    docIDs, D, L, T = segmentDocuments(D, L, T, predicate=predicate, policy=policy_segment, 
        inclusive=include_endpoint, drop_nullcut=drop_nullcut, segment_by_visit=tSegmentByVisit)

    nDs = len(D)
    print('[Q1] n_docs total: %d, n(has diag info): %d => ratio: %f' % (nD0, nDs, nDs/(nD0+0.0)))
    
    vocab_size = 10000+1 
    D_int = tokenizeDoc(D, 
               test_level=1, throw_=True, customized=False, pad_empty_doc=True, 
               num_words=vocab_size, split=' ', oov_token=TDoc.token_unknown)
    n_tokens = countTokens(D_int, n=10)
    print('info> n_uniq_tokens: %d =?= %d (desired)' % (n_tokens, vocab_size))

    rp = random.randint(0, len(D_int))
    print('... example doc:\n%s\n' % D_int[rp])

    # note the order of the input: D/documents, T/timestamps, L/labels
    tsHandler.save_mcs(D_int, T=[], L=L, index=0, docIDs=docIDs, meta='tokenized') # other params:  sampledIDs=[], inputdir, inputdir

    # test consistency

    return

def t_docs0(**kargs): 
    def countTokens(D, n=10): 
        counter = collections.Counter()
        for d in D: 
            counter.update(d)

        print('info> topn:\n%s\n' % counter.most_common(n))
        return len(counter)
    def compare(Ds, Dt):
        assert len(Ds) == len(Dt), "size(Ds):%d != size(Dt):%d" % (len(Ds), len(Dt))
        for i, ds in enumerate(Ds):
            print('(source)    %d: %s' % (i, ds))
            print('(tokenized) %d: %s' % (i, Dt[i]))
        return

    import collections 

    D = ['Well done!',
                'Enjoy life!!',
                'Great effort!!!',
                'Balance between science and spirituality',
                'Work-life balance!', 
                'Enjoy what you do!',
                'The universe is teeming with life.',
                'x y z', 
                '120.7, 250.0 250.11 toxic scam', 
                'Change your life for the better', 
                'Enjoy the moment!', 
                'From moment to moment',
                '250.0 represents diabetes.', 
                'u',]
    nDoc = len(D)
    vocab_size = 5
    oov_token = 'unknown'
    D_int = tokenizeDoc(D, 
               test_level=1, throw_=True, 
               customized=False, # set to False to use Kera's tokenizer

               # params for tokenzier
               num_words=vocab_size, split=' ', oov_token=oov_token)
    nDocTokenized = len(D_int)
    print('info> n_docs (orig): %d =?= n_docs (tokenized): %d' % (nDoc, nDocTokenized))

    n_tokens = countTokens(D_int, n=10)
    print('info> n_uniq_tokens: %d vs %d (desired)' % (n_tokens, vocab_size))

    compare(D, D_int)

    return

def t_stratify(**kargs):
    def policy_relabel():  # ad-hoc
        # CKD stage classes
        lmap = {}
        lmap['Others'] = ['CKD G1-control', 'CKD G1A1-control', 'Unknown', ]  # ... control data 
        lmap['CKD Stage 5'] = ['ESRD after transplant', 'ESRD on dialysis', ]
        return lmap
    def stratum_stats(): # <- units
        labels = units.keys() # documents = D; timestamps = T
        n_labels = len(set(labels))

        nD = nT = 0
        sizes = {}
        for label, entry in units.items():
            Di = entry['sequence']
            Ti = entry['timestamp']
            Li = entry['label']
            nD += len(Di)
            nT += len(Ti)
            sizes[label] = len(Di)
            nDi, nTi = len(Di), len(Ti)
            assert nDi == nTi, "nD=%d <> nT=%d" % (nDi, nTi)
        # assert nD == nT, "nD=%d <> nT=%d" % (nD, nT)
        print('stratum_stats> nD=%d, nT=%d, n_labels=%d ...\n  + sizes:\n%s\n' % (nD, nT, n_labels, sizes))
        return 

    cohort_name = kargs.get('cohort', 'CKD')

    ### stratify cohorts
    # units = stratifyDocuments(cohort=cohort_name, seq_ptype=kargs.get('seq_ptype', 'regular'),
    #                  predicate=kargs.get('predicate', None), simplify_code=kargs.get('simplify_code', False))  # [params] composition
    # stratum_stats()
    
    # relabeld? 
    lmap = policy_relabel()
    units = stratifyDocuments(cohort=cohort_name, seq_ptype=kargs.get('seq_ptype', 'regular'),
                    label_map=lmap, 
                    predicate=kargs.get('predicate', None), 
                    simplify_code=kargs.get('simplify_code', False))  # [params] composition
    stratum_stats()

    return

def runQuery(**kargs):
    """
    Answer research questions. 
    """ 
    t_docs()  # the fraction of MCS documents carrying CKD-related diagnostic codes.

    return

def test(**kargs): 
    """

    Memo
    ----
    1. also see seqClassify
    """

    ### document processing 
    # t_process_docs(**kargs)

    t_docs0() # similar to t_docs(), dealing with short documents
    # t_docs() # tokenization, filtering infrequent tokens, integer repr


    ### stratify documents 
    # t_stratify(**kargs)

    return 

if __name__ == "__main__": 
    # test()

    runQuery(cohort='CKD', policy_segment='prior')


