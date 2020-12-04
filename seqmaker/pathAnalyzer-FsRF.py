# encoding: utf-8

import numpy as np
import multiprocessing

import os, sys, re, random, time, math, collections
import collections, gc
from datetime import datetime
from collections import defaultdict 

# pandas 
from pandas import DataFrame, Series 
import pandas as pd 

# gensim 
from gensim.models import Doc2Vec
import gensim.models.doc2vec

# tpheno
import seqReader as sr
import seqparams, seqAlgo
import vector
import labeling 
from batchpheno.utils import div

# data persistence (e.g. analyzeMCS(), analyzeLCS3() in incremental code)
try:
    import cPickle as pickle
except:
    import pickle

gHasConfig = True
try: 
    from config import seq_maker_config, sys_config
except: 
    print("tdoc> Could not find default configuration package.")
    gHasConfig = False

from tdoc import TDoc
from tset import TSet

# module configuration 
from seqparams import Pathway
from seqConfig import lcsHandler
from system import utils as sysutils

################################################################################################################
#
#  PathAnalyzer
#     answers the following queries 
# 
#
#  Use 
#  ---
#  1. prediagnostic sequence
#
#
################################################################################################################





def readDocFromCSV(**kargs):  # ported from seqReader
    """
    Read coding sequences from .csv files 

    Input
    -----
    cohort: name of the cohort (e.g. PTSD) ... must be provided; no default
            used to figure out file names
    ifiles: sources of coding sequences 
            if ifiles are given, the cohort is igonored 
    (o) basedir: directory from which sources are stored (used when ifiles do not include 'prefixes')

    Memo
    ----
    import seqReader as sr 
    sr.readDocFromCSV(**kargs)
    
    """
    def get_sources(complete=True):
        ifiles = kargs.get('ifiles', [])
        if not ifiles: assert cohort_name is not None

        # [note] doctype has to be 'timed' because timestamps are expected to be in the .csv file
        if complete and not ifiles: # cohort-specific
            for doctype in ['labeled', 'timed', 'doc', 'visit', ]:  # the more complete (more meta data), the higher the precedence
                docfiles = TDoc.getPaths(cohort=cohort_name, basedir=docSrcDir, doctype=doctype, 
                    ifiles=[], ext='csv', verfiy_=True)  # if ifiles is given, then cohort is ignored
                if len(docfiles) > 0: break
        else: # doctpye specific (or ifiles are given)
            # [use] 
            # 1. ifiles <- a list of file names (of the source documents) then use basedir to figure out full paths
            # 2. ifiles <- full paths to the document sources (including prefixes)
            docfiles = TDoc.getPaths(cohort='n/a', basedir=docSrcDir, doctype=kargs.get('doctype', 'timed'), 
                ifiles=ifiles, ext='csv', verfiy_=True)  # if ifiles is given, then cohort is ignored
        assert len(docfiles) > 0, "No %s-specific sources (.dat) are available under:\n%s\n" % (cohort_name, docSrcDir)
        return docfiles   
    def str_to_seq(df, col='sequence', sep=','):
        seqx = [] # a list of (lists of tokens)
        for seqstr in df[col].values: 
            tokens = seqstr.split(sep)
            seqx.append(tokens)
        return seqx

    header = ['sequence', 'timestamp', 'label', ] 
    listOfTokensFormat = TDoc.fListOfTokens # ['sequence', 'timestamp', ]

    cohort_name = kargs.get('cohort', None)  # needed if ifiles is not provided
    basedir = docSrcDir = kargs.get('inputdir', sys_config.read('DataExpRoot'))  # sys_config.read('DataIn') # document in data-in shows one-visit per line

    # if 'complete' is True => try the most information-rich format first i.e. in the order of doctype='labeled', 'timed', 'doc', 'visit'
    fpaths = get_sources(complete=kargs.get('complete', True)) # [params] cohort, basedir
    print('read> reading from %d source files:\n%s\n' % (len(fpaths), fpaths))

    # [output]
    ret = {h: [] for h in header}
    # sequences, timestamps, labels = [], [], []

    # [policy] if there are multiple sources, their contents will be consolidated
    for fpath in fpaths: 
        df_seq = pd.read_csv(fpath, sep='|', header=0, index_col=False, error_bad_lines=True)
        if not df_seq.empty: 
            for h in header: 
                if h in header: 
                    # or use seqparams.TDoc.strToSeq(df_seq, col='sequence')
                    ret[h].extend(str_to_seq(df_seq, col=h, sep=','))  # assuming that ';', '$' are all being replaced

    assert len(ret) > 0, "No data found using the given attributes: %s (columns expected: %s)" % (header, df_seq.columns.values)
    return ret # keys <- header

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
    return st.transformDocuments(docs, L=L, T=T, policy=policy, 
        seq_ptype=seq_ptype, predicate=predicate, simplify_code=simplify_code)

def transformDocuments2(D, L=[], T=[], **kargs):  # this is not the same as seqTransform.transform()
    """
    Transform the document (as does transformDocuments() but also save a copy of the transformed document)
    """
    import seqTransform as st 
    return st.transformDocuments2(D, L=L, T=T, **kargs)

def stratifyDocuments(**kargs): 
    """
    Same as processDocuments() but also allows for stratification by labels. 

    Output
    ------
    a dictionary with keys -> labels 
    values -> data dictionary with following keys: 
            sequence: D
            timestamp: T 
            label: L 

        Recap: D, T, L? see processDocuments()

    Memo
    ----
    1. CKD Data 
       {'CKD Stage 3a': 263, 'Unknown': 576, 'CKD Stage 3b': 159, 'ESRD on dialysis': 43, 'CKD G1-control': 136, 
        'CKD G1A1-control': 118, 'CKD Stage 5': 44, 'CKD Stage 4': 84, 'ESRD after transplant': 691, 
        'CKD Stage 2': 630, 'CKD Stage 1': 89}
    """
    import docProc as dp  # document processor module (wrapper app of seqReader, seqTransformer) 
    return dp.stratifyDocuments(**kargs)

def loadDocuments(**kargs):
    """
    Load documents (and labels)

    Params
    ------
    cohort
    use_surrogate_label: applies only when no labels found in the dataframe

    result_set: a dictionary with keys ['sequence', 'timestamp', 'label', ]
    ifiles: paths to document sources

    Output: a 2-tuple: (D, l) where 
            D: a 2-D np.array (in which each document is a list of strings/tokens)
            l: labels in 1-D array

    Related
    -------
    stratifyDocuments 
    sampleDocuments 

    """
    import seqClassify as sclf
    return sclf.loadDocuments(**kargs)

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

    Output: a 3-tuple: (D, T, L) where 
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
    return dp.processDocuments(**kargs)  # (D, L, T)

def processDocumentsUtil(**kargs):
    """
    A wrapper of processDocuments(). 

    lcsHandler must be configured. 

    """
    def config(): 
        # configure all the parameters 
        userFileID = meta = kargs.get('meta', None)
        ctype = seqparams.normalize_ctype(kargs.get('seq_ptype', 'regular'))
        cohort = kargs.get('cohort', lcsHandler.cohort)  # e.g. 'PTSD'

        # this routine can only be called once, subsquent calls are noop 
        sq.sysConfig(cohort=cohort, seq_ptype=ctype, 
                        lcs_type=kargs.get('lcs_type', 'global') , lcs_policy=kargs.get('lcs_policy', 'df'), 
                        consolidate_lcs=kargs.get('consolidate_lcs', True), 
                        slice_policy=kargs.get('slice_policy', 'noop'), 
                        simplify_code=kargs.get('simplify_code', False), 
                        meta=userFileID)
        return
    def get_global_cohort_dir(): # source MDS
        cohort = kargs.get('cohort', lcsHandler.cohort)  # e.g. 'PTSD'
        return seqparams.getCohortGlobalDir(cohort) # sys_config.read('DataExpRoot')/<cohort>
    def process_docs(inputdir=None): 
        ### load + transfomr + (ensure that labeled_seq exists)
      
        # params: cohort, seq_ptype, ifiles, doc_filter_policy
        #         min_ncodes, simplify_code

        # first check if already provided externally? 
        # use case: sample a subset of documents (D, L, T) and use the result in analyzeLCSDistribution
        D, L, T = kargs.get('D', []), kargs.get('L', []), kargs.get('T', [])
        if len(D) > 0: 
            print('process_docs> Given input documents of size: %d' % len(D))
            assert len(D) == len(T), "size(docs): %d <> size(times): %d" % (len(D), len(T))
            if len(L) == 0: L = [1] * len(D)
            return (D, L, T)

        # otherwise, load from the source
     
        # use the derived MCS set (used to generate d2v training data) as the source
        src_dir = get_global_cohort_dir() if inputdir is None else inputdir # orginal source: get_global_cohort_dir()
        ifiles = kargs.get('ifiles', ['condition_drug_labeled_seq-CKD.csv', ])

        assert os.path.exists(src_dir), "Invaild MCS source dir:\n%s\n" % src_dir
        ctype =  kargs.get('seq_ptype', 'regular')  # kargs.get('seq_ptype', 'regular')
        cohort = kargs.get('cohort', lcsHandler.cohort)  # 'PTSD'

        minDocLength = kargs.pop('min_ncodes', 10)
        tSimplified = kargs.get('simplify_code', False)
        D, L, T = processDocuments(cohort=cohort, seq_ptype=ctype, 
                    inputdir=src_dir, 
                    ifiles=ifiles,
                    # meta=kargs.get('meta', None), 
                    
                    # document-wise filtering 
                    policy=kargs.get('doc_filter_policy', 'minimal_evidence'),  
                    min_ncodes=minDocLength,  # retain only documents with at least n codes

                    # content modification
                    # predicate=kargs.get('predicate', None), # reserved for segment_docs()
                    simplify_code=tSimplified, 

                    source_type='default', 
                    create_labeled_docs=False)  # [params] composition

        # Dl = labelize(D, label_type='doc', class_labels=L)  # use class labels for document label prefixing
        is_labeled = len(np.unique(L)) > 1
        print('  + nD: %d | cohort=%s, ctype=%s, labeled? %s, simplified? %s' % (len(D), cohort, ctype, is_labeled, tSimplified))
        return (D, L, T)
    def stratify_docs(inputdir=None, lmap=None): # params: cohort
        ### load + transfomr + (ensure that labeled_seq exists)

        # use the derived MCS set (used to generate d2v training data) as the source
        src_dir = get_global_cohort_dir() if inputdir is None else inputdir # orginal source: get_global_cohort_dir()
        assert os.path.exists(src_dir), "Invaild MCS source dir:\n%s\n" % src_dir
        ifiles = kargs.get('ifiles', ['condition_drug_labeled_seq-CKD.csv', ])
        ctype =  kargs.get('seq_ptype', lcsHandler.ctype)  # kargs.get('seq_ptype', 'regular')
        # cohort = kargs.get('cohort', 'CKD')  # 'PTSD'
        
        minDocLength = kargs.pop('min_ncodes', 10)
        tSimplified = kargs.get('simplify_code', lcsHandler.is_simplified)
        slicePolicy = kargs.get('splice_policy', lcsHandler.slice_policy)
        
        if lmap is None: lmap = policy_relabel() 
        stratified = stratifyDocuments(cohort=cohort, seq_ptype=ctype,
                    predicate=kargs.get('predicate', None), 
                    simplify_code=kargs.get('simplify_code', tSimplified), 

                    # source
                    inputdir=src_dir, 
                    ifiles=kargs.get('ifiles', []), 
                    min_ncodes=minDocLength, 

                    # relabeling operation 
                    label_map=lmap,  # noop for now

                    # slice operations {'noop', 'prior', 'posterior', }
                    slice_policy=slicePolicy, 
                    slice_predicate=kargs.get('slice_predicate', None), 
                    cutpoint=kargs.get('cutpoint', None),
                    inclusive=True)

        ### subsampling
        # maxNDocs = kargs.get('max_n_docs', None)
        # if maxNDocs is not None: 
        #     nD0 = len(D)
        #     D, L, T = sample_docs(D, L, T, n=maxNDocs)
        #     nD = len(D)

        nD = nT = 0
        for label, entry in stratified.items():
            Di = entry['sequence']
            Ti = entry['timestamp']
            # Li = entry['label']
            nD += len(Di)
            nT += len(Ti)

        # Dl = labelize(D, label_type='doc', class_labels=L)  # use class labels for document label prefixing
        print('stratify_docs> nD: %d | cohort=%s, ctype=%s, simplified? %s' % (nD, lcsHandler.cohort, lcsHandler.ctype, lcsHandler.is_simplified))
        return stratified
    def policy_relabel():  
        ### an example relabling strategy for CKD cohort
        # CKD stage classes
        lmap = {}
        lmap['Others'] = ['CKD G1-control', 'CKD G1A1-control', 'Unknown', ]  # ... control data 
        lmap['CKD Late'] = ['ESRD after transplant', 'ESRD on dialysis', 'CKD Stage 5']
        lmap['CKD Middle'] = ['CKD Stage 3a', 'CKD Stage 3b', 'CKD Stage 4']
        lmap['CKD Early'] = ['CKD Stage 1', 'CKD Stage 2', ]
        return lmap
    def summary(): 
        cohort = kargs.get('cohort', lcsHandler.cohort)  # e.g. 'PTSD'
        minDocLength = kargs.pop('min_ncodes', 10)
        tSimplified = kargs.get('simplify_code', False)
        slicePolicy = kargs.get('splice_policy', lcsHandler.slice_policy)  # for stratify operation
        print('processDocuments> Cohort: %s, min_length: %d, simplified? %s, slice_policy: %s' % \
            (cohort, minDocLength, tSimplified, slicePolicy))

        inputdir = kargs.get('inputdir', None)
        src_dir = get_global_cohort_dir() if inputdir is None else inputdir # orginal source: get_global_cohort_dir()
        print("  + input dir:\n%s\n" % src_dir)
        return
    def subset_docs(D, L, T): 
        idx = kargs.get('document_ids', [])
        Nd = len(idx)
        if Nd > 0: 
            assert Nd <= len(D)
            D = list(np.array(D)[idx])
            L = list(np.array(L)[idx])
            T = list(np.array(T)[idx])
        return (D, L, T)
    def sample_docs(D, L, T, n, sort_index=True, random_state=53):
        # if n > len(y): n = None # noop
        idx = cutils.sampleByLabels(L, n_samples=n, verbose=True, sort_index=sort_index, random_state=random_state)
        D = list(np.array(D)[idx])
        L = list(np.array(L)[idx])
        T = list(np.array(T)[idx])
        assert len(D) == n 
        print('  + subsampling: select only %d docs.' % len(D))
        return (D, L, T, idx)

    # import seqparams
    from tset import TSet
    import seqConfig as sq
    from seqConfig import lcsHandler
    from seqTest import TestDocs
    import classifier.utils as cutils 

    config()
    summary()
    mode = kargs.get('mode', 'load')  # values: 'load', 'stratify', ... 

    ret = {} # output dictionary
    docIds = []
    D, L, T = process_docs(inputdir=kargs.get('inputdir', None))  # if None, use default source directory: sys_config.read('DataExpRoot')/<cohort>
        
    if len(kargs.get('document_ids', [])) > 0: 
        D, L, T = subset_docs(D, L, T) 
        docIds = kargs['document_ids']
    
    maxNDocs = kargs.get('max_n_docs', None)
    if maxNDocs: 
        D, L, T, docIds = sample_docs(D, L, T, n=maxNDocs)

    # header = ['sequence', 'timestamp', 'label', 'doc_ids']
    if mode.startswith('l'): # load
        # nothing more to do 
        ret['sequence'], ret['label'], ret['timestamp'], ret['doc_ids'] = D, L, T, docIds  
    elif mode.startswith('s'): # stratify 
        # ret = stratify_docs(inputdir=kargs.get('inputdir', None), lmap=kargs.get('label_map', None))
        ret = stratify(D, L, T, document_ids=docIds)
    else: 
        raise NotImplementedError
    # [todo] select only a subset of documents 
    # docIds = kargs.get('document_ids', [])
    return ret

def invertedIndex(lcsmap, document_ids=None): 
    return inverseIndex(lcsmap, document_ids=document_ids)
def inverseIndex(lcsmap, document_ids=None):  # lcs -> docIDs 
    from collections import defaultdict
    assert isinstance(lcsmap, dict)

    lcsmapInv = defaultdict(list) # {i: [] for i in range(nD)} # document ID -> LCSs 
    if document_ids is None: 
        for lcs, docIds in lcsmap.items(): 
            for docId in docIds: 
                lcsmapInv[docId].append(lcs)
    else: 
        for lcs, docIds in lcsmap.items(): 
            for docId in docIds: 
                if docId in document_ids: 
                    lcsmapInv[docId].append(lcs)
    # for docId in lcsmapInv.keys(): 
    #     lcsmapInv[docId] = list(lcsmapInv[docId])
    return lcsmapInv

def analyzeMCS(D, T, lcs_set, **kargs):
    """

    Memo
    ----
    key: index, frequency, inverted_index, color, time 
    index: 
        lcsmap: lcs -> document IDs 
    frequency: 
        lcsmapInvFreq: document ID -> {(LCS, freq)}
    inverted_index: 
        lcsmapInv: document ID -> {lcs_i}

    color: 
        lcsColorMap: document ID -> lcs -> {postiions_i}

        lcsColorMap[i][lcs] = lcs_positions

        # document ID -> {LCS -> LCS time indices} read for plot (xxxx0xx0x000xxxxx0 where 0 ... 0 ~> LCS)

    time: 
        lcsTimeMap: document ID -> {LCS -> list of timestamps (of codes in LCS)} 
    

    """
    attributes = ['index', 'inverted_index', 'color', 'time']
    ret = {a: {} for a in attributes} 

    # [todo] add incremental mode
    (lcsmap, lcsmapInvFreq, lcsColorMap, lcsTimeMap) = analyzeLCS2(D, T, lcs_set, **kargs) 
    
    ret['index'] = lcsmap  # lcs -> {doc<i>}
    ret['frequency'] = lcsmapInvFreq  # doc<i> -> {(LCS<i>, freq)}

    # lcsmapInvFreq -> lcsmapInv 
    lcsmapInv = {}
    for i, lcs_freq_set in lcsmapInvFreq.items(): 
        lcsmapInv[i] = [lcs for lcs, freq in lcs_freq_set]

    ret['inverted_index'] = lcsmapInv   # document ID -> {LCS<i>}
    ret['color'] = lcsColorMap
    ret['time'] = lcsTimeMap
    return ret
def analyzeLCS2a(D, T, lcs_set, **kargs):
    """
    Same as analyzeLCS2() but with the return value in dictionary.  
    """
    return analyzeMCS(D, T, lcs_set, **kargs)
def analyzeLCS2(D, T, lcs_set, **kargs):
    """
    Similar to analyzeLCS() but incorporate time elements. 


    **kargs
    ------- 
    lcs_sep: code delimitor in the input LCS string 
    lcsmap: lcs -> docIDs

    Usage Note
    ----------
    1. usually used after deriveLCS() call, which yields the map from (candidate) LCSs to 
       their host documents where LCSs were selected via given criteria (e.g. minimum 
       document frequency + topn)
    2. lcs_set could be the keys of lcsmap: lcs -> docIDs

    """ 
    def normalize_input(x): 
        if isinstance(x, str): 
            x = x.split(kargs.get('sep', ' '))
        assert hasattr(x, '__iter__'), "Invalid input codes: %s" % x
        return x # a list of code strings
    def to_str(seq, sep=','): 
        return sep.join(str(e) for e in seq)
    def code_str(seq, sep='-'): # convert codes to file naming friendly format 
        s = to_str(seq, sep=sep) 
        # alternative: s = s.translate(None, string.punctuation)  # remove punctuations '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
        return s.replace('.', '')
    def time_to_diagnoses(sid, test_=False): # [params] tseqx, matched_index_set
        pass 
    def time_precode_target(seq, tseq, tid=None): # sid: the sid-th document, need this to find corresponding timestamp
        # assert len(seq) == len(tseq)
        # if tid is None: tid = sd00  # first match (of target), first code
        # # time elapsed (days) between first relevant pre-diagnostic code to the target
        # i0, c0 = 0, '000.0'
        # precode_set = set(precodes)
        # for i, c in enumerate(seq): 
        #     if c in precode_set: 
        #         i0 = i; c0 = c
        #         break

        # # time 
        # assert tid > i0, "Target: %s must occur after pre-diag code: %s" % (codes[0], c0)
        # t0 = datetime.strptime(tseq[i0], "%Y-%m-%d")  # datetime object for the first occurrence of any of the precodes
        # tF = datetime.strptime(tseq[tid], "%Y-%m-%d")
        # delta = (tF-t0).days
        # return delta
        pass
    def eval_lcs_stats(): # [params] lcsmap, lcsmapInv, matched_docIDs
        nM = len(matched_docIDs)
        r = nM/(nD+0.0)
        print('\nanalyzeLCS.eval_lcs_stats> number of docs found match: %d, ratio: %f' % (nM, r))

        # number of documents with multiple LCS labels? 
        n_multimatch = n_uniqmatch = n_nomatch = 0
        for i in range(nD): 
            if i in lcsmapInv: 
                if len(lcsmapInv[i]) > 1: 
                    n_multimatch += 1
                    if n_multimatch <= 10: 
                        print('  + document #%d matches multiple LCSs:\n  + %s ...\n' % (i, lcsmapInv[i][:10]))
                elif len(lcsmapInv[i]) == 1: 
                    n_uniqmatch += 1 
                else: 
                    n_nomatch +=1  # this may not exist 
        print('    + number of documents %d | n_multilabel:%d, n_single: %d, n_nomatch: %d' % (nD, n_multimatch, n_uniqmatch, n_nomatch))

        # most popular LCSs? 
        hotLCSs = sorted([(lcs, len(dx)) for lcs, dx in lcsmap.items()], key=lambda x:x[1], reverse=True)[:10]
        for lcs, cnt in hotLCSs: 
            print('  + LCS: %s found %d matches ...' % (lcs, cnt))

        return 
    def test_input(): 
        # either lcs_set or lcsmap has to be given 
        if len(lcs_set) == 0: 
            assert len(kargs.get('lcsmap', {})) > 0, "Either lcs_set or lcsmap has to be provided."
    def make_colored_repr(i, lcs_seq, index_set, base_color=0, verbose=False): # <- lcsColorMap, D, (T)
        # e.g. xxxx0xx0x000xxxxx0 where 0 ... 0 ~> LCS 
        #      0000c00c0          where c is to be replaced by medical codes (colors) and 0s are white
        assert len(index_set) > 0, "LCS: %s is not a match for docID=%d" % (lcs, i)
        
        # preparing for "gene plots," where genes ~ LCSs (or other meaningful sequence segments)
        seq, tseq = D[i], T[i]  # i-th document 

        # A. Overlapping repr
        # lcs_colors = np.array([base_color] * len(seq)) # len(D[i])
        # for t, idx in enumerate(index_set): # foreach (time) position
        #     assert len(lcs_seq) == len(idx)
        #     # lcs_colors[idx] = 
        #     for p in idx:  # foreach positional index
        #         # A. using real codes as colors 
        #         # lcs_colors[i] = lcs_seq[i] 

        #         # B. using increments (logical colors, easier to express "overlaps")
        #         lcs_colors[p] += 1

        # B. Overlapping
        lcs_positions = []
        tHasOverlap = False
        lcs = lcs_sep.join(lcs_seq)  # redundant? 
        for t, idx in enumerate(index_set): # foreach (time) position 
            assert len(lcs_seq) == len(idx)
            lcs_positions.append(idx)
            if t > 0: 
                if len(set(idx).intersection(index_set[t-1])) > 0: 
                    tHasOverlap = True

        if tHasOverlap: 
            assert len(index_set) > 1
            if verbose: 
                # r = random.sample(index_set, 1)
                v1, v2 = index_set[0], index_set[1]
                if len(seq) < 100: 
                    print('color_map> Found overlapping LCS: %s in %d-th doc:\n%s\n' % (lcs, i, seq))
                else: 
                    print('color_map> Found overlapping LCS: %s in %d-th doc:\n%s ...\n' % (lcs, i, seq[:50]))
                print('           + first two: %s =?= %s' % (np.array(seq)[v1], np.array(seq)[v2]))
                print('           + positions: %s ~~~ %s' % (str(v1), str(v2)))

        lcsColorMap[i][lcs] = lcs_positions  # ith document contains lcs in these positions

        # shows which documents contain overlapping LCSs 
        return lcsColorMap
    def has_overlaps(doc_colors): 
        # shows which documents contain overlapping LCSs 
        # any cell contains a value >= 2
        pass
    def make_timeseries_repr(i, lcs_seq, index_set): # lcsTimeMap, D, T
        # assert len(index_set) > 0, "LCS: %s is not a match for docID=%d" % (lcs, i)
        lcs_times = []  # list of list of timestamps
        seq, tseq = D[i], T[i]  # i-th document 
        assert len(seq) == len(tseq), "n(Di)=%d but n(Ti)=%d\n  + Di=\n%s\n  + Ti=\n%s\n" % (len(seq), len(tseq), seq, tseq) 
        for t, idx in enumerate(index_set): 
            assert len(lcs_seq) == len(idx)
            lcs_times.append([tseq[p] for p in idx])
       
        lcs = lcs_sep.join(lcs_seq)  # redundant? 
        lcsTimeMap[i][lcs] = lcs_times
        return lcsTimeMap

    import seqAlgo
    from seqparams import Pathway

    nD = len(D)
    nL = nLCS = len(lcs_set)  # plus 'No_Match' LCS label
    print('analyzeLCS2> Find matching persons whose records contain a given LCS (n=%d); do this for each LCS ...' % nL)

    docIds = kargs.get('document_ids', [])
    tIncremental = kargs.get('incremental_mode', False) or len(docIds) > 0
    if tIncremental:  
        assert kargs.has_key('document_ids'), "document_ids has to be specified for incremental to keep track of document subsets."
        return analyzeLCS3(D, T, lcs_set, **kargs)

    # test_input()
    lcsmap = {lcs: [] for lcs in lcs_set} # LCS -> document IDs  # {lcs:[] for lcs in df['lcs'].unique()}
    lcsmapInv = {i: [] for i in range(nD)} # document ID -> LCSs 
    lcsmapInvFreq = {i: [] for i in range(nD)}  # document ID -> {(LCS, freq)}   ... to subsume lcsmapInv 
    lcsColorMap = {i: {} for i in range(nD)}  # document ID -> {LCS -> LCS time indices} read for plot (xxxx0xx0x000xxxxx0 where 0 ... 0 ~> LCS)
    lcsTimeMap = {i: {} for i in range(nD)} # document ID -> {LCS -> list of timestamps (of codes in LCS)} 

    matched_docIDs = set()  # [test] docuemnts with any  matched LCSs
    lcs_sep = kargs.get('lcs_sep', Pathway.lcs_sep) # should be ' '

    nD_has_labels = 0
    for i, doc in enumerate(D): # foreach doc, find its LCS labels
        # timestamps = T[i]
        docId = i

        for j, lcs in enumerate(lcs_set): # Pathway.header_global_lcs = ['lcs', 'length', 'count', 'n_uniq']
            lcs_seq = lcs.split(lcs_sep)  # Pathway.strToList(lcs)  
            if len(lcs_seq) > len(doc): continue   # LCS is longer than the doc, can't be a match 

            # if j < 10: test_lcs_match(lcs_seq, doc)
            # if seqAlgo.isSubsequence(lcs_seq, doc): # if LCS was derived from patient doc, then at least one match must exist
            matched_index_set = seqAlgo.traceSubsequence3(lcs_seq, doc)
            tMatched = True if len(matched_index_set) > 0 else False
            if tMatched:   # find matched indices
                # find corresponding timestamps 
                # lcs_tseq = lcs_time_series(lcs_seq, i, D, T) # [output] [(<code>, <time>), ...]
                lcsmap[lcs].append(i)  # add the person index
                lcsmapInv[i].append(lcs) # inverted index, which "genes" does an MCS/doc contain? 
                lcsmapInvFreq[i].append((lcs, len(matched_index_set)))  # lcs + frequency

                # shows how LCS is distributed in the document
                if kargs.get('make_color_time', False):
                    make_colored_repr(i, lcs_seq, matched_index_set, verbose=(i%100==0))  # <- lcsColoarMap
                    make_timeseries_repr(i, lcs_seq, matched_index_set) # <- lcsTimeMap

                matched_docIDs.add(i)  # does the ith document contain any (candiate) LCS/genes?  
                # [test]
                # if len(matched_docIDs) < 10: test_lcs_match(lcs_seq, doc)
        if len(lcsmapInv[i]) > 0: nD_has_labels += 1 
    
    assert nD_has_labels >= 3, "Only %d documents have matching LCSs (nD=%d, nLCS=%d)" % (nD_has_labels, nD, nL)
    assert len(lcsColorMap) > 0 and len(lcsTimeMap) > 0
    eval_lcs_stats() 

    # [note] in order to use readDocFromCSV, need to create .csv from .dat first (see sr.readDocToCSV())
    # ret = readDocFromCSV(cohort=cohort_name, ifiles=docSources, basedir=basedir)

    # byproduct: see analyzePrediagnosticSequence(codes, **kargs)
    # header = ['target', 'time_to_first', 'latency', 'n_visits', 'sequence_to_first', 'has_precodes', 'time_to_last', 'days_elapsed', ]   # time_to_diagnosis: really is time to the FIRST diagnosis
          
    return (lcsmap, lcsmapInvFreq, lcsColorMap, lcsTimeMap)

def analyzeLCS3(D, T, lcs_set, **kargs):
    """
    Similar to analyzeLCS2() but support incremental updates. 
    This routine requires the global document ID (didx) to keep track of completed 
    documents. 


    **kargs
    ------- 
    lcs_sep: code delimitor in the input LCS string 
    lcsmap: lcs -> docIDs

    Usage Note
    ----------
    1. usually used after deriveLCS() call, which yields the map from (candidate) LCSs to 
       their host documents where LCSs were selected via given criteria (e.g. minimum 
       document frequency + topn)
    2. lcs_set could be the keys of lcsmap: lcs -> docIDs

    """ 
    def normalize_input(x): 
        if isinstance(x, str): 
            x = x.split(kargs.get('sep', ' '))
        assert hasattr(x, '__iter__'), "Invalid input codes: %s" % x
        return x # a list of code strings
    def to_str(seq, sep=','): 
        return sep.join(str(e) for e in seq)
    def code_str(seq, sep='-'): # convert codes to file naming friendly format 
        s = to_str(seq, sep=sep) 
        # alternative: s = s.translate(None, string.punctuation)  # remove punctuations '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
        return s.replace('.', '')
    def time_to_diagnoses(sid, test_=False): # [params] tseqx, matched_index_set
        pass 
    def eval_lcs_stats(): # [params] D, lcsmap, lcsmapInv, matched_docIDs
        nM = len(matched_docIDs)
        r = nM/(nD+0.0)
        print('analyzeLCS.eval_lcs_stats> number of docs found match: %d, ratio: %f' % (nM, r))

        # number of documents with multiple LCS labels? 
        n_multimatch = n_uniqmatch = n_nomatch = 0
        for i in range(nD):
            docId = docIds[i] 
            if docId in lcsmapInv: 
                if len(lcsmapInv[docId]) > 1: 
                    n_multimatch += 1
                    if n_multimatch <= 10: 
                        print('  + document #%d matches multiple LCSs:\n  + %s ...\n' % (i, lcsmapInv[docId][:10]))
                elif len(lcsmapInv[docId]) == 1: 
                    n_uniqmatch += 1 
                else: 
                    n_nomatch +=1  # this may not exist 
        print('    + number of documents: %d | n_multilabel:%d, n_single: %d, n_nomatch: %d' % (nD, n_multimatch, n_uniqmatch, n_nomatch))

        # most popular LCSs? 
        hotLCSs = sorted([(lcs, len(dx)) for lcs, dx in lcsmap.items()], key=lambda x:x[1], reverse=True)[:10]
        for lcs, cnt in hotLCSs: 
            print('  + LCS: %s found %d matches ...' % (lcs, cnt))

        return 
    def test_input(): 
        # either lcs_set or lcsmap has to be given 
        if len(lcs_set) == 0: 
            assert len(kargs.get('lcsmap', {})) > 0, "Either lcs_set or lcsmap has to be provided."
    def make_colored_repr(seq, tseq, docId, lcs_seq, index_set, base_color=0, verbose=False): # <- lcsColorMap, D, (T)
        # e.g. xxxx0xx0x000xxxxx0 where 0 ... 0 ~> LCS 
        #      0000c00c0          where c is to be replaced by medical codes (colors) and 0s are white
        assert len(index_set) > 0, "LCS: %s is not a match for docID=%d" % (lcs, i)
        
        # preparing for "gene plots," where genes ~ LCSs (or other meaningful sequence segments)
        # seq, tseq = D[i], T[i]  # i-th document 

        lcs_positions = []
        tHasOverlap = False
        lcs = lcs_sep.join(lcs_seq)  # redundant? 
        for t, idx in enumerate(index_set): # foreach (time) position 
            assert len(lcs_seq) == len(idx)
            lcs_positions.append(idx)
            if t > 0: 
                if len(set(idx).intersection(index_set[t-1])) > 0: 
                    tHasOverlap = True

        if tHasOverlap: 
            assert len(index_set) > 1
            if verbose: 
                # r = random.sample(index_set, 1)
                v1, v2 = index_set[0], index_set[1]
                if len(seq) < 100: 
                    print('\ncolor_map> Found overlapping LCS: %s in %d-th doc:\n%s\n' % (lcs, i, seq))
                else: 
                    print('\ncolor_map> Found overlapping LCS: %s in %d-th doc:\n%s ...\n' % (lcs, i, seq[:50]))
                print('           + first two: %s =?= %s' % (np.array(seq)[v1], np.array(seq)[v2]))
                print('           + positions: %s ~~~ %s' % (str(v1), str(v2)))

        lcsColorMap[docId][lcs] = lcs_positions  # ith document contains lcs in these positions

        # shows which documents contain overlapping LCSs 
        return lcsColorMap
    def has_overlaps(doc_colors): 
        # shows which documents contain overlapping LCSs 
        # any cell contains a value >= 2
        pass
    def make_timeseries_repr(seq, tseq, docId, lcs_seq, index_set): # lcsTimeMap, D, T
        # assert len(index_set) > 0, "LCS: %s is not a match for docID=%d" % (lcs, i)
        lcs_times = []  # list of list of timestamps
        # seq, tseq = D[i], T[i]  # i-th document 
        assert len(seq) == len(tseq), "n(Di)=%d but n(Ti)=%d\n  + Di=\n%s\n  + Ti=\n%s\n" % (len(seq), len(tseq), seq, tseq) 
        for t, idx in enumerate(index_set): 
            assert len(lcs_seq) == len(idx)
            lcs_times.append([tseq[p] for p in idx])
       
        lcs = lcs_sep.join(lcs_seq)  # redundant? 
        lcsTimeMap[docId][lcs] = lcs_times
        return lcsTimeMap
    def load_completed(name, keys=None, value_type='list', inputdir=None): 
        return initvar(name, keys=keys, value_type=value_type, inputdir=inputdir)
    def initvar(name, keys=None, value_type='list', inputdir=None):
        if inputdir is None: inputdir = Pathway.getPath(cohort=kargs.get('cohort', lcsHandler.cohort))

        ctype = seqparams.normalize_ctype(kargs.get('seq_ptype', lcsHandler.ctype))  # 'diag', 'med'
        fname = '%s-%s.pkl' % (name, ctype)
        fpath = os.path.join(inputdir, fname)
        
        newvar = {}
        if keys is not None: 
            if value_type == 'list': 
                newvar = {k: [] for k in keys}
            else: 
                # assuming nested dictionary 
                newvar = {k: {} for k in keys}
        else: 
            if value_type == 'list': 
                newvar = defaultdict(list)
            else: 
                newvar = defaultdict(dict) # nested dictionary

        if os.path.exists(fpath) and os.path.getsize(fpath) > 0: 
            # if old dictionary has the same key, it'll overwrite newvar's
            oldvar = pickle.load(open(fpath, 'rb'))
            print('initvar> var: %s > load %d entries' % (name, len(oldvar)))

            # [test]
            assert isinstance(oldvar, dict), "TypeError: %s" % str(oldvar)[:100]
            if value_type == 'list': 
                assert isinstance(next(iter(oldvar.values())), list), \
                    "TypeError (var=%s) > sample value: %s" % (name, next(iter(oldvar.values())))
            else: 
                assert isinstance(next(iter(oldvar.values())), dict), \
                    "TypeError (var=%s) > sample value: %s" % (name, next(iter(oldvar.values())))

            newvar.update(oldvar)
            return newvar
        else: 
            print('initvar> init a brand new dictionary for %s' % name)
        
        return newvar # nested dictionary for lcsColorMap, lcsTimeMap
    def inverse(lcsmap, keys=[]):  # <- D
        lcsmapInv = defaultdict(list) 
        if not lcsmap: 
            if not keys: 
                return lcsmapInv
            else: 
                return {k:[] for k in keys}

        for lcs, didx in lcsmap.items(): 
            for di in didx: 
                lcsmapInv[di].append(lcs)  # this cannot have dups
        return lcsmapInv 

    def save_completed(outputdir=None, content_sep=','):  # lcsmap, (lcsmapInv), lcsmapInvFreq, lcsColorMap, lcsTimeMap
        if outputdir is None: outputdir = Pathway.getPath(cohort=kargs.get('cohort', lcsHandler.cohort))

        # candidate file name stem
        # TDoc.getNameByContent(cohort=None, doctype=None, seq_ptype='regular', ext='dat', doc_basename=None, meta='')

        if kargs.get('make_color_time', False): 
            lcsMaps = {'lcsmap': lcsmap, 'lcsmapInvFreq': lcsmapInvFreq, 'lcsColorMap': lcsColorMap, 'lcsTimeMap': lcsTimeMap, }
        else: 
            lcsMaps = {'lcsmap': lcsmap, 'lcsmapInvFreq': lcsmapInvFreq}

        ctype = seqparams.normalize_ctype(kargs.get('seq_ptype', lcsHandler.ctype))  # 'diag', 'med'
        for varname, var in lcsMaps.items(): 
            fname = '%s-%s.pkl' % (varname, ctype)
            fpath = os.path.join(outputdir, fname)
            # data = locals()[varname]  # not found, why?

            print('analyzeLCS3> saving %s (size=%d) to:\n%s\n' % (varname, len(var), fpath))
            pickle.dump(var, open(fpath, "wb" ))
        return

    def load_stats(name, inputdir=None):
        # Same as initvar but load lcs map contents from .csv files
        if inputdir is None: inputdir = Pathway.getPath(cohort=kargs.get('cohort', lcsHandler.cohort))
        ctype = seqparams.normalize_ctype(kargs.get('seq_ptype', lcsHandler.ctype))  # 'diag', 'med'

        fname = '%s-%s.csv' % (name, ctype)
        fpath = os.path.join(inputdir, fname)

        if os.path.exists(fpath) and os.path.getsize(fpath) > 0: 
            df = pd.read_csv(fpath, sep='|', header=0, index_col=False, error_bad_lines=True)
            print('load_stats> found %d existing entries for %s from %s' % (df.shape[0], name, fname))
            return df 
        return DataFrame() 
    def initvarcsv(name, keys=None, value_type='list', inputdir=None, content_sep=','): 
        # Same as initvar but load lcs map contents from .csv files
        if inputdir is None: inputdir = Pathway.getPath(cohort=kargs.get('cohort', lcsHandler.cohort))
        ctype = seqparams.normalize_ctype(kargs.get('seq_ptype', lcsHandler.ctype))  # 'diag', 'med'

        fname = '%s-%s.csv' % (name, ctype)
        fpath = os.path.join(inputdir, fname)

        newvar = {}
        if keys is not None: 
            if value_type == 'list': 
                newvar = {k: [] for k in keys}
            else: 
                # assuming nested dictionary 
                newvar = {k: {} for k in keys}
        else: 
            if value_type == 'list': 
                newvar = defaultdict(list)
            else: 
                newvar = defaultdict(dict) # nested dictionary

        if os.path.exists(fpath) and os.path.getsize(fpath) > 0: 
            df = pd.read_csv(fpath, sep='|', header=0, index_col=False, error_bad_lines=True)
            print('initvarcsv> var: %s > found %d existing entries from %s' % (name, df.shape[0], fname))
            
            # parse 
            if name == 'lcsmap': 
                header = ['lcs', 'doc_ids']
                idx = []
                for idstr in df['doc_ids'].values: 
                    idx.append(idstr.split(content_sep))  # [[], [], ...]
                newvar.update(dict(zip(df['lcs'], idx)))
                return newvar 
            elif name == 'lcsmapInvFreq': 
                header = ['doc_id', 'lcs', 'freq']  # where doc_id can have duplicates 
                # for di in df['doc_id'].unique(): 
                cols = ['doc_id', ]
                for di, chunk in df.groupby(cols):  
                    newvar[di] = zip(chunk['lcs'], chunk['freq'])

            else: 
                raise NotImplementedError

        if len(newvar) > 0: 
            print('initvarcsv> example:\n%s\n' % sysutils.sample_dict(newvar, n_sample=1))
        return newvar

    def completed_set(name='lcsmapInvFreq', inputdir=None, sep='|', content_sep=','): 
        if inputdir is None: inputdir = Pathway.getPath(cohort=kargs.get('cohort', lcsHandler.cohort))
        ctype = seqparams.normalize_ctype(kargs.get('seq_ptype', lcsHandler.ctype))  # 'diag', 'med'

        query_col = 'doc_id'
        fname = '%s-%s.csv' % (name, ctype)
        fpath = os.path.join(inputdir, fname)
        if os.path.exists(fpath) and os.path.getsize(fpath) > 0:

            df = pd.read_csv(fpath, sep='|', header=0, index_col=False, error_bad_lines=True)
            assert query_col in df.columns, "%s is not in %s dataframe" % (query_col, name)
            return set(df[query_col].values)
        return set() 

    def save_stats(outputdir=None, content_sep=',', sep='|'):  # save data in dataframe format
        # Same as save_completed but lcs-related maps are saved in dataframe format to .csv files
        # only support lcsmap, lcsmapInvFreq for creating lcs feature set  ... 04.18
        if outputdir is None: outputdir = Pathway.getPath(cohort=kargs.get('cohort', lcsHandler.cohort))
        ctype = seqparams.normalize_ctype(kargs.get('seq_ptype', lcsHandler.ctype))  # 'diag', 'med'

        # 1. save lcsmap 
        varname = 'lcsmap'
        header = ['lcs', 'doc_ids']  # use content_sep: ','
        adict = {h: [] for h in header}
        for lcs, idx in lcsmap.items(): 
            val = content_sep.join(str(i) for i in idx)  # need to parse to restore
            adict['lcs'].append(lcs)
            adict['doc_ids'].append(val)

        df = DataFrame(adict, columns=header) 
        # nameMaps = {'lcsmap': lcsmap, 'lcsmapInvFreq': lcsmapInvFreq}
        # if kargs.get('make_color_time', False): 
        #     nameMaps = {'lcsmap': lcsmap, 'lcsmapInvFreq': lcsmapInvFreq, 'lcsColorMap': lcsColorMap, 'lcsTimeMap': lcsTimeMap, }
        fname = '%s-%s.csv' % (varname, ctype)
        fpath = os.path.join(outputdir, fname)

        # load existing 
        # df0 = load_stats(varname, inputdir=outputdir)
        # N0 = df0.shape[0]
        # df = pd.concat([df0, df], ignore_index=True)
        df.to_csv(fpath, sep=sep, index=False, header=True)   
        print('save_stats> saving %s (size=%d) to:\n%s\n' % (varname, df.shape[0], fpath))
        df =  None; gc.collect()

        # 2. save lcsmapInvFreq
        varname = 'lcsmapInvFreq'
        header = ['doc_id', 'lcs', 'freq']  # where doc_id can have duplicates 
        adict = {h: [] for h in header}
        for di, entries in lcsmapInvFreq.items(): 
            for entry in entries: 
                lcs, freq = entry
                adict['doc_id'].append(di)
                adict['lcs'].append(lcs)
                adict['freq'].append(freq)

        df = DataFrame(adict, columns=header) 
        fname = '%s-%s.csv' % (varname, ctype)
        fpath = os.path.join(outputdir, fname)

        # load existing 
        # df0 = load_stats(varname, inputdir=outputdir)
        # N0 = df0.shape[0]
        # df = pd.concat([df0, df], ignore_index=True)
        df.to_csv(fpath, sep=sep, index=False, header=True)  
        print('save_stats> saving %s (size=%d) to:\n%s\n' % (varname, df.shape[0], fpath)) 
        df = None; gc.collect()

        # save lcsColorMap 
        # header = ['doc_id', 'lcs', 'time_indices']  

        # save lcsTimeMap
        # header = ['doc_id', 'lcs', 'times']
        if kargs.get('make_color_time', False): 
            complexMap = {'lcsColorMap': lcsColorMap, 'lcsTimeMap': lcsTimeMap}
            for varname, var in lcsMaps.items(): 
                fname = '%s-%s.pkl' % (varname, ctype)
                fpath = os.path.join(outputdir, fname)
                # data = locals()[varname]  # not found, why?

                print('analyzeLCS3> saving %s (size=%d) to:\n%s\n' % (varname, len(var), fpath))
                pickle.dump(var, open(fpath, "wb" ))
        return
    def delete_completed(): 
        lcsMaps = ['lcsmap', 'lcsmapInvFreq', 'lcsColorMap', 'lcsTimeMap', ]
        ctype = seqparams.normalize_ctype(kargs.get('seq_ptype', lcsHandler.ctype))  # 'diag', 'med'
        for varname in lcsMaps: 
            fname = '%s-%s.pkl' % (varname, ctype)
            fpath = os.path.join(outputdir, fname)
            if os.path.exists(fpath) and os.path.getsize(fpath): 
                print('  + deleting pre-computed lcs-related maps: %s' % fpath)
                os.remove(fpath)
        return

    import seqAlgo
    from seqparams import Pathway
    from collections import defaultdict
    # from system import utils as sysutils
    # from seqConfig import lcsHandler

    nD = len(D)
    nL = nLCS = len(lcs_set)  # plus 'No_Match' LCS label
    # print('analyzeLCS3> Find matching persons whose records contain a given LCS (n=%d); do this for each LCS ...' % nL)
    
    docIds = kargs.get('document_ids', range(nD)) # document IDs for the input documents (a subset of the source)
    assert len(docIds) == nD
    print('analyzeLCS3> min docId: %d, max docId: %d' % (min(docIds), max(docIds)))

    if kargs.get('reset_', False): delete_completed()

    # {lcs: [] for lcs in lcs_set} # LCS -> document IDs  # {lcs:[] for lcs in df['lcs'].unique()}
    lcsmap = initvarcsv('lcsmap', value_type='list') # keys=lcs_set, suggested not to insert them because the set can be very large

    # {i: [] for i in docIds} # document ID -> LCSs 
    lcsmapInv = inverse(lcsmap) # keys=docIds
    
    # {i: [] for i in docIds}  # document ID -> {(LCS, freq)}   ... to subsume lcsmapInv 
    lcsmapInvFreq = initvarcsv('lcsmapInvFreq', value_type='list') # keys=docIds
    
    # {i: {} for i in docIds}, document ID -> {LCS -> LCS time indices} read for plot (xxxx0xx0x000xxxxx0 where 0 ... 0 ~> LCS)

    lcsColorMap, lcsTimeMap = {}, {}
    if kargs.get('make_color_time', False): 
        lcsColorMap =  initvar('lcsColorMap', keys=docIds, value_type='dict')

        # {i: {} for i in docIds} # document ID -> {LCS -> list of timestamps (of codes in LCS)} 
        lcsTimeMap = initvar('lcsTimeMap', keys=docIds, value_type='dict')

    matched_docIDs = set()  # [test] docuemnts with any  matched LCSs
    lcs_sep = kargs.get('lcs_sep', Pathway.lcs_sep) # should be ' '

    # completedDocIds = completed_set(name='lcsmapInvFreq')

    nD_has_labels = 0
    n_processed = n_skipped = 0
    for i, doc in enumerate(D): # foreach doc, find its LCS labels
        docId  = docIds[i]  # dodId: document ID (shouldn't be identicial to i)
        tdoc = T[i]

        if len(lcsmapInvFreq.get(docId, [])) > 0:  # [todo] keep track of processed document IDs? 
            n_skipped += 1
            matched_docIDs.add(docId) # already matached
            if n_skipped % 1000 == 0: print('... skipped %d documents' % n_skipped)
            continue  # completed

        # timestamps = T[i]
        for j, lcs in enumerate(lcs_set): # Pathway.header_global_lcs = ['lcs', 'length', 'count', 'n_uniq']
            lcs_seq = lcs.split(lcs_sep)  # Pathway.strToList(lcs)  
            if len(lcs_seq) > len(doc): continue   # LCS is longer than the doc, can't be a match 

            # if j < 10: test_lcs_match(lcs_seq, doc)
            # if seqAlgo.isSubsequence(lcs_seq, doc): # if LCS was derived from patient doc, then at least one match must exist
            matched_index_set = seqAlgo.traceSubsequence3(lcs_seq, doc)
            tMatched = True if len(matched_index_set) > 0 else False
            if tMatched:   # find matched indices
                # find corresponding timestamps 
                # lcs_tseq = lcs_time_series(lcs_seq, i, D, T) # [output] [(<code>, <time>), ...]
                lcsmap[lcs].append(docId)  # add the person index
                lcsmapInv[docId].append(lcs) # inverted index, which "genes" does an MCS/doc contain? 
                lcsmapInvFreq[docId].append((lcs, len(matched_index_set)))  # lcs + frequency

                # shows how LCS is distributed in the document
                # i ~ D, T (NOT the same as docId)
                if kargs.get('make_color_time', False):  
                    make_colored_repr(doc, tdoc, docId, lcs_seq, matched_index_set, verbose=(len(matched_docIDs)%200==0))  # <- lcsColoarMap
                    make_timeseries_repr(doc, tdoc, docId, lcs_seq, matched_index_set) # <- lcsTimeMap

                matched_docIDs.add(docId)  # does the ith document contain any (candiate) LCS/genes?  
                # [test]
                # if len(matched_docIDs) < 10: test_lcs_match(lcs_seq, doc)
        if len(lcsmapInv[docId]) > 0: nD_has_labels += 1 

        n_processed += 1
        if n_processed % 1000 == 0: 
            save_stats() # save_completed()
            
        if n_processed % 50 == 0: print '%d ' % n_processed,  
    
    # assert nD_has_labels >= 3, "Only %d documents have matching LCSs (nD=%d, nLCS=%d)" % (nD_has_labels, nD, nL)
    if kargs.get('make_color_time', False): assert len(lcsColorMap) > 0 and len(lcsTimeMap) > 0
    eval_lcs_stats() 

    # save results in dataframes 
    save_stats() # save_completed()

    # byproduct: see analyzePrediagnosticSequence(codes, **kargs)
    # header = ['target', 'time_to_first', 'latency', 'n_visits', 'sequence_to_first', 'has_precodes', 'time_to_last', 'days_elapsed', ]   # time_to_diagnosis: really is time to the FIRST diagnosis
          
    return (lcsmap, lcsmapInvFreq, lcsColorMap, lcsTimeMap)

def analyzeLCS(D, lcs_set, **kargs): 
    """
    Find the relationship between input documents and the set of candidate LCSs (used as class labels). 

    Input
    -----
    lcs_set: a list/sequence of lcs, each of which is a string (of codes delimited by space by default, i.e. lcs_sep <- ' ')

    Params 
    ------
    lcs_sep

    Output
    ------

    Related 
    -------
    1. analyzePrediagnosticSequence


    """
    def test_lcs_match(lcs_seq, doc): 
        assert isinstance(doc, list) and isinstance(lcs_seq, list), "illed-formatted input"
        print('  + lcs_seq: %s' % lcs_seq)
        print('  + doc:\n%s\n' % doc)
        return
    def eval_lcs_stats(): # [params] lcsmap, lcsmapInv, matched_docIDs
        nM = len(matched_docIDs)
        r = nM/(nD+0.0)
        print('analyzeLCS.eval_lcs_stats> number of docs found match: %d, ratio: %f' % (nM, r))

        # number of documents with multiple LCS labels? 
        n_multimatch = n_uniqmatch = n_nomatch = 0
        for i in range(nD): 
            if i in lcsmapInv: 
                if len(lcsmapInv[i]) > 1: 
                    n_multimatch += 1
                    if n_multimatch <= 10: 
                        print('  + document #%d matches multiple LCSs:\n  + %s ...\n' % (i, lcsmapInv[i][:10]))
                elif len(lcsmapInv[i]) == 1: 
                    n_uniqmatch += 1 
                else: 
                    n_nomatch +=1  # this may not exist 
        print('    + number of documents: %d | n_multilabel:%d, n_single: %d, n_nomatch: %d' % (nD, n_multimatch, n_uniqmatch, n_nomatch))

        # most popular LCSs? 
        hotLCSs = sorted([(lcs, len(dx)) for lcs, dx in lcsmap.items()], key=lambda x:x[1], reverse=True)[:10]
        for lcs, cnt in hotLCSs: 
            print('  + LCS: %s found %d matches ...' % (lcs, cnt))

        return 
    import seqAlgo
    from seqparams import Pathway

    nD = len(D)
    nL = nLCS = len(lcs_set)  # 'No_Match' is a label, not an LCS
    print('analyzeLCS> Find matching persons whose records contain a given LCS (n=%d); do this for each LCS ...' % nL)

    lcsmap = {lcs: [] for lcs in lcs_set} # LCS -> document IDs  # {lcs:[] for lcs in df['lcs'].unique()}
    lcsmapInv = {i: [] for i in range(nD)} # document ID -> LCSs 
    matched_docIDs = set()  # [test] docuemnts with any  matched LCSs
    lcs_sep = kargs.get('lcs_sep', Pathway.lcs_sep) # should be ' '

    nD_has_labels = 0
    for i, doc in enumerate(D): # foreach doc, find its LCS labels 
        for j, lcs in enumerate(lcs_set): # Pathway.header_global_lcs = ['lcs', 'length', 'count', 'n_uniq']
            if not lcsmap.has_key(lcs): lcsmap[lcs] = []

            lcs_seq = lcs.split(lcs_sep)  # Pathway.strToList(lcs)  
            if len(lcs_seq) > len(doc): continue   # LCS is longer than the doc, can't be a match 

            # if j < 10: test_lcs_match(lcs_seq, doc)

            # if seqAlgo.traceSubsequence3(lcs_seq, doc):   # find matched indices
            if seqAlgo.isSubsequence(lcs_seq, doc): # if LCS was derived from patient doc, then at least one match must exist

                # find corresponding timestamps 
                # lcs_tseq = lcs_time_series(lcs_seq, i, D, T) # [output] [(<code>, <time>), ...]
                lcsmap[lcs].append(i)  # add the person index
                lcsmapInv[i].append(lcs)
                matched_docIDs.add(i)  # global  

                # [test]
                # if len(matched_docIDs) < 10: test_lcs_match(lcs_seq, doc)
        
        if len(lcsmapInv[i]) > 0: nD_has_labels += 1 
    
    assert nD_has_labels >= 3, "Only %d documents have matching LCSs (nD=%d, nLCS=%d)" % (nD_has_labels, nD, nL)

    eval_lcs_stats() 

    # analyze LCS -> document IDs
    n_personx = []
    maxNIDs = None  # retain at most only this number of patient IDs; set to None if no limit imposed
    if maxNIDs is not None: 
        for lcs, docIDs in lcsmap.items(): 
            nd = len(docIDs)
            n_personx.append(nd)  # number of persons sharing the same LCS 

            # subsetting document IDs
            if len(lcsmap[lcs]) > maxNIDs: 
                lcsmap[lcs] = random.sample(docIDs, min(maxNIDs, nd))   

    nDR = nD_has_labels/(nD+0.0)
    print('analyzeLCS> nD_has_labels: %d (r=%f) > skipped %d persons/documents (without any matched LCS)' % (nD_has_labels, nDR, nD-nD_has_labels))

    return lcsmap # document<i> -> {lcs}  # call inverseIndex(lcsmap) to get its inverse

def sortedSequence(s, sep=' ', reverse=False): 
    """
    Input
    -----
    a string of symbols (usu. codes) delimited by 'sep'

    Output
    ------
    a string of sorted symobol

    Use
    ---
    Find the equivalence string up to permutation
    """
    slist = sorted(s.split(sep), reverse=reverse)
    return sep.join(slist)
def findEquivalenceMap():
    pass 

def make_comparator(less_than):
    # e.g. if x<-lcs1 is a subset of y<-lcs2, i.e. lcs1 contains codes that are a subset of those in lcs2
    def compare(x, y):
        if less_than(x, y):
            return -1
        elif less_than(y, x):
            return 1
        else:
            return 0
    return compare

def deriveLCS3(D, T, **kargs): # stratified LCSs
    pass

def deriveLCS2(D, **kargs):
    """
    Similar to deriveLCS but will attempt to derive as many LCSs as possible within given 
    document set D and apply filtering criteria on the dataframe containing all the found 
    LCS candidates (rather than filter them upfront)


    Todo
    ----
    1. Refactor LCS file naming utitiles to Pathway
    """
    def process_label(l):  
        # remove spaces e.g. CKD Stage 4
        return ''.join(str(e) for e in l.split())
    def verify_lcs(cid): 
        div(message='[cluster %s] Finished computing pairwise LCSs (%d non-empty pairs out of total %d)' % (cid, n_pairs, n_total_pairs))
        print("  + Found %d void pairs (at least one doc is empty)" % n_void)
        
        n_sample = 5
        print("  + example LCSs found (n_sample=%d) ......" % n_sample)
        for lcs in random.sample(lcsMap[cid], min(n_sample, len(lcsMap[cid]))): 
            print('    : %s' % str(lcs))  # a list of strings (of ' '-separated codes)
        return
    def do_slice(): # only for file ID, see secondary_id()
        if not kargs.has_key('slice_policy'): return False
        if kargs['slice_policy'].startswith(('noop', 'reg', 'compl', )):  # effective operations 
            return False 
        return True
    def lcs_file_id(): # params: {cohort, lcs_type, lcs_policy, seq_ptype, slice_policy, consolidate_lcs, length}
        # [note] lcs_policy: 'freq' vs 'length', 'diversity'
        #        suppose that we settle on 'freq' (because it's more useful) 
        #        use pairing policy for the lcs_policy parameter: {'random', 'longest_first', }
        adict = {}
        adict['cohort'] = kargs.get('cohort', 'generic')  # this is a mandatory arg in makeLCSTSet()
        adict['lcs_type'] = ltype = kargs.get('lcs_type', 'global')  # global: Common LCSs defined in global scope (as opposed to cluster or local scopes)
        adict['lcs_policy'] = kargs.get('lcs_policy', 'df') # definition of common LCSs

        # use {seq_ptype, slice_policy, length,} as secondary id
        adict['suffix'] = adict['meta'] = suffix = secondary_id()
        return adict
    def secondary_id(): # attach extra info in suffix
        ctype = kargs.get('seq_ptype', 'regular')
        
        suffix = ctype 
        if do_slice(): suffix = '%s-%s' % (suffix, kargs['slice_policy'])
        if kargs.get('consolidate_lcs', True): suffix = '%s-%s' % (suffix, 'iso') # LCSs up to permutations consider as the same? 
        # suffix = '%s-len%s' % (suffix, kargs.get('min_length', 3))

        label_id = kargs.get('label', None)
        if label_id is not None: suffix = '%s-L%s' % (suffix, label_id)
        
        meta = kargs.get('meta', None)
        if meta is not None: suffix = '%s-U%s' % meta  # U: user-provided

        # if kargs.get('simplify_code', False):  suffix = '%s-simple' % suffix
        return suffix   
    def load_lcs(): # [note] for now, only used in makeLCSTSet()
        # if kargs.get('overwrite_lcs', False): return None  # load unless overwrite is set to False explicitly
        adict = lcs_file_id()
        cohort = adict['cohort']
        ltype, lcs_select_policy = adict['lcs_type'], adict['lcs_policy']
        suffix = adict['meta'] 
        print('  + pathway params > ltype: %s, lcs_select_policy: %s, suffix/meta: %s' % (ltype, lcs_select_policy, suffix))
        return Pathway.load(cohort, scope=ltype, policy=lcs_select_policy, suffix=suffix, dir_type='pathway')
    def save_lcs(df):  
        adict = lcs_file_id()
        cohort = adict['cohort']
        ltype, lcs_select_policy = adict['lcs_type'], adict['lcs_policy']
        suffix = adict['meta']         
        # fpath = Pathway.getFullPath(cohort=cohort, scope=ltype, policy=lcs_select_policy, suffix=suffix) # [params] dir_type='pathway' 
        print('  + saving LCS labels ...')
        Pathway.save(df, cohort=cohort, scope=ltype, policy=lcs_select_policy, suffix=suffix, dir_type='pathway')
        return
    def filter_lcs_by_docfreq(topn=20, min_df=50): # params: lcsmap, minNDoc
        # preserve the lcs that is present in at least a minimum number of documents
        candidates = []
        for lcs, dx in lcsmap.items():  # lcsmap is the result of comparing to the entire document set
            nd = len(dx)
            if nd >= min_df: 
                candidates.append(lcs)
        nl = len(candidates)
        print('filter_lcs_by_docfreq> Found %d LCSs present in >= %d documents' % (nl, min_df))

        lcsmap2 = {}
        if nl > topn: 
            print('  + sample a subset of LCSs (target=%d)' % topn)

            ### policy #1 select those that occur most often document wide
            ranked = sorted([(c, len(lcsmap[c])) for c in candidates], key=lambda x: x[1], reverse=True)
            maxLCS, maxRankScore = ranked[0][0], ranked[0][1]
            print("  + maximum DF: %d, LCS=\n%s\n" % (maxRankScore, maxLCS))

            ### policy #2 random subsampling
            # candidates = random.sample(candidates, topn)
            
            candidates = [lcs for lcs, cnt in ranked[:topn]]
            # [code] try to break the tie
            
            for lcs in candidates: 
                lcsmap2[lcs] = lcsmap[lcs]  # lcs -> {doc}

        else: 
            if nl > 0: # at least some LCSs exist with DF > min_df
                print('  + warning: candidates (n=%d) less than target (topn=%d) => use all LCSs' % (nl, topn))
                for lcs in candidates: 
                    lcsmap2[lcs] = lcsmap[lcs]  # lcs -> {doc}
            else: 
                print('  + Warning: No LCS found with DFreq >= %d' % min_df)
                # lcsmap2 empty
        
        print('  + concluded with n:%d <=? topn:%d' % (len(lcsmap2), topn))
        return lcsmap2
    def filter_by_anchor(): # [params] lcsmap: lcs -> docIDs
        # eligible LCSs that contain the target diag code (works for cases where diag codes are accurate)
        # e.g. 309.81 Posttraumatic stress disorder (PTSD)
        # cohort dependent 
        cohort_name = kargs.get('cohort', '?')
        print('filter_by_anchor> cohort=%s' % cohort_name)
        N0 = len(lcsmap)
        lcsmap2 = {}
        target = kargs.get('anchor', '309.81') 
        if cohort_name == 'PTSD':      
            # target = kargs.get('anchor', '309.81') #
            for lcs, dx in lcsmap.items():
                if target in lcs: 
                    lcsmap2[lcs] = dx 
            # N = len(lcsmap2)
        else: 
            lcsmap2 = lcsmap
            print('filter_by_anchor> cohort=%s > noop!' % cohort_name)
            # noop 

        N = len(lcsmap2)
        print('filter_by_anchor> LCS ~ target=%s > N: %d -> %d' % (target, N0, N))

        return lcsmap2
    def filter_lcs_by_uniq(topn=20, min_df=50): # params: lcsmap
        # "try" to preserve LCSs with different code set 
        # see constraints 1 - 3

        ### constraint #1: DF
        candidates = []
        for lcs, dx in lcsmap.items():  # lcsmap is the result of comparing to the entire document set
            nd = len(dx)
            if nd >= min_df: 
                candidates.append(lcs)
        nl = len(candidates)
        print('filter_lcs_by_uniq> Found %d LCSs present in >= %d documents (label=%s)' % \
            (nl, min_df, kargs.get('label', 'n/a'))) # [log] 6494 LCSs present in >= 50 documents

        if nl > topn: 
            print('  + sample a subset of LCSs by diversity (target=%d)' % topn)
            
            ### constraint #2: Diversity
            # candidates2 = apply_diversity_constraint(candidates) # but this has a flaw 
            # nl2 = len(candidates2)

            ### constraint Diversity 2a: No LCS should contain codes that are a subset of another LCS 
            #       When two LCSs are similar (one LCS contains codes that are a subset of those of the other LCS),  
            #       keep the more informative LCS (longer one) as the candidate LCS label. 
            lcsToSubsumed = subsume_subset(candidates)  # lcs to other lcs<i> s.t. lcs > lcs<i>

            # [log] 7138 with sufficient DF down to 2927 diversified LCSs (objective topn: 9)
            candidates2 = lcsToSubsumed.keys() # these LCSs are definitely not a subset of one another
            nl2 = len(candidates2)
            print('  + %d with sufficient DF down to %d diversified LCSs (objective topn: %d)' % (nl, nl2, topn))

            if nl2 < topn:  # need to pad additional (topn-nl2)
                # [policy] if not enough LCSs selected, pick random LCS from original candidate set
                # a. pick those with greatest diff in freq distr 
                # b. pick those with greatest edit distance
                
                undiversified = set(candidates)-set(candidates2)
                candidates3 = random.sample(undiversified, min(topn-nl2, len(undiversified)))
                print('  + padding additoinal %d (=? objective: %d) LCSs that do not satisfy diversity requirement ...' % \
                    (len(candidates3), topn-nl2))
                candidates = candidates3 + candidates2 
            else: # surplus, need to select a subset
                # [policy]
                # always choose the longest? OR random? 
                
                # candidates = random.sample(set(candidates2), topn)
                ### constraint #3: Length
                lcsLengths = [(s, len(s.split(lcs_sep))) for s in candidates2]  # lcsmap is the result of comparing to the entire document set
                lcsLengths = sorted(lcsLengths, key=lambda x: x[1], reverse=True)[:topn]
                print('  + select %d longest LCSs among %d (diversified) candidates with ranks:\n%s\n' % (topn, nl2, lcsLengths))
                candidates = [lcs for lcs, _ in lcsLengths]
            
            assert len(candidates) == topn, "  + size of LCS set (n=%d) <> topn=%d!" % (len(candidates), topn)
        else:  # n(LCS) < topn
            if nl > 0:  # at some LCSs exist with DF >= min_df
                print('  + warning: candidates (n=%d) less than target (topn=%d) => use all LCSs' % (nl, topn))
            else: 
                print('  + Warning: No LCS found with DFreq >= %d' % min_df)
                # lcsmap2 empty; candidates is an empty set
                
        lcsmap2 = {}
        for lcs in candidates: 
            lcsmap2[lcs] = lcsmap[lcs]  # lcs -> {doc}
        
        print('  + concluded with n:%d <=? topn:%d' % (len(lcsmap2), topn))     
        return lcsmap2
    def filter_by_policy(lcsmap, topn=20, min_df=50): # params: lcsmap
        # use only filters above
        N0 = len(lcsmap)  # direct access of lcsmap won't work with nested calls below
        policy = kargs.get('lcs_policy', 'uniqness')
        if policy.startswith('u'):  # uniqnuess
            lcsmap = filter_lcs_by_uniq(topn=topn, min_df=min_df) # select top n LCSs, all of which occurr in at least m documents 
        elif policy.startswith('d'):  # df, document frequency
            lcsmap = filter_lcs_by_docfreq(topn=topn, min_df=min_df)
        elif policy == 'noop': 
            pass
        else: 
            raise NotImplementedError, "unrecognized LCS filter strategy: %s" % policy
        print("filter_by_policy> lcsmap size %d -> %d | policy=%s, topn=%d, min_df=%d" % (N0, len(lcsmap), policy, topn, min_df))
        return lcsmap
    def analyze(): 
        # [note] design 
        T = kargs.get('T', [])
        ret = {}
        lcs_set = df['lcs'].values
        if len(T) == 0: 
            lcsmap = analyzeLCS(D, lcs_set)
            ret['index'] = lcsmap
        else:  # T is available
            ret = analyzeMCS(D, T, lcs_set)
            # lcsmap, icsmapInv = ret['index'], ret['inverted_index']
            # lcsColorMap, lcsTimeMap = ret['color'], ret['time']
        return ret

    import motif as mf
    from seqparams import Pathway
    import itertools
    # import vector

    # LCS paramters
    topNLCS = kargs.get('topn_lcs', 50000) # set to a high number
    minLength, maxLength = 5, 15
    minDocLength = 10  # minimum document lengths (from which to compute pairwise LCSs)
    maxNPairs = 300000 # previously, 250000
    removeDups = kargs.get('remove_duplicates', True) # remove duplicate codes with consecutive occurrences (and preserve only the first one) 
    # overwrite = kargs.get('overwrite_', False)

    # file ID 
    seq_ptype = kargs.get('seq_ptype', 'regular')
    slice_policy = kargs.get('slice_policy', 'noop')  # slice operations {'noop', 'prior', 'posterior', } 
    cohort_name = kargs.get('cohort', 'CKD')

    tLoadLCS = kargs.get('load_lcs', False)
    df = load_lcs() if tLoadLCS else None  # try loading first | params: overwrite_lcs, 
    tNewLCS = False
    if df is None:  # df: ['length', 'lcs', 'n_uniq', 'count', 'df', ]
        ### generate candidate LCSs from scratch
        #[note] see t_lcs for the stratified version (i.e. search candidate LCSs within each class partition separately)
        print('docToLCS> computing new set of LCS candidates (n=%d) ... ' % topNLCS)
        minLCSDocFreq = 5     # min(len(D)/10, 10)
        df = deriveLCS(D=D, # label=process_label(label) 
                        topn_lcs=topNLCS,  # max(topNLCS, 100000),   # set a large number to save as many LCS candiates as possible
                        min_length=kargs.get('min_length', minLength), max_length=kargs.get('max_length', maxLength), 
                        min_ndocs=kargs.get('min_ndocs', minLCSDocFreq),  # depends on the size of D
                         
                        # min_ncodes=kargs.get('min_ncodes', minDocLength),  # minimum document length

                        max_n_pairs=kargs.get('max_n_pairs', maxNPairs),  
                        remove_duplicates=removeDups,
                        pairing_policy=kargs.get('pairing_policy', 'random'),  # {'longest_first', 'random', }
                        consolidate_lcs=kargs.get('consolidate_lcs', True),  # ordering important? used for LCS (surrogate) labels 
                        lcs_policy=kargs.get('lcs_policy', 'df'), # lcs filtering policy

                        # update policy: compute a brand new set or add to the existing? 
                        incremental_update=True, 

                        ### file ID paramters
                        seq_ptype=seq_ptype,
                        slice_policy=slice_policy,  # same as the param in stratifyDocuments; only for file ID purpose
                        cohort=cohort_name) # used for file ID and op such as filter_by_anchor()
        tNewLCS = True 
        print('... LCS analysis completed ...')
    
    ### given df, compute lcsmap
    assert df.shape[0] > 0

    # return only topN
    sort_keys = ['length', 'count'] # ['df', 'n_uniq']
    df = df.sort_values(sort_keys, ascending=False).head(topNLCS)

    return df

def deriveNGram(D, **kargs):
    import seqCluster as sc
    # topn=100, min_length=1, max_length=8, partial_order=False


    return sc.deriveNgram(D, **kargs)

def deriveLCS(D, **kargs):
    """
    Given a set of documents (coding sequences), find their LCSs and rank them 
    according to frequencies (where frequency refers to the number of documents having the LCS). 

    Params
    ------
    D: corpus, list of lists of strings/tokens
    document_clusters

    max_n_pairs 
    remove_duplicates
    min_length
    max_length
    topn_lcs: preserve only this many LCSs (ranked by frequency)

    consolidate_lcs: if True, LCSs up to permutations are considered as identicial 

    policy
    label 

    Output
    ------
    Pathway.save(df, cohort=cohort, scope=ltype, policy=lcs_select_policy, suffix=suffix, dir_type='pathway')

    use meta to add user-defined file ID

    Use
    ---
    1. load document first 
        D, T, L = processDocuments(cohort=cohort_name, seq_ptype=seq_ptype, 
            predicate=kargs.get('predicate', None),
            simplify_code=kargs.get('simplify_code', False), 
            ifiles=kargs.get('ifiles', []))

    2. find top N LCS by global frequency
       train classifier by treating these LCS either as binary labels (e.g. is_lcs_0 or not)
       or by treating them as multiple class labels 
       so for instance, if topn_lcs = 20, there are effectively 20+1=21 labels {lcs1, lcs2, ..., lcs20, none_of_the_above}
    """
    def filter_by_length(lcs_counts=None): # lcsMinLength, lcsMaxLength, lcsFMapGlobal (could use lcsFMap)
        if lcs_counts is None: lcs_counts = lcsFMapGlobal # or use local/cluster-level map: lcsFMap

        sep = lcs_sep # ' '
        if lcsMinLength is not None and lcsMaxLength is not None: # screen shorter LCS (not as useful for observing disease progression)
            ngr_cnts = []  # adpated from LCS analysis within the space of common n-grams (seqmaker.pathwayAnalyzer)
            for s in lcs_counts.keys(): 
                ntok = len(s.split(sep))  # number of tokens
                if ntok >= lcsMinLength and ntok <= lcsMaxLength: 
                    ngr_cnts.append((s, lcs_counts[s]))  # number of times the LCS (s) was matched
        else: 
            ngr_cnts = [(s, cnt) for s, cnt in lcs_counts.items()]
        return ngr_cnts
    def filter_by_length2(): # [params] lcsmapEst
        # screen shorter LCSs (which are not as useful for observing disease progression)
        hasLenConstraint = True if (lcsMinLength is not None) and (lcsMaxLength is not None) else False 

        # choose 
        if hasLenConstraint: 
            for s, docIds in lcsmapEst.items(): 
                ntok = len(s.split(lcs_sep))  # number of tokens           
                if ntok >= lcsMinLength and ntok <= lcsMaxLength: 
                    ngr_cnts.append((s, len(docIds)))  # number of documents the LCS (s) appear in the document set
        else: 
            ngr_cnts = [(s, len(docIds)) for s, docIds in lcsmapEst.items()]
        return ngr_cnts
    def select_lcs_by_length(): # [params] lcsMap
        # get all eligible LCSs that satisfy the length constraint 
        hasLenConstraint = True if (lcsMinLength is not None) and (lcsMaxLength is not None) else False 
        S = set()
        if hasLenConstraint:
            for cid, lcs_set in lcsMap.items(): # lcsMap: cluster ID -> LCSs set but here, we focus on only 'one cluster'
                for s in lcs_set: 
                    ntok = len(s.split(lcs_sep))  # number of tokens 
                    if ntok >= lcsMinLength and ntok <= lcsMaxLength: 
                        S.add(s)
        else: 
            for cid, lcs_set in lcsMap.items(): 
                S.update(lcs_set)
        return S 
    def stratify_by_length():
        assert lcsMinLength is not None and lcsMaxLength is not None
        
        ldict = {l: {} for l in range(lcsMinLength, lcsMinLength+1)}  
        for s, docIds in lcsmapEst.items(): 
            ntok = len(s.split(lcs_sep))  # number of tokens           
            if ntok >= lcsMinLength and ntok <= lcsMaxLength: 
                ldict[ntok][s] = docIds
        return ldict
    def peek_sep(): 
        # peek data type 
        doc = DCluster[random.sample(cIDs, 1)[0]]
        if isinstance(random.sample(doc, 1)[0], str): 
            print('test> find the doc data type and its delimiter ...')
            
            # infer seperator? 
            sep_candidates = [',', ' ']
            found_sep = False
            the_sep = ' '
            for cid, documents in DCluster.items(): 
                assert len(documents) > 0
                longest_doc = sorted([(i, len(doc)) for i, doc in enumerate(documents)], key=lambda x:x[1], reverse=True)[0][1]

                # try splitting it
                for tok in sep_candidates: 
                    # select hte longest one 
                    if longest_doc.find(tok) > 0:  # e.g. '300.01 317 311' and tok = ' '
                        if len(longest_doc.split(tok)) > 1: 
                            the_sep = tok
                            found_sep = True
                            break 
                if found_sep: break 
            print("status> determined cluster_ngram seperator to be '%s' =?= '%s' (default) ... " % (the_sep, sep))
            sep = the_sep
        return sep
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
    def normalize_docs(): # convert input (D) into a canonical form
        # D: either document clusters (cid -> docs) or documents (a list of list of strings/tokens)
        if isinstance(D, dict): 
            # do nothing
            raise ValueError, "use stratifyDocuments() to consider multiple clusters"
            # member = D.itervalues().next()
            # assert hasattr(member, '__iter__')
            # DCluster = D
        else: 
            # [test] D must be a list of lists of tokens
            x = random.randint(0, len(D)-1)  # nDoc
            assert len(D) > 0 and hasattr(D[x], '__iter__'), "Invalid input D[%d]:\n%s\n" % (x, D[x])
            print('  + Aggregate %d documents as one single cluster' % len(D))
            DCluster = {0: D}  # one cluster

        return DCluster
    def make_pairs(D_subset, cid=0, policy='random'): # [params] maxNPairs, policy={'longest_first', 'random'}
        # select pairs of documents from which to derive LCSs 
        n = len(D_subset)
        # pairwise lcs   
        n_total_pairs = (n*(n-1))/2.  # approx.
        pairs = [(i, j) for i in range(n) for j in range(i+1, n)]  # likely data source: condition_drug_timed_seq-PTSD-regular.csv
        if maxNPairs is None: return pairs

        # condition: maxNPairs is a number
        if policy.startswith('rand'): # randomized
            npt = len(pairs)
            pairs = random.sample(pairs, min(maxNPairs, npt))  # use lengths 
        elif policy.startswith('long'): # longest first 
            # priority list
            plist = sorted([(i, len(d)) for i, d in enumerate(D_subset)], key=lambda x: x[1], reverse=True) # descending order, longest first
            n = (math.sqrt(8 * maxNPairs)+1)/2  # number of documents needed to make max pairs
            n = int(math.ceil(n)) # overesimate a bit 
            pairs = list(itertools.combinations([i for i, _ in plist[:n]], 2))
            # pairs = itertools.combinations([i for i in plist[:n]], 2)  # return an iterator
        else: 
            raise NotImplementedError, "make_pairs> unrecognized document pairing policy: %s" % policy

        cid = normalize_cluster_id(cid)
        print('  + [cluster %s] choose %d out of %d possible document pairs' % (cid, len(pairs), n_total_pairs))
        return pairs
    def verify_lcs(cid): 
        div(message='[cluster %s] Finished computing pairwise LCSs (%d non-empty pairs out of total %d)' % (cid, n_pairs, n_total_pairs))
        print("  + Found %d void pairs (at least one doc is empty)" % n_void)
        
        n_sample = 5
        print("  + example LCSs found (n_sample=%d) ......" % n_sample)
        for lcs in random.sample(lcsMap[cid], min(n_sample, len(lcsMap[cid]))): 
            print('    : %s' % str(lcs))  # a list of strings (of ' '-separated codes)
        return
    def consolidate_permutations(lcsmap): # lcsmap: doc -> {lcs}
        """
        sequences that consist of the same codes are considered as identical. 
        """
        if not tAggLCS: 
            # noop 
            return lcsmap
        
        sep = Pathway.lcs_sep
        N0 = len(lcsmap)
        eqvmap = {}  # equivalence map: (re-ordered) lcs -> docId 
        sortedlcsmap = {}  # lcs -> (re-ordered) lcs
        for lcs, docIds in lcsmap.items(): 
            sorted_lcs = sortedSequence(lcs, sep=sep)  # Pathway.lcs_sep is typically a space
            sortedlcsmap[lcs] = sorted_lcs
            # lcs_list = sorted(lcs.split(lcs_sep))
            # lcs_str = lcs_sep.join(lcs_list)
            if not eqvmap.has_key(sorted_lcs): eqvmap[sorted_lcs] = set(docIds)
            eqvmap[sorted_lcs].update(docIds)  # document IDs with LCSs up to permutations are consolidated

        # [test]
        N = len(eqvmap)        
        for slcs in eqvmap.keys(): 
            eqvmap[slcs] = sorted(eqvmap[slcs])  # sorted() return a list, which is what we want
        print('consolidate_permutations> numeber of LCS entries: %d -> %d (smaller?)' % (N0, N))
        assert N <= N0

        # use the original lcs with the right ordering as entries
        lcsmap2 = {}
        mappedSet = set()  # keep only one entry for {lcs} that share the same document IDs 
        for lcs, docIds in lcsmap.items(): 
            # if not sortedlcsmap[lcs] in mappedSet: # the sorted/standardized lcs has been mapped (to docIDs) yet? 

            # LCSs up to permutations will reference the same document IDs (and get updated multiple times)
            lcsmap2[lcs] = eqvmap[sortedlcsmap[lcs]] # lcs -> sorted lcs -> (consolidated) docIDs (use the equivalent document IDs) 
            # mappedSet.add(sortedlcsmap[lcs]) 

        # [log] lcsmap2 (n=54631) should have the same size as eqvmap (n=53410)
        assert len(lcsmap2) >= N, "lcsmap2 (n=%d) should have the same size as eqvmap (n=%d)" % (len(lcsmap2), N)

        return lcsmap2  # condition: many entries (equivalent seq) refer to the same docIDs
    def filter_lcs_by_docfreq(topn=20, min_df=50): # params: lcsmap, minNDoc
        # preserve the lcs that is present in at least a minimum number of documents
        candidates = []
        for lcs, dx in lcsmap.items():  # lcsmap is the result of comparing to the entire document set
            nd = len(dx)
            if nd >= min_df: 
                candidates.append(lcs)
        nl = len(candidates)
        print('filter_lcs_by_docfreq> Found %d LCSs present in >= %d documents' % (nl, min_df))

        lcsmap2 = {}
        if nl > topn: 
            print('  + sample a subset of LCSs (target=%d)' % topn)

            ### policy #1 select those that occur most often document wide
            ranked = sorted([(c, len(lcsmap[c])) for c in candidates], key=lambda x: x[1], reverse=True)
            maxLCS, maxRankScore = ranked[0][0], ranked[0][1]
            print("  + maximum DF: %d, LCS=\n%s\n" % (maxRankScore, maxLCS))

            ### policy #2 random subsampling
            # candidates = random.sample(candidates, topn)
            
            candidates = [lcs for lcs, cnt in ranked[:topn]]
            # [code] try to break the tie
            
            for lcs in candidates: 
                lcsmap2[lcs] = lcsmap[lcs]  # lcs -> {doc}

        else: 
            if nl > 0: # at least some LCSs exist with DF > min_df
                print('  + warning: candidates (n=%d) less than target (topn=%d) => use all LCSs' % (nl, topn))
                for lcs in candidates: 
                    lcsmap2[lcs] = lcsmap[lcs]  # lcs -> {doc}
            else: 
                print('  + Warning: No LCS found with DFreq >= %d' % min_df)
                # lcsmap2 empty
        
        print('  + concluded with n:%d <=? topn:%d' % (len(lcsmap2), topn))
        return lcsmap2
    def filter_by_anchor(): # [params] lcsmap: lcs -> docIDs
        # eligible LCSs that contain the target diag code (works for cases where diag codes are accurate)
        # e.g. 309.81 Posttraumatic stress disorder (PTSD)
        # cohort dependent 
        cohort_name = kargs.get('cohort', '?')
        print('filter_by_anchor> cohort=%s' % cohort_name)
        N0 = len(lcsmap)
        lcsmap2 = {}
        target = kargs.get('anchor', '309.81') 
        if cohort_name == 'PTSD':      
            # target = kargs.get('anchor', '309.81') #
            for lcs, dx in lcsmap.items():
                if target in lcs: 
                    lcsmap2[lcs] = dx 
            # N = len(lcsmap2)
        else: 
            lcsmap2 = lcsmap
            print('filter_by_anchor> cohort=%s > noop!' % cohort_name)
            # noop 

        N = len(lcsmap2)
        print('filter_by_anchor> LCS ~ target=%s > N: %d -> %d' % (target, N0, N))

        return lcsmap2
    def filter_lcs_by_uniq(topn=20, min_df=50): # params: lcsmap
        # "try" to preserve LCSs with different code set 
        # see constraints 1 - 3

        ### constraint #1: DF
        candidates = []
        for lcs, dx in lcsmap.items():  # lcsmap is the result of comparing to the entire document set
            nd = len(dx)
            if nd >= min_df: 
                candidates.append(lcs)
        nl = len(candidates)
        print('filter_lcs_by_uniq> Found %d LCSs present in >= %d documents (label=%s)' % \
            (nl, min_df, kargs.get('label', 'n/a'))) # [log] 6494 LCSs present in >= 50 documents

        if nl > topn: 
            print('  + sample a subset of LCSs by diversity (target=%d)' % topn)
            
            ### constraint #2: Diversity
            # candidates2 = apply_diversity_constraint(candidates) # but this has a flaw 
            # nl2 = len(candidates2)

            ### constraint Diversity 2a: No LCS should contain codes that are a subset of another LCS 
            #       When two LCSs are similar (one LCS contains codes that are a subset of those of the other LCS),  
            #       keep the more informative LCS (longer one) as the candidate LCS label. 
            lcsToSubsumed = subsume_subset(candidates)  # lcs to other lcs<i> s.t. lcs > lcs<i>

            # [log] 7138 with sufficient DF down to 2927 diversified LCSs (objective topn: 9)
            candidates2 = lcsToSubsumed.keys() # these LCSs are definitely not a subset of one another
            nl2 = len(candidates2)
            print('  + %d with sufficient DF down to %d diversified LCSs (objective topn: %d)' % (nl, nl2, topn))

            if nl2 < topn:  # need to pad additional (topn-nl2)
                # [policy] if not enough LCSs selected, pick random LCS from original candidate set
                # a. pick those with greatest diff in freq distr 
                # b. pick those with greatest edit distance
                
                undiversified = set(candidates)-set(candidates2)
                candidates3 = random.sample(undiversified, min(topn-nl2, len(undiversified)))
                print('  + padding additoinal %d (=? objective: %d) LCSs that do not satisfy diversity requirement ...' % \
                    (len(candidates3), topn-nl2))
                candidates = candidates3 + candidates2 
            else: # surplus, need to select a subset
                # [policy]
                # always choose the longest? OR random? 
                
                # candidates = random.sample(set(candidates2), topn)
                ### constraint #3: Length
                lcsLengths = [(s, len(s.split(lcs_sep))) for s in candidates2]  # lcsmap is the result of comparing to the entire document set
                lcsLengths = sorted(lcsLengths, key=lambda x: x[1], reverse=True)[:topn]
                print('  + select %d longest LCSs among %d (diversified) candidates with ranks:\n%s\n' % (topn, nl2, lcsLengths))
                candidates = [lcs for lcs, _ in lcsLengths]
            
            assert len(candidates) == topn, "  + size of LCS set (n=%d) <> topn=%d!" % (len(candidates), topn)
        else:  # n(LCS) < topn
            if nl > 0:  # at some LCSs exist with DF >= min_df
                print('  + warning: candidates (n=%d) less than target (topn=%d) => use all LCSs' % (nl, topn))
            else: 
                print('  + Warning: No LCS found with DFreq >= %d' % min_df)
                # lcsmap2 empty; candidates is an empty set
                
        lcsmap2 = {}
        for lcs in candidates: 
            lcsmap2[lcs] = lcsmap[lcs]  # lcs -> {doc}
        
        print('  + concluded with n:%d <=? topn:%d' % (len(lcsmap2), topn))     
        return lcsmap2
    def filter_by_policy(lcsmap, topn=20, min_df=50): # params: lcsmap
        # use only filters above
        N0 = len(lcsmap)  # direct access of lcsmap won't work with nested calls below
        policy = kargs.get('lcs_policy', 'uniqness')
        if policy.startswith('u'):  # uniqnuess
            lcsmap = filter_lcs_by_uniq(topn=topn, min_df=min_df) # select top n LCSs, all of which occurr in at least m documents 
        elif policy.startswith('d'):  # df, document frequency
            lcsmap = filter_lcs_by_docfreq(topn=topn, min_df=min_df)
        elif policy == 'noop': 
            pass
        else: 
            raise NotImplementedError, "unrecognized LCS filter strategy: %s" % policy
        print("filter_by_policy> lcsmap size %d -> %d | policy=%s, topn=%d, min_df=%d" % (N0, len(lcsmap), policy, topn, min_df))
        return lcsmap
    def diversity_by_set(lcs): # params: lcs_sep 
        # output: a lcs string, sorted, repeated codes removed
        return lcs_sep.join( str(e) for e in sorted(set(lcs.split(lcs_sep))) )
    def apply_diversity_constraint(candidates):
        selected = set()
        for lcs in candidates: 
            s = diversity_by_set(lcs) # set uniquess; sequence of uniq codes in a string
            if not s in selected: 
                # lcsmap2[lcs] = lcsmap[lcs] # lcs -> {doc}
                candidates.append(lcs)  # caveat: subset lcs can still be an entry 
        return candidates
    def is_subset(lcs1, lcs2): # lcs1 < lcs2? 
        lcss1 = set(lcs1.split(lcs_sep))
        lcss2 = set(lcs2.split(lcs_sep))   
        return lcss1 < lcss2  # lcss1 is a subset of lcss2?  
    def subsume_subset0(alist):  # assuming alist is ordered by x >= y >= z in the sense of subset relations
        # e.g. ['x y x z', 'x d', 'x y a b z', 'y x z', 'x y', 'x z', 'z']
        #      'x y a b z' subsumes S: {'y x z', 'x y', 'x z', 'z'} because 'x y a b z' > e in S
        nl = len(alist)
        alist = sorted(alist, cmp=make_comparator(is_subset), reverse=True) # '>' relation; least subset to most subset
        ulist = []  # uniq lcs
        i = 0
        while i < nl: 
            lcs = alist[i]
            ulist.append(lcs)
            if i == nl-1: break 
            for j, lcs2 in enumerate(alist[i+1:]):  # loop until lcs > lcs2 is untrue
                # if lcs<i> is not a subset of anyone (! i < {j}), then it's definitely a candidate
                # if i < j, then j subsumes i to be the candidate
                inext = j+(i+1)
                if is_subset(lcs2, lcs): # lcs subsumes lcs2 
                    # continue
                    pass 
                else: 
                    # ulist.append(lcs)
                    break  # cannot subsume anynore 
            i = inext  # next search point (which is at least i+1)   
        return ulist
    def subsume_subset(alist):
        nl = len(alist)
        alist = sorted(alist, cmp=make_comparator(is_subset), reverse=True) # '>' relation; least subset to most subset
        
        adict = {}
        selected = set() # has the lcs been subsumed by any other lcs? 
        for i, lcs in enumerate(alist):
            if lcs in selected: continue
            adict[lcs] = []
            for j, lcs2 in enumerate(alist[i+1:]): 
                if is_subset(lcs2, lcs): # lcs subsumes lcs2 
                    # if not adict.has_key(lcs): adict[lcs] = []
                    adict[lcs].append(lcs2); selected.add(lcs2)
        return adict    
    def is_subset_to_any(s, aset, n=100): 
        # does the LCS (s) contains codes that are a subset of any of the LCSs in aset? 
        me = set(s.split(lcs_sep))
        aset0 = random.sample(aset, min(len(aset), n))
        for s0 in aset0: 
            you = set(s0.split(lcs_sep))
            if me < you: 
                return True
        return False
    def test_lcs_consistency(): # lcsmapEst vs lcsmap, lcsmapEst should have more counts
        print('... test_lcs_consistency ...')
        print('  + total LCS of all lengths: %d, filtered: %d' % (len(lcsmapEst), len(lcsmap)))
        n, nT = 0, 100
        for s, dx in lcsmapEst.items(): 
            if s in lcsmap: # filtered by min and max lengths, and matched against all docs by analyzeLCS
               dx2 = lcsmap[s]
               assert len(dx2) >= len(dx), \
                   "matching with full document set should get more count but lcsmap[s]=%d < lcsmapEst[s]=%d" % (len(dx2), len(dx))
               print('  + random match: %s -> %s (%d)' % (s, dx, len(dx)))
               print('  + full match:   %s -> %s (%d)' % (s, dx2, len(dx2)))
               n += 1 
            if n >= nT: break
        return  
    def test_min_docs(): # lcsmap (after filter_lcs operation), minNDoc
        # nL0, nL = len(lcsmapEst), len(lcs_candidates)  # must follow select_lcs_by_length
        print("deriveLCS> nLCS: %d (initial) -> %d (length qualified) -> %d (consolidated? %s) -> %d (topn)" % \
            (nL0, nL, nLc, tAggLCS, len(lcsmap))) # [log] 89435 -> 51216 -> 20

        for lcs, dx in lcsmap.items():
            assert len(dx) >= minNDoc, "df is only at %d for LCS: %s" % (len(dx), lcs)
        return 
    def show_params(cid='n/a'): 
        cid = normalize_cluster_id(cid)
        print('  + [cluster=%s] topn_lcs: %d, max_n_pairs: %d, remove_duplicates? %s, pairing_policy=%s' % \
            (cid, topn_lcs, maxNPairs, removeDups, pairing_policy))
        print('  +              min_length: %d, max_length: %d' % (lcsMinLength, lcsMaxLength))
        print('  +              min doc freqency: %d' % minDocFreq)
        print('  +              min doc length: %s' % minDocLength if minDocLength is not None else '?')
        print('  +              slice policy: %s' % kargs.get('slice_policy', 'noop'))
        print('  +              lcs filter policy: %s' % kargs.get('lcs_filter_policy', 'uniq'))
        print('  +              consolidate similar LCS? %s' % tAggLCS)
        return
    def normalize_cluster_id(cid): 
        label_id = kargs.get('label', 'n/a')
        if label_id != 'n/a': cid = process_label(label_id)
        return cid
    def process_label(l): 
        # remove spaces e.g. CKD Stage 4
        return ''.join(str(e) for e in l.split())
    def do_slice(): # only for file ID, see secondary_id()
        if not kargs.has_key('slice_policy'): return False
        if kargs['slice_policy'].startswith(('noop', 'reg', 'compl', )):  # effective operations 
            return False 
        return True
    def lcs_file_id(): # params: {cohort, scope, policy, seq_ptype, slice_policy, consolidate_lcs, length}
        # [note] lcs_policy: 'freq' vs 'length', 'diversity'
        #        suppose that we settle on 'freq' (because it's more useful) 
        #        use pairing policy for the lcs_policy parameter: {'random', 'longest_first', }
        adict = {}
        adict['cohort'] = kargs.get('cohort', 'generic')  # this is a mandatory arg in makeLCSTSet()

        # e.g. {'global', 'local'}
        adict['lcs_type'] = ltype = kargs.get('lcs_type', 'global')  # global: Common LCSs defined in global scope (as opposed to cluster or local scopes)
        adict['lcs_policy'] = kargs.get('lcs_policy', 'df') # definition of common LCSs; e.g. {'uniq', 'df', }

        # use {seq_ptype, slice_policy, length,} as secondary id
        adict['suffix'] = adict['meta'] = suffix = secondary_id()
        return adict
    def secondary_id(): # attach extra info in suffix
        ctype = kargs.get('seq_ptype', 'regular')
        suffix = ctype # kargs.get('suffix', ctype) # user-defined, vector.D2V.d2v_method
        if do_slice(): suffix = '%s-%s' % (suffix, kargs['slice_policy'])
        if kargs.get('consolidate_lcs', True): suffix = '%s-%s' % (suffix, 'iso') # LCSs up to permutations consider as the same? 
        # suffix = '%s-len%s' % (suffix, kargs.get('min_length', 3))

        label_id = kargs.get('label', None)
        if label_id is not None: suffix = '%s-L%s' % (suffix, label_id)
        # if kargs.get('simplify_code', False):  suffix = '%s-simple' % suffix
        meta = kargs.get('meta', None)
        if meta is not None: suffix = "%s-U%s" % (suffix, meta) 

        return suffix 
    def load_lcs(): # [note] for now, only used in makeLCSTSet()
        # [todo] use seqConfig.load_lcs()
        adict = lcs_file_id()
        cohort = adict['cohort']
        ltype, lcs_select_policy = adict['lcs_type'], adict['lcs_policy']
        suffix = adict['meta'] 
        return Pathway.load(cohort, scope=ltype, policy=lcs_select_policy, suffix=suffix, dir_type='pathway')
    def save_lcs(df):  
        adict = lcs_file_id()
        cohort = adict['cohort']
        ltype, lcs_select_policy = adict['lcs_type'], adict['lcs_policy']
        suffix = adict['meta']         
        # fpath = Pathway.getFullPath(cohort=cohort, scope=ltype, policy=lcs_select_policy, suffix=suffix) # [params] dir_type='pathway' 
        print('deriveLCS> saving LCS labels ...')
        Pathway.save(df, cohort=cohort, scope=ltype, policy=lcs_select_policy, suffix=suffix, dir_type='pathway')
        return
    def eff_docs(): # params: lcsmap
        # find document subset eventually included that lead to the final set of LCSs 
        deff = set()
        for lcs, dx in lcsmap.items(): 
            deff.update(dx)
        return deff
    def select_by_size(docs, min_length=None):
        if min_length is not None: 
            return [doc for doc in docs if len(doc) >= min_length]
        return docs

    import motif as mf
    from seqparams import Pathway
    import itertools
    # import vector
    import seqConfig as sq

    # userFileID = meta = kargs.get('meta', None)
    # ctype = seqparams.normalize_ctype(kargs.get('seq_ptype'))
    # sq.sysConfig(cohort=kargs.get('cohort', 'generic'), seq_ptype=ctype, 
    #     lcs_type=kargs.get('lcs_type', 'global') , lcs_policy=kargs.get('lcs_policy', 'df'), 
    #     consolidate_lcs=kargs.get('consolidate_lcs', True), 
    #     slice_policy=kargs.get('slice_policy', 'noop'), 
    #     simplify_code=kargs.get('simplify_code', False), 
    #     meta=userFileID)

    topn_lcs = kargs.get('topn_lcs', 20)   # but predominent will be shorter ones
    minNDoc = minDocFreq = kargs.get('min_ndocs', 50)  # candidate LCSs must appear in >= min_ndocs documents
    maxNPairs = kargs.get('max_n_pairs', 100000) # Running pair-wise LCS comparisons is expensive, may need to set an upperbound for maximum parings
    removeDups = kargs.get('remove_duplicates', True) # remove duplicate codes with consecutive occurrences (and preserve only the first one)
    pairing_policy = kargs.get('pairing_policy', 'random') # policy = {'longest_first', 'random', }
    minDocLength = kargs.get('min_ncodes', None)

    # only see LCS of length >= 5; set to None to prevent from filtering
    lcsMinLength, lcsMaxLength = kargs.get('min_length', 2), kargs.get('max_length', 20) 
    lcsMinCount = kargs.get('min_count', 10)  # min local frequency (used for 'diveristy' policy)
    tAggLCS = kargs.get('consolidate_lcs', True)
    # lcsSelectionPolicy  # options: frequency (f), global frequency (g), longest (l)
    show_params()    

    nDoc = 0
    DCluster = normalize_docs() # D can be a cluster or a list of documents
    assert len(DCluster) == 1, "deriveLCS> use stratifyDocuments() to consider multiple clusters"

    lcsPersonMap = {}   # LCS -> person ids  # can be very large
    lcsFMapGlobal = {}  # LCS -> count (global across all clusters)
    lcsFMap = {cid:{} for cid in DCluster.keys()}  # LCS frequency map: cid -> lcs -> count
    lcsMap = {} # cid -> canidate LCS
    lcs_sep = Pathway.lcs_sep # ' '  # lcs_sep = peek_sep()

    lcsmapEst = {}  # LCS -> document IDs
    label_id = kargs.get('label', None)  # used for label data which subsumes cid in deriveLCS where only 1 cluster is considered
    for cid, documents in DCluster.items(): # unless specified otherwise, assume that all documents (D) are one big cluster
        assert cid < 1 
        # cid = normalize_cluster_id(cid) # use label if available as cid

        lcsx = set()  # per-cluster LCS set

        # filter document lengths 
        if minDocLength is not None: 
            n_prior = len(documents)
            documents = select_by_size(documents, min_length=minDocLength); n_posterior = len(documents)
            print('deriveLCS> delta(document set | min_ncodes=%d): %d -> %d' % (minDocLength, n_prior, n_posterior))
        
        nDoc += len(documents)

        # pairwise lcs  # [todo] 
        pairs = make_pairs(D_subset=documents, cid=cid, policy=pairing_policy) # [params] policy='longest_first'
        n_total_pairs = len(pairs)
        show_params(cid=cid)

        n_pairs = n_void = 0
        for pair in pairs:  # likely data source: condition_drug_timed_seq-PTSD-regular.csv
            i, j = pair

            doc1, doc2 = documents[i], documents[j]

            if len(doc1) == 0 or len(doc2) == 0: continue 
            # assert isinstance(doc1, list) and isinstance(doc2, list), "doc1=%s, doc2=%s" % (doc1, doc2)

            # [expensive]
            sl = mf.lcs(doc1, doc2)  # sl is a list of codes since doc1: list, doc2: list
            if removeDups: sl = [e[0] for e in itertools.groupby(sl)]

            # convert list of codes to a string for indexing
            s = lcs_sep.join(sl)  

            # count local frequencies
            if s: # don't add emtpy strings
                lcsx.add(s) 
                
                # cannot do this if there were > 1 clusters
                if not lcsmapEst.has_key(s): lcsmapEst[s] = set()

                # count local/cluster frequencies
                if not s in lcsFMap[cid]: lcsFMap[cid][s] = 0
                lcsFMap[cid][s] += 1 

                # count global frequencies
                if not s in lcsFMapGlobal: lcsFMapGlobal[s] = 0
                lcsFMapGlobal[s] += 1

                # if not s in lcsPersonMap: lcsPersonMap[s] = set()
                # lcsPersonMap[s].update([i, j])

                n_pairs += 1   # effective pairs excluding empty ones
                r = n_pairs/(n_total_pairs+0.0)
                
                # r_percent = int(math.ceil(r*100))
                # percentages = {interval: 0 for interval in range(0, 100, 10)}
                if n_pairs % 500 == 0: 
                    print('  + [cluster %s] finished computing (%d out of %d ~ %f%%) pairwise LCS ...' % \
                        (cid, n_pairs, n_total_pairs, r*100))
            else: 
                n_void += 1  # at least one empty 
        
        lcsMap[cid] = lcsx
        verify_lcs(cid)  # [test] 
    ### end foreach (cid, doc)

    # filter LCSs  (length -> (consolidate permutations) -> )
    # [note] may need to just focus on certain lengths
    #        high frequency LCSs tend to be short
    # n_pairs0 = len(lcsFMapGlobal)  # or use lcsFMap: cluster level map
    # lcsCounts = filter_by_length(lcs_counts=lcsFMapGlobal, sep=lcs_sep) # [params] lcsFMapGlobal

    ### filter LCS candidates by their lengths (e.g. focus on L=5 ~ 10)
    lcs_candidates = select_lcs_by_length() # lcsMap, lcsMinLength, lcsMaxLength
    nL0, nL = len(lcsmapEst), len(lcs_candidates)

    # lcsmap: lcs -> docId  ... given an LCS, find which documents contain it? 
    lcsmap = analyzeLCS(D, lcs_set=lcs_candidates)  # lcs_sep <- Pathway.lcs_sep
    test_lcs_consistency()
    
    ### consolidate LCSs of exactly the same set of codes (but different ordering)
    # no-op if consolidate_lcs is set to False
    # [note] if an LCS:x ~ gene, then permutation(x) ~ allele of the same gene (x)
    lcsmap = consolidate_permutations(lcsmap) # lcsmap
    nLc = len(lcsmap)

    # lcsmap: lcs => documents (containing the lcs)

    ### now, filter the LCSs
    # lcsCounts = filter_by_length2() # lcsmapEst
    # lcsmap = filter_lcs_by_frequency(topn=topn_lcs)  # return only topn_lcs entries from lcsmap to make labels
    # lcsmap = filter_lcs_by_docfreq(topn=topn_lcs, min_df=minNDoc)

    #> baseline filter
    # lcsmap = filter_by_anchor()  # e.g. PTSD-related LCS must contain 309.81

    ### strategy-based filter e.g. document frequency, uniqueness, lengths
    # [note] it's necessary to pass on lcsmap due to nested calls within the inner function
    lcsmap = filter_by_policy(lcsmap, topn=topn_lcs, min_df=minNDoc)  # option: lcs_policy
    test_min_docs() # params: nL0, nL, nLc, lcsmap

    # compute statistics and save data
    header = ['length', 'lcs', 'n_uniq', 'count', 'df', ] # Pathway.header_global_lcs # ['length', 'lcs', 'count', 'n_uniq', 'df', ] 
    adict = {h:[] for h in header}

    nDEff = len(eff_docs()) # <- lcsmap
    print('  + effective number of documents where candidate LCSs are present: %d (out of %d => r: %f)' % (nDEff, nDoc, nDEff/(nDoc+0.0)))

    df0 = load_lcs() if kargs.get('incremental_update', True) else None # incremental_update: always load if already existed
    lcsCnt = {}
    if df0 is None: 
        df0 = DataFrame(columns=header)
    else: 
        lcsCnt = dict(zip(df0['lcs'].values, df0['count'].values))

    nNew = 0 
    for lcs, dx in lcsmap.items(): 
        ss = lcs.split(lcs_sep)
        cnt = len(dx)
            
        if not lcs in lcsCnt: 
            adict['length'].append(len(ss))
            adict['lcs'].append(lcs)
            adict['count'].append(cnt)            # number of documents containing the LCS
            adict['df'].append( round(cnt/(nDoc+0.0), 3) )    # document frequency
            adict['n_uniq'].append(len(set(ss)))  # diversity: number of unique tokens within the LCS
            nNew += 1
        else: 
            pass # consolidate? 
    print('deriveLCS> Found existing %d LCSs; adding %d new' % (df0.shape[0], nNew))
    dfnew = DataFrame(adict, columns=header)
    df = df0.append(dfnew, ignore_index=True)
    df = df.sort_values(['length', 'count'], ascending=True)
    save_lcs(df)

    return df # (df, lcsmap) # LCS -> document IDs  (this is needed to label the document set)

def sampleDocuments(cohort, **kargs): 
    """
    Sample documents. 

    Use 
    ---
    1. use stratify() to convert to a stratified format; i.e. a dictionary 
       containing the following keys {'sequence', 'timestamp', 'label'}

    """
    def sample_docs(D, L, T, n, sort_index=True, random_state=53):
        # if n > len(y): n = None # noop
        idx = cutils.sampleByLabels(L, n_samples=n, verbose=True, sort_index=sort_index, random_state=random_state)
        D = list(np.array(D)[idx])
        L = list(np.array(L)[idx])
        T = list(np.array(T)[idx])
        assert len(D) == n 
        print('  + subsampling: select only %d docs.' % len(D))
        return (D, L, T, idx)
    def get_global_cohort_dir(): # source MDS
        return seqparams.getCohortGlobalDir(cohort) # sys_config.read('DataExpRoot')/<cohort> 
        
    # import seqparams
    from tdoc import TDoc  # labeling info is also maintained in document sources of doctype = 'labeled'
    from seqConfig import lcsHandler  # system default
    import classifier.utils as cutils 

    # use the derived MCS set (used to generate d2v training data) as the source
    inputdir = kargs.get('inputdir', None)
    src_dir = get_global_cohort_dir() if inputdir is None else inputdir # orginal source: get_global_cohort_dir()
    ifiles = kargs.get('ifiles', ['condition_drug_labeled_seq-CKD.csv', ])

    assert os.path.exists(src_dir), "Invaild MCS source dir:\n%s\n" % src_dir
    ctype =  kargs.get('seq_ptype', lcsHandler.ctype)  # kargs.get('seq_ptype', 'regular')
    minDocLength = kargs.pop('min_ncodes', 10)
    tSimplified = kargs.get('simplify_code', lcsHandler.is_simplified)
    
    D, L, T = processDocuments(cohort=cohort, seq_ptype=ctype, 
                    inputdir=src_dir, 
                    ifiles=ifiles,
                    # meta=kargs.get('meta', None), 
                    
                    # document-wise filtering 
                    policy=kargs.get('doc_filter_policy', 'minimal_evidence'),  
                    min_ncodes=minDocLength,  # retain only documents with at least n codes

                    # content modification
                    # predicate=kargs.get('predicate', None), # reserved for segment_docs()
                    simplify_code=tSimplified, 

                    source_type='default', 
                    create_labeled_docs=False)  # [params] composition

    # docIds is used to keep track of which documents are selected. 
    maxNDocs = kargs.get('max_n_docs', None)
    if maxNDocs is not None: 
        nD0 = len(D)
        D, L, T, docIds = sample_docs(D, L, T, n=maxNDocs)
        nD = len(D)
    else: 
        docIds = [] # no need to keep track

    is_labeled = len(np.unique(L)) > 1
    # Dl = labelize(D, label_type='doc', class_labels=L)  # use class labels for document label prefixing
    print('  + nD: %d | cohort=%s, ctype=%s, labeled? %s, simplified? %s' % (len(D), cohort, ctype, is_labeled, tSimplified))
    return (D, L, T, docIds)

def mergeLabels(L, lmap={}): 
    import seqClassify as sc
    if lmap is None: lmap = {}
    return sc.mergeLabels(L, lmap=lmap)
def merge(ts, lmap={}):
    import seqClassify as sc
    if lmap is None: lmap = {}
    return sc.merge(ts, lmap=lmap)

def stratify(D, L, T, document_ids=[]): 
    """
    Stratify documents. Similar to stratifyDocuments() but primarily used with 
    sampleDocuments(). 

    Params
    ------
    document_ids: a subset of document IDs to consider; if None or empty, then consider all

    """
    def stratify_docs(inputdir=None, lmap=None): # params: cohort
        ### load + transfomr + (ensure that labeled_seq exists)

        # use the derived MCS set (used to generate d2v training data) as the source
        src_dir = get_global_cohort_dir() if inputdir is None else inputdir # orginal source: get_global_cohort_dir()
        assert os.path.exists(src_dir), "Invaild MCS source dir:\n%s\n" % src_dir
        ifiles = kargs.get('ifiles', ['condition_drug_labeled_seq-CKD.csv', ])
        ctype =  kargs.get('seq_ptype', 'regular')  # kargs.get('seq_ptype', 'regular')
        # cohort = kargs.get('cohort', 'CKD')  # 'PTSD'
        
        minDocLength = kargs.pop('min_ncodes', 10)
        tSimplified = kargs.get('simplify_code', lcsHandler.is_simplified)
        
        lmap = kargs.get('label_map', {}) # policy_relabel() 
        stratified = stratifyDocuments(cohort=cohort, seq_ptype=ctype,
                    predicate=kargs.get('predicate', None), 
                    simplify_code=kargs.get('simplify_code', tSimplified), 

                    # source
                    inputdir=src_dir, 
                    ifiles=kargs.get('ifiles', []), 
                    min_ncodes=minDocLength, 

                    # relabeling operation 
                    label_map=lmap,   # noop for now ... 04.18

                    # slice operations {'noop', 'prior', 'posterior', }
                    slice_policy=lcsHandler.slice_policy, 
                    slice_predicate=kargs.get('slice_predicate', None), 
                    cutpoint=kargs.get('cutpoint', None),
                    inclusive=True)

        nD = nT = 0
        for label, entry in stratified.items():
            Di = entry['sequence']
            Ti = entry['timestamp']
            # Li = entry['label']
            nD += len(Di)
            nT += len(Ti)

        # Dl = labelize(D, label_type='doc', class_labels=L)  # use class labels for document label prefixing
        print('stratify_docs> nD: %d | cohort=%s, ctype=%s, simplified? %s' % (nD, cohort, ctype, tSimplified))
        return stratified

    docIds = document_ids # input documents could be a sampled subset
    hasUserDocIds = False
    if docIds: # document IDs/positions wrt the source ('condition_drug_labeled_seq-CKD.csv')
        assert len(docIds) == len(D)
        hasUserDocIds = True
    else: 
        docIds = range(len(D))

    # stratified[label]['sequence'] -> D
    labelSet = np.unique(L)
    Ls = Series(L)
    stratified = {l:{} for l in labelSet}
    for label in labelSet: 
        idx = Ls[Ls == label].index.values 
        stratified[label]['sequence'] = list(np.array(D)[idx])
        stratified[label]['label'] = list(np.array(L)[idx])
        stratified[label]['timestamp'] = list(np.array(T)[idx])
        # stratified[label]['doc_ids'] = list(idx)

        if hasUserDocIds: 
            stratified[label]['doc_ids'] = list(np.array(docIds)[idx])
        else: 
            stratified[label]['doc_ids'] = list(idx)

    return stratified

def d2lcs(cohort, **kargs):
    return docToLCS(cohort, **kargs)
def docToLCS(cohort, **kargs): 
    def verify_doc(): # params: D, L, T
        assert len(D) > 0, "No input documents found (cohort=%s)" % cohort
        x = random.randint(0, len(D)-1)  # nDoc
        assert isinstance(D[x], list), "Invalid input D[%d]:\n%s\n" % (x, D[x])
        assert len(T) == len(D), "inconsistent number of timestamps (nT=%d while nD=%d)" % (len(T), len(D))
        assert len(L) == len(D), "inconsistent number of labels (nL=%d while nD=%d)" % (len(L), len(D))
        print('docToLCS> Found %d documents (nT=%d, nL=%d, cohort=%s)' % (len(D), len(T), len(L), cohort))
        return
    def analyze(): # <- D, T | df  
        # given derived LCSs (df), find their associations with LCSs (i.e. index: lcs->{d}, inverted_index: d->{lcs}, color, time)
        # [note] design 
        ret = {}
        lcs_set = df['lcs'].values
        if len(T) == 0:  
            lcsmap = analyzeLCS(D, lcs_set)
            ret['index'] = lcsmap
        else:  # T is available
            assert len(D) == len(T)
            ret = analyzeMCS(D, T, lcs_set)
            # lcsmap, icsmapInv = ret['index'], ret['inverted_index']
            # lcsColorMap, lcsTimeMap = ret['color'], ret['time']
        return ret
    def is_labeled_data(lx): 
        # if lx is None: return False
        nL = len(np.unique(lx))
        if nL <= 1: 
            return False 
        return True
    def sample_docs(D, L, T, n, sort_index=True, random_state=53):
        # if n > len(y): n = None # noop
        idx = cutils.sampleByLabels(L, n_samples=n, verbose=True, sort_index=sort_index, random_state=random_state)
        D = list(np.array(D)[idx])
        L = list(np.array(L)[idx])
        T = list(np.array(T)[idx])
        assert len(D) == n 
        print('  + subsampling: select only %d docs.' % len(D))
        return (D, L, T)
    def process_docs(inputdir=None): # params: cohort
        ### load + transfomr + (ensure that labeled_seq exists)

        # use the derived MCS set (used to generate d2v training data) as the source
        src_dir = get_global_cohort_dir() if inputdir is None else inputdir # orginal source: get_global_cohort_dir()
        ifiles = kargs.get('ifiles', ['condition_drug_labeled_seq-CKD.csv', ])

        assert os.path.exists(src_dir), "Invaild MCS source dir:\n%s\n" % src_dir
        ctype =  kargs.get('seq_ptype', 'regular')  # kargs.get('seq_ptype', 'regular')
        # cohort = kargs.get('cohort', 'CKD')  # 'PTSD'

        minDocLength = kargs.pop('min_ncodes', 10)
        tSimplified = kargs.get('simplify_code', False)
        D, L, T = processDocuments(cohort=cohort, seq_ptype=ctype, 
                    inputdir=src_dir, 
                    ifiles=ifiles,
                    # meta=kargs.get('meta', None), 
                    
                    # document-wise filtering 
                    policy=kargs.get('doc_filter_policy', 'minimal_evidence'),  
                    min_ncodes=minDocLength,  # retain only documents with at least n codes

                    # content modification
                    # predicate=kargs.get('predicate', None), # reserved for segment_docs()
                    simplify_code=tSimplified, 

                    source_type='default', 
                    create_labeled_docs=False)  # [params] composition

        maxNDocs = kargs.get('max_n_docs', None)
        if maxNDocs is not None: 
            nD0 = len(D)
            D, L, T = sample_docs(D, L, T, n=maxNDocs)
            nD = len(D)

        is_labeled = len(np.unique(L)) > 1
        # Dl = labelize(D, label_type='doc', class_labels=L)  # use class labels for document label prefixing
        print('  + nD: %d | cohort=%s, ctype=%s, labeled? %s, simplified? %s' % (len(D), cohort, ctype, is_labeled, tSimplified))
        return (D, L, T)
    def get_tset_dir(): # derived MDS 
        # params: cohort, dir_type

        tsetPath = TSet.getPath(cohort=kargs.get('cohort', 'CKD'), dir_type=kargs.get('dir_type', 'combined'))  # ./data/<cohort>/
        print('docToLCS> training set dir: %s' % tsetPath)
        return tsetPath 
    def get_global_cohort_dir(): # source MDS
        return seqparams.getCohortGlobalDir(kargs.get('cohort', 'CKD')) # sys_config.read('DataExpRoot')/<cohort> 

    from tset import TSet # mark training data
    from seqparams import Pathway # set pathway related parameters, where pathway refers to sequence objects such as LCS
    from tdoc import TDoc  # labeling info is also maintained in document sources of doctype = 'labeled'
    # from seqConfig import tsHandler
    import classifier.utils as cutils

    ctype = seq_ptype = kargs.get('seq_ptype', 'regular') 
    d2v_method = kargs.get('d2v_method', seqparams.D2V.d2v_method)
    verify_labeled_file = True

    # [params] LCS 
    slice_policy = kargs.get('slice_policy', 'noop')  # slice operations {'noop', 'prior', 'posterior', } 
    topNLCS = kargs.get('topn_lcs', 10000)  
    lcsMinLength, lcsMaxLength = kargs.get('min_length', 5), kargs.get('max_length', 15)

    # D: corpus, T: timestamps, L: class labels
    # load + transform + (labeled_seq file)
    D, L, T = process_docs(inputdir=None) # cohort, ctype, *inputdir (None to use default), *max_n_docs

    print('... document processing completed (load+transform+label)... ')
    verify_doc()

    df = deriveLCS2(D=D, 
                    topn_lcs=kargs.get('topn_lcs', topNLCS), 
                    min_length=lcsMinLength, max_length=lcsMaxLength, 

                    max_n_pairs=kargs.get('max_n_pairs', 250000), 
                    pairing_policy=kargs.get('pairing_policy', 'random'), 
                    remove_duplicates=True,   # remove consecutive codes from within the same visit?

                    consolidate_lcs=kargs.get('consolidate_lcs', True),  # ordering important? used for LCS (surrogate) labels 
                    lcs_policy=kargs.get('lcs_policy', 'df'), # lcs filtering policy

                    ### file ID paramters
                    seq_ptype=seq_ptype,
                    slice_policy=slice_policy,  # same as the param in stratifyDocuments; only for file ID purpose
                    cohort=cohort, 
                    meta=kargs.get('meta', None), # used for file ID and op such as filter_by_anchor()
                    load_lcs=kargs.get('load_lcs', False)) # if False, always compute new set of LCSs;  overwrite_lcs=False

    ### analyze these LCS (index, inverted_index, color, time)
    summary = {}
    if kargs.get('time_analysis', True): 
        summary = analyze() # D, T, L | df | -> analyzeMCS()
 
    ### include document sources? 
    if kargs.get('include_docs', False): 
        summary['D'], summary['T'], summary['L'] = D, T, L

    return (df, summary)

def makeLCSFeatureTSet(cohort, **kargs):
    """
    Find LCSs shared by the cohort-specific documents (i.e. D(cohort)) and use them 
    as features, each of which takes on {0, 1} i.e. present or abscent 


    Related
    -------
    1. makeLCSTSet(): LCSs are used as labels instead of features

    """ 
    def eval_feature_set(rank_type='length'): # df
        lcs_sep = Pathway.lcs_sep
        # df: ['length', 'lcs', 'n_uniq', 'count', 'df', ]
        
        ### sort order? 

        # a. Sort by multiple criteria 
        # df.sort_values(['length', 'count'], ascending=False).head(topNLCS)

        
        # b. Sort by lengths  
        fset = df['lcs'].values  # a set of LCS strings
        ranks = map(lambda lcs: len(lcs.split(lcs_sep)), fset)  # use document frequency instead
        fsetp = fset[np.argsort(ranks)] # sort according to the order suggested by fset_ (ascending order by default)

        print('  + feature set (n=%d) | minR: %d, maxR: %d < rank type: %s' % (len(fsetp), min(ranks), max(ranks), rank_type))
        print('  + example:\n%s ...\n' % fsetp[:20])
        return fsetp
    def do_slice(): 
        if not kargs.has_key('slice_policy'): return False
        if kargs['slice_policy'].startswith(('noop', 'reg', 'compl', )):  # effective operations 
            return False 
        return True
    def secondary_id(): # attach extra info in suffix: {seq_ptype, slice_policy, label_id}
        ctype = kargs.get('seq_ptype', 'regular')
        suffix = kargs.get('suffix', ctype) # vector.D2V.d2v_method
        if do_slice(): suffix = '%s-%s' % (suffix, kargs.get('slice_policy', 'noop'))
        if kargs.get('consolidate_lcs', True): suffix = '%s-%s' % (suffix, 'iso') # LCSs up to permutations consider as the same? 

        label_id = kargs.get('label', None)  # stratified dataset by labels
        if label_id is not None: suffix = '%s-L%s' % (suffix, label_id)
    
        return suffix 
    def save_tset(X, fset=None, overwrite=True): # <- nD, mark training data 
        # d2v_method = kargs.get('d2v_method', seqparams.D2V.d2v_method) # pv-dm2
        # suffix = secondary_id()

        # load + suffix <- 'Flcs'
        ts = TSet.loadLCSFeatureTSet(cohort=cohort, d2v_method=d2v_method, seq_ptype=seq_ptype) # use default 'Flcs' keyword
        if ts is not None and not ts.empty: print('  + Found pre-computed tset (dim:%s)' % str(ts.shape))
        if ts is None or overwrite:  # if the training set does not exist
            print('  + Creating new LCS-feature tset (cohort=%s, d2v=%s, ctype=%s)' % (cohort, d2v_method, seq_ptype))

            # base training set (with regular labels) 
            # ts = TSet.loadCombinedTSet(cohort=cohort, d2v_method=d2v_method, seq_ptype=seq_ptype, suffix=suffix) 

            # may need to set index to 0
            ts = TSet.loadBaseTSet(cohort=cohort, d2v_method=d2v_method, seq_ptype=seq_ptype) # do not include suffix 
            if ts is None or ts.empty: 
                # [note] directory params: cohort, dir_type <- 'combined' by default
                fpath = TSet.getFullPath(cohort, d2v_method=d2v_method, seq_ptype=seq_ptype)  # do not include suffix (secondary ID)
                raise ValueError, "Primary tset (c:%s, d2v:%s, ctype:%s) does not exist at:\n%s\n" % \
                    (cohort, d2v_method, seq_ptype, fpath) 

            Xv, y = TSet.toXY(ts)  # or use TSet.toXY2() if n_features is known 
            # assert len(L) == len(y), "Inconsistent number of labels: nL=%d, nrows=%d" % (len(L), X.shape[0])

            ### Use y but replace Xv: doc2vec representation
            TSet.saveLCSFeatureTSet(X, y=y, header=fset, cohort=cohort, d2v_method=d2v_method, seq_ptype=seq_ptype)
        else: 
            assert ts.shape[0] == nD, "Inconsistent data size: nD=%d <> nrows=%d" % (nD, ts.shape[0])
            # assert ts.shape[1] == seqparams.D2V.n_features * 2
            assert X.shape[0] == ts.shape[0], "Inconsistent number of rows: nL=%d, nrows=%d" % (len(L), ts.shape[0])

        # save a copy? 
        return ts  # or (X, L)
    def verify_lcs(i): 
        lcs_set = lcsmapInv[i]
        for lcs in lcs_set: 
            lcs_seq = lcs.split(Pathway.lcs_sep)  # default: ' '  [note] do not use Pathway.sep <- '|'
            if not seqAlgo.isSubsequence(lcs_seq, D[i]): 
                raise ValueError, "%s is not a LCS of D:\n%s\n" % (lcs, Pathway.lcs_sep.join(D[i]))
    def verify_lcs_analysis(): # df, summary, n_features
        lcsmap = summary['index']
        D = summary['D']
        print('  + size(lcsmap):%d =?= nrow:%d | D (size=%d, cohort=%s)' % (len(lcsmap), df.shape[0], len(D), cohort))

        ranked = sorted([(lcs, len(idx)) for lcs, idx in lcsmap.items()], key=lambda x: x[1], reverse=True)
        maxLCS, maxRankScore = ranked[0][0], ranked[0][1]
        minLCS, minRankScore = ranked[-1][0], ranked[-1][1]
        print("  + (topn=%d) max DF: %d\n  + LCS: %s\n" % (n_features, maxRankScore, maxLCS))
        print("    (topn=%d) min DF: %d\n  + LCS: %s\n" % (n_features, minRankScore, minLCS))
        # candidates = [lcs for lcs, cnt in ranked[:topNLCS]]
        return 

    import seqAlgo
    from tset import TSet
    from seqparams import Pathway

    # [params] LCS feature
    n_features = topNLCS = kargs.get('n_features', 10000)  # CKD has a small cohort; minimize the risk of overfitting
    seq_ptype=kargs.get('seq_ptype', 'regular') # LCS features are probably better represented via mixed type
    d2v_method = kargs.get('d2v_method', seqparams.D2V.d2v_method)

    ### get candidate LCS set
    df, summary = docToLCS(cohort, seq_ptype=seq_ptype, 
                            topn_lcs=n_features, 
                            min_length=kargs.get('min_length', 5), max_length=kargs.get('max_length', 15), 
 
                            consolidate_lcs=kargs.get('consolidate_lcs', True),  # ordering important? used for LCS (surrogate) labels 
                            lcs_policy=kargs.get('lcs_policy', 'df'), # lcs filtering policy
                            slice_policy=kargs.get('slice_policy', 'noop'), 

                            # overwrite_lcs=kargs.get('overwrite_lcs', False), # if True, the re-compute LCSs
                            load_lcs=kargs.get('load_lcs', False), # if False, the re-compute LCSs
                            include_docs=True) # include source in order to verify document property; expensive for large corpus! 
    
    verify_lcs_analysis() # <- df, summary

    D, T, L = summary['D'], summary['T'], summary['L']
    nD = len(D)
    lcsmap, lcsmapInv, lcsmapInvFreq = summary['index'], summary['inverted_index'], summary['frequency']
    nF, nD = len(lcsmap), len(lcsmapInv)
    assert len(D) == nD, "Inconsistent lengths between D(n=%d) and inverted index(n=%d)" % (len(D), nD) 
    # lcsColorMap, lcsTimeMap = ret['color'], ret['time'] 

    # order the feature set 
    fset = eval_feature_set()
    fpos = {f: i for i, f in enumerate(fset)}  # feature positions

    fvec = np.zeros((nD, nF))
    test_points = set(random.sample(range(nD), min(10, nD)))
    for i in range(nD): 
        if i in test_points: verify_lcs(i)
        positions = [fpos[lcs] for lcs, _ in lcsmapInvFreq[i]]  # active position
        values = [freq for _, freq in lcsmapInvFreq[i]] 
        fvec[i][positions] = values

    # save feature vectors
    ts = save_tset(fvec, fset=fset, overwrite=True) # assuming that "base" training set already exists (i.e. d2v training set)

    return ts 

def makeLCSFeatureTSet2(cohort, **kargs): 
    """
    Represent each document by the given LCS features using their frequencies of occurrences. 

    Assumption
    ----------
    1. feature set (global LCS candidates) have been determined. 

    Params
    ------
        feature_set
        topn

        - load documents: 
            cohort, seq_ptype, ctype, ifiles, doc_filter_policy, min_ncodes, simplify_code

        - load LCS set 
            cohort, lcs_type, lcs_policy, ctype, consolidate_lcs, label, meta

    Memo
    ----
    1. return value of analyzeMCS() is a dictionary with the following key, values

    key: index, frequency, inverted_index, color, time 
    
    Specifically, 

    index: 
        lcsmap: lcs -> document IDs 
    frequency: 
        lcsmapInvFreq: document ID -> {(LCS, freq)}
    inverted_index: 
        lcsmapInv: document ID -> {lcs_i}

    color: 
        lcsColorMap: document ID -> lcs -> {postiions_i}
        lcsColorMap[i][lcs] = lcs_positions

        # document ID -> {LCS -> LCS time indices} read for plot (xxxx0xx0x000xxxxx0 where 0 ... 0 ~> LCS)

    time: 
        lcsTimeMap: document ID -> {LCS -> list of timestamps (of codes in LCS)} 

    Output
    ------
    * Persistent 
        1. ranked feature set: LCS features are saved to a dataframe consisting of two attributes ['lcs', 'score', ]
           where score be any rank criteria (e.g. docuemnt frequency)
        2. (sparse) training data

    """
    def config(): 
        # configure all the parameters 
        userFileID = meta = kargs.get('meta', None)
        ctype = seqparams.normalize_ctype(kargs.get('seq_ptype', 'regular'))

        # this routine can only be called once, subsquent calls are noop 
        sq.sysConfig(cohort=cohort, seq_ptype=ctype, 
                        lcs_type=kargs.get('lcs_type', 'global') , lcs_policy=kargs.get('lcs_policy', 'df'), 
                        consolidate_lcs=kargs.get('consolidate_lcs', True), 
                        slice_policy=kargs.get('slice_policy', 'noop'), 
                        simplify_code=kargs.get('simplify_code', False), 
                        meta=userFileID)
        return 
    def process_docs(inputdir=None): 
        ### load + transfomr + (ensure that labeled_seq exists)
      
        # params: cohort, seq_ptype, ifiles, doc_filter_policy
        #         min_ncodes, simplify_code

        # first check if already provided externally? 
        # use case: sample a subset of documents (D, L, T) and use the result in analyzeLCSDistribution
        D, L, T = kargs.get('D', []), kargs.get('L', []), kargs.get('T', [])
        if len(D) > 0: 
            print('makeLCSFeatureTSet2> Given input documents of size: %d' % len(D))
            assert len(D) == len(T), "size(docs): %d <> size(times): %d" % (len(D), len(T))
            if len(L) == 0: L = [1] * len(D)
            return (D, L, T)

        # otherwise, load from the source
     
        # use the derived MCS set (used to generate d2v training data) as the source
        src_dir = get_global_cohort_dir() if inputdir is None else inputdir # orginal source: get_global_cohort_dir()
        ifiles = kargs.get('ifiles', ['condition_drug_labeled_seq-CKD.csv', ])

        assert os.path.exists(src_dir), "Invaild MCS source dir:\n%s\n" % src_dir
        ctype =  kargs.get('seq_ptype', 'regular')  # kargs.get('seq_ptype', 'regular')
        # cohort = kargs.get('cohort', 'CKD')  # 'PTSD'

        minDocLength = kargs.pop('min_ncodes', 10)
        tSimplified = kargs.get('simplify_code', False)
        D, L, T = processDocuments(cohort=cohort, seq_ptype=ctype, 
                    inputdir=src_dir, 
                    ifiles=ifiles,
                    # meta=kargs.get('meta', None), 
                    
                    # document-wise filtering 
                    policy=kargs.get('doc_filter_policy', 'minimal_evidence'),  
                    min_ncodes=minDocLength,  # retain only documents with at least n codes

                    # content modification
                    # predicate=kargs.get('predicate', None), # reserved for segment_docs()
                    simplify_code=tSimplified, 

                    source_type='default', 
                    create_labeled_docs=False)  # [params] composition

        # maxNDocs = kargs.get('max_n_docs', None)
        # if maxNDocs is not None: 
        #     nD0 = len(D)
        #     D, L, T = sample_docs(D, L, T, n=maxNDocs)
        #     nD = len(D)

        # Dl = labelize(D, label_type='doc', class_labels=L)  # use class labels for document label prefixing
        is_labeled = len(np.unique(L)) > 1
        print('  + nD: %d | cohort=%s, ctype=%s, labeled? %s, simplified? %s' % (len(D), cohort, ctype, is_labeled, tSimplified))
        return (D, L, T)
    def get_tset_dir(): # derived MDS 
        # params: cohort, dir_type
        tsetPath = TSet.getPath(cohort=kargs.get('cohort', lcsHandler.cohort), dir_type=kargs.get('dir_type', 'combined'))  # ./data/<cohort>/
        print('makeLCSFeatureTSet2> training set dir: %s' % tsetPath)
        return tsetPath 
    def get_global_cohort_dir(): # source MDS
        return seqparams.getCohortGlobalDir(kargs.get('cohort', lcsHandler.cohort)) # sys_config.read('DataExpRoot')/<cohort>
    def to_freq_vector(doc, vec): # <- sortedTokens
        # modify vec based on word counts in doc
        wc = collections.Counter()
        wc.update(doc)
        for i, var in enumerate(sortedTokens):
            if var in wc:   
                vec[i] = wc[var]
            else: 
                pass
        return vec
    def load_ranked_fset(topn=None): 
        # load 
        df = TSet.loadRankedLCSFSet(cohort=cohort, d2v_method=lcsHandler.d2v_method, 
                     seq_ptype=lcsHandler.ctype, suffix=lcsHandler.meta)
        if df is not None and not df.empty: 
            if topn is not None: df = df.head(topn)
        return df  # dataframe or None (if not found)
    def rank_fset(fset, analyze_mcs={}, save_=True, topn=None):  # <- D, T 
        # rank by document frequency (among other possible criteria) ... 
        # order features by documet frequency (and save it)
        if not analyze_mcs: analyze_mcs = analyzeMCS(D, T, fset) 
        ret = analyze_mcs # result of the analyzeMCS call
        # lcsToDocIDs = resource['index']
        
        docToFreq = ret['frequency']  # docID -> {(lcs, freq)}
        lcsToWeightedFreq = {}  # if an LCS occurs in the same document N times, then it's (weighted) doc freq is N (instead of just 1)
        
        # init frequency
        for var in fset: 
            lcsToWeightedFreq[var] = 0 

        # poll
        for docID, values in docToFreq.items(): 
            for lcs, freq in values: 
                if not lcsToWeightedFreq.has_key(lcs): 
                    print('order_by_df> Warning: LCS %s never appeared in (sampled) documents.' % lcs) # this should not happen
                    lcsToWeightedFreq[lcs] = 0
                lcsToWeightedFreq[lcs] += freq

        header = ['lcs', 'score']  # LCS, global term frequency
        df = DataFrame(lcsToWeightedFreq, columns=header)
        sort_keys = ['score', 'lcs'] # ['df', 'n_uniq']
        if topn is not None: 
            df = df.sort_values(sort_keys, ascending=False).head(topn)
        else: # take all LcS 
            df = df.sort_values(sort_keys, ascending=False)

        if save_: 
            # this calls saveDataFrame with special keyword padded to 'meta' (Rlcs)
            TSet.saveRankedLCSFSet(df, cohort=cohort, d2v_method=lcsHandler.d2v_method, 
                seq_ptype=lcsHandler.ctype, suffix=lcsHandler.meta) # dir_type/'combined'
       
        # sorted_ = sorted([(lcs, len(dx)) for lcs, dx in lcsToDocIDs], key=lambda x: x[1], reverse=True) # descending order
        return df
    def test_lcs_frequency(i, ret):  # <- D
        dtf = ret['frequency'] # docID -> {(lcs, freq)}
        print('[doc #%d] %s' % (i, abridge(D[i])))
        for i, (lcs, freq) in enumerate(dtf[i].items()): 
            print('  + [lcs_%d] (f=%d) %s' % ((i+1), freq, lcs))
        return
    def abridge(doc, max_len=200, sep=' '): 
        length = len(doc) 
        dp = doc
        midpoint = max_len/2
        if length > max_len: 
            # ', '.join(e for e in doc[:100])
            front = sep.join(e for e in doc[:midpoint])
            back = sep.join(e for e in doc[-midpoint:])
            dp = '[' + front + '...' + back + ']'
        return dp
    def save_tset(X, y):
        # file ID: cohort, d2v_method, seq_ptype, index, suffix
        # directory: cohort, dir_type/'combined'
        if not isinstance(y, np.ndarray): y = np.array(y)
        TSet.saveSparseLCSFeatureTSet(X, y=y, cohort=cohort, d2v_method=lcsHandler.d2v_method, 
            seq_ptype=lcsHandler.ctype, suffix=lcsHandler.meta)

        # PS1: feature set is defined according to load_rank_fset (precomputed variables) or rank_fset (new variables)

        # PS2: load method
        # X, y = TSet.loadSparse(cohort, **kargs)
        return

    import seqparams
    from tset import TSet
    import seqConfig as sq
    from seqConfig import lcsHandler
    from seqTest import TestDocs
    from scipy.sparse import csr_matrix # coo_matrix
    # use sq.sysConfig

    config()  # if sq.sysConfig() was run first, then this is a noop
    D, L, T = process_docs(inputdir=None)  # set inputdir to None to use default

    # load existing LCS features (derived from docToLCS, deriveLCS, etc)
    lcsCandidates = kargs.get('feature_set', [])  # try user input first
    if len(lcsCandidates) == 0: 
        # params: cohort, lcs_type, lcs_policy, ctype, consolidate_lcs, label, meta
        df = lcsHandler.load_lcs()  # load the global LCS training set
        assert df is not None and not df.empty, "Could not load pre-computed LCS feature set."

        # load lcs set
        lcsCandidates = df['lcs'].values
        print('makeLCSFeatureTSet2> found %d LCS canidates ...' % df.shape[0])

    ### global evaluation
    print('makeLCSFeatureTSet2> Analyze MCS with %d input docs.' % len(D))

    # the document IDs must be consistent with the initial source
    docIds = kargs.get('document_ids', []) # if provided => incremental_mode -> True

    ### find LCS candiate frequencies in given document set
    # params: set make_color_time to False to disable color map and time map (space consuming)
    ret = analyzeMCS(D, T, lcsCandidates, document_ids=docIds, reset_=False, make_color_time=False)  # initial candidates (to be filtered by ranked score)

    # use frequencies as feature values
    # lcsToDocIDs = ret['index']

    # compute ordered feature set (and its associated frequency values)
    topNFSet = kargs.get('topn', 20000)

    # policy A: rank feature by specified criteria e.g. term frequency
    df = load_ranked_fset(topn=topNFSet)
    if df is None or df.empty: 
        df = rank_fset(lcsCandidates, analyze_mcs=ret, save_=True, topn=topNFSet) 

    # lcsFSet is a filtered, ranked version of lcsCandidates
    lcsFSet = df['lcs'].values # this is feature order
    nD, nF = len(D), len(lcsFSet)

    # sparse format by default 
    row, col, data = [], [], []
    lcsToFPos = {lcs:i for i, lcs in enumerate(lcsFSet)}  # lcsToFPos: LCS to its feature position/index in lcsFSet
    docToFreq = ret['frequency']  # docID -> {(lcs, freq)}
    testIDs = random.sample(range(len(D)), 20)
    for i, doc in enumerate(D):
        if i in testIDs: test_lcs_frequency(i, ret)

        idx = [lcsToFPos[lcs] for lcs, _ in docToFreq[i]]  #  docID -> {(lcs, freq)} -> {(pos, freq)}
        values = [freq for _, freq in docToFreq[i]]

        row.extend([i] * len(idx))  # ith document
        col.extend(idx)  # jth attribute
        data.extend(values)  # count 
    Xs = csr_matrix((data, (row, col)), shape=(nD, nF))
    print('makeLCSFeatureTSet2> Found %d coordinates, %d active values, ' % (len(data), Xs.nnz))

    assert Xs.shape[0] == len(L)
    if kargs.get('save_', True): 
        save_tset(Xs, L)
    
    return (Xs, L)

def loadLCSStats(name, document_ids=[], value_type='list', inputdir=None, **kargs): 
    """
    Load pre-computed LCS maps. 
    """
    def initvarcsv(name, keys=None, value_type='list', inputdir=None, content_sep=','): 
        # Same as initvar but load lcs map contents from .csv files
        cohort = kargs.get('cohort', lcsHandler.cohort)
        if inputdir is None: inputdir = Pathway.getPath(cohort=cohort)
        ctype = seqparams.normalize_ctype(kargs.get('seq_ptype', lcsHandler.ctype))  # 'diag', 'med'

        fname = '%s-%s.csv' % (name, ctype)
        fpath = os.path.join(inputdir, fname)

        newvar = {}
        if keys is not None: # keys: e.g. document IDs
            if value_type == 'list': 
                newvar = {k: [] for k in keys}
            else: 
                # assuming nested dictionary 
                newvar = {k: {} for k in keys}
        else: 
            if value_type == 'list': 
                newvar = defaultdict(list)
            else: 
                newvar = defaultdict(dict) # nested dictionary

        if os.path.exists(fpath) and os.path.getsize(fpath) > 0: 
            df = pd.read_csv(fpath, sep='|', header=0, index_col=False, error_bad_lines=True)
            print('initvarcsv> var: %s > found %d existing entries from %s' % (name, df.shape[0], fname))
            
            # parse 
            if name == 'lcsmap': 
                header = ['lcs', 'doc_ids']
                idx = []
                for idstr in df['doc_ids'].values: 
                    idx.append(idstr.split(content_sep))  # [[], [], ...]
                if keys: idx = list(idx.intersection(keys)) # only consider these document IDs
                newvar.update(dict(zip(df['lcs'], idx)))
                return newvar 
            elif name == 'lcsmapInvFreq': 
                header = ['doc_id', 'lcs', 'freq']  # where doc_id can have duplicates 
                # for di in df['doc_id'].unique(): 
                cols = ['doc_id', ]
                if keys: 
                    for di, chunk in df.groupby(cols):  
                        if di in keys: 
                            newvar[di] = zip(chunk['lcs'], chunk['freq'])
                else: 
                    for di, chunk in df.groupby(cols):  
                        newvar[di] = zip(chunk['lcs'], chunk['freq'])

            else: 
                raise NotImplementedError

        if len(newvar) > 0: 
            print('initvarcsv> example:\n%s\n' % sysutils.sample_dict(newvar, n_sample=1))
        return newvar
    # from seqparams import Pathway
    # from system import utils as sysutils

    adict = initvarcsv(name=name, keys=document_ids, value_type=value_type, inputdir=inputdir)  # lcs -> docIDs
    return adict


def featureSelectTfidf(topn, **kargs): 
    """
    
    Related
    -------
    1. featureSelect()
    """
    return chooseLCSFeatureSet(topn, **kargs)
def chooseLCSFeatureSet(topn, **kargs): 
    """
    Chooose a subset of (precomputed) LCS candidates. 
    
    Run incremental mode for analyzeMCS() first to gather LCS statistics. 

    Prior
    -----
    1. total feature set (i.e. all possible LCS candidates) has been pre-computed.
       run docToLCS() multiple times

    Use
    ---
    makeLCSFeatureTSet2a()


    """
    def get_global_cohort_dir(): # source MDS
        return seqparams.getCohortGlobalDir(kargs.get('cohort', lcsHandler.cohort)) # sys_config.read('DataExpRoot')/<cohort>
    def process_docs(inputdir=None): 
        ### load + transfomr + (ensure that labeled_seq exists)
      
        # params: cohort, seq_ptype, ifiles, doc_filter_policy
        #         min_ncodes, simplify_code

        # first check if already provided externally? 
        # use case: sample a subset of documents (D, L, T) and use the result in analyzeLCSDistribution
        D, L, T = kargs.get('D', []), kargs.get('L', []), kargs.get('T', [])
        if len(D) > 0: 
            print('process_docs> Given input documents of size: %d' % len(D))
            assert len(D) == len(T), "size(docs): %d <> size(times): %d" % (len(D), len(T))
            if len(L) == 0: L = [1] * len(D)
            return (D, L, T)

        # otherwise, load from the source
     
        # use the derived MCS set (used to generate d2v training data) as the source
        src_dir = get_global_cohort_dir() if inputdir is None else inputdir # orginal source: get_global_cohort_dir()
        ifiles = kargs.get('ifiles', ['condition_drug_labeled_seq-CKD.csv', ])

        assert os.path.exists(src_dir), "Invaild MCS source dir:\n%s\n" % src_dir
        ctype =  kargs.get('seq_ptype', 'regular')  # kargs.get('seq_ptype', 'regular')
        cohort = kargs.get('cohort', lcsHandler.cohort)  # 'PTSD'

        minDocLength = kargs.pop('min_ncodes', 10)
        tSimplified = kargs.get('simplify_code', False)
        D, L, T = processDocuments(cohort=cohort, seq_ptype=ctype, 
                    inputdir=src_dir, 
                    ifiles=ifiles,
                    # meta=kargs.get('meta', None), 
                    
                    # document-wise filtering 
                    policy=kargs.get('doc_filter_policy', 'minimal_evidence'),  
                    min_ncodes=minDocLength,  # retain only documents with at least n codes

                    # content modification
                    # predicate=kargs.get('predicate', None), # reserved for segment_docs()
                    simplify_code=tSimplified, 

                    source_type='default', 
                    create_labeled_docs=False)  # [params] composition

        # Dl = labelize(D, label_type='doc', class_labels=L)  # use class labels for document label prefixing
        is_labeled = len(np.unique(L)) > 1
        print('  + nD: %d | cohort=%s, ctype=%s, labeled? %s, simplified? %s' % (len(D), cohort, ctype, is_labeled, tSimplified))
        return (D, L, T)
    def load_ranked_fset(n_features): 
        cohort = kargs.get('cohort', lcsHandler.cohort)  # 'PTSD'
        df = TSet.loadRankedLCSFSet(cohort=cohort, d2v_method=lcsHandler.d2v_method, 
                     seq_ptype=lcsHandler.ctype, suffix=lcsHandler.meta)
        if df is not None and not df.empty: 
            if n_features is not None: df = df.head(n_features)

        # header = ['lcs', 'score']  # LCS, (adjusted) global term frequency
        return df  # dataframe or None (if not found)
    def rank_fset(n_features, docIds=None, inputdir=None, content_sep=','): 
        # Same as initvar but load lcs map contents from .csv files 
        cohort = kargs.get('cohort', lcsHandler.cohort)  # 'CKD'

        # n_features = topn 
        if inputdir is None: inputdir = Pathway.getPath(cohort=cohort)
        ctype = seqparams.normalize_ctype(kargs.get('seq_ptype', lcsHandler.ctype))  # 'diag', 'med'
        
        Nd = len(D) if not docIds else len(docIds) # total number of documents

        # first check lcs to docIDs ...
        name = 'lcsmap'
        fname = '%s-%s.csv' % (name, ctype)
        fpath = os.path.join(inputdir, fname)

        gDocFreq = {}  # global document frequency table
        if os.path.exists(fpath) and os.path.getsize(fpath) > 0: 
            df = pd.read_csv(fpath, sep='|', header=0, index_col=False, error_bad_lines=True) # header = ['lcs', 'doc_ids']
            gDocFreq = dict(zip(df['lcs'], df['doc_ids']))
            # for r, row in df.iterrows(): 

            df = None; gc.collect()
            for lcs, idstr in gDocFreq.items(): 
                # lcs, idstr = row['lcs'], row['doc_ids']
                ids = idstr.split(content_sep)
                gDocFreq[lcs] = len(ids)

        # ... then check docID -> {(lcs, freq)}
        name = 'lcsmapInvFreq'
        fname = '%s-%s.csv' % (name, ctype)
        fpath = os.path.join(inputdir, fname)
        gTermFreq = {}
        if os.path.exists(fpath) and os.path.getsize(fpath) > 0: 
            # header = ['doc_id', 'lcs', 'freq']  # where doc_id can have duplicates 
            df = pd.read_csv(fpath, sep='|', header=0, index_col=False, error_bad_lines=True)
            gc.collect()
            for i, lcs in enumerate(gDocFreq.keys()): # for lcs in set(df['lcs'].values): 
                chunk = df.loc[df['lcs']==lcs]   # [note] this subsetting op is slow when size(data) is large
                if not chunk.empty: 
                    gTermFreq[lcs] = sum(chunk['freq'].values)  # global TF
                else: 
                    print('rank_fset> Warning: Could not find %s in any documents!' % lcs)
                    gTermFreq[lcs] = 0 
                if i > 0 and i % 100 == 0: print '%d, ' % i
        df = None; gc.collect()
        
        assert len(gTermFreq) > 0, "rank_fset> No feature set found."
        ### rank features according to (discounted) frequencies 
        for lcs, gtf in gTermFreq.items(): 
            w = (1+Nd)/(1+gDocFreq[lcs]+0.0)            
            score = gtf * (math.log(w)+1.0)
            gTermFreq[lcs] = score # adjusted gtf

        lcs_scores = sorted([(lcs, score) for lcs, score in gTermFreq.items()], key=lambda x:x[1], reverse=True)[:n_features] # descending order
        print('rank_fset> fset size=%d (requested %d) example scores:\n%s\n' % (len(lcs_scores), n_features, lcs_scores[:10]))

        header = ['lcs', 'score']  # LCS, (adjusted) term frequency
        adict = {h:[] for h in header}
        for lcs, s in lcs_scores: 
            adict['lcs'].append(lcs)
            adict['score'].append(s)
        df = DataFrame(adict, columns=header)

        # this calls saveDataFrame with special keyword padded to 'meta' (Rlcs)
        TSet.saveRankedLCSFSet(df, cohort=cohort, d2v_method=lcsHandler.d2v_method, 
                seq_ptype=ctype, suffix=lcsHandler.meta) # dir_type/'combined'
       
        # sorted_ = sorted([(lcs, len(dx)) for lcs, dx in lcsToDocIDs], key=lambda x: x[1], reverse=True) # descending order
        return df

    import seqparams
    from seqparams import Pathway
    from tset import TSet
    import seqConfig as sq
    from seqConfig import lcsHandler
    from seqTest import TestDocs
    # from scipy.sparse import csr_matrix # coo_matrix
    # use sq.sysConfig

    # config()  # if sq.sysConfig() was run first, then this is a noop
    D, L, T = process_docs(inputdir=None)  # set inputdir to None to use default

    # load existing LCS features (derived from docToLCS, deriveLCS, etc)
    lcsCandidates = kargs.get('feature_set', [])  # try user input first
    if len(lcsCandidates) == 0: 
        # params: cohort, lcs_type, lcs_policy, ctype, consolidate_lcs, label, meta
        df = lcsHandler.load_lcs()  # load the global LCS training set
        assert df is not None and not df.empty, "Could not load pre-computed LCS feature set."

        # load lcs set
        lcsCandidates = df['lcs'].values
        # print('chooseLCSFeatureSet> found %d LCS canidates ...' % df.shape[0])

    ### global evaluation
    nCandidates = len(lcsCandidates)
    print('chooseLCSFeatureSet> Choose from among %d LCS candidate features for modeling MCS data of %d input docs.' % \
        (nCandidates, len(D)))

    # policy A: rank feature by specified criteria e.g. term frequency
    df = load_ranked_fset(n_features=topn)
    if df is None or df.empty: # ['lcs', 'score']
        # the document IDs must be consistent with the initial source
        df = rank_fset(n_features=topn, docIds=kargs.get('document_ids', []), inputdir=None, content_sep=',')

    return df

def loadLCSFeatureTSet(**kargs):
    """
    Load LCS-feature training data. 

    Operations: load, scale, modify (classes), subsampling

    """
    def load_tset(policy_fs='sorted'):
        policyFeatureRank = kargs.get('policy_fs', policy_fs)  # sorted, tfidf, tf, ... 
        fileID = meta = lcsHandler.meta
        if policyFeatureRank.startswith('so'):
            fileID = 'sorted' if meta is None else '%s-sorted' % lcsHandler.meta
            
        cohort = kargs.get('cohort', lcsHandler.cohort)
        
        # X, y = lcsHandler.loadSparse()
        X, y = TSet.loadSparseLCSFeatureTSet(cohort=cohort, d2v_method=lcsHandler.d2v_method, 
                    seq_ptype=lcsHandler.ctype, suffix=fileID)
        return (X, y)
    def modify_tset(X, y): 
        focused_classes = kargs.get('focused_labels', None) # None: N/A
        if focused_classes is not None: # only preserve these classes and everything else becomes "Others"
            # e.g. In CKD data, one may only want to focus on predicting ‘Stage 1’, ‘Stage 2’, ‘Stage 3a’, ... 'Stage 5'
            y = sclf.focusLabels(y, labels=focused_classes, other_label='Others')  # output in np.ndarray

        lmap = kargs.get('label_map', seqparams.System.label_map)
        if lmap: 
            # profile(X, y)
            y = sclf.mergeLabels(y, lmap=lmap)
            print('> After re-labeling ...')
        return (X, y)
    def remove_classes(X, y, labels=[], other_label='Others'):
        exclude_set = labels if len(labels)>0 else [other_label, ]
        
        N0 = len(y)
        ys = Series(y)
        cond_pos = ~ys.isin(exclude_set)

        idx = ys.loc[cond_pos].index.values
        y = ys.loc[cond_pos].values 
        X = X[idx]  # can sparse matrix be indexed like this?

        print('... remove labels: %s > size(ts): %d -> %d' % (labels, N0, X.shape[0]))        
        return (X, y)
    def subsample(X, y, n=None, sort_index=False, random_state=53):
        """
        Subsampling operation applied after doc vec X is derived. 
        """
        if n is not None: 
            idx = cutils.samplePerClass(y, n_per_class=n, sort_index=sort_index, random_state=random_state)
            print('  + X (dim0: %s -> dim: %s)' % (X.shape[0], len(idx)))
            return (X[idx], y[idx])
        return (X, y)
    def profile(X, y): 
        n_classes = len(np.unique(y))
        no_throw = kargs.get('no_throw', True)

        print('loadSparseTSet> X (dim: %s), y (dim: %d)' % (str(X.shape), len(y)))
        print("                + number of store values (X): %d" % X.nnz)
        print('                + number of classes: %d' % n_classes)

        # assert X.shape[0] == len(y)
        return 

    import seqConfig as sq
    import classifier.utils as cutils
    from seqConfig import lcsHandler
    from tset import TSet
    from seqTest import TestDocs
    from system import utils as sysutils
    from sklearn.preprocessing import StandardScaler, MaxAbsScaler # to scale (dummy) scores 
    import seqClassify as sclf

    ### load 
    X, y = load_tset()
    profile(X, y)  # initial 

    ### scaling X
    if kargs.get('scale_', False): # could destroy sparsity 
        print('  + scaling feature values (z-scores) ...')
        # scaler = StandardScaler(with_mean=False)  # centering will destropy sparsity, set it off
        scaler = MaxAbsScaler()
        X = scaler.fit_transform(X)

    ### modify classes
    if X is not None and y is not None:
        print('  + modifying classes ...') 
        X, y = modify_tset(X, y)

    n_classes0 = len(np.unique(y))

    # drop explicit control data in multiclass
    if kargs.get('drop_ctrl', False): # drop control group (e.g. label: 'Others')
        assert n_classes0 > 2, "Not a multiclass domain (n_classes=%d)" % n_classes0
        X, y = remove_classes(X, y, labels=['Others', ])

    ### subsampling per class
    maxNPerClass = kargs.get('n_per_class', None)
    if maxNPerClass is not None: 
        print('  + subsampling classes to max: %d' % maxNPerClass)
        # prior to samplig, need to convert to canonical labels first 
        assert maxNPerClass > 1
        X, y = subsample(X, y, n=maxNPerClass)

    profile(X, y) # final

    return (X, y)

def makeLCSFeatureTSet2a(cohort, **kargs):  # [note] maybe do away with the explicit arg: cohort
    """
    Same as makeLCSFeatureTSet2() but take the feature statistics from the pre-computed results
    instead of calling analyzeMCS(). 
    """
    def rank_fset(fset, save_=True, topn=None, policy_fs='sorted'): # fset: initial feature (super)set 
        policyFeatureRank = kargs.get('policy_fs', policy_fs)  # sorted, tfidf, tf, ... 
        if policyFeatureRank.startswith('so'):
            return rank_fset_sorted(fset, save_=save_, topn=topn)  # output: df | header= ['lcs', 'score']
        return rank_fset_tfidf(fset, save_=save_, topn=topn)  # output: df | header= ['lcs', 'score']    
    def rank_fset_sorted(fset, save_=True, topn=None):  # <- D, T 
        meta = lcsHandler.meta 
        fileID = 'sorted' if meta is None else '%s-sorted' % meta
        topNFSet = kargs.get('topn', None)
        nT = len(fset)
        if topNFSet is not None and topNFSet < nT: 
            fset = random.sample(fset, topNFSet)
        
        # just alphanumeric ordering
        fset = sorted(fset)

        header = ['lcs', 'score']  # LCS, global term frequency
        adict = {h:[] for h in header}
        scores = []
        for i, lcs in enumerate(fset): 
            adict['lcs'].append(lcs)
            scores.append(1/(i+1.0))  # dummy score
        
        scores = np.array(scores).reshape(-1, 1)   # to column vector
        # fit > transform 
        std_scale = StandardScaler().fit(scores)
        adict['score'] = std_scale.transform(scores).reshape(1, len(fset))[0]
        df = DataFrame(adict, columns=header)
        if save_: 
            # this calls saveDataFrame with special keyword padded to 'meta' (Rlcs)
            TSet.saveRankedLCSFSet(df, cohort=cohort, d2v_method=lcsHandler.d2v_method, 
                seq_ptype=lcsHandler.ctype, suffix=fileID) # dir_type/'combined'
       
        # sorted_ = sorted([(lcs, len(dx)) for lcs, dx in lcsToDocIDs], key=lambda x: x[1], reverse=True) # descending order
        return df
    def rank_fset_tfidf(fset, save_=True, topn=None): # fset: total
        kargs['save_'] = save_ 
        df = chooseLCSFeatureSet(topn, **kargs)
        lcsCandidatesPrime = df['lcs'].values # this must be a subset of the original candidates 
        assert set(lcsCandidatesPrime).issubset(fset), "inconsistent feature set" 
        return df # ['lcs', 'score']  
    def load_fset(topn=None, policy_fs='sorted'):
        policyFeatureRank = kargs.get('policy_fs', policy_fs)  # sorted, tfidf, tf, ... 
        if policyFeatureRank.startswith('so'):
            return load_sorted_fset(topn=topn)
        return load_ranked_fset(topn=topn)
    def load_sorted_fset(topn=None): 
        # load 
        meta = lcsHandler.meta
        fileID = 'sorted' if meta is None else '%s-sorted' % meta
        df = TSet.loadRankedLCSFSet(cohort=cohort, d2v_method=lcsHandler.d2v_method, 
                     seq_ptype=lcsHandler.ctype, suffix=fileID)
        if df is not None and not df.empty: 
            if topn is not None: df = df.head(topn)  # this is not as meaningful when LCSs are just alphanumerically sorted
        return df  # dataframe or None (if not found)
    def load_ranked_fset(topn=None): 
        # load 
        fileID = lcsHandler.meta 
        df = TSet.loadRankedLCSFSet(cohort=cohort, d2v_method=lcsHandler.d2v_method, 
                     seq_ptype=lcsHandler.ctype, suffix=fileID)
        if df is not None and not df.empty: 
            if topn is not None: df = df.head(topn)
        return df  # dataframe or None (if not found): 
    def initvarcsv(name, keys=None, value_type='list', inputdir=None, content_sep=','): 
        # Same as initvar but load lcs map contents from .csv files
        if inputdir is None: inputdir = Pathway.getPath(cohort=cohort)
        ctype = seqparams.normalize_ctype(kargs.get('seq_ptype', lcsHandler.ctype))  # 'diag', 'med'

        fname = '%s-%s.csv' % (name, ctype)
        fpath = os.path.join(inputdir, fname)

        newvar = {}
        if keys is not None: # keys: e.g. document IDs
            if value_type == 'list': 
                newvar = {k: [] for k in keys}
            else: 
                # assuming nested dictionary 
                newvar = {k: {} for k in keys}
        else: 
            if value_type == 'list': 
                newvar = defaultdict(list)
            else: 
                newvar = defaultdict(dict) # nested dictionary

        if os.path.exists(fpath) and os.path.getsize(fpath) > 0: 
            df = pd.read_csv(fpath, sep='|', header=0, index_col=False, error_bad_lines=True)
            print('initvarcsv> var: %s > found %d existing entries from %s' % (name, df.shape[0], fname))
            
            # parse 
            if name == 'lcsmap': 
                header = ['lcs', 'doc_ids']
                idx = []
                for idstr in df['doc_ids'].values: 
                    idx.append(idstr.split(content_sep))  # [[], [], ...]
                    # if keys: idx = list(idx.intersection(keys))
                newvar.update(dict(zip(df['lcs'], idx)))
                return newvar 
            elif name == 'lcsmapInvFreq': 
                header = ['doc_id', 'lcs', 'freq']  # where doc_id can have duplicates 
                # for di in df['doc_id'].unique(): 
                cols = ['doc_id', ]
                for di, chunk in df.groupby(cols):  
                    # if keys and not di in keys: continue
                    newvar[di] = zip(chunk['lcs'], chunk['freq'])
            else: 
                raise NotImplementedError

        if len(newvar) > 0: 
            print('initvarcsv> example:\n%s\n' % sysutils.sample_dict(newvar, n_sample=1))
        return newvar
    def eval_feature_value(fset, inputdir=None, content_sep=','): 
        # Same as initvar but load lcs map contents from .csv files 
        # cohort = kargs.get('cohort', lcsHandler.cohort)  # 'CKD'

        # n_features = topn 
        if inputdir is None: inputdir = Pathway.getPath(cohort=cohort)
        ctype = seqparams.normalize_ctype(kargs.get('seq_ptype', lcsHandler.ctype))  # 'diag', 'med'
        
        nD, nF = len(D), len(fset)  # total number of documents (which can be < total)

        # first check lcs to docIDs ...
        name = 'lcsmap'
        gDocFreq = initvarcsv(name='lcsmap')  # lcs -> docIDs
        for lcs, idx in gDocFreq.items(): 
            gDocFreq[lcs] = len(idx)

        # ... then check docID -> {(lcs, freq)}
        name = 'lcsmapInvFreq'
        gTermFreq = initvarcsv(name='lcsmapInvFreq')  # id -> {(lcs, tf)}

        assert len(gTermFreq) > 0, "Could not load recomputed %s" % name
        # sparse format by default 
        row, col, data = [], [], []
        lcsToFPos = {lcs:i for i, lcs in enumerate(fset)}  # lcsToFPos: LCS to its feature position/index in lcsFSet
        
        print('eval_feature_value> nD: %d, nF: %d' % (nD, nF))
        testIDs = random.sample(range(nD), 20)
        for i, doc in enumerate(D):
             # if i in testIDs: test_lcs_frequency(i, ret)

            col_idx = [lcsToFPos[lcs] for lcs, _ in gTermFreq[i]]  #  docID -> {(lcs, freq)} -> {(pos, freq)}
                
            # values = [tf or _, tf in gTermFreq[i]] # term frequency only
            values = []
            for lcs, tf in gTermFreq[i]: 
                w = (1+nD)/(1+gDocFreq[lcs]+0.0)
                score = tf * (math.log(w)+1.0)
                values.append(score)  # tf-idf scores

            row.extend([i] * len(col_idx))  # ith document
            col.extend(col_idx)  # jth attribute
            data.extend(values)  # scores
        Xs = csr_matrix((data, (row, col)), shape=(nD, nF))
        assert Xs.shape[0] == nD and Xs.shape[1] == nF
        
        print('makeLCSFeatureTSet2a> Found %d coordinates, %d active values, ' % (len(data), Xs.nnz))
        gc.collect()
        return Xs
    def save_tset(X, y, policy_fs='sorted'):
        # file ID: cohort, d2v_method, seq_ptype, index, suffix
        # directory: cohort, dir_type/'combined'
        policyFeatureRank = kargs.get('policy_fs', policy_fs)  # sorted, tfidf, tf, ... 
        fileID = meta = lcsHandler.meta
        if policyFeatureRank.startswith('so'):
            fileID = 'sorted' if meta is None else '%s-sorted' % lcsHandler.meta
            
        if not isinstance(y, np.ndarray): y = np.array(y)
        TSet.saveSparseLCSFeatureTSet(X, y=y, cohort=cohort, d2v_method=lcsHandler.d2v_method, 
            seq_ptype=lcsHandler.ctype, suffix=fileID)

        # PS1: feature set is defined according to load_rank_fset (precomputed variables) or rank_fset (new variables)

        # PS2: load method
        # X, y = TSet.loadSparseLCSFeatureTSet(cohort, **kargs)
        return
    def load_tset(policy_fs='sorted'):
        policyFeatureRank = kargs.get('policy_fs', policy_fs)  # sorted, tfidf, tf, ... 
        fileID = meta = lcsHandler.meta
        if policyFeatureRank.startswith('so'):
            fileID = 'sorted' if meta is None else '%s-sorted' % lcsHandler.meta
            
        X, y = TSet.loadSparseLCSFeatureTSet(cohort=cohort, d2v_method=lcsHandler.d2v_method, 
                    seq_ptype=lcsHandler.ctype, suffix=fileID)
        return (X, y)
        
    # import seqparams
    from tset import TSet
    import seqConfig as sq
    from seqConfig import lcsHandler
    from seqTest import TestDocs
    from system import utils as sysutils
    from scipy.sparse import csr_matrix # coo_matrix
    # from seqparams import Pathway
    from sklearn.preprocessing import StandardScaler # to scale (dummy) scores 
    # use sq.sysConfig

    kargs['cohort'] = cohort
    ret = processDocumentsUtil(**kargs)  # params: 'max_n_docs' for subsampling; 'document_ids' for subsetting
    D, T, L = ret['sequence'], ret['timestamp'], ret['label']

    # the document IDs must be consistent with the initial source
    docIds = ret['doc_ids'] 
    if docIds: assert len(D) == len(docIds), "size(D):%d but given %d docIDs" % (len(D), len(docIds))

    # load existing LCS features (derived from docToLCS, deriveLCS, etc)
    lcsCandidates = kargs.get('feature_set', [])  # try user input first
    if len(lcsCandidates) == 0: 
        # params: cohort, lcs_type, lcs_policy, ctype, consolidate_lcs, label, meta
        df = lcsHandler.load_lcs()  # load the global LCS training set
        assert df is not None and not df.empty, "Could not load pre-computed LCS feature set."

        # load lcs set
        lcsCandidates = df['lcs'].values
        print('makeLCSFeatureTSet2a> found %d LCS canidates ...' % df.shape[0])
    nT = len(lcsCandidates)

    # just use sorted order
    # policy A: rank feature by specified criteria e.g. term frequency
    topNFSet = kargs.get('topn', None)
    df = load_fset(topn=topNFSet)  # params: 'policy_fs' (sorted, tfidf, tf, ... )
    if df is None or df.empty: 
        # params: 'policy_fs'
        print('makeLCSFeatureTSet2a> creating a new set of ranked LCS features ...')
        df = rank_fset(lcsCandidates, save_=True, topn=topNFSet) # alphanumeric sorted order
    else: 
        print('  + df(lcs_fset) header: %s' % ' '.join(df.columns.values))

    lcsFSet = df['lcs'].values
    nD, nF = len(D), len(lcsFSet)

    # final feature set
    print('makeLCSFeatureTSet2a> nD=%d, nF=%d (total=%d):\n    + dim(1):\n%s\n    + dim(-1):\n%s\n' % \
        (nD, nF, nT, lcsFSet[0], lcsFSet[1]))

    ### lcsFSet is a filtered, ranked, and/or sorted version of lcsCandidates
    
    # sparse format by default 
    Xs = eval_feature_value(fset=lcsFSet)  # feature values are not scaled
    assert Xs is not None, "Could not compute feature values"
    if kargs.get('scale_', False): # could destroy sparsity 
        print('  + scaling feature values (z-scores)')
        std_scaler = StandardScaler(with_mean=False)
        Xs = std_scaler.fit_transform(Xs)
    
    assert Xs.shape[0] == len(L)
    if kargs.get('save_', True): 
        save_tset(Xs, L)
    
    return (Xs, L)

def analyzeLCSDistribution(cohort, **kargs): 
    """
    Analyze LCS distributions within each stratified cohort (e.g. CKD cohort stratified by severity stages)
    The analysis is very similar to chooseLCSFeatureSet(topn, **kargs)

    Params
    ------
        - feature selection 
            topn 

        - stratify documents 
            min_ncodes 

            inputdir, ifiles
          
            cohort, seq_ptype, d2v_method

            slice_policy
          


    """
    def config(): 
        # configure all the parameters 
        userFileID = meta = kargs.get('meta', None)
        ctype = seqparams.normalize_ctype(kargs.get('seq_ptype', 'regular'))

        # this routine can only be called once, subsquent calls are noop 
        sq.sysConfig(cohort=cohort, seq_ptype=ctype, 
                        lcs_type=kargs.get('lcs_type', 'global') , lcs_policy=kargs.get('lcs_policy', 'df'), 
                        consolidate_lcs=kargs.get('consolidate_lcs', True), 
                        slice_policy=kargs.get('slice_policy', 'noop'), 
                        simplify_code=kargs.get('simplify_code', False), 
                        meta=userFileID)
        return
    def get_global_cohort_dir(): # source MDS
        return seqparams.getCohortGlobalDir(cohort) # sys_config.read('DataExpRoot')/<cohort>
    def load_fset(topn=None, policy_fs='sorted', col='lcs'):
        policyFeatureRank = kargs.get('policy_fs', policy_fs)  # sorted, tfidf, tf, ...
        
        df = None 
        if policyFeatureRank.startswith('so'):
            df = load_sorted_fset(topn=topn)
        else: 
            df = load_ranked_fset(topn=topn)
        
        if df is None or df.empty: 
            msg = 'load_fset> Could not find precomputed LCS candidates ...'
            raise ValueError, msg
        fset = df[col].values 
        df = None; gc.collect()
        return fset
    def load_sorted_fset(topn=None): 
        # load 
        meta = lcsHandler.meta
        fileID = 'sorted' if meta is None else '%s-sorted' % meta
        df = TSet.loadRankedLCSFSet(cohort=cohort, d2v_method=lcsHandler.d2v_method, 
                     seq_ptype=lcsHandler.ctype, suffix=fileID)
        if df is not None and not df.empty: 
            if topn is not None: df = df.head(topn)  # this is not as meaningful when LCSs are just alphanumerically sorted
        print('analyzeLCSDistribution> sorted lcs fset > dim: %s' % str(df.shape))
        return df  # dataframe or None (if not found)
    def load_ranked_fset(topn=None): 
        # load 
        fileID = lcsHandler.meta 
        df = TSet.loadRankedLCSFSet(cohort=cohort, d2v_method=lcsHandler.d2v_method, 
                     seq_ptype=lcsHandler.ctype, suffix=fileID)
        if df is not None and not df.empty: 
            if topn is not None: df = df.head(topn)
        return df  # dataframe or None (if not found): 
    def get_lcs_fset(topn=None):
        df = load_ranked_fset(topn=topn)
        return df['lcs'].values 
    def eval_local_freq(fset, document_ids): # params: lcsmapInvFreq 
        # params: stratified_stats: output of analyzeMCS evaluated on a stratified cohort

        lcsToWeightedFreq = {var: 0 for var in fset}  # if an LCS occurs in the same document N times, then it's (weighted) doc freq is N (instead of just 1)
        lcsFreq = {var: 0 for var in fset}

        print('eval_local_freq> After init, lcsToWeightedFreq has %d entries ...' % len(lcsToWeightedFreq))

        # compute weighted frequency
        for i, docID in enumerate(document_ids): 
            
            # a subset of document IDs may not appear because of the min_ncodes constraint
            if not docID in lcsmapInvFreq: continue 

            for j, (lcs, tf) in enumerate(lcsmapInvFreq[docID]): 
                
                # [test]
                if i < 10 and j < 2: print('  + lcs: %s => tf: %d' % (lcs, tf))

                if not lcsToWeightedFreq.has_key(lcs): 
                    print('eval_local_docfreq> Warning: %s never appeared in (sampled) documents.' % lcs) # this should not happen
                    lcsToWeightedFreq[lcs] = 0
                lcsToWeightedFreq[lcs] += tf 

        # compute non-weighted document frequency (i.e. if an LCS occurs multiple times in the same document, count as 1)
        for docID in document_ids: 

            if not docID in lcsmapInvFreq: continue
            for lcs, tf in lcsmapInvFreq[docID]: 
                lcsFreq[lcs] += 1 
        
        return (lcsToWeightedFreq, lcsFreq)

    def save_stats(df, label, is_weighted=True): 
        TSet.saveLCSFreqDistribution(df, is_weighted=is_weighted, label=label,
                cohort=cohort, d2v_method=lcsHandler.d2v_method, 
                seq_ptype=lcsHandler.ctype, suffix=lcsHandler.meta) # dir_type/'combined'

        # PS: load 
        # TSet.loadLCSFreqDistribution(is_weighted=is_weighted, 
        #         cohort=cohort, d2v_method=lcsHandler.d2v_method, 
        #         seq_ptype=lcsHandler.ctype, suffix=lcsHandler.meta)
        return 
    def policy_relabel(): 
        # CKD stage classes
        lmap = {}
        lmap['Others'] = ['CKD G1-control', 'CKD G1A1-control', 'Unknown', ]  # ... control data 
        lmap['CKD Late'] = ['ESRD after transplant', 'ESRD on dialysis', 'CKD Stage 5']
        lmap['CKD Middle'] = ['CKD Stage 3a', 'CKD Stage 3b', 'CKD Stage 4']
        lmap['CKD Early'] = ['CKD Stage 1', 'CKD Stage 2', ]
        return lmap
    def merge_strata(stratified, lmap={}): 
        if not lmap: lmap = policy_relabel()
        
        stratifiedPrime = {}
        header = ['sequence', 'timestamp', 'label', 'doc_ids']
        for target, labels in lmap.items(): # map labels to target
            
            # init 
            stratifiedPrime[target] = {h: [] for h in header}

            n_merged = 0
            for label in labels:  # foreach label, incorperate their datasets into stratifiedPrime
                if label in stratified: 
                    # entries.update(stratified[label]) # this may not work
                    entry = stratified[label]
                    # Di = entry['sequence']
                    # Ti = entry['timestamp']
                    # Li = entry['label']
                    # Ii = entry['doc_ids']
                    for h in header: 
                        stratifiedPrime[target][h].extend(entry[h])  # stratified[label]['sequence'] <- a list ([[], [], ...])

                    n_merged += 1
            # stratifiedPrime[target] = entries
            print('... merged %d labels to target: %s ...' % (n_merged, target))
        print('merge_strata> New stratified has %d labels' % len(stratifiedPrime))
        # assert len(stratifiedPrime) == len(lmap)
        return stratifiedPrime

    import seqparams
    from tset import TSet
    import seqConfig as sq
    from seqConfig import lcsHandler
    from seqTest import TestDocs
    import collections 

    config() # configuration will be done only once (if sq.sysConfig was called, then noop here)

    # load (ranked) LCS feature set 
    topNFSet = kargs.get('topn', None) # None to select all 
    topNLCS = kargs.get('topn_lcs', 100)  # used for plotting LCS distribution
    lcsFSet = load_fset(topn=topNFSet, policy_fs=kargs.get('policy_fs', 'sorted'))
    # lcsFSet = get_lcs_fset(topNFSet)
    assert hasattr(lcsFSet, '__iter__') and len(lcsFSet) > 2

    ### Find frequency distribution of LCS features in each stratification/class 
    #   and find if they are significantly different (plot the distribution of top N LCS in terms of document frequencies)
    #   use subsampling, boostrapping to enforce equal sample size

    ### local evaluation
    # label -> entry: Di, Ti, Li
    stratified = kargs.get('stratified_docs', {}) # try user input first: label -> entry: ['sequence', 'timestamp', 'label']
    if not stratified: 
        kargs['mode'] = 'stratify'
        stratified = processDocumentsUtil(**kargs) # stratify_docs(inputdir=None)
    else: 
        assert isinstance(stratified, dict)
        nl = len(stratified)
        print('analyzeLCSDistribution> Received user-provided input (nL=%d) ...' % nl)

    TestDocs.stratum_stats(stratified) # [test] size of docs? how many labels? 

    lcsmapInvFreq = loadLCSStats(name='lcsmapInvFreq')
    print('analyzeLCSDistribution> lcsmapInvFreq n_entries: %d' % len(lcsmapInvFreq))

    ### control params
    tMerge = True 

    if tMerge: 
        stratified = merge_strata(stratified)

    nD = nT = n_saved = 0
    for label, entry in stratified.items():
        Di = entry['sequence']
        Ti = entry['timestamp']
        Li = entry['label']
        Ii = entry['doc_ids']
        nD += len(Di)
        nT += len(Ti)
        print('analyzeLCSDistribution> searching LCSs for label=%s | N=%d' % (label, len(Di)))

        # ret = analyzeMCS(Di, Ti, lcsFSet)  # this is too slow, use precomputed results
        lcsWeightedFreq, lcsFreq = eval_local_freq(fset=lcsFSet, document_ids=Ii)

        # save the data 
        if kargs.get('save_', True):
            header = ['lcs', 'frequency', ]
            adict = {h: [] for h in header}

            # convert to dataframe-ready format 
            for lcs, freq in lcsWeightedFreq.items(): 
                adict['lcs'].append(lcs) 
                adict['frequency'].append(freq)

            df = DataFrame(adict, columns=header)
            save_stats(df, label=label, is_weighted=True) 
            n_saved += 1

    #     lcsColorMap, lcsTimeMap = ret['color'], ret['time']
    #     sample_dict(lcsColorMap); sample_dict(lcsTimeMap)
    print('... completed analysis and saved %d label-specific datasets' % n_saved)

        # plot LCS time series
    return

def makeLCSLabeledTSet(cohort, **kargs):
    return makeLCSTSet(cohort, **kargs) 
def makeLCSTSet(cohort, **kargs):
    """
    Use globally frequent LCSs to create (surrogate) labels. 

    Params
    ------
    a. training set identifier
    cohort
    seq_ptype 
    d2v_method

    b. 
    max_n_pairs 
    remove_duplicates
    min_length
    max_length
    topn_lcs: preserve only this many LCSs (ranked by frequency)

    Memo
    ----
    1. also create an interface in labeling module

    2. time matching
       use seqAlgo.traceSubsequence3

       query seq: ['720.0', '123.5'] is in ref seq: 
                  ['720.0', '123.5', 'x', 'x', 'y', '720.0', 'z', 'z', '123.5', 'x', '720.0', '123.5', 'y', '720.0']
          output ~> [[0, 1], [5, 8], [10, 11]] 

    Related
    -------
    labeling.labelByLCS: given input sequences (D), find their LCS labels 

    Todo
    ----
    1. abstraction for naming, loading and saving LCS label files 
       see secondary_id, load_lcs, deriveLCS.save_lcs

    """
    def do_slice(): 
        if not kargs.has_key('slice_policy'): return False
        if kargs['slice_policy'].startswith(('noop', 'reg', 'compl', )):  # effective operations 
            return False 
        return True
    def secondary_id(): # attach extra info in suffix: {seq_ptype, slice_policy, label_id}
        ctype = kargs.get('seq_ptype', 'regular')
        suffix = ctype
        # suffix = kargs.get('suffix', ctype) # vector.D2V.d2v_method
        if do_slice(): suffix = '%s-%s' % (suffix, kargs['slice_policy'])
        if kargs.get('consolidate_lcs', True): suffix = '%s-%s' % (suffix, 'iso') # LCSs up to permutations consider as the same? 
        # suffix = '%s-len%s' % (suffix, kargs.get('min_length', 3))
        label_id = kargs.get('label', None)  # stratified dataset by labels
        if label_id is not None: suffix = '%s-L%s' % (suffix, label_id)

        meta = kargs.get('meta', None)
        if meta is not None: suffix = '%s-U%s' % (suffix, meta) 
        return suffix 
    def load_lcs(): # lcs_type
        ltype = kargs.get('lcs_type', 'global')  # global: Common LCSs defined in global scope (as opposed to cluster or local scopes)
        lcs_select_policy = kargs.get('lcs_policy', 'freq')
        # seq_ptype = kargs.get('seq_ptype', 'regular')

        # use {seq_ptype, slice_policy, length,} as secondary id
        suffix = secondary_id()
        return Pathway.load(cohort, scope=ltype, policy=lcs_select_policy, suffix=suffix, dir_type='pathway')
    def save_lcs(df): # maybe this should be incorporated into deriveLCS() itself 
        # see load_lcs and deriveLCS
        # assert 'n_docs' in df.columns, "n_docs attribute has not been added."  # this is deferred to makeLCSTSet
        raise ValueError, "LCS creation has been delegated to deriveLCS."
    def verify_doc(): # params: D, L, T
        assert len(D) > 0, "No input documents found (cohort=%s)" % cohort
        x = random.randint(0, len(D)-1)  # nDoc
        assert isinstance(D[x], list), "Invalid input D[%d]:\n%s\n" % (x, D[x])
        assert len(T) == len(D), "inconsistent number of timestamps (nT=%d while nD=%d)" % (len(T), len(D))
        assert len(L) == len(D), "inconsistent number of labels (nL=%d while nD=%d)" % (len(L), len(D))
        print('markTSetByLCS> Found %d documents (nT=%d, nL=%d, cohort=%s)' % (len(D), len(T), len(L), cohort))
        return 
    def verify_tset(): # params: D, T, L
        # d2v_method = kargs.get('d2v_method', seqparams.D2V.d2v_method) # pv-dm2
        # seq_ptype = kargs.get('seq_ptype', 'regular') 
        nD = len(D)
        ts = TSet.loadLCSTSet(cohort=cohort, d2v_method=d2v_method, seq_ptype=seq_ptype)
        if ts is None: 
            ts = TSet.loadCombinedTSet(cohort=cohort, d2v_method=d2v_method, seq_ptype=seq_ptype) 
            assert ts is not None, "No relevant training set found (cohort=%s, d2v=%s, ctype=%s)" % (cohort, d2v_method, seq_ptype)
        assert ts.shape[0] == nD, "Training set size: %d <> nDoc: %d" % (ts.shape[0], nD)
    def test_lcs_match(lcs_seq, doc): 
        assert isinstance(doc, list) and isinstance(lcs_seq, list), "illed-formatted input"
        print('  + lcs_seq: %s' % lcs_seq)
        print('  + doc:\n%s\n' % doc)
        return 
    def lcs_time_series(lcs, ith, D, T): 
        # [params] lcs: target LCS (in list format)
        #          ith: ith document where LCS has a match 
        #          D: corpus 
        #          T: time corpus (i.e. timestamps of D)  
        assert len(T) > 0 and len(D) == len(T)
        matched_positions = seqAlgo.traceSubsequence3(lcs, D[ith])  # find matched positions of 'lcs' in the document 'D[ith'
        matched_times = [] 
        if len(matched_positions) > 0: 
            for positions in matched_positions: 
                matched_times.append(list(np.array(T[ith])[positions]))
        else: 
            # print('  + could not find matched positions')
            raise ValueError, \
            "Could not find matched positions between lcs:\n%s\nand doc:\n%s\nUse isSubsequence to verify match first." % (lcs, D[ith])
        return matched_times
    def label_document(L, overwrite=True):  # label source document
        # pattern: condition_drug_<file_type>-<cohort>.csv 
        # fname = TDoc.getName(cohort=cohort, doctype='labeled', ext='csv')  # [params] doc_basename/'condition_drug'
        prefix = kargs.get('basedir', TDoc.prefix)
        fpath = TDoc.getPath(cohort=cohort, seq_ptype=seq_ptype, doctype='labeled', ext='csv', basedir=prefix)  # usually there is one file per cohort  
        assert os.path.exists(fpath), "labeled docuemnt source should have been created previously (e.g. by processDocuments)"
        # assert len(fpaths) > 0, "Could not find any document sources associated with cohort=%s" % cohort

        print('markTSetByLCS.label_document> Now filling in the labels to %s (? n_paths=1)' % fpath)
        df_src = pd.read_csv(fpath, sep='|', header=0, index_col=False, error_bad_lines=True)
        print('info> header: %s' % list(df_src.columns))

        lcs_col = 'label_lcs'; assert lcs_col in TDoc.fLabels, "Invalid label attribute: %s" % lcs_col
        if not (lcs_col in df_src.columns) or overwrite: 
            df_src[lcs_col] = L
            df_src.to_csv(fpath, sep='|', index=False, header=True)
            print('  + IO: saved label_seq (cohort=%s) to:\n%s\n' % (cohort, fpath))
        else: 
            print('  + Info: %s already existed (and NOT ovewriting existing labels)' % lcs_col)  
        return df_src
    def save_lcs_tset(L, overwrite=True): # mark training data 
        # d2v_method = kargs.get('d2v_method', seqparams.D2V.d2v_method) # pv-dm2
        suffix = secondary_id()
        ts = TSet.loadLCSTSet(cohort=cohort, d2v_method=d2v_method, seq_ptype=seq_ptype, suffix=suffix)
        if ts is not None and not ts.empty: print('  + Found pre-computed tset (dim:%s)' % str(ts.shape))
        if (ts is None) or overwrite:  # if the training set does not exist
            print('  + Creating new LCS tset (cohort=%s, d2v=%s, ctype=%s)' % (cohort, d2v_method, seq_ptype))

            # base training set (with regular labels) 
            # ts = TSet.loadCombinedTSet(cohort=cohort, d2v_method=d2v_method, seq_ptype=seq_ptype, suffix=suffix) 
            ts = TSet.loadBaseTSet(cohort=cohort, d2v_method=d2v_method, seq_ptype=seq_ptype) # do not include suffix 
            if ts is None or ts.empty: 
                # [note] directory params: cohort, dir_type <- 'combined' by default
                fpath = TSet.getFullPath(cohort, d2v_method=d2v_method, seq_ptype=seq_ptype)  # do not include suffix (secondary ID)
                raise ValueError, "Primary tset (c:%s, d2v:%s, ctype:%s) does not exist at:\n%s\n" % \
                    (cohort, d2v_method, seq_ptype, fpath) 

            X, y = TSet.toXY(ts)  # or use TSet.toXY2() if n_features is known 
            assert len(L) == len(y), "Inconsistent number of labels: nL=%d, nrows=%d" % (len(L), X.shape[0])
            TSet.saveLCSTSet(X, y=L, cohort=cohort, d2v_method=d2v_method, seq_ptype=seq_ptype, suffix=suffix)
        else: 
            assert ts.shape[0] == nD, "Inconsistent data size: nD=%d <> nrows=%d" % (nD, ts.shape[0])
            # assert ts.shape[1] == seqparams.D2V.n_features * 2
            assert len(L) == ts.shape[0], "Inconsistent number of labels: nL=%d, nrows=%d" % (len(L), ts.shape[0])

        # save a copy? 
        return ts  # or (X, L)
    def inverse_lcsmap(): 
        assert isinstance(lcsmap, dict) and len(lcsmap) > 0
 
        nD = len(D)
        lcsmapInv = {i: set() for i in range(nD)} # document ID -> LCSs 
        for lcs, docIds in lcsmap.items(): 
            for docId in docIds: 
                lcsmapInv[docId].add(lcs)  # this cannot have dups
        for docId in lcsmapInv.keys(): 
            lcsmapInv[docId] = list(lcsmapInv[docId])
        return lcsmapInv
    def test_inverse_lcsmap():  # lcsmap
        # note that a similar test is done in analyzeLCS, which had not gone through filtering criteria
        acc = 0 
        fdistr = [] # n_doc frequency distribution
        n_multimatch = n_uniqmatch = n_nomatch = 0
        for docId, lcsx in lcsmapInv.items(): 
            # lcsmapInv[docId] = list(lcsmapInv[docId])
            n_lcs = len(lcsx)
            if n_lcs >= 1: fdistr.append(n_lcs)  # only care about those with labels
            # acc += n_lcs 

            if n_lcs > 1: 
                n_multimatch += 1
                if n_multimatch <= 10: 
                    print('makeLCSTSet.test> document #%d matches multiple LCSs:\n    %s\n' % (docId, lcsmapInv[docId]))
            elif n_lcs == 1: 
                n_uniqmatch += 1 
            else: 
                n_nomatch += 1

        acc, nl = sum(fdistr), len(fdistr)
        avg_lcs = acc/(nl+0.0)
        print('  + expected number of labels: %f' % avg_lcs)
        print('  + distr: %s' % fdistr[:50])
        return
    def label_by_longest_lcs():  # lcsmapInv, L, lcsmap
        L = [token_nomatch] * nD  # redefinition of L
        for i, doc in enumerate(D): 
            if len(lcsmapInv[i]) == 0: 
                # noop 
                pass 
            elif len(lcsmapInv[i]) == 1:  
                L[i] = lcsmapInv[i][0]
            else: 
                # choose the longest? 
                longestFirst = sorted([(lcs, len(lcs.split(lcs_sep))) for lcs in lcsmapInv[i]], key=lambda x:x[1], reverse=True)
                maxlength = longestFirst[0][1]
                
                # avoid aways picking the same labeling (same LCS shared by multiple documents)
                candidates = []
                for lcs, length in longestFirst: 
                    if length >= maxlength: 
                        candidates.append(lcs)
                L[i] = random.sample(candidates, 1)[0]
        return L
    def label_by_relative_uniqueness(): # lcsmapInv, L, lcsmap
        print('makeLCSTSet> apply labeling policy: label_by_relative_uniqueness() ...')
        L = [token_nomatch] * nD  # redefinition of L
        j = random.randint(0, nD-1)
        nml = 0
        for i, doc in enumerate(D): 
            if len(lcsmapInv[i]) == 0: 
                # noop 
                pass 
            elif len(lcsmapInv[i]) == 1:  
                L[i] = lcsmapInv[i][0]
            else: 
                # choose the least popular LCS (present in least number of documents)
                lcnt = []
                for lcs in lcsmapInv[i]: 
                    dx = lcsmap[lcs]
                    lcnt.append((lcs, len(dx)))

                lcnt = sorted(lcnt, key=lambda x:x[1], reverse=False) # ascending order
                L[i] = lcnt[0][0] # choose the one present in least number of docs  

                # [test]
                if nml < 5: print('  + sorted lcnt:\n%s\n' % lcnt)
                nml += 1  # number of docs multiple labels
        return L 
    def label_by_consolidated_single_label(): 
        # [note] just like a chomosome containing multiple genes (~LCSs), an MCS carries several genes and therefore it may not be 
        #        appropriate to label a document with a single representative LCS  

        # similar to label_by_relative_uniqueness() but also consolide multiple label into a single one
        print('makeLCSTSet> apply labeling policy: label_by_consolidated_single_label() ...')
    
        L = [token_nomatch] * nD  # redefinition of L
        # j = random.randint(0, nD-1)
        n_multimatch = n_uniqmatch = n_nomatch = 0
        for i, doc in enumerate(D):  # doc: a list of tokens
            if len(lcsmapInv[i]) == 0: 
                # noop: just use default negative label (as a control set) 
                n_nomatch += 1 
            elif len(lcsmapInv[i]) == 1:  
                L[i] = lcsmapInv[i][0]
                n_uniqmatch += 1
            else: 
                # choose the least popular LCS (present in least number of documents)
                eff_labels = set()  # effective label representations (e.g. two LCSs up to a permutation are considered identical)
                candidates = []
                for lcs in lcsmapInv[i]:  # LCSs up to permutations are considered as the same
                    sorted_lcs = sortedSequence(lcs, sep=lcs_sep)  # Pathway.lcs_sep is typically a space
                    if not sorted_lcs in eff_labels: 
                        candidates.append(lcs)
                if n_multimatch < 10: 
                    print('  + narrow %d labels to %d (policy=permutation)' % (len(lcsmapInv[i]), len(candidates)))
                
                if len(candidates) == 1: 
                    L[i] = candidates[0]
                else: 
                    # lcs_sep = Pathway.lcs_sep
                    print('  + still many (n=%d):\n%s\n' % (len(candidates), candidates[:10]))

                    # policy #1: sample one of the longest 
                    # ranked = sorted([(lcs, len(lcs.split(lcs_sep))) for lcs in candidates], key=lambda x:x[1], reverse=True)

                    # policy #2: more unique codes first
                    ranked = sorted([(c, len(c.split(lcs_sep))) for c in candidates], key=lambda x:x[1], reverse=True) # high to low diversity
                    
                    # policy #3: more occurrences within the documents first (~ term frequency)
                    # [note] most of the frequencies (if not all) are identical
                    # ranked = sorted([(c, len(seqAlgo.traceSubsequence3(lcs.split(lcs_sep), doc))) for c in candidates], 
                    #                     key=lambda x:x[1], reverse=True)
                    # print('  + use traceSubsequence3 > top-10 ranked scores:\n%s\n' % ranked[:10])
                    
                    # ranked = sorted(candidates, cmp=make_comparator(has_less_ucodes_than), reverse=True) 
                    maxScore = ranked[0][1]
                    assert maxScore > 0, "max ranked score should be greater than 0:\n%s\n" % ranked

                    # avoid aways picking the same labeling of the same length (same LCS shared by multiple documents)
                    candidates2 = []
                    for lcs, score in ranked: 
                        # length = len(lcs.split(lcs_sep))  
                        if score >= maxScore: 
                            candidates2.append(lcs)
                    if len(candidates2) > 1: 
                        print('  + found %d candidate labels of max length=%d' % (len(candidates2), maxScore))
                    L[i] = random.sample(candidates2, 1)[0]

                n_multimatch += 1 
        return L 

        return
    def has_less_ucodes_than(s1, s2): # s1 has less unique codes than s2? 
        # used with sorted(alist, cmp=make_comparator(is_subset), reverse=True)
        sl1 = s1.split(lcs_sep)  # Pathway.lcs_sep
        sl2 = s2.split(lcs_sep)
        return len(set(sl1)) < len(set(sl2))
    def lcs_label_stats(n=5):  # lcsmap, L
        n_entries = len(lcsmap)
        nL = len(set(L))
        print('  + number of LCSs as labels: %d =?= nL: %d' % (n_entries, nL))
        # lcsx = random.sample(L, n)
        # print('  + example LCS labels:')
        # for lcs in lcsx: 
        #     print('    + %s' % lcs)
        counter = collections.Counter(L)
        label_distr = {}
        for l, cnt in counter.items(): 
            codeset = lcs_sep.join(sorted(set(l.split(lcs_sep))))
            print('  + label: %s | n_docs: %d | set: %s' % (l, cnt, codeset))
            label_distr[l] = cnt
        print('  + label frequency distribution:\n%s\n' % label_distr.values())
        # label vs size 
        return
    def filter_lcs_by_frequency(topn=20): # params: lcsmap, (topn_lcs)
        lcsCounts = [(s, len(dx)) for s, dx in lcsmap.items()]  # lcsmap is the result of comparing to the entire document set
        lcsCounts = sorted(lcsCounts, key=lambda x: x[1], reverse=True)[:topn]
        return {lcs: lcsmap[lcs] for lcs, _ in lcsCounts}     

    ### makeLCSTSet routine 
    from tset import TSet # mark training data
    from seqparams import Pathway # set pathway related parameters, where pathway refers to sequence objects such as LCS
    from tdoc import TDoc  # labeling info is also maintained in document sources of doctype = 'labeled'
    # import seqTransform as st

    verify_labeled_file = True
    seq_ptype = seqparams.normalize_ctype(kargs.get('seq_ptype', 'regular')) 
    d2v_method = kargs.get('d2v_method', seqparams.D2V.d2v_method)
    src_dir =  seqparams.getCohortGlobalDir(cohort)  # sys_config.read('DataExpRoot')/<cohort>

    # D: corpus, T: timestamps, L: class labels
    # load + transform + (labeled_seq file)
    # [todo] set save_to True to also save a copy of transformed document (with labels)? 
    D, L, T = processDocuments(cohort=cohort, seq_ptype=seq_ptype, 
               predicate=kargs.get('predicate', None),
               simplify_code=kargs.get('simplify_code', False), 

               inputdir=src_dir,
               ifiles=kargs.get('ifiles', []), 

               # slice operations {'noop', 'prior', 'posterior', }
               slice_policy=kargs.get('slice_policy', 'noop'), 
               slice_predicate=kargs.get('slice_predicate', None), 
               cutpoint=kargs.get('cutpoint', None),
               inclusive=True)  

    nD = len(D)

    # [test]
    verify_doc(); verify_tset()  # d2v tset must already exists

    # the following can also be achieved by calling labeling.labelByLCS() ... 
    # [note] ... but this introduces mutual dependency since labelByLCS <- deriveLCS

    df = load_lcs() if kargs.get('load_lcs', True) else None  # try loading first
    S = []
    lcsmap = {}
    if df is None:  
        # [note] 125000 allows for comparing 500+ documents pairwise
        # df: ['lcs', 'length', 'count', 'n_uniq']
        # max_n_pairs: 125000
        print('markTSetByLCS> Deriving candidate LCS labels from %d documents ...' % nD)
        df = deriveLCS(D=D, 
                        topn_lcs=kargs.get('topn_lcs', 10), 
                        min_length=kargs.get('min_length', 3), max_length=kargs.get('max_length', 1e5), 
                        max_n_pairs=kargs.get('max_n_pairs', 125000), min_ndocs=kargs.get('min_ndocs', 50), 
                    remove_duplicates=True,
                    pairing_policy=kargs.get('pairing_policy', 'random'),  

                    slice_policy=kargs.get('slice_policy', 'noop'),  # only for file ID purpose 
                    consolidate_lcs=kargs.get('consolidate_lcs', True),  # ordering important? 
                    cohort=cohort)
    assert df.shape[0] > 0
    print('markTSetByLCS> LCS labels ready (dim: %s)...' % str(df.shape))
    S = list(df['lcs'].values)
    lcsmap = analyzeLCS(D, lcs_set=S)  # which documents match with which LCS? lcs_sep=Pathway.lcs_sep
        
    # condition: LCS dataframe ready 
    assert len(lcsmap) > 0
    # nLDesired = 10  # we want only this many labels

    # [m1] same document could have multiple LCS labels
    # [m2] lcsmap has all LCS entries (big!)

    ### save LCS labels to labeled_seq file
    # use LCSs themselves as labels 
    # generate a list of labels matching D
    lcs_sep = Pathway.lcs_sep
    token_nomatch = Pathway.lcs_nomatch  # 'No_Match'
    lcsmapInv = inverse_lcsmap(); test_inverse_lcsmap()
   
    ### [code] once inverse lcsmap is obtained, one can resolve multiple LCS labels here

    # [policy] label_by_relative_uniqueness()
    L = label_by_consolidated_single_label() 
  
    # final verification
    # [test]
    lcs_label_stats()  # find number of labels

    # automatically load labeled_seq file e.g. condition_drug_labeled_seq-CKD.csv 
    label_document(L, overwrite=True) # labeling.write(L, cohort=cohort, seq_ptype=seq_ptype, **kargs) # [params] doctype='labeled', ext='csv'
    
    ts = save_lcs_tset(L, overwrite=True) # load if existed else create
    return ts

def analyzePrediagnosticSequence(codes, **kargs):
    """
    Find the sequence of codes that occur prior to the mention of target codes (codes)
    I.e. given a set of target codes (usually only one code, say 720.00), find their 
    prediagnostic sequence. 

    Usage Note
    ----------
    1. this is only useful when a specific cutpoint is known; what happens when we cannot pinpoint 
       whe location sigifying a key diagnosis associated with a disease? (e.g. 309.81 ~ PTSD)

    2. also see analyzeLCS() 

    Params
    ------
    codes: target codes

    """
    return apseq(codes, **kargs) 
def apseq(codes, **kargs):
    def normalize_input(x): 
        if isinstance(x, str): 
            x = x.split(kargs.get('sep', ' '))
        assert hasattr(x, '__iter__'), "Invalid input codes: %s" % x
        return x # a list of code strings
    def to_str(seq, sep=','): 
        return sep.join(str(e) for e in seq)
    def code_str(seq, sep='-'): # convert codes to file naming friendly format 
        s = to_str(seq, sep=sep) 
        # alternative: s = s.translate(None, string.punctuation)  # remove punctuations '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
        return s.replace('.', '')
    def time_to_diagnoses(sid, test_=False): # [params] tseqx, matched_index_set
        # use sid to index timestamps accordingly 
        if not tseqx: 
            print('time_to_diagnoses> no input timestamps > aborting ...')
            return None
        
        ret = {} # keys: ['time_to_first', 'time_to_last', 'days_elapsed', 'n_visits'] 

        # datetime.strptime("2017-08-11", "%Y-%m-%d")
        tseq = tseqx[sid]  # corresponds to the sid-th sequence
        t0 = datetime.strptime(tseq[0], "%Y-%m-%d")  # datetime object ready for arithmetic
        tseq_arr = np.array(tseq)
        tlist = []
        for idx in matched_index_set: 
            times = tuple(tseq_arr[idx])

            # keys: occurrence of the first code; values: occurrences of all input codes
            td0 = times[0]
            # tdict[t0] = times
            tlist.append((td0, times))  # times in string format

        tlist = sorted(tlist, key=lambda x: datetime.strptime(x[0], '%Y-%m-%d'))
        if test_: print('  + sorted times:\n%s\n' % tlist[:10])

        # time to first diagnosis
        td0 = datetime.strptime(tlist[0][0], "%Y-%m-%d") # first observed, first code
        tdf = datetime.strptime(tlist[-1][0], "%Y-%m-%d") # last observed, first code
        ret['time_to_first'] = (td0-t0).days   # (td0 - t0) ~> datetime.timedelta object
        ret['time_to_last'] = (tdf-t0).days
        ret['n_visits'] = len(dict(tlist))  # constraint: different dates
        ret['n_total_visits'] = len(tlist)  
        if test_: assert ret['n_total_visits'] >= ret['n_visits']
        ret['days_elapsed'] = ret['time_to_last']-ret['time_to_first']
        assert ret['days_elapsed'] >= 0

        # all questions answered? 
        if test_: assert len(set(time_dependent_fields)-set(ret.keys()))==0

        return ret
    def has_precodes(seq): # any of the prediagnostic codes occur in the pre-diag sequences (i.e. sequence up to first occurence of target)
        # precodes = kargs.get('precodes', [])
        if not precodes: 
            print('  + Warning: No pre-diagnostic codes given => set to False.')
            return False 
        
        n0 = len(set(precodes))
        n1 = len(set(precodes)-set(seq))
        return n0 > n1  # then a subset of precodes must be in 'seq'
    def time_precode_target(seq, tseq, tid=None): # sid: the sid-th document, need this to find corresponding timestamp
        assert len(seq) == len(tseq)
        if tid is None: tid = sd00  # first match (of target), first code
        # time elapsed (days) between first relevant pre-diagnostic code to the target
        i0, c0 = 0, '000.0'
        precode_set = set(precodes)
        for i, c in enumerate(seq): 
            if c in precode_set: 
                i0 = i; c0 = c
                break

        # time 
        assert tid > i0, "Target: %s must occur after pre-diag code: %s" % (codes[0], c0)
        t0 = datetime.strptime(tseq[i0], "%Y-%m-%d")  # datetime object for the first occurrence of any of the precodes
        tF = datetime.strptime(tseq[tid], "%Y-%m-%d")
        delta = (tF-t0).days
        return delta
    def eval_common_codes_prior(seq): # the most common codes before target
        commonCodesPrior.update(seq)
        return
    def eval_common_ngrams_prior(seq, max_length=4): # [note] cannot just operate on 'seqx' becaues each 'seq' may be different
        # length -> Counter: {(ngr, count)}
        commonNGramsPrior.update(seqAlgo.count_ngrams2([seq, ], min_length=1, max_length=max_length, partial_order=False)) # length => {(ngr, count)}
        return 
    def test_matched(seq, seq0, seqF, ithmatch): 
        print('  + this is the %d-th matches so far ...' % ithmatch)
        print('  + example seq (prior to FIRST mention, n=%d):\n%s\n' % (len(seq0), seq0))
        print('  + example seq (prior to LAST mention, n=%d):\n%s\n' % (len(seqF), seqF))
        print('  + lengths (to first mention): %d vs (to last): %d' % (len(pre0_seq), len(preF_seq)))
        return
    def summary_report(topn=10, max_length=4): # has to be the last statement of the outer function 
        n_total = sum(n_persons_batch)
        n_total_precodes = sum(n_persons_precodes_batch)
        n_total_precodes_last = sum(n_precodes_last_batch)
        r_precodes = n_total_precodes/(n_total+0.0)
        r_precodes_last = n_total_precodes_last/(n_total+0.0)

        print('tdoc.analyzer> Found %d eligible persons (grand total: %d) with target codes: %s' % (n_total, nGrandTotal, to_str(codes)))
        print('               + among all eligible, %d (r=%f) of them has precodes (prior to FIRST mention)' % (n_total_precodes, r_precodes))
        print('               + among all eligible, %d (r=%f) of them has precodes (prior to LAST mention)' % \
            (n_total_precodes_last, r_precodes_last))

        # most common codes 
        topn_codes = commonCodesPrior.most_common(topn)
        print('               ++ Top %d codes:\n%s\n' % (topn, topn_codes))

        # most common bigrams 
        # for i in range(1, max_length+1): 
        for length in [2, ]:  
            topn_ngrams = commonNGramsPrior[length].most_common(topn)
            print('               ++ Top %d %d-grams:\n%s\n' % (topn, length, topn_ngrams))
        return

    global gHasConfig   # an attempt to run this module without config package
    import collections
    # import seqAlgo
    # import seqReader as sr 
    # from datetime import datetime
    # import time 
    # import pandas as pd
    precodes = kargs.get('precodes', [])
    print('input> target codes:\n%s\ninput> prediagnostic codes (e.g. BP prior to SpA):\n%s\n' % (codes, precodes))

    docSources = kargs.get('ifiles', [])
    cohort_name = kargs['cohort'] if not docSources else 'n/a' # must provide cohort (to determine file names of the document sources)
    basedir = os.path.join(sys_config.read('DataExpRoot'), 'sequencing') if gHasConfig else os.getcwd()
    if not docSources: 
        docSources = TDoc.getPathsByCohort(cohort)  # source documents are all in default dir: tpheno/data-exp 

    # [note] in order to use readDocFromCSV, need to create .csv from .dat first (see sr.readDocToCSV())
    # ret = readDocFromCSV(cohort=cohort_name, ifiles=docSources, basedir=basedir)
    codes = normalize_input(codes)

    # [output]
    header = ['target', 'time_to_first', 'latency', 'n_visits', 'sequence_to_first', 'has_precodes', 'time_to_last', 'days_elapsed', ]   # time_to_diagnosis: really is time to the FIRST diagnosis
    adict = {h:[] for h in header} # [output]
    # condition_drug_seq-group-N.dat where N = 1 ~ 10 
    # [note] query document source one by one since they are large
    nD = 10
    allow_partial_match = False
    n_total_matched = 0
    time_dependent_fields = ['time_to_first', 'time_to_last', 'days_elapsed', 'n_visits', ]  # answered by time_to_diagoses operation
    nGrandTotal = 0  # number of perons in the entire sequenced data (DB)
    n_persons_batch, n_persons_precodes_batch, n_precodes_last_batch = [], [], []
    lenToFirst_precodes_distr, lenToLast_precodes_distr = [], []
    commonCodesPrior = collections.Counter()
    commonNGramsPrior = {}  # length -> Counter: {(ngr, count)}
    for ithdoc, fpath in enumerate(docSources):
        # ret = TDoc.loadGeneric(filenames=[fname, ]) # prefix: tpheno/data-exp/sequencing

        # normalize input path
        prefix, fname = os.path.dirname(fpath), os.path.basename(fpath) 
        if prefix is None: prefix = basedir # default is to analyze sequencing dir 
        fpath = os.path.join(prefix, fname)
        assert os.path.exists(fpath), "Invalid path: %s" % fpath
                
        # read timed coding sequences
        # [note] load + parse vs TDoc.loadGeneric() does not include parsing (coding sequences require some interpretations)
        #        if no_throw <- True, then skip visit segments with parsing errors (i.e. cannot separate time)
        seqx, tseqx = sr.readTimedDocPerPatient(cohort='n/a', ifiles=[fname, ], inputdir=basedir, no_throw=True) 
        assert not tseqx or len(seqx) == len(tseqx)

        # hasLabel = True if len(labels) > 0 else False
        div(message='Found %d sequences in %d-th src: %s | has timestamp? %s,' % (len(seqx), ithdoc, fname, len(tseqx)>0))
        nGrandTotal += len(seqx)
        # analyze this file    
        # n_persons: number of persons matching target codes 
        # n_persons_precodes: matched targets and contain predignositc codes (prior to first mention)
        # n_precodes_last: matched targets and contain pre-diag codes (prior to LAST mention)
        n_persons = n_persons_precodes = n_precodes_last = 0  
        for i, seq in enumerate(seqx): 
            q, r = codes, seq
            if i < 10: assert isinstance(r, list)

            # is it matched? matched positions 
            matched_index_set = seqAlgo.traceSubsequence3(q, r)
            tMatched = True if len(matched_index_set) > 0 else False            

            # find the time of the first mention (of the input coding seqments, ordering important)
            # number of occurrences (of input codes)
            # compute time to diagnosis
            # number of mentions corresponding to different dates/visits (assumption)

            if tMatched: # matched => found target/input codes in the sequence
                sd00 = matched_index_set[0][0] # position of first match (in the entire sequence), index of the first code in codes 
                sf = matched_index_set[-1][0] # last match, first code

                subseq = to_str(seq[:sd00+1])

                # query ['time_to_first', 'time_to_last', 'days_elapsed', 'n_visits']
                tdict = time_to_diagnoses(sid=i, test_=(i % 5==0)); assert tdict is not None, "No timestamps."
                adict['target'].append(to_str(codes)) 
                adict['sequence_to_first'].append(subseq)
                for h in time_dependent_fields: 
                    adict[h].append(tdict[h])
                n_persons += 1  # per source
                n_total_matched += 1  # overall 

                # [query] pre-diagnostic codes analysis
                pre0_seq = seq[:sd00]; lenToFirst_precodes_distr.append(len(pre0_seq)) # [test]
                if has_precodes(pre0_seq):  # params: kargs['precodes'] 
                    adict['has_precodes'].append(1)
                    n_persons_precodes += 1 
                    
                    # [query] time elapsed from first precode to target? 
                    delta = time_precode_target(seq, tseq=tseqx[i], tid=sd00) # seq, timestamp, tid: target position
                    adict['latency'].append(delta)  # unit: days

                else: 
                    adict['has_precodes'].append(0)
                    adict['latency'].append(-1) # this will be removed
                    
                preF_seq = seq[:sf]; lenToLast_precodes_distr.append(len(preF_seq)) # [test]
                if has_precodes(preF_seq):
                    n_precodes_last += 1    # number of persons | matched + has pre-diagnostic codes before last mention (of target) 

                # most common codes prior to targets
                # eval_common_codes_prior(pre0_seq) # updates commonCodesPrior 714.0, 715.0
                commonCodesPrior.update(pre0_seq) # don't count target

                # most common n-grams prior to targets
                eval_common_ngrams_prior(pre0_seq, max_length=4)
                    
                # [test]
                if n_total_matched % 5 == 0: 
                    test_matched(seq, seq0=pre0_seq, seqF=preF_seq, ithmatch=n_total_matched)
        ### end foreach sequence in 'ithdoc'-th source
        
        # collect number-of-matches-statistics 
        n_persons_batch.append(n_persons)  
        n_persons_precodes_batch.append(n_persons_precodes) 
        n_precodes_last_batch.append(n_precodes_last)
        div(message='End of %d-th sequencing file: Found %d candidates, in which %d contain precodes.' % (ithdoc, n_persons, n_persons_precodes))

    # save report 
    df = DataFrame(adict, columns=header); N0 = df.shape[0]; assert N0 == sum(n_persons_batch)
    
    # save only those with precodes (prior to first mention)? 
    # [filter]
    df = df.loc[df['has_precodes']==1]; NHasPC = df.shape[0]; assert NHasPC == sum(n_persons_precodes_batch)
    df.drop(['has_precodes'], axis=1, inplace=True)
    print("  + n_persons: %d => n_persons(has precodes): %d" % (N0, NHasPC)) 

    # [note] tpheno/seqmaker/data/SpA/prediag_analysis_C7200.csv
    fpath = os.path.join(seqparams.getCohortDir(cohort='SpA'), 'prediag_analysis_C%s.csv' % code_str(codes, sep='-'))
    df.to_csv(fpath, sep='|', index=False, header=True)  
    print("  + IO: saved summary to:\n%s\n" % fpath)
    
    # total persons found 
    assert len(n_persons_batch) == len(docSources)
    summary_report() # has to be the last statement
                
    return 

def t_subsequence(**kargs):
    def get_sequencing_files(n=10): 
        fpat = 'condition_drug_timed_seq-group-%d.dat'
        files = []
        for i in range(1, n+1):
            # files.append(fpat % i)
            # return files
            yield os.path.join(basedir, fpat % i)
    def get_bp_codes(): 
        bp_codes = ['724', '724.0', '724.00', '724.01', 724.02, 724.03, 724.09, 724.1, 724.2, 724.3, 724.4, 724.5, 724.6, 724.8, 724.9]
        return [str(e) for e in bp_codes]

    import seqAlgo
    basedir = os.path.join(sys_config.read('DataExpRoot'), 'sequencing') if gHasConfig else os.getcwd()

    bp_codes = get_bp_codes()
    target_codes = ['720.0', ]

    # analyze 720.0, SaP
    apseq(target_codes, precodes=bp_codes, ifiles=list(get_sequencing_files()), cohort='n/a')

    return

def t_analyze_lcs(**kargs):
    def load_lcs(): # lcs_type
        ltype = kargs.get('lcs_type', 'global')  # global: Common LCSs defined in global scope (as opposed to cluster or local scopes)
        lcs_select_polcy = kargs.get('lcs_policy', 'freq')
        # seq_ptype = kargs.get('seq_ptype', 'regular')
        identifier = kargs.get('identifier', seq_ptype)
        return Pathway.load(cohort, scope=ltype, policy=lcs_select_polcy, suffix=identifier, dir_type='pathway')
    def verify_doc(): # params: D, L, T
        assert len(D) > 0, "No input documents found (cohort=%s)" % cohort
        x = random.randint(0, len(D)-1)  # nDoc
        assert isinstance(D[x], list), "Invalid input D[%d]:\n%s\n" % (x, D[x])
        assert len(T) == len(D), "inconsistent number of timestamps (nT=%d while nD=%d)" % (len(T), len(D))
        assert len(L) == len(D), "inconsistent number of labels (nL=%d while nD=%d)" % (len(L), len(D))
        print('markTSetByLCS> Found %d documents (nT=%d, nL=%d, cohort=%s)' % (len(D), len(T), len(L), cohort))
        return 
    def verify_tset(): # params: D, T, L
        # d2v_method = kargs.get('d2v_method', seqparams.D2V.d2v_method) # pv-dm2
        # seq_ptype = kargs.get('seq_ptype', 'regular') 
        nD = len(D)
        ts = TSet.loadLCSTSet(cohort=cohort, d2v_method=d2v_method, seq_ptype=seq_ptype)
        if ts is None: 
            ts = TSet.loadCombinedTSet(cohort=cohort, d2v_method=d2v_method, seq_ptype=seq_ptype) 
            assert ts is not None, "No relevant training set found (cohort=%s, d2v=%s, ctype=%s)" % (cohort, d2v_method, seq_ptype)
        assert ts.shape[0] == nD, "Training set size: %d <> nDoc: %d" % (ts.shape[0], nD)
    def label_document(L):
        # pattern: condition_drug_<file_type>-<cohort>.csv 
        # fname = TDoc.getName(cohort=cohort, doctype='labeled', ext='csv')  # [params] doc_basename/'condition_drug'
        prefix = kargs.get('basedir', TDoc.prefix)

        # header: ['sequence', 'timestamp', 'label', 'label_lcs']
        fpath = TDoc.getPath(cohort=cohort, seq_ptype=seq_ptype, doctype='labeled', ext='csv', basedir=prefix)  # usually there is one file per cohort  
        assert os.path.exists(fpath), "labeled docuemnt source should have been created previously (e.g. by processDocuments)"
        # assert len(fpaths) > 0, "Could not find any document sources associated with cohort=%s" % cohort

        print('markTSetByLCS.label_document> Now filling in the labels to %s (? n_paths=1)' % fpath)
        df_src = pd.read_csv(fpath, sep='|', header=0, index_col=False, error_bad_lines=True) # ['sequence', 'timestamp', 'label']
        print('info> header: %s' % list(df_src.columns))

        lcs_col = 'label_lcs'; assert lcs_col in TDoc.fLabels, "Invalid label attribute: %s" % lcs_col
        if lcs_col in df_src.columns: 
            print('  + Warning: %s already existed.' % lcs_col)
        df_src[lcs_col] = L
        df_src.to_csv(fpath, sep='|', index=False, header=True) # ['sequence', 'timestamp', 'label', 'label_lcs']
        print('  + IO: saved label_seq (cohort=%s) to:\n%s\n' % (cohort, fpath))
        
        return df_src

    from tset import TSet # mark training data
    from seqparams import Pathway # set pathway related parameters, where pathway refers to sequence objects such as LCS
    from tdoc import TDoc  # labeling info is also maintained in document sources of doctype = 'labeled'

    cohort = kargs.get('cohort', 'CKD')  # 'PTSD'
    seq_ptype = kargs.get('seq_ptype', 'regular') 
    d2v_method = kargs.get('d2v_method', seqparams.D2V.d2v_method)
    verify_labeled_file = True

    # D: corpus, T: timestamps, L: class labels
    # load + transform + (labeled_seq file)
    D, L, T = processDocuments(cohort=cohort, seq_ptype=seq_ptype, 
            predicate=kargs.get('predicate', None),
            simplify_code=kargs.get('simplify_code', False), 
            ifiles=kargs.get('ifiles', []), save_=True)  # also save a copy of transformed document (with labels)
    print('... document processing completed (load+transform+label)... ')

    # [test]
    verify_doc() 
    # verify_tset()

    # the following can also be achieved by calling labeling.labelByLCS() ... 
    # [note] ... but this introduces mutual dependency since labelByLCS <- deriveLCS

    # LCS params
    slice_policy = kargs.get('slice_policy', 'noop')  # slice operations {'noop', 'prior', 'posterior', } 

    ### Version 1
    # df = load_lcs()  # try loading first
    # tNewLCS = False
    # if df is None:  
    #     [note] 125000 allows for comparing 500+ documents pairwise
    #     df: ['lcs', 'length', 'count', 'n_uniq']
    #     max_n_pairs: 125000
    #     df = deriveLCS(D=D, topn_lcs=kargs.get('topn_lcs', 20), 
    #             min_length=kargs.get('min_length', 5), max_length=kargs.get('max_length', 15), 
    #             max_n_pairs=kargs.get('max_n_pairs', 10000), remove_duplicates=True, 
    #             pairing_policy=kargs.get('pairing_policy', 'random'), 
    #             cohort=cohort)

    #     tNewLCS = True 
    #     print('... LCS analysis completed ...')

    ### Version 2 saves larger amount of LCSs prior to filtering
    df = deriveLCS2(D=D, 
                    topn_lcs=kargs.get('topn_lcs', 5000), 
                    min_length=kargs.get('min_length', 5), max_length=kargs.get('max_length', 15), 
                    max_n_pairs=kargs.get('max_n_pairs', 250000), 
                    remove_duplicates=True, 
                    pairing_policy=kargs.get('pairing_policy', 'random'), 
                    consolidate_lcs=kargs.get('consolidate_lcs', True),  # ordering important? used for LCS (surrogate) labels 
                    lcs_policy=kargs.get('lcs_policy', 'df'), # lcs filtering policy

                    ### file ID paramters
                    seq_ptype=seq_ptype,
                    slice_policy=slice_policy,  # same as the param in stratifyDocuments; only for file ID purpose
                    cohort=cohort) # used for file ID and op such as filter_by_anchor()

    return

def t_classify0(**kargs):
    def make_file_id(): # [classifier_name, cohort, d2v_method, seq_ptype, suffix]
        identifier = kargs.get('identifier', None)
        if identifier is None: 
            d2v_method = vector.D2V.d2v_method
            identifier = seqparams.makeID(params=[get_clf_name(), cohort_name, d2v_method, 
                seq_ptype, kargs.get('suffix', None)])  # null characters and None will not be included
        return identifier
    def get_clf_name(): # can be called before or after model selection (<- select_classifier)
        # classifier = kargs.get('classifier', None)
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
    def get_tset(fname=None, dir_type='combined'):  # go to seqmaker/data/PTSD/combined folder by default
        if fname is None:  
            fname = TSet.getName(cohort=cohort, d2v_method=d2v_method, index=kargs.get('index', None), 
                            seq_ptype=seq_ptype, suffix=kargs.get('suffix', None))

        # 1. given file name
        #    'dir_type' <- 'combined' by default
        ts = TSet.loadLCSTSet(cohort=cohort, d2v_method=d2v_method, seq_ptype=seq_ptype, fname=fname)

        # 2. without fname 
        # ts = TSet.loadLCSTSet(cohort=cohort, d2v_method=d2v_method, seq_ptype=seq_ptype, suffix=kargs.get('suffix', None))
        return ts
    def tset_stats(ts, f_label='target'):
        uL = [1, ]
        try: 
            uL = ts[f_label].unique()  
        except: 
            pass 
        n_classes = len(uL) 
        if n_classes > 1:  
            print('  + Stats: n_docs: %d, n_classes:%d | cohort: %s' % (ts.shape[0], n_classes, cohort))
        else: 
            print('  + Stats: n_docs: %d, n_classes:? (no labeling info) | cohort: %s' % (ts.shape[0], cohort)) 
        return  

    import seqClassify as sc
    from tset import TSet
    import vector

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

    cohort = kargs.get('cohort', 'PTSD')
    seq_ptype = kargs.get('seq_ptype', 'diag') 
    d2v_method = kargs.get('d2v_method', vector.D2V.d2v_method)
    
    # diag only: tset-IDdiag-pv-dm2-Llcs-GPTSD.csv, 'tset-IDdiag-pv-dm2-Llcs-prior-GPTSD.csv' 
    # regular:   tset-IDregular-pv-dm2-Llcs-GPTSD.csv
    fname = 'tset-IDregular-pv-dm2-Llcs-GPTSD.csv' # 'tset-IDdiag-pv-dm2-Llcs-GPTSD.csv'
    ts = get_tset(fname) # 'tset-IDdiag-pv-dm2-Llcs-prior-GPTSD.csv'
    tset_stats(ts)

    classifier = RandomForestClassifier(n_estimators=100)  # supports multiclass 
    # random_state = np.random.RandomState(0)
    # clf = svm.SVC(kernel='linear', probability=True, random_state=random_state)

    # plot file id: (cohort_name, d2v_method, seq_ptype, suffix)
    file_id = TSet.getFileId(fname) # 'IDdiag-pv-dm2-Llcs-prior-GPTSD' # 'ID%s-%s-Llcs-prior-G%s' % (seq_ptype, d2v_method, cohort_name)
    sc.multiClassEvaluate(ts=ts, classifier=classifier, identifier=file_id) 

    return

def t_classify0a(**kargs): 
    """

    Memo
    ----
    1. Example regular training sets: 
        a. trained with labeled data only (cohort=CKD)
           tpheno/seqmaker/data/CKD/combined/tset-n0-IDregular-pv-dm2-U-GCKD.csv
        b. labeled + augmented data
           tpheno/seqmaker/data/CKD/combined/tset-n0-IDregular-pv-dm2-A-GCKD.csv
    """
    def config(): 
        scls.tsHandler.config(cohort=kargs.get('cohort', 'CKD'), 
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
    def do_slice(): 
        if not kargs.has_key('slice_policy'): return False
        if kargs['slice_policy'].startswith(('noop', 'reg', 'compl', )):  # effective operations 
            return False 
        return True
    def secondary_id(): # attach extra info in suffix: {seq_ptype, slice_policy, label_id}
        ctype = kargs.get('seq_ptype', 'regular')
        suffix = kargs.get('suffix', ctype) # vector.D2V.d2v_method
        if do_slice(): suffix = '%s-%s' % (suffix, kargs.get('slice_policy', 'noop'))
        if kargs.get('consolidate_lcs', True): suffix = '%s-%s' % (suffix, 'iso') # LCSs up to permutations consider as the same? 

        label_id = kargs.get('label', None)  # stratified dataset by labels
        if label_id is not None: suffix = '%s-L%s' % (suffix, label_id)
    
        return suffix 
    def load_tset(fpath=None, index=0):
        ts = None
        if fpath is not None and (os.path.exists(fpath) and os.path.getsize(fpath) > 0): 
            ts = pd.read_csv(fpath, sep=TSet.sep, header=0, index_col=False, error_bad_lines=True)
        else: 
            ### load default
            # suffix = secondary_id()
            # load + suffix <- 'Flcs'

            # e.g. tset-IDregular-pv-dm2-Flcs-GCKD.csv
            ts = TSet.loadLCSFeatureTSet(cohort=kargs.get('cohort', 'CKD'), 
                        d2v_method=kargs.get('d2v_method', vector.D2V.d2v_method), 
                        seq_ptype=seqparams.normalize_ctype(kargs.get('seq_ptype', 'regular'))) # use default 'Flcs' keyword

            # ts = tsHandler.load(index)
        assert ts is not None and not ts.empty, "t_classify_lcs> Null training set."
        print('  + LCS-feature training set dim=%s' % str(ts.shape))
        return ts
    def relabel(): 
        # CKD stage classes
        lmap = {}
        lmap['Others'] = ['CKD G1-control', 'CKD G1A1-control', 'Unknown', ]  # ... control data 
        lmap['CKD Stage 5'] = ['ESRD after transplant', 'ESRD on dialysis', ]
        lmap['CKD Stage 3'] = ['CKD Stage 3a', 'CKD Stage 3b', ]
        print('  + Relabeling the data set according to the following map:\n%s\n' % lmap)
        return lmap 
    def select_features_lasso(n=100):  # <- ts
        print('  ... common LCSs ~ LassoCV ... ')
        active_lcsx, positions = evaluate.select_features_lasso(ts, n_features=n)
        for i, lcs in enumerate(active_lcsx): 
            print('  + [#%d, pos=%d] %s' % ((i+1), positions[i], lcs))
        return (active_lcsx, positions)
    def select_features_l2(n=100):
        print('  ... common LCSs ~ logistic regression ...') 

        # [todo]
        return
    def get_logistic(): 
        clf = LogisticRegression(penalty='l2', class_weight='balanced', solver='sag')
        clf.set_params(multi_class='multinomial', solver='saga', max_iter=1000)
        return clf

    import evaluate, vector
    import seqClassify as sclf
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
    from sklearn.linear_model import LogisticRegression, Lasso, LassoCV

    # [params]
    #    set is_augmented to True to use augmented training set
    cohort_name = kargs.get('cohort', 'CKD')
    seq_ptype = kargs.get('seq_ptype', 'regular')
    sclf.tsHandler.config(cohort=cohort_name, seq_ptype=seq_ptype, is_augmented=False) # 'regular' # 'diag', 'regular'
    mode = 'multiclass'  # values: 'binary', 'multiclass'
    param_grid = None

    focusedLabels = None # None to include ALL  # focus(), merge()
    classesOnROC = ['CKD Stage 1', 'CKD Stage 3', 'CKD Stage 5', ] # assuming that CKD Stage 3a and 3b are merged into CKD Stage 3

    # [todo] loop through cartesian product of seq_ptype and d2v_method? 
    ts = load_tset()
    indices = None
    # fset, indices = select_features_lasso(n=200)  # params: max_iter (for LassoCV), meta_fields (use TSet's default)
    
    #        params: classifier, classifier_name
    sclf.multiClassEvaluate(ts=ts, cohort=cohort_name, seq_ptype=seq_ptype, # not needed 
        classifier=get_logistic(), 
        classifier_name=None,   # eithr provide classifer or its name {'l2_logistic', 'l1_logistic'}
        support=indices,  # active feature positions/indices
        focused_labels=focusedLabels, 
        roc_per_class=classesOnROC,
        param_grid=param_grid, 
        label_map=relabel(), 

        # use suffix or identifier to distinguish different classificaiton tasks; set to None to use default
        # if identifier is given, it'll subsume all the other parameter-related file id including suffix
        suffix='Flcs', identifier=None)  
    
    return

# [todo]
def t_lcs_timeseries(**kargs):
    """
    Assuming target LCSs are given, find corresponding occurences of these medical codes in terms of times. 


    Memo
    ----
    1. Use the following modules: 
        deriveLCS2
        analyzeMCS(D, T, lcs_set, **kargs)

    """
    delimit_lcs = ' '

    # [input] lcsmap: LCS -> 
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

def t_lcs(**kargs): 
    """


    Memo
    ----
    makeLCSTSet common parameters: 
                topn_lcs=5, min_length=6, max_length=25, min_ndocs=250, 
                max_n_pairs=250000, pairing_policy='random', 
                load_lcs=False, 
                slice_policy='noop', slice_predicate=None, 
                consolidate_lcs=True
    processDocument 
        D, L, T = processDocuments(cohort=cohort, seq_ptype=seq_ptype, 
               predicate=kargs.get('predicate', None),
               simplify_code=kargs.get('simplify_code', False), 
               ifiles=kargs.get('ifiles', []), 

               # slice operations {'noop', 'prior', 'posterior', }
               slice_policy=kargs.get('slice_policy', 'noop'), 
               slice_predicate=kargs.get('slice_predicate', None), 
               cutpoint=kargs.get('cutpoint', None),
               inclusive=True)  
    """
    def process_label(l): 
        # remove spaces e.g. CKD Stage 4
        return ''.join(str(e) for e in l.split())
    def policy_relabel(): 
        # CKD stage classes
        lmap = {}
        lmap['Others'] = ['CKD G1-control', 'CKD G1A1-control', 'Unknown', ]  # ... control data 
        lmap['CKD Late'] = ['ESRD after transplant', 'ESRD on dialysis', 'CKD Stage 5']
        lmap['CKD Middle'] = ['CKD Stage 3a', 'CKD Stage 3b', 'CKD Stage 4']
        lmap['CKD Early'] = ['CKD Stage 1', 'CKD Stage 2', ]
        return lmap
    def stratum_stats(units): # <- units
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
        print('  + number of labels: %d' % n_labels)
        return
    def sample_dict(lmap, n=5):  # <- Di, Ti 
        lmap2 = sampling.sample_dict(lmap, n_sample=n)  # select n docs 
        for di, entry in lmap2.items(): 
            print("sample_dict> Document #%d:\n%s\n" % (di, Di[di][:5]+[' ... ']+Di[di][-5:]))  # first n ... last n
            for lcs, alist in entry.items(): 
                print("  + LCS: %s" % lcs)
                print("  + n_occurrence(%d):\n%s\n" % (len(alist), alist))
        return
    def get_global_cohort_dir(name='CKD'): 
        return seqparams.getCohortGlobalDir(name) # sys_config.read('DataExpRoot')/<cohort> 

    from sampler import sampling 
    ### parameters 
    # A. Documents 
    simplifyCode = False
    cohort_name = kargs.get('cohort', 'CKD')  # PTSD for experimenting on LCS-based labeling
    seq_ptype = seqparams.normalize_ctype('diag') # opt: 'regular', 'diag', 'med'
    slice_policy = kargs.get('slice_policy', 'noop')

    # A.1 source 
    inputdir = get_global_cohort_dir(name='CKD0')  # CKD0: smaller annotated cohort n=2833
    ifiles = kargs.get('ifiles', ['condition_drug_labeled_seq-CKD.csv', ])

    # B. LCS 
    topNLCS = 200
    minLength, maxLength = 5, 10
    maxNPairs = 250000 
     
    # C. Labeling 
    lmap = policy_relabel()
    # label -> entry: Di, Ti, Li
    stratified = stratifyDocuments(cohort=cohort_name, seq_ptype=seq_ptype,
                    predicate=kargs.get('predicate', None), 
                    simplify_code=kargs.get('simplify_code', simplifyCode), 

                    # source
                    inputdir=inputdir, 
                    ifiles=ifiles, 

                    # relabeling operation 
                    label_map=lmap, 

                    # slice operations {'noop', 'prior', 'posterior', }
                    slice_policy=slice_policy, 
                    slice_predicate=kargs.get('slice_predicate', None), 
                    cutpoint=kargs.get('cutpoint', None),
                    inclusive=True)
    stratum_stats(stratified) # [test]
    labels = stratified.keys() # documents = D; timestamps = T
    n_labels = len(set(labels))

    nD = nT = 0
    for label, entry in stratified.items():
        Di = entry['sequence']
        Ti = entry['timestamp']
        Li = entry['label']
        nD += len(Di)
        nT += len(Ti)
        print('t_lcs> searching LCSs for label=%s | N=%d' % (label, len(Di)))

        minLCSDocFreq = min(len(Di)/10, 3)

        # ['length', 'lcs', 'n_uniq', 'count', 'df', ] 
        df = deriveLCS(D=Di, label=process_label(label), 
                    topn_lcs=kargs.get('topn_lcs', topNLCS), 
                    min_length=kargs.get('min_length', minLength), max_length=kargs.get('max_length', maxLength), 
                    min_ndocs=kargs.get('min_ndocs', minLCSDocFreq),  # depends on the size of D
                    max_n_pairs=kargs.get('max_n_pairs', maxNPairs),  
                    remove_duplicates=True,
                    pairing_policy=kargs.get('pairing_policy', 'random'),  # {'longest_first', 'random', }
                    consolidate_lcs=kargs.get('consolidate_lcs', True),  # ordering important? used for LCS (surrogate) labels
                    lcs_type='local', 
                    lcs_policy=kargs.get('lcs_policy', 'df'), # lcs filtering policy

                    # file ID paramters
                    seq_ptype=seq_ptype, 
                    slice_policy=slice_policy,  # same as the param in stratifyDocuments; only for file ID purpose 
                    cohort=cohort_name) # used for file ID and op such as filter_by_anchor()

        # df = deriveNGram(Di, 
        #                 topn=100,   # retain only topn n-grams for each length (n)
        #                 min_length=1, max_length=8, partial_order=False,
        #                 ng_type='local',
        #                 ng_policy=kargs.get('ng_policy', 'freq'), 
        #                 seq_ptype=seq_ptype)

        print('  + label=%s > %d LCSs' % (label, df.shape[0]))
        lcs_set = df['lcs'].values
        ret = analyzeMCS(Di, Ti, lcs_set)  # alias: analyzeLCS2
        lcsColorMap, lcsTimeMap = ret['color'], ret['time']
        sample_dict(lcsColorMap); sample_dict(lcsTimeMap)

        # plot LCS time series

    return

def t_lcs2(**kargs):
    """
    Similar to t_lcs() but this template function gathers the frequent LCS set in the global scope
    and subsequently match them with the data within each stratum/class labels to determine local LCSs. 
    """
    def get_global_cohort_dir(name='CKD'): 
        return seqparams.getCohortGlobalDir(name) # sys_config.read('DataExpRoot')/<cohort> 
    def policy_relabel(): 
        # CKD stage classes
        lmap = {}
        lmap['Others'] = ['CKD G1-control', 'CKD G1A1-control', 'Unknown', ]  # ... control data 
        lmap['CKD Late'] = ['ESRD after transplant', 'ESRD on dialysis', 'CKD Stage 5']
        lmap['CKD Middle'] = ['CKD Stage 3a', 'CKD Stage 3b', 'CKD Stage 4']
        lmap['CKD Early'] = ['CKD Stage 1', 'CKD Stage 2', ]
        return lmap 
    def precondition(i=0):
        print('\n...... Trial #%d (cohort=%s) ......\n' % (i, lcsHandler.cohort))
        if maxNDocs: 
            print('precondition> sample %d docments ...' % maxNDocs)
        print('  + minDocLength: %d, min LCS Document Freq: %d' % (minDocLength, minLCSDocFreq))
        return
    def stratum_stats(units): # <- units
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
        print('  + number of labels: %d' % n_labels)
        return
    def stratify_docs(inputdir=None, lmap=None): # params: cohort
        ### load + transfomr + (ensure that labeled_seq exists)

        # use the derived MCS set (used to generate d2v training data) as the source
        src_dir = get_global_cohort_dir() if inputdir is None else inputdir # orginal source: get_global_cohort_dir()
        assert os.path.exists(src_dir), "Invaild MCS source dir:\n%s\n" % src_dir
        ifiles = kargs.get('ifiles', ['condition_drug_labeled_seq-CKD.csv', ])
        ctype =  kargs.get('seq_ptype', lcsHandler.ctype)  # kargs.get('seq_ptype', 'regular')
        cohort = kargs.get('cohort', lcsHandler.cohort)  # 'PTSD'
        
        minDocLength = kargs.pop('min_ncodes', 10)
        tSimplified = kargs.get('simplify_code', lcsHandler.is_simplified)
        
        if lmap is None: lmap = policy_relabel() 
        stratified = stratifyDocuments(cohort=cohort, seq_ptype=ctype,
                    predicate=kargs.get('predicate', None), 
                    simplify_code=kargs.get('simplify_code', tSimplified), 

                    # source
                    inputdir=src_dir, 
                    ifiles=kargs.get('ifiles', []), 
                    min_ncodes=minDocLength, 

                    # relabeling operation 
                    label_map=lmap, 


                    # slice operations {'noop', 'prior', 'posterior', }
                    slice_policy=lcsHandler.slice_policy, 
                    slice_predicate=kargs.get('slice_predicate', None), 
                    cutpoint=kargs.get('cutpoint', None),
                    inclusive=True)

        ### subsampling
        # maxNDocs = kargs.get('max_n_docs', None)
        # if maxNDocs is not None: 
        #     nD0 = len(D)
        #     D, L, T = sample_docs(D, L, T, n=maxNDocs)
        #     nD = len(D)

        nD = nT = 0
        for label, entry in stratified.items():
            Di = entry['sequence']
            Ti = entry['timestamp']
            # Li = entry['label']
            nD += len(Di)
            nT += len(Ti)

        # Dl = labelize(D, label_type='doc', class_labels=L)  # use class labels for document label prefixing
        print('stratify_docs> nD: %d | cohort=%s, ctype=%s, simplified? %s' % (nD, cohort, ctype, tSimplified))
        return stratified
    def get_tset_dir(): # derived MDS 
        # params: cohort, dir_type
        tsetPath = TSet.getPath(cohort=kargs.get('cohort', lcsHandler.cohort), dir_type=kargs.get('dir_type', 'combined'))  # ./data/<cohort>/
        print('t_lcs2> training set dir: %s' % tsetPath)
        return tsetPath 
    def get_global_cohort_dir(): # source MDS
        return seqparams.getCohortGlobalDir(kargs.get('cohort', lcsHandler.cohort)) # sys_config.read('DataExpRoot')/<cohort> 

    import gc
    from sampler import sampling   
    import seqConfig as sq
    from seqConfig import lcsHandler

    ### parameters 
    
    # A. Documents (use lcsHandler to enforce file naming consistency)
    # file naming 
    userFileID = meta = None
    ctype = seqparams.normalize_ctype('diag')
    sq.sysConfig(cohort=kargs.get('cohort', 'CKD'), seq_ptype=ctype, 
        lcs_type='global', lcs_policy='df', 
        consolidate_lcs=True, 
        slice_policy=kargs.get('slice_policy', 'noop'), 

        simplify_code=kargs.get('simplify_code', False), 
        meta=userFileID)

    minDocLength = 10  # only retain documents with at least 10 codes
    maxNDocs = 10000

    # A.1 source 
    # inputdir = get_global_cohort_dir(name=lcsHandler.cohort)  # CKD0: smaller annotated cohort n=2833
    # ifiles = kargs.get('ifiles', ['condition_drug_labeled_seq-CKD.csv', ])

    # B. LCS 
    topNLCS = 5000
    minLength, maxLength = 5, 15
    maxNPairs = 250000 
    minLCSDocFreq = 5

    nTrials = 0

    # if kargs.get('overwrite_lcs', False):
    #     df0 = lcsHandler.load_lcs() 

    for i in range(nTrials): 
        precondition(i)
        df, summary = docToLCS(lcsHandler.cohort, seq_ptype=lcsHandler.ctype, 
                                topn_lcs=topNLCS, 
                                min_length=kargs.get('min_length', minLength), max_length=kargs.get('max_length', maxLength), 
                                min_ndocs=kargs.get('min_ndocs', minLCSDocFreq),  # mininum number of docs in which the LCS candiates should be present

                            # document selection (e.g. min length)
                            max_n_docs=kargs.get('max_n_docs', maxNDocs),  # subsampling documents to compute LCSs, 
                            min_ncodes=kargs.get('min_ncodes', minDocLength), 
 
                            consolidate_lcs=lcsHandler.consolidate_lcs,  # ordering important? used for LCS (surrogate) labels 
                            lcs_policy=lcsHandler.lcs_policy, # lcs filtering policy
                            slice_policy=lcsHandler.slice_policy, 

                            # overwrite_lcs=kargs.get('overwrite_lcs', True), # if True, the re-compute LCSs
                            load_lcs=False, # False => recompute a new set of LCS
                            include_docs=False, 
                            time_analysis=False,  # analysiz occurrences of LCS elements in time

                            meta=lcsHandler.meta) # include source in 'summary' in order to verify document property; expensive for large corpus!
        print('  + LCS computaiton complete => size of df=%d' % df.shape[0])
    df = None; gc.collect()

    # load lcs set
    dfG = lcsHandler.load_lcs()
    lcsCandidates = dfG['lcs'].values
    print('  + found %d LCS canidates ...' % dfG.shape[0])

    # C. Labeling 

    # label -> entry: Di, Ti, Li
    # stratified = stratify_docs(inputdir=None)
    # stratum_stats(stratified) # [test]
    # labels = stratified.keys() # documents = D; timestamps = T
    # n_labels = len(set(labels))
    # print('  + found %d unique labels' % n_labels)

    # nD = nT = 0
    # for label, entry in stratified.items():
    #     Di = entry['sequence']
    #     Ti = entry['timestamp']
    #     Li = entry['label']
    #     nD += len(Di)
    #     nT += len(Ti)
    #     print('t_lcs2> searching LCSs for label=%s | N=%d' % (label, len(Di)))

    #     # minLCSDocFreq = min(len(Di)/10, 3)

    #     # ['length', 'lcs', 'n_uniq', 'count', 'df', ] 
    #     # df = dfG  # deriveLCS(...)

    #     lcs_set = lcsCandidates
    #     ret = analyzeMCS(Di, Ti, lcs_set)  # alias: analyzeLCS2
    #     lcsColorMap, lcsTimeMap = ret['color'], ret['time']
    #     sample_dict(lcsColorMap); sample_dict(lcsTimeMap)

        # plot LCS time series

    return 

def t_lcs2a(**kargs):
    return t_make_lcs_fset(**kargs) 
def t_make_lcs_fset(**kargs):
    """
    Assuming that the global feature set has been determined (see t_lcs2), 
    construct the (sparse) feature set based on these frequenct LCS patterns.

    """
    def summary(): 
        # if X is None: return
        nrow, ncol = X.shape[0], X.shape[1]
        print('  + X (%d by %d) type=%s' % (nrow, ncol, type(X)))

        nL = len(np.unique(y))
        print('  + nL: %d' % nL)
        return

    import seqparams
    import seqConfig as sq
    # from seqConfig import lcsHandler

    # params
    # topn
    #     - load documents: 
    #         cohort, seq_ptype, ctype, ifiles, doc_filter_policy, min_ncodes, simplify_code
    #     - load LCS set 
    #         cohort, lcs_type, lcs_policy, ctype, consolidate_lcs, label, meta
    userFileID = meta = None
    cohort = kargs.get('cohort', 'CKD')
    # use 'regular' for creating feature set but separate 'diag' and 'med' when doing LCS analysis
    ctype = seqparams.normalize_ctype('regular')  # 'diag', 'med'
    sq.sysConfig(cohort=cohort, seq_ptype=ctype, 
        lcs_type='global', lcs_policy='df', 
        consolidate_lcs=True, 
        slice_policy=kargs.get('slice_policy', 'noop'), 

        simplify_code=kargs.get('simplify_code', False), 
        meta=userFileID)

    tCompleteData = True
    tMakeTSet = True

    # given the feature set derived from t_lcs2(), create training data using LCSs as features
    kargs['min_ncodes'] = min_ncodes = 10
    X = y = None
    if tCompleteData: 
        if tMakeTSet:
            print('t_lcs2a> compute feature set with a complete document set ...')
            X, y = makeLCSFeatureTSet2a(cohort, **kargs)  # if non-incremental, use makeLCSFeatureTSet2(cohort, **kargs) 
            summary()
        print('t_lcs2a> analyze lcs distributions in different strata')
        analyzeLCSDistribution(cohort, **kargs) # Analyze LCS distributions within each stratified cohort (e.g. CKD cohort stratified by severity stages)
    else: 
        kargs['max_n_docs'] = max_ndocs = 5000
        kargs['topn'] = topn = 1000
        topn_lcs = 100

        print('> sampling a subset of documents (n=%d)' % max_ndocs)

        Dsub, Lsub, Tsub, docIds = sampleDocuments(cohort, min_ncodes=min_ncodes, max_n_docs=max_ndocs)
        nDsub = len(Dsub)

        # stratified = stratify(Dsub, L=Lsub, T=Tsub)
        userFileID = 'sub%s' % max_ndocs
        assert nDsub <= max_ndocs

        if tMakeTSet: 
            X, y = makeLCSFeatureTSet2a(cohort, min_ncodes=min_ncodes, max_n_docs=max_ndocs, topn=topn, document_ids=docIds) 
            assert X.shape[0] == nDsub
            summary()
        kargs['stratified_docs'] = stratify(D=Dsub, L=Lsub, T=Tsub, document_ids=docIds)
        analyzeLCSDistribution(cohort, topn=topn, topn_lcs=topn_lcs)


    # analyzeLCSDistribution(cohort, **kargs) # Analyze LCS distributions within each stratified cohort (e.g. CKD cohort stratified by severity stages)
    return

def t_characterize_strata(): 

    analyzeLCSDistribution(cohort, **kargs)

    return 

def t_analyze_mcs(**kargs):  
    """
    Run this prior to makeLCSFeatureSet2(). 


    """
    def chunks(l, n):
        """Yield successive n-sized chunks from l."""
        for i in range(0, len(l), n):
            yield l[i:i + n]
    def process_docs(inputdir=None): 
        ### load + transfomr + (ensure that labeled_seq exists)
      
        # params: cohort, seq_ptype, ifiles, doc_filter_policy
        #         min_ncodes, simplify_code
        # use the derived MCS set (used to generate d2v training data) as the source
        src_dir = get_global_cohort_dir() if inputdir is None else inputdir # orginal source: get_global_cohort_dir()
        ifiles = kargs.get('ifiles', ['condition_drug_labeled_seq-CKD.csv', ])

        assert os.path.exists(src_dir), "Invaild MCS source dir:\n%s\n" % src_dir
        ctype =  kargs.get('seq_ptype', lcsHandler.ctype)  # kargs.get('seq_ptype', 'regular')
        cohort = kargs.get('cohort', lcsHandler.cohort)  # 'PTSD'

        minDocLength = kargs.pop('min_ncodes', 10)
        tSimplified = kargs.get('simplify_code', lcsHandler.is_simplified)
        D, L, T = processDocuments(cohort=cohort, seq_ptype=ctype, 
                    inputdir=src_dir, 
                    ifiles=ifiles,
                    # meta=kargs.get('meta', None), 
                    
                    # document-wise filtering 
                    policy=kargs.get('doc_filter_policy', 'minimal_evidence'),  
                    min_ncodes=minDocLength,  # retain only documents with at least n codes

                    # content modification
                    # predicate=kargs.get('predicate', None), # reserved for segment_docs()
                    simplify_code=tSimplified, 

                    source_type='default', 
                    create_labeled_docs=False)  # [params] composition

        # Dl = labelize(D, label_type='doc', class_labels=L)  # use class labels for document label prefixing
        is_labeled = len(np.unique(L)) > 1
        print('  + nD: %d | cohort=%s, ctype=%s, labeled? %s, simplified? %s' % (len(D), cohort, ctype, is_labeled, tSimplified))
        return (D, L, T)
    def get_global_cohort_dir(): # source MDS
        return seqparams.getCohortGlobalDir(cohort) # sys_config.read('DataExpRoot')/<cohort> 

    import gc 
    import seqConfig as sq

    userFileID = meta = None
    cohort = kargs.get('cohort', 'CKD')
    # use 'regular' for creating feature set but separate 'diag' and 'med' when doing LCS analysis
    ctype = seqparams.normalize_ctype('regular')  # 'diag', 'med'
    sq.sysConfig(cohort=cohort, seq_ptype=ctype, 
        lcs_type='global', lcs_policy='df', 
        consolidate_lcs=True, 
        slice_policy=kargs.get('slice_policy', 'noop'), 

        simplify_code=kargs.get('simplify_code', False), 
        meta=userFileID)

    D, L, T = process_docs(inputdir=None)  # set inputdir to None to use default
    nD = len(D)

    # load existing LCS features (derived from docToLCS, deriveLCS, etc)
    lcsCandidates = kargs.get('feature_set', [])  # try user input first
    if len(lcsCandidates) == 0: 
        # params: cohort, lcs_type, lcs_policy, ctype, consolidate_lcs, label, meta
        df = lcsHandler.load_lcs()  # load the global LCS training set
        assert df is not None and not df.empty, "Could not load pre-computed LCS feature set."

        # load lcs set
        lcsCandidates = df['lcs'].values
        print('t_analyze_mcs> found %d LCS canidates (ctype=%s) ...' % (df.shape[0], ctype))

    ### global evaluation
    print('t_analyze_mcs> Analyze MCS with %d input docs.' % len(D))

    # the document IDs must be consistent with the initial source
    docIds = kargs.get('document_ids', []) # if provided => incremental_mode -> True
    seti = 1
    for docIds in chunks(range(0, nD), 5000): 
        Dsub = list(np.array(D)[docIds])
        Tsub = list(np.array(T)[docIds])
        print('... processing set %d of size %d (docId %d ~ %d)' % (seti, len(Dsub), min(docIds), max(docIds)))
        ret = analyzeMCS(Dsub, Tsub, lcsCandidates, document_ids=docIds, 
                reset_=False, incremental_mode=True, make_color_time=False)  # initial candidates (to be filtered by ranked score)
        seti += 1
        gc.collect()

    gc.collect()
    ### next step: make feature vector 
    min_ncodes = 10
    topn = 10000

    X, y = makeLCSFeatureTSet2(cohort, D=D, L=L, T=T, 
            min_ncodes=min_ncodes, topn=topn, incremental_mode=True)  # document_ids: (range(nD) by default)
        
    return 

def t_lcs_feature_select0(**kargs): 
    import gc 
    import seqConfig as sq

    userFileID = meta = None
    cohort = kargs.get('cohort', 'CKD')
    # use 'regular' for creating feature set but separate 'diag' and 'med' when doing LCS analysis
    ctype = seqparams.normalize_ctype('regular')  # 'diag', 'med'
    sq.sysConfig(cohort=cohort, seq_ptype=ctype, 
        lcs_type='global', lcs_policy='df', 
        consolidate_lcs=True, 
        slice_policy=kargs.get('slice_policy', 'noop'), 

        simplify_code=kargs.get('simplify_code', False), 
        meta=userFileID)

    topn = 10000
    min_ncodes = 10
    df = chooseLCSFeatureSet(topn, **kargs)
    print('t_choose_lcs_fset> example lcs feature:\n%s\n' % df.head(10))
    return

def t_analyze_mcs2(**kargs):
    """

    Memo
    ----
    1. path to the (precomputed) global LCS statistics: 
        tpheno/seqmaker/data/CKD/pathway/lcsmapInvFreq-regular.csv

    """
    def initvarcsv2(name, keys=None, value_type='list', inputdir=None, content_sep=','): 
        # Same as initvar but load lcs map contents from .csv files
        if inputdir is None: inputdir = Pathway.getPath(cohort=kargs.get('cohort', lcsHandler.cohort))
        ctype = seqparams.normalize_ctype(kargs.get('seq_ptype', lcsHandler.ctype))  # 'diag', 'med'

        fname = '%s-%s.csv' % (name, ctype)
        fpath = os.path.join(inputdir, fname)

        newvar = {}
        if keys is not None: 
            if value_type == 'list': 
                newvar = {k: [] for k in keys}
            else: 
                # assuming nested dictionary 
                newvar = {k: {} for k in keys}
        else: 
            if value_type == 'list': 
                newvar = defaultdict(list)
            else: 
                newvar = defaultdict(dict) # nested dictionary

        if os.path.exists(fpath) and os.path.getsize(fpath) > 0: 
            df = pd.read_csv(fpath, sep='|', header=0, index_col=False, error_bad_lines=True)
            print('initvarcsv2> var: %s > found %d existing entries from %s' % (name, df.shape[0], fname))
            
            # parse 
            if name == 'lcsmap':  # lcs -> docIDs
                header = ['lcs', 'doc_ids']
                idx = []
                for idstr in df['doc_ids'].values: 
                    idx.append(idstr.split(content_sep))
                newvar.update(dict(zip(df['lcs'], idx)))
                return newvar 
            elif name == 'lcsmapInvFreq': 
                header = ['doc_id', 'lcs', 'freq']  # where doc_id can have duplicates 
                # for di in df['doc_id'].unique(): 
                cols = ['lcs', ]
                adict = {}
                for lcs in set(df['lcs'].values): 
                    adict[lcs] = sum(df.loc[df['lcs']==lcs]['freq'].values)
                print('initvarcsv2> Found %d entries ...' % len(adict))     

            else: 
                raise NotImplementedError

        if len(newvar) > 0: 
            print('initvarcsv> example:\n%s\n' % sysutils.sample_dict(newvar, n_sample=1))
        return newvar
    def compute_tfidf(lcs):
        # tf(d, lcs) * idf(lcs)
        # 
        pass

    # load temporary file 
    # variables: lcsmap, lcsmapInvFreq
    initvarcsv(name=lcsmapInvFreq)    


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
       ./data/CKD/combined/tset-n0-IDdiag-pv-dm2-GCKD.csv

    """
    import seqClassify as sclf
    import vector

    cohort_name = kargs.get('cohort', 'CKD')  # PTSD for experimenting on LCS-based labeling
    d2v_method = kargs.get('d2v_method', vector.D2V.d2v_method)
    print('info> cohort=%s, d2v_method=%s' % (cohort_name, d2v_method))
    seq_ptype = seqparams.normalize_ctype(kargs.get('seq_ptype', 'reg')) # 'regular', 'med'

    # [control]
    tMakeTSet = kargs.get('make_tset', False) 
    tMakeLCSTSet = kargs.get('make_lcs_tset', False)   # ts with LCSs as labels 
    tMakeLCSFeatureTSet = kargs.get('make_lcs_feature_tset', True)   # ts with LCSs as features 

    ### LCS 
    minLength, maxLength = 5, 15

    # [options]
    # makeTSet: train test split (randomized subsampling, n_trials is usu. 1)
    # makeTSetCombined: no separation of train and test data on the d2v level 
    # makeTSetCV: 

    if tMakeTSet: 
        ts = sclf.makeTSetCombined(d2v_method=d2v_method, 
                                    seq_ptype=seq_ptype, 
                                    model_id=kargs.get('model_id', seq_ptype), # distinguish models based on sequence contents
                                    test_model=kargs.get('test_model', True), 
                                    load_model=kargs.get('load_model', False), 
                                    cohort=cohort_name) # [note] this shouldn't revert back and call load_tset() again 

    # to load training set use the following ...
    # ts = loadTSetCombined(cohort=cohort_name, d2v_method=vector.D2V.d2v_method)  # ./data/<cohort>/combined

    if tMakeLCSTSet: 
        # [use] if slice_policy in {'prior', 'posterior'}, only use the corresponding subsequence to derive LCS labels
        # [memo] n=5000, max_n_pairs ~ 12,497,500
        #        min_ndocs: min n_docs in which an LCS is present
        ts = makeLCSTSet(cohort=cohort_name, d2v_method=d2v_method, seq_ptype=seq_ptype, 
                topn_lcs=5, min_length=5, max_length=25, min_ndocs=200, 
                max_n_pairs=250000, pairing_policy='random', 
                load_lcs=False, 
                slice_policy='noop', slice_predicate=None,  # {'noop', 'prior', 'posterior', }; only for file ID purpose 
                consolidate_lcs=False)  # consolidate_lcs <- True, ordering not important (permutation isomorphic)  

    topNLCS = 1000  # use LCSs with higher ranks (e.g. doc freq) as features
    if tMakeLCSFeatureTSet: 
        makeLCSFeatureTSet(cohort=cohort_name, n_features=topNLCS, seq_ptype=seq_ptype, 
            min_length=minLength, max_length=maxLength,   # these will only take effect if overwrite_lcs <- True
            load_lcs=True) # set to False to recompute LCSs

    return

def featureSelectNoop(**kargs):
    """
    Choose all LCS features available (i.e. feature selection itself is a noop)
    """ 
    def load_tset(n_per_class=None): 
        X, y = kargs.get('X', None), kargs.get('y', None)
        if X is None or y is None: 
            X, y = loadLCSFeatureTSet(scale_=kargs.get('scale_', True), 
                label_map=seqparams.System.label_map, 
                drop_ctrl=kargs.get('drop_ctrl', True), 
                n_per_class=n_per_class) # subsamping
        assert X is not None
        return (X, y)
    def load_fset(topn=None, policy_fs='sorted'):
        policyFeatureRank = kargs.get('policy_fs', policy_fs)  # sorted, tfidf, tf, ...
        df = None 
        if policyFeatureRank.startswith('so'):
            df = load_sorted_fset(topn=topn)
        else: 
            df = load_ranked_fset(topn=topn)
        return df
    def load_sorted_fset(topn=None): 
        # load 
        meta = lcsHandler.meta
        fileID = 'sorted' if meta is None else '%s-sorted' % meta
        df = TSet.loadRankedLCSFSet(cohort=lcsHandler.cohort, d2v_method=lcsHandler.d2v_method, 
                     seq_ptype=lcsHandler.ctype, suffix=fileID)
        if df is not None and not df.empty: 
            if topn is not None: df = df.head(topn)  # this is not as meaningful when LCSs are just alphanumerically sorted
        return df  # dataframe or None (if not found)
    def load_ranked_fset(topn=None): 
        # load 
        fileID = lcsHandler.meta 
        df = TSet.loadRankedLCSFSet(cohort=lcsHandler.cohort, d2v_method=lcsHandler.d2v_method, 
                     seq_ptype=lcsHandler.ctype, suffix=fileID)
        if df is not None and not df.empty: 
            if topn is not None: df = df.head(topn)
        return df  # dataframe or None (if not found): 

    X, y = load_tset(n_per_class=kargs.get('n_per_class', None))
    df = load_fset(topn=None, policy_fs=kargs.get('policy_fs', 'sorted')) # set topn to None to load ALL
    lcsFSetTotal = df['lcs'].values
    assert X.shape[1] == len(lcsFSetTotal)

    n_classes = len(np.unique(y))
    print('featureSelectNoop> X (dim: %s), y (n_classes: %d), n_features: %d' % (str(X.shape), n_classes, len(lcsFSetTotal)))
    return (X, y, lcsFSetTotal)

def featureSelect(topn, **kargs): 
    """
    Select features based on importance weights derived from LASSO (and other methods). 

    Memo
    ----
    1. example (ranked) feature file
        tset-IDregular-pv-dm2-Rlcs-sorted-GCKD.csv    ... generic
        tset-IDregular-pv-dm2-Rlcs-sorted-lasso-GCKD.csv   ... lasso-selected

    2. RF picked 11293 features out of 43405 (n_estimator=5000)
       Lasso picked 

    """
    def load_tset(n_per_class=None): 
        X, y = kargs.get('X', None), kargs.get('y', None)

        # e.g. tset-IDregular-pv-dm2-Rlcs-sorted-GCKD.csv 
        if X is None or y is None: 
            X, y = loadLCSFeatureTSet(scale_=kargs.get('scale_', True), 
                label_map=seqparams.System.label_map, 
                drop_ctrl=kargs.get('drop_ctrl', True), 
                n_per_class=n_per_class) # subsamping
        assert X is not None
        return (X, y)
    def load_fset(topn=None, policy_fs='sorted'):
        policyFeatureRank = kargs.get('policy_fs', policy_fs)  # sorted, tfidf, tf, ...
        df = None 
        if policyFeatureRank.startswith('so'):
            df = load_sorted_fset(topn=topn)
        else: 
            df = load_ranked_fset(topn=topn)
        return df
    def update_fset(df, policy_fs='sorted'):
        policyFeatureRank = kargs.get('policy_fs', policy_fs)
        if policyFeatureRank.startswith('so'):
            meta = lcsHandler.meta
            fileID = 'sorted' if meta is None else '%s-sorted' % meta

            TSet.saveRankedLCSFSet(df, cohort=lcsHandler.cohort, d2v_method=lcsHandler.d2v_method, 
                seq_ptype=lcsHandler.ctype, suffix=fileID) # dir_type/'combined'
        else: 
            TSet.saveRankedLCSFSet(df, cohort=lcsHandler.cohort, d2v_method=lcsHandler.d2v_method, 
                    seq_ptype=lcsHandler.ctype, suffix=lcsHandler.meta) # dir_type/'combined'
        nf = collections.Counter(df['lasso'].values)[1]
        print('update_fset> policy=%s, fileID=%s, df(dim=%s) => n_features: %d ...' % (policyFeatureRank, fileID, str(df.shape), nf))
        return
    def load_sorted_fset(topn=None): 
        # load 
        meta = lcsHandler.meta
        fileID = 'sorted' if meta is None else '%s-sorted' % meta
        df = TSet.loadRankedLCSFSet(cohort=lcsHandler.cohort, d2v_method=lcsHandler.d2v_method, 
                     seq_ptype=lcsHandler.ctype, suffix=fileID)
        if df is not None and not df.empty: 
            if topn is not None: df = df.head(topn)  # this is not as meaningful when LCSs are just alphanumerically sorted
        return df  # dataframe or None (if not found)
    def load_ranked_fset(topn=None): 
        # load 
        fileID = lcsHandler.meta 
        df = TSet.loadRankedLCSFSet(cohort=lcsHandler.cohort, d2v_method=lcsHandler.d2v_method, 
                     seq_ptype=lcsHandler.ctype, suffix=fileID)
        if df is not None and not df.empty: 
            if topn is not None: df = df.head(topn)
        return df  # dataframe or None (if not found): 
    def select_features(X, y, n=10000, fset=None, policy_fs='sorted', 
                estimator=None, n_sample_per_class=3000, threshold=None, 
                n_iter=10, test_=False):  # <- ts
        print('  ... LCS selection via Lasso-like operations ... ')
        if fset is None: 
            fset = load_fset(topn=None, policy_fs=policy_fs) # set topn to None to load ALL
        active_lcsx, positions = evaluate.select_features(X, y, 
            n_features=n, feature_set=fset, 
            n_iter=n_iter, # number of cycles of feature selection, each of which looks at a different portion of X
            n_per_class=n_sample_per_class, threshold=threshold, 
            estimator=estimator, test_importance=test_)  # threshold 1e-5 by default if penalty='l1'
        displayMax = 100
        for i, lcs in enumerate(active_lcsx): 
            if i < displayMax: 
                print('  + [#%d, pos=%d] %s' % ((i+1), positions[i], lcs))
        print('select_features> n_selected: %d, requested: %d' % (len(active_lcsx), n))
        return (active_lcsx, positions)
    def load_selected_features(df=None, col='lasso'):
        if df is None: df = load_fset(topn=None, policy_fs=kargs.get('policy_fs', 'sorted')) # set topn to None to load ALL
        assert col in df.columns.values, "selection vector (col=%s) has not been added to the fset dataframe yet." % col
        active_lcsx = df.loc[df['lasso']==1]['lcs'].values
        positions = df.loc[df['lasso']==1].index.values
        print('select_features> n_selected: %d' % len(active_lcsx))
        return (active_lcsx, positions)

    from sklearn.feature_selection import SelectFromModel
    from sklearn.linear_model import LassoCV, RandomizedLogisticRegression, LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    import evaluate  # feature, model selections
    # from tset import TSet

    ### (X, y)
    X, y = load_tset(n_per_class=kargs.get('n_per_class', None))

    # We use the base estimator LassoCV since the L1 norm promotes sparsity of features.
    # clf = LassoCV()

    # load candidate feature set  
    # prerequisite: run makeLCSFeatureTSet2a to obtain a ranked/sorted feature set first
    df = load_fset(topn=None, policy_fs=kargs.get('policy_fs', 'sorted')) # set topn to None to load ALL
    lcsFSetTotal = df['lcs'].values
    assert X.shape[1] == len(lcsFSetTotal)

    lcsFSet, pos = lcsFSetTotal, None
    if kargs.get('load_selected', False):
        lcsFSet, pos = load_selected_features(df=df)

        if topn == len(lcsFSet): 
            Xp = X[:, pos]
            return (Xp, y, lcsFSet) 
        else: 
            print('featureSelectLasso> precomputed high-ranked features has a different size: (%d) but requested %d' % (len(lcsFSet), topn))

    maxNSamplePerClass = kargs.get('n_sample_per_class', 2500)  # use only a subset in each classs to speed up

    # [params][todo]
    # # RandomizedLogisticRegression() # stability selection 
    # clf_default = LogisticRegression(class_weight='balanced', solver='saga', penalty='l1', max_iter=250)  # default max_iter=100
    clf_default = RandomForestClassifier(n_estimators=5000, n_jobs=15)  # avg importance: 2.304e-05, very small
    clf = kargs.get('estimator', clf_default)  # will use LassoCV by default
    importance_threshold = None # 0.1, None  # the larger => the smaller the feature set
    nIterFS = 2   # typically 10+
    
    ### feature selection starts here
    lcsFSet, pos = select_features(X, y, n=topn, fset=lcsFSetTotal, 
            estimator=clf, threshold=importance_threshold, 
            n_sample_per_class=maxNSamplePerClass, n_iter=nIterFS, test_=True)    
    print('featureSelectLasso> size(total): %d, size(selected): %d =?= requested: %d' % (len(lcsFSetTotal), len(lcsFSet), topn))

    # update feature set 
    posSet = set(pos)
    sv = [] 
    for p in range(X.shape[1]):
        if p in posSet:  
            sv.append(1)
        else: 
            sv.append(0)
    df['lasso'] = sv 
    update_fset(df, policy_fs=kargs.get('policy_fs', 'sorted'))  # update selected features

    # lcsFSetTotal[pos] => lcsFSet
    Xp = X[:, pos]
    assert len(lcsFSet) == Xp.shape[1], "size(lcsFSet): %d <> Xp: %d (dim: %s)" % (len(lcsFSet), Xp.shape[1], str(Xp.shape))
    return (Xp, y, lcsFSet) 

def t_classify(**kargs):
    """

    Memo
    ----
    1. Example training sets: 

        <path> LCS feature set
           sorted: tpheno/seqmaker/data/CKD/combined/tset-IDregular-pv-dm2-Rlcs-sorted-GCKD.csv

        <path> X: tpheno/seqmaker/data/CKD/combined/tset-IDregular-pv-dm2-SFlcs-sorted-GCKD.npz
               y: tpheno/seqmaker/data/CKD/combined/tset-IDregular-pv-dm2-SFlcs-sorted-GCKD.csv


    """
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
                (lcsHandler.cohort, lcsHandler.ctype, lcsHandler.d2v_method, lcsHandler.meta)
        msg += "  + is_simplified? %s, ... \n" % lcsHandler.is_simplified
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
    def remove_classes(X, y, labels=[], other_label='Others'):
        exclude_set = labels if len(labels)>0 else [other_label, ]
        
        N0 = len(y)
        ys = Series(y)
        cond_pos = ~ys.isin(exclude_set)

        idx = ys.loc[cond_pos].index.values
        y = ys.loc[cond_pos].values 
        X = X[idx]  # can sparse matrix be indexed like this?

        print('... remove labels: %s size(ts): %d -> %d' % (labels, N0, X.shape[0]))        
        return (X, y)
    def select_features_lasso(n=100):  # <- ts
        print('  ... common LCSs ~ LassoCV ... ')
        active_lcsx, positions = evaluate.select_features_lasso(ts, n_features=n)
        for i, lcs in enumerate(active_lcsx): 
            print('  + [#%d, pos=%d] %s' % ((i+1), positions[i], lcs))
        return (active_lcsx, positions)
    def get_logistic(): 
        clf = LogisticRegression(penalty='l2', class_weight='balanced', solver='sag')
        clf.set_params(multi_class='multinomial', solver='saga', max_iter=1000)
        return clf
    def choose_classifier(name='random_forest'):
        if name.startswith(('rand', 'rf')):  # n=389K
            # max_features: The number of features to consider when looking for the best split; sqrt by default
            # sample_leaf_options = [1,5,10,50,100,200,500] for tuning minimum sample leave (>= 50 usu better)
            clf = RandomForestClassifier(n_jobs=12, 
                    # random_state=53, 
                    n_estimators=1000,   # default: 10
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
            clf = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, 
                # random_state=53, 
                min_samples_split=250, min_samples_leaf=50, max_depth=8,  # prevent overfitting
                max_features = 'sqrt', # Its a general thumb-rule to start with square root.
                subsample=0.80)
        else: 
            raise ValueError, "Unrecognized classifier: %s" % name
        return clf

    import evaluate
    import classifier.utils as cutils
    from tset import TSet
    import seqClassify as sclf

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

    # if mode.startswith('bin'): # binary class (need to specifiy positive classes to focus on)
    #     return t_binary_classify(**kargs)    

    ### classification 
    clf_list = []

    # choose classifiers
    # random forest: n_estimator = 100, oob_score = TRUE, n_jobs = -1,random_state =50, max_features = "auto", min_samples_leaf = 50

    # 1. logistic 
    # clf = LogisticRegression(class_weight='balanced', solver='saga', penalty='l2')

    # 2. random forest 
    # clf = RandomForestClassifier(n_estimators=100, n_jobs=5, random_state=53)  # n_estimators=500, 

    # 3. SGD classifier
    #    when tol is not None, max_iter:1000 by default
    # clf = SGDClassifier(loss='log', penalty='elasticnet', max_iter=1500, tol=1e-3) # l1_ratio: elastic net mixing param: 0.15 by default
    
    clf = choose_classifier(name=kargs.pop('clf_name', 'rf')) # rf: random forest
    clf_list.append( clf )  # supports multiclass 

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
    ts_dtype = kargs.get('tset_dtype', 'sparse')
    maxNPerClass = kargs.get('n_per_class', None) # classification
    maxNSamplePerClass = kargs.get('n_sample_per_class', 3000)  # LASSO feature selection has n iterations; use only a data subset within each
    userFileID = kargs.get('meta', lcsHandler.meta) 
    topNFSet = kargs.get('n_features', 10000)
    if ts_dtype.startswith('d'): 
        raise NotImplementedError
    else:   # sparse representation
        X, y = kargs.get('X', None), kargs.get('y', None)
        if X is None or y is None: 
            X, y = loadLCSFeatureTSet(scale_=kargs.get('scale_', True), 
                label_map=seqparams.System.label_map, 
                drop_ctrl=kargs.get('drop_ctrl', True), 
                n_per_class=maxNPerClass) # subsamping
        else: 
            assert X.shape[0] == len(y)
            y = sclf.mergeLabels(y, lmap=seqparams.System.label_map) # y: np.ndarray

            # subsampling 
            if maxNPerClass:
                y = np.array(y)
                X, y = subsample(X, y, n=maxNPerClass)

            if kargs.get('scale_', True): 
                # scaler = StandardScaler(with_mean=False)
                scaler = MaxAbsScaler()
                X = scaler.fit_transform(X)
        
        assert X is not None and y is not None

        # [note] request 'topNFSet' features but may not actually get that many
        if kargs.get('apply_fs', True): 
            # [params] estimator
            X, y, fset = featureSelect(topn=topNFSet, X=X, y=y, policy_fs='sorted', 
                    n_sample_per_class=maxNSamplePerClass, load_selected=kargs.get('load_selected_lcs', False))
        else: # bypass feature selection 
            X, y, fset = featureSelectNoop(X=X, y=y, policy_fs='sorted')

        # [test]
        # clf_list = [LogisticRegression(class_weight='balanced', solver='saga', penalty='l2'), ]
        summary(X=X, y=y)
        identifier = kargs.get('identifier', 'Flcs-tfidf')
        for clf in clf_list: 
            sclf.multiClassEvaluateSparse(X=X, y=y, cohort=lcsHandler.cohort, seq_ptype=lcsHandler.ctype, # not needed 
                classifier=clf, 
                # classifier_name='l2_logistic',   # if not None, will try this first
                focused_labels=focusedLabels, 
                roc_per_class=classesOnROC,
                param_grid=param_grid,    # grid-based model selection
                label_map=seqparams.System.label_map, # use sysConfig to specify
                meta=userFileID, identifier=identifier) # use meta (global) or identifier (multiclass tasks) to distinguish different classificaiton tasks

    # [todo] loop through cartesian product of seq_ptype and d2v_method? 

    return

def t_lcs_feature_select(**kargs):
    maxNPerClass = kargs.get('n_per_class', None) # classification
    maxNSamplePerClass = kargs.get('n_sample_per_class', 5000)  # feature selection has n iterations; use only a data subset within each
    userFileID = kargs.get('meta', lcsHandler.meta) 
    topNFSet = kargs.get('n_features', 10000) 

    ### load training data 
    if True: 
        X, y = kargs.get('X', None), kargs.get('y', None)
        if X is None or y is None: 
            X, y = loadLCSFeatureTSet(scale_=kargs.get('scale_', True), 
                label_map=seqparams.System.label_map, 
                drop_ctrl=kargs.get('drop_ctrl', True), 
                n_per_class=maxNPerClass) # subsamping
        else: 
            assert X.shape[0] == len(y)
            y = sclf.mergeLabels(y, lmap=seqparams.System.label_map) # y: np.ndarray

            # subsampling 
            if maxNPerClass:
                y = np.array(y)
                X, y = subsample(X, y, n=maxNPerClass)

            if kargs.get('scale_', True): 
                # scaler = StandardScaler(with_mean=False)
                scaler = MaxAbsScaler()
                X = scaler.fit_transform(X)
        
        assert X is not None and y is not None


    X, y, fset = featureSelect(topn=topNFSet, X=X, y=y, policy_fs='sorted', 
                    n_sample_per_class=maxNSamplePerClass, load_selected=kargs.get('load_selected_lcs', False)) 

    return (X, y, fset)

def test(**kargs): 
    import seqConfig as sq

    # params
    # topn
    #     - load documents: 
    #         cohort, seq_ptype, ctype, ifiles, doc_filter_policy, min_ncodes, simplify_code
    #     - load LCS set 
    #         cohort, lcs_type, lcs_policy, ctype, consolidate_lcs, label, meta
    
    userFileID = meta = None  # lcs_policy, (per-class) sample size
    identifier = 'Flcs-fs' # use this for classification results instead of userFileID for now, feature selection
    cohort = kargs.get('cohort', 'CKD')
    # use 'regular' for creating feature set but separate 'diag' and 'med' when doing LCS analysis
    ctype = seqparams.normalize_ctype('regular')  # 'diag', 'med'
    sq.sysConfig(cohort=cohort, seq_ptype=ctype, 
        lcs_type='global', lcs_policy='df', 
        consolidate_lcs=True, 
        slice_policy=kargs.get('slice_policy', 'noop'), 

        simplify_code=kargs.get('simplify_code', False), 
        meta=userFileID)

    tCompleteData = True
    maxNPerClass = None # 10000  # None to include all
    maxNSamplePerClass = 3000
    topNFSet = 10000 # 10000

    ### make training set based on path analysis (e.g. LCS labeling)
    # t_model(make_tset=True, make_lcs_feature_tset=False, make_lcs_tset=False)

    ### classification 
    # t_classify(**kargs)

    ### global LCS given a cohort-specific documents 
    # t_analyze_lcs(cohort='CKD', seq_ptype='regular')
    # t_analyze_lcs(cohort='CKD', seq_ptype='diag')
    # t_analyze_lcs(cohort='CKD', seq_ptype='med')

    ### stratified LCS patterns
    # t_lcs()  

    ### global -> stratified LCS patterns
    # t_lcs2()

    ### create LCS feature set (followed by logistic regression and observe which LCSs in each stratum have relatively higher coefficients)
    # t_model(cohort='CKD', seq_ptype='regular', make_lcs_feature_tset=True, make_lcs_tset=False, make_tset=False)
    # fset = t_classify_lcs(**kargs)

    ### create LCS feature set given LCSs derived from t_lcs2()
    # t_analyze_mcs() # precompute LCSs first i.e. run analyzeMCS()
    t_lcs_feature_select()  # after running incremental mode of analyzeMCS(), choose from among the LCS candidates
    # t_make_lcs_fset() # t_lcs2a()

    tLoadSelectedLCS, tApplyFS = True, True
    if tLoadSelectedLCS: tApplyFS=True
    # t_classify(identifier=identifier, 
    #     clf_name='gbt',
    #     n_per_class=maxNPerClass,  # classification
    #     n_sample_per_class=maxNSamplePerClass, # feature selection (used only when apply_fs is True)
    #     n_features=topNFSet, apply_fs=tApplyFS, load_selected_lcs=tLoadSelectedLCS, 
    #     drop_ctrl=True)  

    # exploratory cluster analysis 
    # t_cluster()  # feature selection -> frequency analaysis by plotting 2D comparison plots
    
    ### time series of LCSs + sequence colormap
    # t_lcs_timeseries()

    return

def test_batch(**kargs):
    nTrials = 20
    for i in range(nTrials):
        nt = i+1
        div(message='Beginning Trial #%d' % nt, symbol='%')
        test(load_model=True)
        div(message='End Trial #%d' % nt, symbol='%')
    return

if __name__ == "__main__": 
    test()
    # test_batch()

