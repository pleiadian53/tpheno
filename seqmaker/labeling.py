# encoding: utf-8

import numpy as np 

from operator import itemgetter
import collections
from collections import OrderedDict, namedtuple
# from sklearn.feature_extraction.text import TfidfVectorizer

import os, random
import string
import pandas as pd 
from pandas import DataFrame

# from batchpheno import sampling
from batchpheno.utils import div
from pattern import medcode as pmed

import seqparams

# d2v, w2v
import gensim


# [todo]
class TDoc(seqparams.TDoc):  # this defines a different inheritance line from tdoc.TDoc
    def __init__(self, docs, **kargs):  # [todo]
        """
        Allows for creating instances of TDoc that depend on the cohort
        """
        self.cohort = kargs.get('cohort', None)
        self.nDoc = len(docs)

        return 

    @staticmethod
    def isLabeled(docs, n_test=10):
        # ensure that documents were already labeled
        nD = len(docs)
        if not nD: 
            print('warning> empty corpus!')
            return False

        is_labeled = False
        idx = np.random.randint(nD, size=n_test)  # sampling with replacement
        for n, i in enumerate(idx): 

            # test two essential attributes of a labeled document (via gensim.models.doc2vec.TaggedDocument
            try: 
                labels = docs[i].tags  # ith document; each document is a namedtuple 
                assert len(labels) > 0, "isLabeled> empty labeling for %d-th doc: %s" % (i, docs[i][:50])
                # can also test doc.words
                is_labeled = True
            except: 
                is_labeled = False 
                # break
        
            # condition: trust that the 'tags' attribute is consistent throughout all documents 
            if is_labeled or (n > n_test): break
        
        # assert is_labeled, "Documents are not labeled yet. Try using makeD2VLabels()"
        return is_labeled

    @staticmethod 
    def isListOfTokens(docs, n_test=10): # [todo] allow for 2D array format? 
        assert hasattr(docs, '__iter__'), "Input document set is not a 2D array: %s" % docs[:10]
        idx = np.random.randint(len(docs), size=n_test)
        tval = True
        for i in idx: 
            if not hasattr(docs[i], '__iter__'): # isinstance(docs[i], np.ndarray)
                tval = False; break 
        return tval

class TDocTag(object):  # [todo] a component of TDoc; could have been an 'inner class' of TDoc

    doc_label_prefix = 'client'  # use class labels if available, or 'train'/'test' in the case of CV, or 'doc' as generic prefix
    label_sep = '_'  # separator between label_prefix and label_index 

    @staticmethod
    def canonicalize(tags): 
        """
        The labels read from structured sequence files (*.csv) 
        are for classification use but not typically for 
        serving the purpose of document labeling required by 
        Doc2Vec. 

        So we'll tag integer document IDs (0 to n_doc-1) to each label
        such that each label is a list of >=2 elements i.e. [docID, label, ...]

        Output: a list of tags, each of which is a list where the first entry 
                is an integer document ID (0 ~ N-1); N: number of documents

                to be inserted as tagl in TaggedDocument(sen.split(), tagl)
        """
        # test format 
        r = random.randint(0, len(tags)-1)

        # in multilabel format? each tag is in the form of [e1, e2, ...]
        isMultiLabel = True if hasattr(tags[r], '__iter__') else False

        if isMultiLabel: # i.e. each label is a list
            print('TDocTag.canonicalize> input labels in multilabel format.')
            docTags = []
            for i, tag in enumerate(tags): 
                
                # docId = TDocTag.getDocID(i)
                docId = i  # set docId here
                if tag[0] == docId: 
                    # do nothing, first element is already the intended docId
                    pass 
                else: 
                    tag.insert(0, docId)
                docTags.append(tag)
        else: 
            docTags = []
            for i, tag in enumerate(tags): 
                if i < 3: assert isinstance(tag, str)
                docId = i # docId = TDocTag.getDocID(i) 
                docTags.append([docId, tag, ])            
        return docTags 

    @staticmethod
    def labelAsIs(tags): # no padding of integer document IDs 
        # test format 
        r = random.randint(0, len(tags)-1)

        # in multilabel format? each tag is in the form of [e1, e2, ...]
        isMultiLabel = True if hasattr(tags[r], '__iter__') else False    
        
        if isMultiLabel: 
            print('TDocTag.labelAsIs> input labels in multilabel format.')
            docTags = []
            for i, tag in enumerate(tags): 
                
                # docId = TDocTag.getDocID(i)
                docId = tag[0]  # document ID is the first tag 
                docTags.append([docId, ])
        else: 
            docTags = []
            for i, tag in enumerate(tags): 
                if i < 3: assert isinstance(tag, str)
                # docId = tag # docId = TDocTag.getDocID(i) 
                docTags.append([tag, ])            
        return docTags 

    @staticmethod
    def nameDocID(x): # name documet ID by integer x
        return '%s_%s' % (TDocTag.doc_label_prefix, x)

    @staticmethod
    def getDocIDs(docs, pos=0): # given a list of docs, return their document IDs (e.g. integer IDs, annotated labels, etc.)
        assert TDoc.isLabeled(docs, n_test=10), "Input documents (n=%d) are not labeled yet." % len(docs)
        return [doc.tags[pos] for doc in docs]

    @staticmethod
    def getIntIDs(n):
        return [[i, ] for i in range(n)]  # use multilabel format

    @staticmethod
    def toMultiLabel(tags): # from single-label format to multi-label format 
        # test format 
        r = random.randint(0, len(tags)-1)

        # in multilabel format? each tag is in the form of [e1, e2, ...]
        isMultiLabel = True if hasattr(tags[r], '__iter__') else False
        assert pos < len(tags[r]) and pos >= 0, "Invalid index pos=%d" % pos     
        if isMultiLabel: 
            # noop 
            print('TDocTag.toMultiLabel> input tags is already in multilabel format.') # [[l1, ], [l2, ], ...]
            return tags 
        
        return [[tag, ] for tag in tags]
        
    @staticmethod
    def toSingleLabel(tags, pos=0): 
        """
        Convert a list of tags to labels used by single-label classification algorithms. 

        Default: the first element (pos=1) in each tag is the label
        """
        # test format 
        r = random.randint(0, len(tags)-1)

        # in multilabel format? each tag is in the form of [e1, e2, ...]
        isMultiLabel = True if hasattr(tags[r], '__iter__') else False
        
        labels = []
        if isMultiLabel: 
            assert pos < len(tags[r]) and pos >= 0, "Invalid index pos=%d" % pos 
            for i, tag in enumerate(tags): 
                labels.append(tag[pos])
        else: 
            # noop, already done 
            print('TDocTag.toSingleLabels> Already in single-label format:\n%s\n' % tags[:10])
            labels = tags
        return labels

### end class TDocTag

def focus(ts, labels, other_label='Others'): 
    """
    Consolidate multiclass labels into smaller categories. 
    Focuse only on the target labels and classify all of the other labels
    using as others.  

    Related
    -------

    """
    raise ValueError, "Use seqClassifier.focus"

def binarize(labels, positive): 
    """
    Turn multiclass labels into binary labels 

    Memo
    ----
    This is NOT the same as seqClassifier.binarize()

    """
    # n_positive, n_negative = len(positive), len(negative)
    # assert n_positive > 0 or n_negative > 0
    # assert (n_positive + n_negative) == len(labels)

    n_active = 0

    # make deep copy
    # binLabels = np.empty_like(labels)
    binLabels = np.ones(len(labels))
    for i, label in enumerate(labels):     
        if label in positive: 
            # labels[i] = 1    # this modifies the list in place
            binLabels[i] = 1
            n_active += 1 
        else: 
            # labels[i] = 0
            binLabels[i] = 0
    assert n_active > 0, "None of the labels is detected positive defined by %s" % str(positive)
    return binLabels

def count(labels):
    unique, counts = np.unique(labels, return_counts=True) 
    return dict(zip(unique, counts))

def toStr(seq, sep=','): 
    return sep.join(str(e) for e in seq)

def labelDocByFile(fpath, person_ids, id_field, label_field, **kargs): 
    """


    Preliminary Steps 
    -----------------
    For a given cohort, first use the ID file(s) to get relevant person_ids (see getPersonIDs)

    Params
    ------
    cohort: 
        if 'fpath' is not a full path to the data source, then 'cohort' may be needed

    Memo
    ----
    1. CKD example    
       header = ['patientId', 'Case_Control_Unknown_Status', 'NKF_Stage', ]
    """
    def normalize_path(): 
        rootdir, fname = os.path.dirname(fpath), os.path.basename(fpath) 
        if not rootdir: 
            # if cohort is given => data-in/<cohort>; if not => data-in (data-in symlinked to data)
            rootdir = kargs.get('inputdir', seqparams.getCohortSrcDir(cohort=kargs.get('cohort', None)))

        # [condition] input file can be read into a dataframe
        return os.path.join(rootdir, fname)

    fpath = normalize_path()
    sep = kargs.get('sep', ',')

    df = pd.read_csv(fpath, sep=sep, header=0, index_col=False, error_bad_lines=True)
    print('io> Loaded data source of dimension: %s' % str(df.shape))
    assert id_field in df.columns and label_field in df.columns
   
    # [test] unique lables
    labelset = list(df[label_field].unique())
    cohort_name = kargs.get('cohort', None)
    print('info> cohort: %s | labels (n=%d):\n%s\n' % (cohort_name, len(labelset), labelset))
    # [log] 7 labels for CKD

    return labelDocByDataFrame(df, person_ids=person_ids, id_field=id_field, label_field=label_field)

def getPersonIDs(cohort, **kargs):
    """
    Given a cohort, find the corresponding patient IDs (via .id files)

    Input
    -----
    cohort-dependent source file (with label information)
       e.g. cohort=CKD: data/CKD/eMerge_NKF_Stage_20170818.csv

    Params
    ------
    cohort 
    inputdir: where cohort-dependent source file is kept 
    seq_compo: sequence composition; 'condition_drug' by default
    sep

    """
    def read_ids(fpath):
        fname = os.path.basename(fpath) 
        assert fname.find('.id') > 0
        assert os.path.exists(fpath), 'Invalid input: %s' % fpath
        df = pd.read_csv(fpath, sep=sep, header=0, index_col=False, error_bad_lines=True)
        assert id_field in df.columns, 'Invalid ID field: %s | header=%s' % (id_field, df.columns.values)
        return df[id_field].values

    # cohort_name = kargs['cohort']
    sep = kargs.get('sep', ',')
    basedir = inputdir = kargs.get('inputdir', seqparams.getCohortDir(cohort=cohort))
    fname = seqparams.getIDFile(cohort, **kargs)
    fpath = os.path.join(basedir, fname)
    id_field = kargs.get('id_field', 'person_id')

    return read_ids(fpath)

def labelDocByDataFrame(df, person_ids, id_field, label_field): 
    """
    Get labels from dataframe (from which coding sequences were formulated). 

    Assume that person_ids are available to match the desired entries (from which 
    labels are obtained). 

    Memo
    ----
    1. Coding sequences (made via seqMaker2 or seqMakerGeneric) 
       are arranged according to person_ids in ascedning order 
       by default. 

    """    
    assert id_field in df.columns and label_field in df.columns

    # filter, sort, extract
    sort_keys = [id_field, ]
    df = df.loc[df[id_field].isin(person_ids)]
 
    if df.empty: 
        print('warning> No qualified entries found according to the input IDs: %s ...' % person_ids[:10])
        return []  # 

    df.sort_values(sort_keys, ascending=True, inplace=True)  # [1]

    return df[label_field].values

def labelDocByFreq(seqx, **kargs): 
    """
    Label documents by most frequent tokens (regardless of token type i.e. diagnostic codes, med codes 
    or what have you)
    """
    # tFilterByDiag = kargs.get('filter_diag', False) 
    lNeg = '-1'  # label unknown
    n_doc = len(seqx)

    labels = [lNeg] * n_doc  # use default (negative) label if no ICD-9 code is found
    for i, seq in enumerate(seqx): 
        if len(seq) > 0: 
            labels[i] = counter_diag.most_common(1)[0][0]

    return labels

def labelDocByFreqDiag(seqx, **kargs):  # refactored from seqAnalyzer
    """
    Label the documents by most frequent diagnostic codes. 

    Input: documents (list of lists of tokens)
    Output: document labels (a list)
   
    Params
    ------
    filter_diag: if True, preserve only the input sequences with diagnostic codes 

    Input
    -----

    Memo
    ----
    This could potentially produce too many labels.
    """
    def to_str(seq, sep=','): 
        return sep.join(str(e) for e in seq)
    # from pattern import medcode as pmed

    tFilterByDiag = kargs.get('filter_diag', False) 
    lNeg = '-1'  # not a diagnostic code
    n_doc = len(seqx)

    if tFilterByDiag: 
        n_anomaly = 0

        # preserve only diag codes for each document 
        for i, seq in enumerate(seqx): 
            # [c for c in sequence if pmed.isICD(e)] 
            seqx[i] = filter(pmed.isICD, seq)  # use diagnostic codes to label the sequence
            if len(seqx[i]) == 0: 
                # seqx[i] = ['', ]  # use the most frequent diag code later
                print('warning> No diagnostic code found in %d-th sequence/doc:\n%s\n' % (i, toStr(seq)))
                n_anomaly += 1 
        div(message='A total of %d out of %d documents have valid diag codes.' % (n_doc-n_anomaly, n_doc), symbol='%')

        # in case if the sequence does not contain any diagnostic code, then replace empty list with the most common diag code on global scope
        counter = collections.Counter()
        for i, seq in enumerate(seqx): 
             counter.update(seq)
        assert len(counter) > 0
        top1code = counter.most_common(1)[0][0]
        for i, seq in enumerate(seqx): 
            if not seq: 
                seq[i] = [top1code]

        labels = []
        for seq in seqx: 
            counter_diag = collections.Counter(seq)
            labels.append(counter_diag.most_common(1)[0][0])
    else: 
        topn = 10

        # a. just use the most common code 
        # labels.append(counter_diag.most_common(1)[0][0])

        # b. only check topn codes, within which, the most frequent diag code is used as the label
        labels = [lNeg] * n_doc  # use default (negative) label if no ICD-9 code is found
        for i, seq in enumerate(seqx): 
            l0 = lNeg
            tDiagFound = False
            if len(seq) > 0: 
                counter = collections.Counter(seq)
                topn_counter = counter.most_common(topn)
                for tok, cnt in topn_counter: 
                    if pmed.isICD(tok): 
                        l0 = tok; tDiagFound = True
                        break 
            # [test]
            if tDiagFound: 
                labels[i] = l0
            else: 
                print('warning> No diagnostic code found in %d-th sequence/doc:\n%s\n' % (i, toStr(seq)))
            
    return labels

def labelByLCS(seqx, **kargs): 
    def load_lcs(): 
        typ = kargs.get('lcs_type', 'global_lcs')
        policy = kargs.get('lcs_policy', 'freq')
        seq_ptype = kargs.get('seq_ptype', 'regular')
        identifier = kargs.get('identifier', 'CType%s' % seq_ptype)
        return Pathway.load(cohort, lcs_type=tye, lcs_policy=policy, identifier=identifier, dir_type='pathway')
    def verify_doc(): 
        assert len(D) > 0, "No input documents found (cohort=%s)" % cohort
        x = random.randint(0, len(D)-1)  # nDoc
        assert isinstance(D[x], list), "Invalid input D[%d]:\n%s\n" % (x, D[x])
        assert len(T) == len(D), "inconsistent number of timestamps (nT=%d while nD=%d)" % (len(T), len(D))
        assert len(L) == len(D), "inconsistent number of labels (nL=%d while nD=%d)" % (len(L), len(D))
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
    def eval_lcs_stats(): # [params] lcsmap, lcsmapInv, matched_docIDs
        nM = len(matched_docIDs)
        r = nM/(nD+0.0)
        print('eval_lcs_stats> number of docs found match: %d, ratio: %f' % (nM, r))

        # number of documents with multiple LCS labels? 
        n_multimatch = 0
        for i in range(nD): 
            if i in lcsmapInv and len(lcsmapInv[i]) > 1: 
                n_multimatch += 1
                if n_multimatch <= 10: 
                    print('  + document #%d matches multiple LCSs:\n    %s\n' % (i, lcsmapInv[i]))
        print('    + number of documents with multiple LCS labels? %d' % n_multimatch)

        # most popular LCSs? 
        hotLCSs = sorted([(lcs, len(dx)) for lcs, dx in lcsmap.items()], key=lambda x:x[1], reverse=True)[:10]
        for lcs, cnt in hotLCSs: 
            print('  + LCS: %s found %d matches ...' % (lcs, cnt))

        return

    import pathAnalyzer as pa 
    from seqparams import Pathway
    import seqAlgo

    print('labelByLCS> Loading target LCS ...')
    lcsSet = kargs.get('lcs_set', []) 
    if not lcsSet:  # None or empty
        df = load_lcs()  # try loading first
        tNewLCS = False
        if df is None:  
            # [note] 125000 allows for comparing 500+ documents pairwise
            # params: remove duplicate codes with consecutive occurrences (and preserve only the first one)
            df = pa.analyzeLCS(D=seqx, topn_lcs=kargs.get('topn_lcs', 20), 
                        min_length=kargs.get('min_length', 2), max_length=kargs.get('max_length', 15), 
                        max_n_pairs=kargs.get('max_n_pairs', 125000), remove_duplicates=True)
        tNewLCS = True
        assert len(df['lcs'].unique()) == len(df['lcs'].values), "LCS is not unique in the input dataframe:\n%s\n" % df['lcs'].values
        lcsSet = list(df['lcs'].values)  

    nD = len(seqx)
    nL = len(lcsSet) + 1  # plus 'No_Match'
    print('labelByLCS> Find matching persons whose records contain given LCSs (n=%d); do this for each LCS ...' % nL)
    assert len(lcsSet) > 0

    lcsmap = {lcs: [] for lcs in lcsSet} # LCS -> document IDs  # {lcs:[] for lcs in df['lcs'].unique()}
    lcsmapInv = {i: [] for i in range(nD)} # document ID -> LCSs 
    matched_docIDs = set()  # docuemnts with any  matched LCSs
    lcs_sep = kargs.get('lcs_sep', Pathway.lcs_sep) # should be ' '
    nD_has_labels = 0
    for i, doc in enumerate(seqx): # foreach doc, find its LCS labels 
        for j, lcs in enumerate(lcsSet): # Pathway.header_global_lcs = ['lcs', 'length', 'count', 'n_uniq']
            if not lcsmap.has_key(lcs): lcsmap[lcs] = []

            lcs_seq = lcs.split(lcs_sep)  # Pathway.strToList(lcs)  
            if len(lcs_seq) > len(doc): continue   # LCS is longer than the doc, can't be a match 

            if seqAlgo.isSubsequence(lcs_seq, doc): # if LCS was derived from patient doc, then at least one match must exist
                
                # find corresponding timestamps 
                # lcs_tseq = lcs_time_series(lcs_seq, i, D, T) # [output] [(<code>, <time>), ...]
                lcsmap[lcs].append(i)  # add the person index
                lcsmapInv[i].append(lcs)
                matched_docIDs.add(i)  # global  
        if len(lcsmapInv[i]) > 0: nD_has_labels += 1 
    
    # analyze LCS -> document IDs
    n_personx = []
    maxNIDs = None  # retain at most only this number of patient IDs; set to None if no limit imposed
    for lcs, docIDs in lcsmap.items(): 
        nd = len(docIDs)
        n_personx.append(nd)  # number of persons sharing the same LCS 

        # subsetting document IDs
        if maxNIDs is not None and len(lcsmap[lcs]) > maxNIDs: 
            lcsmap[lcs] = random.sample(docIDs, min(maxNIDs, nd))  

    nDR = nD_has_labels/(nD+0.0)
    print('info> nD_has_labels: %d (r=%f) > skipped %d persons/documents (without any matched LCS)' % (nD_has_labels, nDR, nD-nD_has_labels))
    eval_lcs_stats() 

    # same document could have multiple LCS labels

    ### save LCS labels to labeled_seq file
    # use LCSs themselves as labels 
    # generate a list of labels matching D
    token_nomatch = Pathway.token_lcs_null  # 'No_Match'
    L = [token_nomatch] * nDoc 
    for i, doc in enumerate(D): 
        if len(lcsmapInv[i]) == 0: 
            # noop 
            pass 
        elif len(lcsmapInv[i]) == 1:  
            L[i] = lcsmapInv[i][0]
        else: 
            # choose the longest? 
            longestFirst = sorted([(lcs, len(lcs.split(lcs_sep))) for lcs in lcsmapInv[i]], key=lambda x:x[1], reverse=True)
            L[i] = longestFirst[0][0]

    return L 

def write(L, cohort='generic', seq_ptype='regular', doctype='labeled', ext='csv', **kargs): 
    import tdoc.TDoc as td # not the same as this TDoc 
    import pandas as pd

    # pattern: condition_drug_<file_type>-<cohort>.csv 
    # fname = TDoc.getName(cohort=cohort, doctype='labeled', ext='csv')  # [params] doc_basename/'condition_drug'
    prefix = kargs.get('basedir', td.prefix)
    fpath = TDoc.getPath(cohort=cohort, seq_ptype=seq_ptype, doctype='labeled', ext='csv', basedir=prefix)  # usually there is one file per cohort  
    assert os.path.exists(fpath), "labeled docuemnt source should have been created previously (e.g. by processDocuments)"
    # assert len(fpaths) > 0, "Could not find any document sources associated with cohort=%s" % cohort

    print('labeling.write> Now filling in the labels to %s (n=1?)' % fpath)
    df_src = pd.read_csv(fpath, sep='|', header=0, index_col=False, error_bad_lines=True)
    print('info> header: %s' % list(df_src.columns))

    lcs_col = 'label_lcs'; assert lcs_col in TDoc.fLabels, "Invalid label attribute: %s" % lcs_col
    if lcs_col in df_src.columns: 
        print('  + Warning: %s already existed.' % lcs_col)
    df_src[lcs_col] = L
    df_src.to_csv(fpath, sep='|', index=False, header=True)
    print('  + IO: saved label_seq (cohort=%s) to:\n%s\n' % (cohort, fpath))
        
    return df_src

def getSurrogateLabels(docs, **kargs): 
    """
    Use labeling utilities from pattern.medcode module. Each disease (cohort) must provide a
    generate() method. 

    Dependency
    ----------
    cohort -> disease -> generate(): which produces a mapping from labels to document positions
    """
    import pattern.medcode as med

    # if labels are given, map them to the right format (i.e. res: label -> document positions)

    # labels = kargs.get('labels', None)
    # if labels is not None: 
    #     assert len(labels) == len(docs)
    #     res = {} # {'dummy_label': [ ]}
    #     for i, l in enumerate(labels): 
    #         if not res.has_key(l): res[l] = []
    #         res[l].append(i)
    #     return res

    # cohort_name = kargs['cohort']
    assert kargs.has_key('cohort'), "No cohort specified."

    nDoc = len(docs)
    # ret = med.getSurrogateLabels(docs, **kargs)  # cohort select appropriate disease which must provide generate()
    # [output] ret: labels -> docs IDs 

    labels = med.label(docs, **kargs)  # [params] cohort

    # convert to a list of labels matching docs positions 
    # labels = []
    # if isinstance(ret, dict):  # a dict mapping from labels to document IDs (positional IDs)
    #     labels = [None] * len(docs)
    #     for label, docIDs in ret.items(): 
    #         for i in docIDs: 
    #             labels[i] = label
    assert hasattr(labels, '__iter__')  # desired format, a list of labels, one for each document
    assert len(labels) == nDoc

    # [condition] no unlabeled docs 
    unlabeled = [i for i, l in enumerate(labels) if l is None]
    assert len(unlabeled) == 0, "There exist unlabeled documents at positions: %s" % unlabeled

    n_classes = len(set(labels))
    if n_classes == 1: print('getSurrogateLabels> One class only!')
    
    return labels  # a list of class labels

def labelize(docs, label_type='doc', class_labels=[], offset=0):
    """
    Automatic labeling for input documents. 
    makeD2VLabels is an older version of this routine. 

    Params
    ------
    label_type: use this string as doc label prefix if class_labels is not provided
    class_labels: if provided, then label the documents using 
                  class labels (but still make each document label 
                  unique) i.e. use class labels as label_type

    """
    # import gensim
    assert TDoc.isListOfTokens(docs, n_test=10), "Ill-formated input docs: %s" % docs

    TaggedDocument = gensim.models.doc2vec.TaggedDocument
    labeledDocs = []

    # testing
    labelx = []
    if len(class_labels) > 0: 
        assert len(docs) == len(class_labels)
        # docLabels = [] # test uniqueness only

        counter = {l: 0 for l in np.unique(class_labels)}
        for i, doc in enumerate(docs): 
            dID = counter[class_labels[i]]
            dID = dID + offset
            label = '%s_%s' % (class_labels[i], dID); labelx.append(label)
            labeledDocs.append(TaggedDocument(doc, [label, ]))
            
            # update document ID of the same class label
            counter[class_labels[i]] += 1
    else: 
        for i, doc in enumerate(docs):
            dID = i + offset
            label = '%s_%s' % (label_type, dID); labelx.append(label)
            labeledDocs.append(TaggedDocument(doc, [label, ]))

    nuniq, ntotal = len(np.unique(labelx)), len(labelx)
    # print('labelize> n_uniq: %d =?= n_total: %d' % (nuniq, ntotal))
    assert len(np.unique(labelx)) == len(labelx), "labels are not unique %d vs %d" % (nuniq, ntotal)
    return labeledDocs

# def combine(tagged_doc1, tagged_doc2):
#     from gensim.models.doc2vec import TaggedDocument
#     docs1, tags1 = tagged_doc1.words, tagged_doc1.tags 
#     docs2, tags2 = tagged_doc2.words, tagged_doc2.tags
        
#     docs = np.concatenate((docs1, docs2))
#     tags = np.concatenate((tags1, tags2))

#     return TaggedDocument() 

def labelize0(sequences, **kargs):
    return makeD2VLabels(sequences, **kargs) 
def makeD2VLabels(sequences, **kargs):   # refactored from seqAnalzyer
    """
    Label the input sequences for the purpose of using Doc2Vec. 

    Adapted from vector.makeD2VLabels() and seqCluster.makeD2VLabels()

    Input
    -----
    sequences: a list of sequence (each of which is a  list of tokens)

    labels: 
    ? a routine that creates labels 

    Output
    ------
    labeled sequences: each sequence becomes a named tuple with attribtues ['words', 'tags', ]

    Related 
    -------
    Base function of the wrapper labelDocuments()

    Memo
    ----
    1. TaggedDocument (& deprecated LabeledSentence) ... 10.23.17

    a. The input to Doc2Vec is an iterator of LabeledSentence objects. Each such object represents a single sentence, 
    and consists of two simple lists: a list of words and a list of labels:
        
        sentence = LabeledSentence(words=[u'some', u'words', u'here'], labels=[u'SENT_1'])

    b. LabeledSentence is an older, deprecated name for the same simple object-type to encapsulate a text-example that is now called TaggedDocument. 
       Any objects that have words and tags properties, each a list, will do. (words is always a list of strings; 
       tags can be a mix of integers and strings, but in the common and most-efficient case, is just a list with a single id integer, 
       starting at 0.)

    """ 
    # from collections import namedtuple  # can customize your own attributes (instead of using gensim's attributes such as words and tags)
    import gensim
    def index_label(i): 
        return '%s_%s' % (label_prefix, i)

    # [params] redundant? 
    # cohort_name = kargs.get('cohort', 'diabetes')
    # seq_ptype = kargs.get('seq_ptype', 'regular')  # values: regular, random, diag, med, lab ... default: regular

    # attributes = D2V.label_attributes # ['codes', 'labels', ]  

    # [old] use gensim.models.doc2vec.TaggedDocument
    # LabelDoc = namedtuple('LabelDoc', attributes) # a namedtuple with 2 attributes words and tags
    # LabelDoc = namedtuple('LabelDoc', ['words', 'labels'])
    label_prefix = seqparams.TDoc.doc_label_prefix 
    exclude = set(string.punctuation)
    all_docs = []

    # [input]
    assert sequences is not None and len(sequences) > 0

    labels = kargs.get('labels', []) # precomputed sentence labels 
    if not labels:  
        # df_ldoc = labelDoc(sequences, load_=load_label, seqr='full', sortby='freq', seq_ptype=seq_ptype)
        raise ValueError, "No user-defined labels given."
        
        # [note] below is for generating surrogate class labels 
        # labeling_routine = kargs.get('labeler', labelDocByFreqDiag)  # any labelDoc*
        # assert hasattr(labeling_routine, '__call__'), "Invalid labeler: %s" % labeling_routine
        # labels = mlabels = labeling_routine(sequences, **kargs)
        # labelx = labelize()
    else: 
        assert len(labels) == len(sequences)

    # label normalization: ensure that each label is a list 
    labelx = TDocTag.labelAsIs(labels) # TDocTag.canonicalize(labels)
    print('makeD2VLabels> doc tag examples:\n%s\n' % labelx[:10])
    # each element in tagx should be a list

    for i, sen in enumerate(sequences):
        if isinstance(sen, str): 
            word_list = sen.split() 
        else: 
            word_list = sen  # split is already done

        # For every sentences, if the length is less than 3, we may want to discard it
        # as it seems too short. 
        # if len(word_list) < 3: continue   # filter short sentences
    
        tagl = labelx[i] # condition tagl is in the list (multilabel) format
        assert isinstance(tagl, list)
        if isinstance(sen, str): 
            sen = ''.join(ch for ch in sen if ch not in exclude)  # filter excluded characters

            all_docs.append(gensim.models.doc2vec.TaggedDocument(sen.split(), tagl))
            # all_docs.append(LabelDoc(sen.split(), tagl))  # format: sequence (list of tokens) + labels (a list of labels)
        else:  

            all_docs.append(gensim.models.doc2vec.TaggedDocument(sen, tagl))
            # all_docs.append(LabelDoc(sen, tagl)) # assuming unwanted char already filetered 

    # Print out a sample for one to view what the structure is looking like    
    # print all_docs[0:10]
    for i, doc in enumerate(all_docs[0:5]+all_docs[-5:]): 
        print('> doc #%d: %s' % (i, doc))
    # [log] e.g. doc #3: LabelDoc(words=['583.81', '250.41', 'V45.81', ... , '48003'], tags=['362.01_599.0_250.51'])

    return all_docs


###### Disease Labeling 
# use pathwayLabeler.py 

def t_labeling(**kargs):
    def get_global_cohort_dir(): 
        return seqparams.getCohortGlobalDir(cohort_name) # basedir: sys_config.read('DataExpRoot')
    def cohort_to_src_file(name='CKD0'): 

        # [todo] keep this in config file
        map_ = {'CKD0': 'eMerge_NKF_Stage_20170818.csv',  # small, n=2388
                'CKD': 'CdwCkdCohort_20170817.csv'}   
        print('  + fname: %s' % map_[name])
        return map_[name]

    import cohort, docProc
    from tdoc import TDoc
    import random

    header = TDoc.header_labeled_seq  # ['sequence', 'timestamp', 'label']
    header_src = ['patientId','Case_Control_Unknown_Status','NKF_Stage'] # ['person_id', 'NKF_Stage', 'CaseControlUnknownStatus', ]

    # use batchpheno.icd9utils.preproc_code() to process CCS ICD9 string
    cohort_name = kargs.get('cohort', 'CKD0')
    outputdir = get_global_cohort_dir()

    # diagCodes = ccs.getCohort(cohort_name)
    # CKD_codes = ['585', '585.1', '585.2', '585.3', '585.4', '585.5', '585.6', '585.9', '792.5', 'V42.0', 'V45.1', 'V45.11', 
    #              'V45.12', 'V56.0', 'V56.1', 'V56.2', 'V56.31', 'V56.32', 'V56.8']
    
    ### load annotated source(s)
    fname = cohort_to_src_file(name=cohort_name) # 'CdwCkdCohort_20170817.csv' (large)
    # dfsrc = cohort.readAnnotatedDataSource(fname=fname, cohort=cohort_name)  # load_labeled_candidates(suffix='query_ids-src')
    fpath = os.path.join(outputdir, fname)
    dfsrc = pd.read_csv(fpath, sep=',', header=0, index_col=False, error_bad_lines=True)

    header_src = list(dfsrc.columns.values)
    labeledPersonIDs = dfsrc['patientId'].values 
    labelset = list(dfsrc['NKF_Stage'].unique())

    ### load MCSs 
    ctype = 'regular'
    src_dir = outputdir
    inputs = ['condition_drug_labeled_seq-CKD.csv', ] 
    D, L, T = docProc.processDocuments(cohort=cohort_name, seq_ptype=ctype, 
                    inputdir=src_dir, 
                    ifiles=kargs.get('ifiles', inputs),
                    
                    # document-wise filtering 
                    policy=kargs.get('doc_filter_policy', 'minimal_evidence'),  
                    min_ncodes=kargs.pop('min_ncodes', 10),  # retain only documents with at least n codes

                    # content modification
                    predicate=kargs.get('predicate', None), 
                    simplify_code=False, 

                    source_type='default', 
                    create_labeled_docs=True)  # [params] composition
    print('t_labeling> Read %d documents' % len(D))
    assert len(labelset) == len(set(L)), "label inconsistent"
    
    Dl = labelize(D, label_type='doc', class_labels=L)
    r = random.randint(0, len(Dl)-1)
    print('  + example:\n%s\n' % str(Dl[r]))

    return 

def t_labeling_cohort(**kargs):
    """

    Memo
    ----
    1. cohort=CKD 
        + sources
          condition_drug_seq-CKD.dat
          condition_drug_seq-CKD.id     // header: ['person_id', ]

        + labels: 
          available in the input file: eMerge_NKF_Stage_20170818.csv

          sometimes labels need to be computed

    """ 
    def read_ids(fname): 
        assert fname.find('.id') > 0
        fp = os.path.join(basedir, fname)
        assert os.path.exists(fp), 'Invalid input: %s' % fp
        df_id = pd.read_csv(fp, sep=sep, header=0, index_col=False, error_bad_lines=True)
        return df_id['person_id'].values
    def seq_to_str(seqx, sep=','): 
        return [sep.join(str(s) for s in seq) for seq in seqx] 
    def str_to_seq(df, col='sequence', sep=','):
        seqx = []
        for seqstr in df[col].values: 
            s = seqstr.split(sep)
            seqx.append(s)
        return seqx
    def to_str(tokens, sep='+'): 
        return sep.join([str(tok) for tok in tokens])

    # import labeling
    import seqReader as sr

    ### CKD cohort 
    # basedir = sys_config.read('DataIn')  # data-in simlink to data ... 10.17 
    # 'data-in' is reserved for input data not generated from within the system 
    basedir = sys_config.read('DataExpRoot')
    
    # cohort attributes
    cohort_name = 'CKD'
    fname = 'eMerge_NKF_Stage_20170818.csv'    
    header = ['patientId', 'Case_Control_Unknown_Status', 'NKF_Stage', ]
    sep = ','

    fpath = os.path.join(basedir, fname)
    df = pd.read_csv(fpath, sep=sep, header=0, index_col=False, error_bad_lines=True)

    # use stages as labels 
    labelset = list(df['NKF_Stage'].unique())
    print('info> cohort: %s | labels (n=%d):\n%s\n' % (cohort_name, len(labelset), labelset))
    # [log] 7 labels

    labels = df['NKF_Stage']

    # only read documents with data 
    idx = person_ids = getPersonIDs(cohort=cohort_name, inputdir=basedir, sep=sep)  # cohort, inputdir, sep, sep_compo
    seqparams.TDoc.verifyIDs(idx)  # ascending order? duplicates? 

    n_persons = len(idx)
    print('info> n_persons: %d' % n_persons)

    ### find labels
    # don't use the data source's ordering of IDs, which when made via seqMaker2.py was sorted
    # ERROR: labels = df.loc[df['patientId'].isin(idx)]['NKF_Stage'].values
    
    sort_keys = ['patientId', ]
    # df_test1 = df.sort_values(sort_keys, ascending=True)
    # l = df['patientId'].values
    # assert all(l[i] <= l[i+1] for i in xrange(len(l)-1))   # passed
    # assert all(l == idx) # passed

    # filter > sort > extract (good!)
    # output np.array
    # labels = labels_ref = labelDocByDataFrame(, person_ids=idx, id_field='patientId', label_field='NKF_Stage')
    labels = labels_ref = labelDocByFile(fpath, person_ids=idx, id_field='patientId', label_field='NKF_Stage')

    # n_labels = len(labels)
    # print('info> Got %d labels' % n_labels)

    # [test] verify the ID and labels
    # print('status> verifying the match between IDs and labels')
    # for i, (r, row) in enumerate(df_test1.iterrows()): # sorted entries
    #     pid, label = row['patientId'], row['NKF_Stage']
    #     if pid in idx: 
    #         assert label == labels[i], "%d-th label: %s <> %s" % (i, label, labels[i])
    ## [conclusion] the label ordering via df_test1 does not agree!!! 

    # extract labels according to the ID ordering
    # sampleIDs = random.sample(range(n_persons), 50)
    # labels = []
    # for pid in idx: 
    #     row = df.loc[df['patientId']==pid]  # row is a dataframe
    #     assert row.shape[0] == 1, 'Found dups: id=%s => %s' % (pid, row)
    #     l = list(row['NKF_Stage'].values)
    #     labels.extend(l)
    # assert len(labels) == len(labels_ref) == len(idx)    # passed
    # assert all(labels_ref == labels), "ordering inconsistency:\n%s\n VS \n%s\n" % (labels_ref[:50], labels[:50])  # passed

    n_labels = len(labels)
    print('info> verified %d labels' % n_labels)

    # double check with structured version of the sequences produced by seqMaker2 (header: person_id, sequence, timestamp)
    # tfile = 'condition_drug_timed_seq-%s.csv' % cohort_name # test file
    # fpath2 = os.path.join(basedir, tfile)
    # # if os.path.exists(fpath2): 
    # dft = pd.read_csv(fpath2, sep='|', header=0, index_col=False, error_bad_lines=True)
    # print('info> from timed_seq .csv | n_persons: %d =?= n_labels: %d' % (dft.shape[0], n_labels)) # n_persons: 2833 =?= n_labels: 2833

    ### Read Sequences

    print('info> 1. CSeq from .csv')
    ret = readDocFromCSV(cohort=cohort_name, inputdir=basedir)
    print('info> making structured format of the coding sequences (cohort:%s, n_labels:%d)' % (cohort_name, n_labels))
    # df = readToCSV(cohort=cohort_name, labels=labels)
    
    seqx = ret['sequence'] # list(dft['sequence'].values)
    tseqx = ret.get('timestamp', []) # list(dft['timestamp'].values)
    if tseqx: 
        assert len(seqx) == len(tseqx), "len(seqx)=%d, len(times)=%d" % (len(seqx), len(tseqx))

    print('info> 2. CSeq from .dat')
    seqx2, tseqx2 = readDoc(cohort=cohort_name, inputdir=basedir, include_timestamps=True) # ifiles

    # can then create .csv via readDocToCSV()  # [params]  cohort, basedir, ifiles, labels

    if tseqx2: 
        assert len(seqx2) == len(tseqx2), "len(seqx)=%d, len(times)=%d" % (len(seqx2), len(tseqx2))
    n_docs, n_docs2 = len(seqx), len(seqx2)
    
    # print('info> read %d from .dat =?= %d from .csv' % (n_docs2, n_docs))
    assert n_docs == n_docs2, ".dat and .csv formats are not consistent n_doc: %d (csv) <> %d (dat)" % (n_docs, n_docs2)

    # when did they diverge? 
    # n_matched = 0
    # for i, seq in enumerate(seqx): 
    #     s1 = seq # list of tokens

    #     try: 
    #         s2 = seqx2[i] # list of tokens
    #     except: 
    #         s2 = []

    #     if s1 == s2: 
    #         n_matched += 1 
    #     else: 
    #         msg = ".csv not consistent with .dat (n_matched=%d)=>\n%s\nVS\n%s\n" % (n_matched, s1, s2)
    #         raise ValueError, msg 

    n_docs_src = df.shape[0]
    assert n_docs == n_labels, "n_labels: %d <> n_docs: %d ..." % (n_labels, n_docs)

    print('input> n_doc_src (cohort source document): %d, n_doc (parsed, has data in DB): %d' % (n_docs_src, n_docs))

    print('info> writing labels to .csv')
    df2 = sr.readDocToCSV(cohort=cohort_name, labels=labels)
    print('info> created .csv format with columns:\n%s\n' % df2.columns.values)
    n_docs3 = df2.shape[0]
    assert n_docs3 == n_docs

    return

def test(**kargs):

    t_labeling()

    return 

if __name__ == "__main__": 
    test()


