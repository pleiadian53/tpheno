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
from pattern import medcode as pmed  # disease-specific phenotyping utilities

import seqparams

# related 
import labeling

################################################################################################
#
#  1. Use pathwayAnalzyer (and seqCluster) to mine frequent n-grams, LCSs that characterize 
#     the disease-specific coding sequences 
#
#  2. Then use this module to create class labels
#
#

def label(seqx, **kargs): # [todo]
    """

    Input: 

    Params: 
        min_count: minimum number of occurrences of disease codes before declaring positive

    Output: 

    """
    
    # encode documents in phenotypic/class labels 
    labelsets = phenotypeDoc(seqx, **kargs)  # [params] min_count, seq_ptype, cohort
    ret = labelToID(labelsets)  # this is only label-to-ID format

    # convert to a list of labels matching docs positions 
    labels = [None] * len(seqx)
    ulabels = set()
    if kargs.get('numeric_', True)
        for label, docIDs in ret.items():  # 
            if not isinstance(label, int): continue
            for i in docIDs: 
                labels[i] = label
            ulabels.add(label) # [test]
    else: # canonical
        for label, docIDs in ret.items(): 
            if isinstance(label, int): continue
            for i in docIDs: 
                labels[i] = label
            ulabels.add(label) # [test]

    assert len(ulabels) * 2 == len(ret), "Inconsistent canonical and numeric labels (got: %d but total: %d)" % \
        (len(ulabels), len(ret))
    # [condition] no unlabeled docs 
    unlabeled = [i for i, l in enumerate(labels) if l is None]
    assert len(unlabeled) == 0, "There exist unlabeled documents at positions: %s" % unlabeled

    return labels

def decode(multilabel):

    return (1, 1,) 
def labelToID(labelsets):
    """
    Map labels in multilabel format (e.g. [1, 0, 0] as type I diabetes) to: 
        lv: canonical class label 
        lc: numerical class label

    """
    ret = {}
    for i, lset in enumerate(labelsets): 
        lkey = tuple(lset)
        
        lv, lc = decoder(lkey)
        # lv, lc = Diabetes.decode(lkey)  # canonical class label, numeric class label 
        
        for l in [lv, lc, ]: 
            if not ret.has_key(l): ret[l] = [] 
 
        # canonical class label 
        ret[lv].append(i)

        # numeric class label
        ret[lc].append(i)
    return ret 

def getNumericLabels(seqx, **kargs):
    kargs['numeric_'] = True
    return label(seqx, **kargs)

def getCanonicalLabels(seqx, **kargs):
    kargs['numeric_'] = False 
    return label(seqx, **kargs)
def getNamedLabels(seqx, **kargs):
    kargs['numeric_'] = False 
    return label(seqx, **kargs) 

# [policy]
def phenotype(doc, **kargs):
    min_count = kargs.get('min_count', 1) 

    # [policy]
    # n_type1 = n_type2 = n_birth = 0
    # labels = [0, 0, 0]  # default: False for all labels (in the order of type I, II, gestational)
    # for e in doc: 
    #     if is_diabetes_type1(e): 
    #         n_type1 += 1 
    #         if n_type1 >= min_count: labels[0] = 1
    #     if is_diabetes_type2(e): 
    #         n_type2 += 1 
    #         if n_type2 >= min_count: labels[1] = 1
    #     if is_diabetes_gestational(e): 
    #         n_birth += 1
    #         if n_birth >= min_count: labels[2] = 1

    return labels # multilabel

def toBinaryLabel(docs, **kargs): 
    """

    Related
    -------
    phenotypeIndex(labelsets, **kargs)
    """
    # 1. specify the positive class

    pass


# [precond] input sequences/docs should contain diagnostic codes
def phenotypeDoc(docs, **kargs): # this produces mutliple labels
    """
    phenotype input documents (consisting of temporal sequences of codes) according to 
    the mention of related diagnostic codes. 

    This routine generates the labeling in a vector form (represented by tuple)

    Related 
    -------
    1. phenotypeIndex()

    """
    pass

def t_labeling(**kargs):
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

    import labeling  # basic labeling utilities
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
    idx = person_ids = labeling.getPersonIDs(cohort=cohort_name, inputdir=basedir, sep=sep)  # cohort, inputdir, sep, sep_compo
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
    labels = labels_ref = labeling.labelDocByFile(fpath, person_ids=idx, id_field='patientId', label_field='NKF_Stage')

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
    ret = sr.readDocFromCSV(cohort=cohort_name, inputdir=basedir)
    print('info> making structured format of the coding sequences (cohort:%s, n_labels:%d)' % (cohort_name, n_labels))
    # df = readToCSV(cohort=cohort_name, labels=labels)
    
    seqx = ret['sequence'] # list(dft['sequence'].values)
    tseqx = ret.get('timestamp', []) # list(dft['timestamp'].values)
    if tseqx: 
        assert len(seqx) == len(tseqx), "len(seqx)=%d, len(times)=%d" % (len(seqx), len(tseqx))

    print('info> 2. CSeq from .dat') # this is obsolete, .csv is guaranteed to be generated at seqMaker2 level
    seqx2, tseqx2 = sr.readDoc(cohort=cohort_name, inputdir=basedir, include_timestamps=True) # ifiles

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
    return


if __name__ == "__main__":
    test()