import numpy as np
import multiprocessing

import os, sys, re, random, time, math
import collections, gc
from datetime import datetime 

# pandas 
from pandas import DataFrame
import pandas as pd 

# gensim 
from gensim.models import Doc2Vec
import gensim.models.doc2vec

# tpheno
import seqReader as sr
import seqparams, seqAlgo
import labeling 
from batchpheno.utils import div

gHasConfig = True
try: 
    from config import seq_maker_config, sys_config
except: 
    print("tdoc> Could not find default configuration package.")
    gHasConfig = False

from tdoc import TDoc

################################################################################################################
#
#  seq2seq
#     sequence prediction using deep learning framework (e.g. LSTM) 
# 
#
#  Use 
#  ---
#  1. predict CKD labels
#
#
################################################################################################################

def processDocuments0(**kargs):
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
    import seqClassify as sclf 
    return sclf.processDocuments(**kargs)

def getCodeBook(D): # get the set of all vocabulary in a sorted order
    def test_format(): 
        nD = len(D)
        j = random.randint(0, nD-1)
        assert isinstance(D[j], list)
        return
    def code_stats(): # params: Vs 
        print("  + min code (by alpha order): %s, max code: %s" % (Vs[0], Vs[-1]))
        nV = len(Vs)
        print("  + vocabulary size: %d" % nV)
        return

    import seqTransform as st # in case we need to filter out unwanted codes

    test_format()

    V = set()
    for seq in D: 
        V.update(seq)

    Vs = sorted(V, reverse=False) # ascending order
    code_stats()

    return Vs

def processDocuments(): 

    from tset import TSet # mark training data
    from seqparams import Pathway # set pathway related parameters, where pathway refers to sequence objects such as LCS
    from tdoc import TDoc  # labeling info is also maintained in document sources of doctype = 'labeled'
    # import seqTransform as st

    verify_labeled_file = True
    cohort = kargs.get('cohort', 'PTSD')
    seq_ptype = kargs.get('seq_ptype', 'regular') 
    d2v_method = kargs.get('d2v_method', seqparams.D2V.d2v_method)

    # D: corpus, T: timestamps, L: class labels
    # load + transform + (labeled_seq file)
    # [todo] set save_to True to also save a copy of transformed document (with labels)? 
    D, T, L = processDocuments0(cohort=cohort, seq_ptype=seq_ptype, 
               predicate=kargs.get('predicate', None),
               simplify_code=kargs.get('simplify_code', False), 
               ifiles=kargs.get('ifiles', []), 

               # splice operations {'noop', 'prior', 'posterior', }
               splice_policy=kargs.get('splice_policy', 'noop'), 
               splice_predicate=kargs.get('splice_predicate', None), 
               cutpoint=kargs.get('cutpoint', None),
               inclusive=True) 
    Vs = getCodeBook(D)  # make one-hot coding according to this codebook
    

    return 

def test(**kargs):
    return

if __name__ == "__main__": 
    test()

