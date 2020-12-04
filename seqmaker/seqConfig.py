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
from tsHandler import tsHandler

from seqparams import Pathway

import evaluate  # classification
import algorithms, seqAlgo  # count n-grams, sequence-specific algorithms
import labeling

# multicore processing 
import multiprocessing

# evaluation 
from sklearn import metrics
from scipy.spatial import distance
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from scipy import interp
import scipy

##### Set module variables ##### 
GNFeatures = vector.W2V.n_features # [note] descendent: seqparams.W2V -> vector.W2V
GWindow = vector.W2V.window
GNWorkers = vector.W2V.n_workers
GMinCount = vector.W2V.min_count

#########################################################################################################
#
#  An abtraction for handling MCSs and the corresponding training sets. 
#
#
#
#########################################################################################################


# wrapper class: training set handler 
class tsHandler(tsHandler):  # defined only for backward compatibility
    """
    Training data management. 

    This is subsumed by tsHanlder.tsHandler 
    """

    # default parameter values
    is_augmented = False 
    cohort = 'CKD'
    seq_ptype = ctype = 'regular'
    d2v = d2v_method = 'pv-dm2'  # pv-dm, bow
    is_simplified = False
    meta = 'D'   # values: {D/default, A/augmented, S/subset}
    dir_type = 'combined'

    # class state
    is_configured = False

    N_train_max = N_max = 10000  # max training examples
    N_test_max = 5000

### end class tsHandler

class lcsHandler(object): 

    cohort = 'CKD' # tsHandler.cohort
    seq_ptype = ctype = 'regular' # [design] this probably has to be in synch with tsHanlder
    
    # unused
    d2v = d2v_method = 'pv-dm2'  # pv-dm, bow  # ... not used here
    is_simplified = False
    
    meta = None
    label = None  # class label

    slice_policy = 'noop'
    lcs_type = 'global'
    lcs_policy = 'df'  # 'uniq', 'df'/'document_frequency'
    consolidate_lcs = True

    # pathway analysis 
    pattern_types = ['enriched', 'rare', ]   # e.g. shall we use enriched or (relatively) rare patterns to characterize a disease subgroup

    # relabel 
    label_map = {}

    is_configured = False

    @staticmethod
    def config(lcs_type='global', lcs_policy='df', **kargs): 
        if not lcsHandler.is_configured: 
            lcsHandler.cohort = kargs.get('cohort', tsHandler.cohort)
            lcsHandler.seq_ptype = lcsHandler.ctype = kargs.get('seq_ptype', tsHandler.ctype)
            lcsHandler.lcs_type = lcs_type
            lcsHandler.lcs_policy = lcs_policy

            lcsHandler.slice_policy = kargs.get('slice_policy', 'noop')
            lcsHandler.consolidate_lcs = kargs.get('consolidate_lcs', True)

            # meta
            user_file_descriptor = lcsHandler.meta = kargs.get('meta', tsHandler.meta)

            # unused for now 
            lcsHandler.is_simplified = tsHandler.is_simplified
            lcsHandler.d2v = lcsHandler.d2v_method = tsHandler.d2v_method
 
            lcsHandler.is_configured = True
        return 

    @staticmethod
    def set_sequence_type(ctype):
        # import seqparams
        lcsHandler.ctype = lcsHandler.seq_ptype = seqparams.normalize_ctype(ctype) 
        tsHandler.ctype = tsHandler.seq_ptype = lcsHandler.ctype  # [design] probably has to be in synch 

        return

    @staticmethod
    def do_slice(): # only for file ID, see secondary_id()
        if lcsHandler.slice_policy.startswith(('noop', 'reg', 'compl', )):  # effective operations 
            return False 
        return True

    @staticmethod
    def secondary_id(label=None): # attach extra info in suffix
        """

        Memo
        ----
        1. example path 
           <prefix>/tpheno/seqmaker/data/CKD/pathway/lcs_local-df-regular-iso-LControl-Uenriched.csv
               
                file_stem: lcs 
                scope: local 
                policy: df
                suffix: regular-iso-L<label> 
                        where label <- Control 
                meta: a pattern type {'enriched', 'rare', }

        """
        ctype = lcsHandler.ctype
        suffix = ctype # kargs.get('suffix', ctype) # user-defined, vector.D2V.d2v_method
        if lcsHandler.do_slice(): suffix = '%s-%s' % (suffix, lcsHandler.slice_policy)
        if lcsHandler.consolidate_lcs: suffix = '%s-%s' % (suffix, 'iso') # LCSs up to permutations consider as the same? 
        # suffix = '%s-len%s' % (suffix, kargs.get('min_length', 3))

        label_id = lcsHandler.label if label is None else label
        if label_id is not None: 
            suffix = '%s-L%s' % (suffix, label_id)
        # if kargs.get('simplify_code', False):  suffix = '%s-simple' % suffix
        
        meta = lcsHandler.meta
        if meta is not None: suffix = "%s-U%s" % (suffix, meta) 

        return suffix  

    @staticmethod
    def load_lcs(lcs_type=None, lcs_select_policy=None, meta=None, ctype=None): # [note] for now, only used in makeLCSTSet()
        """

        Input
        -----
        lcs_type: {'global', 'local'} 
            global: LCSs are computed under the global scope (i.e. the entire patient cohort regardless of the underlying subgroups)
            local: LCSs are computed under the local scope (i.e. from within each subgroup)

            e.g. Take the CKD cohort as an example, enriched LCS patterns differ from stage to stage. 


        Memo
        ----
        1. example path to the LCS-pattern file
           <prefix>/tpheno/seqmaker/data/CKD/pathway/lcs_local-df-regular-iso-LControl-Uenriched.csv
               
                file_stem: lcs 
                scope: local 
                policy: df
                
                <secondary id> regular-iso-LControl-Uenriched    //seq_ptype/ctype (regular) is implicit here, a class attribute not a function argument (**kargs)
                    suffix: regular-iso-L<label> 
                            where label <- Control 
                    meta: a pattern type {'enriched', 'rare', }

        """
        # params: cohort, lcs_type, lcs_policy, ctype, consolidate_lcs, label, meta
        ctype0 = lcsHandler.ctype
        if ctype is not None: lcsHandler.ctype = ctype  # {'regular', 'med', 'diag'}; this is used implicitly in secondary_id

        if meta is not None: lcsHandler.meta = meta  # e.g. enriched, rare

        adict = lcsHandler.lcs_file_id()
        cohort = adict['cohort']

        if lcs_type is None: lcs_type = adict['lcs_type']
        if lcs_select_policy is None: lcs_select_policy = adict['lcs_policy']

        suffix = adict['meta'] 

        # [design]
        if ctype is not None: lcsHandler.ctype = ctype0  # restore ctype so that this operation does not affect the global setting

        df = Pathway.load(cohort, scope=ltype, policy=lcs_select_policy, suffix=suffix, dir_type='pathway')
        if df is None: 
            return DataFrame() # dummy dataframe
        return df

    @staticmethod
    def save_lcs(df, lcs_type=None, lcs_select_policy=None, ctype=None):  
        """

        Params
        ------
        label, lcs_type, lcs_policy, seq_ptype, meta: {enriched, rare, ...},
        """
        ctype0 = lcsHandler.ctype
        if ctype is not None: lcsHandler.ctype = ctype  # {'regular', 'med', 'diag'}; this is used implicitly in secondary_id
        
        if meta is not None: lcsHandler.meta = meta  # e.g. enriched, rare 
        adict = lcsHandler.lcs_file_id()
        cohort = adict['cohort']

        if lcs_type is None: lcs_type = adict['lcs_type']
        if lcs_select_policy is None: lcs_select_policy = adict['lcs_policy']

        suffix = adict['meta']         
        # fpath = Pathway.getFullPath(cohort=cohort, scope=ltype, policy=lcs_select_policy, suffix=suffix) # [params] dir_type='pathway' 
        print('lcsHandler> saving LCS labels ...')
        Pathway.save(df, cohort=cohort, scope=ltype, policy=lcs_select_policy, suffix=suffix, dir_type='pathway')

        # [design]
        if ctype is not None: lcsHandler.ctype = ctype0  # restore ctype so that this operation does not affect the global setting
        return

    @staticmethod
    def load(): 
        """
        Load dense LCS-feature training set. 
        """
        raise NotImplementedError

    @staticmethod
    def loadSparse(policy_fs='sorted'): 
        """
        Load sparse LCS-feature training set. 
        """
        policyFeatureRank = kargs.get('policy_fs', policy_fs)  # sorted, tfidf, tf, ... 
        fileID = meta = lcsHandler.meta
        if policyFeatureRank.startswith('so'):
            fileID = 'sorted' if meta is None else '%s-sorted' % lcsHandler.meta
            
        X, y = TSet.loadSparseLCSFeatureTSet(cohort=lcsHandler.cohort, d2v_method=lcsHandler.d2v_method, 
                    seq_ptype=lcsHandler.ctype, suffix=fileID)
        return (X, y)

    @staticmethod
    def lcs_file_id(): # params: {cohort, scope, policy, seq_ptype, slice_policy, consolidate_lcs, length}
        # [note] lcs_policy: 'freq' vs 'length', 'diversity'
        #        suppose that we settle on 'freq' (because it's more useful) 
        #        use pairing policy for the lcs_policy parameter: {'random', 'longest_first', }
        adict = {}
        adict['cohort'] = lcsHandler.cohort  # this is a mandatory arg in makeLCSTSet()

        # e.g. {'global', 'local'}
        adict['lcs_type'] = ltype = lcsHandler.lcs_type  # global: Common LCSs defined in global scope (as opposed to cluster or local scopes)
        adict['lcs_policy'] = lcsHandler.lcs_policy # definition of common LCSs; e.g. {'uniq', 'df', }

        # use {seq_ptype, slice_policy, length,} as secondary id
        adict['suffix'] = adict['meta'] = suffix = lcsHandler.secondary_id()
        return adict

### end class lcsHandler


# utility functions 

def updateDescriptor(base, *args, **kargs):
        bp = base
        fsep = '-'

        params = set(bp.split(fsep))
        print('... %s' % params)
        for arg in args: 
            if not arg in params: 
                bp = '%s-%s' % (bp, arg)
            params.add(arg)
        
        for param, value in kargs.items(): 
            assert param is not None and len(param) > 0, "Invalid parameter: %s" % param
            if value is None and (not param in params): 
                bp = '%s-%s' % (bp, param)
                params.add(param)
                print('... + %s' % params)
            else: 
                v = '%s%s' % (param, value)
                if v in params:  # don't include the same descriptor
                    pass
                else: 
                    bp = '%s-%s' % (bp, v)
                    params.add(v)
                    print('... + %s' % params)
        return bp

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
    import seqparams 

    sysparams.cohort = cohort # system parameters are shared across modules 
    sysparams.label_map = relabel()

    # training set parameters 
    if d2v_method is None: d2v_method = seqparams.D2V.d2v_method  # [note] also use vector.D2V.d2v_method
    user_file_descriptor = meta
    tsHandler.config(cohort=sysparams.cohort, d2v_method=d2v_method, 
                    seq_ptype=kargs.get('seq_ptype', 'regular'),
                    meta=user_file_descriptor)  # is_augmented/False, is_simplified/False, dir_type/'combined'

    lcsHandler.config(lcs_type=kargs.get('lcs_type', 'global'), 
        lcs_policy=kargs.get('lcs_policy', 'df'),
        consolidate_lcs=kargs.get('consolidate_lcs', True), 
        slice_policy=kargs.get('slice_policy', 'noop'),
        meta=user_file_descriptor)

    return 


