# encoding: utf-8

import pandas as pd
from pandas import DataFrame, Series

import matplotlib
matplotlib.use('Agg') # use a non-interactive backend such as Agg (for PNGs), PDF, SVG or PS.
from matplotlib import pyplot as plt

import os, sys, collections, re, glob
from scipy.cluster.hierarchy import dendrogram, linkage
import numpy as np

from scipy.cluster.hierarchy import cophenet
from scipy.spatial.distance import pdist

ProjDir = '/Users/pleiades/Documents/work/sprint'
DataDir = os.path.join(ProjDir, 'data')

class TSet(object):
    index_field = 'MASKID'
    date_field = 'date'
    target_field = 'target'  # usually surrogate labels
    annotated_field = 'annotated'
    content_field = 'content'  # representative sequence elements 
    label_field = 'mlabels'  # multiple label repr of the underlying sequence (e.g. most frequent codes as constituent labels)

    meta_fields = [target_field, index_field, ]

# end TSet 

def transform(ts, f_target=None, **kargs): 
    from seqmaker import evaluate 
    if not kargs.has_key('meta_fields'): kargs['meta_fields'] = TSet.meta_fields
    return evaluate.transform(ts, f_target=f_target, **kargs)

def standardize(X, y=None, method='minmax'): 
    from seqmaker import evaluate
    return evaluate.standardize(X=X, y=y, method=method)

def source(**kargs): 
    # input data sources
    ipath = kargs.get('ipath', DataDir) # input path
    ext = 'csv'
    dfiles = [name for name in glob.glob('%s/*.%s' % (ipath, ext))]
    print('io> all data files:\n%s\n' % dfiles) # full paths
    
    return dfiles # full paths

def get_table(name='baseline', **kargs): 
    """

    Assumption
    ----------
    1. at most only one table will match the name
    """
    dfiles = source(**kargs)
    
    for dfile in dfiles: 
        if os.path.basename(dfile).find(name) >= 0: 
            return dfile 
    return None

def get_attributes(name='baseline'):  # attributes under study 
    """

    Memo
    ----
    1. feature set 
       baseline.csv 

       "MASKID","INTENSIVE","NEWSITEID","RISK10YRS","INCLUSIONFRS","SBP","DBP","N_AGENTS","NOAGENTS","SMOKE_3CAT","ASPIRIN","EGFR","SCREAT","SUB_CKD","RACE_BLACK","AGE","FEMALE","SUB_CVD","SUB_CLINICALCVD","SUB_SUBCLINICALCVD","SUB_SENIOR","RACE4","CHR","GLUR","HDL","TRR","UMALCR","BMI","STATIN","SBPTERTILE"
    """
    adict = {}  # table names -> feature sets

    # 6 + (5) eligibility traits && 9 extra 
    adict['baseline'] = ['risk10yrs', 'sbp', 'n_agents', 'egfr', 'age', 'sub_cvd'] + \
                        ['dbp', 'smoke_3cat', 'screat', 'CHR', 'GLUR', 'HDL', 'TRR', 'UMALCR', 'BMI']  

    # baseline: unused from exclusion criteria
    # ['diabetes', 'stroke', 'heart_failure', 'proteinuria', 'end_stage_renal_disease']
    fset = adict.get(name, [])
    return [f.upper() for f in fset]

def make_tset(**kargs): 
    basedir = DataDir
    print('info> basedir: %s' % basedir)
    
    index_field = 'MASKID'
    table_name = 'baseline'
    experiment_id = 'sprint'
    identifier = 'E%s-T%s' % (experiment_id, table_name)

    fpath = get_table(name=table_name, **kargs)
    assert os.path.exists(fpath), "Invalid input: %s" % fpath
    df = pd.read_csv(fpath, sep=',', header=0, index_col=False, error_bad_lines=True)
    print('io> loaded data of dim: %s' % str(df.shape)) # [log] dim: (9361, 30)

    fset = get_attributes(table_name)
    n_features = len(fset)
    assert n_features > 0
    print('info> n_features: %d' % n_features) # 15

    # if not index_field in fset: 
    #     # fset = [index_field] + fset
    #     fset.insert(0, index_field)
    # len(fset)
    assert len(set(fset)-set(df.columns)) == 0, "not all features present!"

    ts = df[fset]
    ts[index_field] = df[index_field]

    nrow0 = ts.shape[0]
    ts = ts.dropna(axis=0, how='any')  # inplace=True
    nrow = ts.shape[0]

    # [log] row before dropna: 9361, after: 8808 > dim: (8808, 16)
    print('info> nrow before dropna: %d, after: %d > dim: %s' % (nrow0, nrow, str(ts.shape)))

    fname = 'tset-%s.csv' % identifier
    fpath = os.path.join(basedir, fname)
    print('io> saving training set file to %s' % fpath)
    ts.to_csv(fpath, sep=',', index=False, header=True)

    return ts

def test(**kargs): 

    ### make SPRINT training data for clustering 
    make_tset()

    return

if __name__ == "__main__": 
    test()


