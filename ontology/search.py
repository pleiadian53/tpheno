# encoding: utf-8

import pymssql
# import mssql_config

import pandas as pd
from pandas import DataFrame, Series
import os, gc, sys
from os import getenv 
import time, re, string, collections

try:
    import cPickle as pickle
except:
    import pickle

import random
import scipy
import numpy as np

# local modules 
from batchpheno import icd9utils, utils, predicate, qrymed2
from batchpheno.utils import div, indent
from config import seq_maker_config, sys_config

from pattern import medcode as pmed
import seqparams

try: 
    import targets
except: 
    msg = 'Could not import target set'
    raise ValueError, msg

def t_map_lab(**kargs): 
    """

    Memo
    ----
    1. refactored from seqmaker.cohort 
    
    """
    import sys, re, gc
    from batchpheno import qrymed2
    from pattern import medcode
    
    cohort_name = 'PTSD'
    ctrl_cohort_name = '%s-Negative' % cohort_name
    projdir = sys_config.read('ProjDir')  # '/phi/proj/poc7002/tpheno'
    basedir = os.path.join(projdir, cohort_name)  # source 
    outputdir = kargs.get('output_dir', sys_config.read('DataExpRoot')) # temporary (output) data dir

    # header: person_id, measurement_date, measurement_time, value_as_number, value_source_value
    # augmented: person_id, measurement_date, measurement_time, value_as_number, value_source_value, source_description
    table_name = 'measurement'
    pivots = ['value_source_value', ]  # lab codes
    fname = '%s-code_desc-%s.csv' % (table_name, cohort_name)
    fpath = os.path.join(outputdir, fname)

    print('io> loading code-to-desc from: %s' % fpath)
    df_lab = pd.read_csv(fpath, sep='|', header=0, index_col=False, error_bad_lines=True)
    print('status> loaded lab table (cohort:%s) > dim: %s' % (cohort_name, str(df_lab.shape)))

    
    div('step 2> compute mapping from code to entity measured and specimen ...')
    # [params]
    # basedir for local data
    basedir = os.path.join(os.getcwd(), 'data') 
    print('status> change basedir to %s' % basedir)
    load_data = True

    # preprocessing 
    superset = qrymed2.getDescendent(1) # '70745'
    print('info> size of total MED codes: %d' % len(superset))
    header = ['code', 'entity_measured', 'specimen']
    edict = {h:[] for h in header}
    n_entries = ne = nes = 0
    
    df = None
    fpath = os.path.join(basedir, 'code_measure_specimen.csv')
    # temp_path = os.path.join(basedir, 'code_measure_specimen.csv')
    if load_data and (os.path.exists(fpath) and os.path.getsize(fpath) > 0): 
        # mdict = pickle.load(open(temp_path, 'rb'))
        df = pd.read_csv(fpath, sep='|', header=0, index_col=False, error_bad_lines=True)
    else: 
        for code in superset: 
            slot = 16
            cmd = 'qrymed -val %s %s' % (slot, code)  # e.g. 35789: CPMC Laboratory Test: Lym  
            entity_measured = qrymed2.eval(cmd, verbose=False)       # e.g. 32028 - Lymphocytes
            if entity_measured is not None: 
                edict['code'].append(code) 
                edict['entity_measured'].append(entity_measured)

                slot = 14  # assesses sample (e.g. 46125: body fluid cell count specimen)
                cmd = 'qrymed -val %s %s' % (slot, code)
                specimen = qrymed2.eval(cmd, verbose=False) 
                if specimen is not None: 
                    nes += 1
            
                edict['specimen'].append(specimen) # could be None? 
                ne += 1 
    
        df = DataFrame(edict, columns=header)
        print('io> saving code-to-lab-measurement map (n_rows:%d) to %s' % (ne, fpath))
        df.to_csv(fpath, sep='|', index=False, header=True)
        # pickle.dump(edict, open(temp_path, "wb" ))
        print("info> n_codes with 'entity measured': %d > %d also has 'assesses sample'" % (df.shape[0], nes))    

    assert df is not None and not df.empty
    edict = None; gc.collect()

    # header: ['value_source_value', 'source_description']
    has_measurements = []  # codes with valid/available measurements
    mdict = {} # code -> measurement and specimen
    cols = ['measure', 'specimen', ]
   
    # first put loaded data in maps 
    edict0 = dict(zip(df['code'].values, df['entity_measured'].values))
    sdict0 = dict(zip(df['code'].values, df['specimen'].values))

    edict, sdict = {}, {}  # map: entity measured | map: specimen
    # for code in df['value_source_value'].values: 
        

    # foreach code that has measurement(s), find its ancetors
    # temp_path = os.path.join(basedir, 'ancestor_map.pkl')
    # ansmap = {}  # ancestor map: code -> ancestors 
    # if load_data and (os.path.exists(temp_path) and os.path.getsize(temp_path) > 0): 
    #     mdict = pickle.load(open(temp_path, 'rb'))
    # else: 
    #     for c, attr in mdict.items():  # for each code with 'entity measured' 
    #         ancestors = qrymed2.getAncestor(c) # default: to_int=True
    #         if ancestors is not None: 
    #             ansmap[c] = ancestors
            
    return

def test(**kargs): 
    return

if __name__ == "__main__": 
    test() 