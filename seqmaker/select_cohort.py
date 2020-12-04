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

######################################################################################################
#
#
#  Reference
#  ---------
#    1. string formatting (https://pyformat.info/)
#
#
######################################################################################################

### System Variables ### 
DBScope = 'ohdsi.west'
tb_condition = 'condition_occurrence'
tb_drug = 'drug_exposure'  # ohdsi.west.drug_exposure
tb_measure = 'measurement' # ohdsi.west.measurement

fp_condition = '%s.csv' % tb_condition
fp_drug = '%s.csv' % tb_drug
fp_measure = '%s.csv' % tb_measure


Q_Condition_Ref = """
SELECT person_id, condition_start_date, condition_source_value
FROM ohdsi.west.condition_occurrence
WHERE person_id in (SELECT DISTINCT person_id
                    FROM ohdsi.west.condition_occurrence
                    WHERE condition_source_value like '250%' or
      			              condition_source_value like '249%' or 
      			              condition_source_value like '791%' or
      			              condition_source_value like '790%' or 
      			              condition_source_value like '648%' or
      			              condition_source_value like 'V65%' or 
      			              condition_source_value like 'V45%' or 
      			              condition_source_value like 'V53%'); 
"""

Q_Condition = \
"""SELECT {select_clause}
FROM {from_clause}
WHERE {where_clause}"""

Q_Condition_ID = \
"""SELECT {select_clause}
FROM {from_clause}
WHERE person_id in (SELECT DISTINCT person_id
                    FROM {from_clause}
                    WHERE {where_clause});"""

Q_Drug = \
"""SELECT {select_clause}
FROM {from_clause}
WHERE {where_clause}"""


Q_Lab = \
"""SELECT {select_clause}
FROM {from_clause}
WHERE {where_clause}"""

###### Example query codes ##### 
q_template_measure = \
"""
DECLARE @Expo table (person_id int, start_date date, source_value varchar(50))

INSERT into @Expo (person_id, start_date, source_value)
{cohort}

SELECT person_id, measurement_date, measurement_time, value_as_number, value_source_value
FROM ohdsi.west.measurement
WHERE person_id in (select DISTINCT person_id from @Expo); 
"""

##### 

def selectFrom(table, columns=None, **kargs):

    # [params]
    # id_field = kargs.get('id_field', 'person_id')
    save_intermediate = kargs.get('save_intermediate', False)
    overwrite_intermediate = kargs.get('overwrite_intermediate', True)
    output_dir = kargs.get('output_dir', sys_config.read('DataExpRoot')) # temporary data dir
    fsep = '|'
    cohort_name = kargs.get('cohort', 'diabetes')
    identifier = '%s' % cohort_name

    # global tb_condition, DBScope
    if table.startswith('cond'): 
        select_ = 'person_id, condition_start_date, condition_source_value'
    elif table.startswith('drug'): # drug_exposure
        select_ = 'person_id, drug_exposure_start_date, drug_source_value' 
    elif table.startswith(('lab', 'measure')):
        if columns is None: 
            select_ = 'person_id, measurement_date, measurement_time, value_as_number, value_source_value' 
        else: 
            select_ = ','.join(columns)

    from_ = '%s.%s' % (DBScope, table)  # e.g. ohdsi.west.condition_occurrence

    q = \
    """SELECT {select_clause}
       FROM {from_clause}"""
    params = {'select_clause': select_,
              'from_clause': from_}
    q = q.format(**params)

    print('select_from> query: %s' % q)
    df = execute_query(q)

    if save_intermediate: 
        fp = os.path.join(output_dir, '%s-no_constraint-%s.csv' % (table, identifier))
        if not os.path.exists(fp) or (os.path.exists(fp) and overwrite_intermediate): 
            print('output> writing querying result to %s' % fp)
            df.to_csv(fp, sep=fsep, index=False, header=True, encoding='utf-8')

    return df

def execute_query(q, **kargs): 
    # table = kargs.get('table', tb_condition)  # tb_condition: diagnositc codes, tb_drug: medications

    # [params] database
    server = seq_maker_config.Server
    user = seq_maker_config.User
    password = seq_maker_config.Password
    database = 'ohdsi'

    # from_ = '%s.%s' % (DBScope, table)  # e.g. ohdsi.west.condition_occurrence

    conn = pymssql.connect(server=server, user=user, password=password, database=database)
    cursor = conn.cursor()

    div('Query: %s\nServer %s\nDatabase: %s\n' % (q, server, database))
    df = pd.read_sql(q, conn)
    assert not df.empty, "query:\n%s\n does not return any data!" % q
    print('verify> dim: %s' % str(df.shape))

    return df

def searchByID(**kargs): 
    """

    """
    global tb_condition, DBScope

    # [params]
    id_field = kargs.get('id_field', 'person_id')
    save_intermediate = kargs.get('save_intermediate', True)
    overwrite_intermediate = kargs.get('overwrite_intermediate', True)
    output_dir = kargs.get('output_dir', sys_config.read('DataExpRoot')) # temporary data dir
    fsep = '|'

    cohort_name = kargs.get('cohort', 'diabetes')
    identifier = '%s' % cohort_name

    ids = kargs.get('ids', targets.target_set)  
    table = kargs.get('table', tb_condition)  # tb_condition: diagnositc codes, tb_drug: medications
    
    # [todo]
    if table.startswith('cond'): 
        select_ = 'person_id, condition_start_date, condition_source_value'
    elif table.startswith('drug'): # drug_exposure
        select_ = 'person_id, drug_exposure_start_date, drug_source_value' 
    elif table.startswith(('lab', 'measure')):
        select_ = 'person_id, measurement_date, measurement_time, value_as_number, value_source_value' 

    from_ = '%s.%s' % (DBScope, table)  # e.g. ohdsi.west.condition_occurrence
    where_ = '{} in ({});'.format(id_field, ','.join(str(e) for e in ids))

    # [todo] make query string
    q = '' 
    if table.startswith('cond'): 
        q = query_condition(select_=select_, from_=from_, where_=where_)
        print('query>\n%s\n' % q)
    elif table.startswith('drug'):    
        q = query_medication(select_=select_, from_=from_, where_=where_)
        print('query>\n%s\n' % q)
    elif table.startswith(('lab', 'measure')):
        q = query_lab(select_=select_, from_=from_, where_=where_) 
        print('query>\n%s\n' % q)
    # sys.exit(0)

    # [query] overwrite query here

    # [params] database
    server = seq_maker_config.Server
    user = seq_maker_config.User
    password = seq_maker_config.Password
    database = 'ohdsi'

    conn = pymssql.connect(server=server, user=user, password=password, database=database)
    cursor = conn.cursor()

    div('Querying table: %s from server %s, database: %s' % (from_, server, database))
    df = pd.read_sql(q, conn)
    assert not df.empty, "query:\n%s\n does not return any data!" % q
    print('verify> dim (tb=%s): %d by %d' % (table, df.shape[0], df.shape[1]))

    if save_intermediate: 
        fp = os.path.join(output_dir, '%s-query_ids-%s.csv' % (table, identifier))
        if not os.path.exists(fp) or (os.path.exists(fp) and overwrite_intermediate): 
            print('output> writing querying result to %s' % fp)
            df.to_csv(fp, sep=fsep, index=False, header=True, encoding='utf-8')

    return df

def constraint_in(alist, attribute='person_id', negate=True): 
    s = ','.join(str(e) for e in alist)
    
    q = ''
    if negate: 
	    q = '{} NOT IN ({})'.format(attribute, s)
    else: 
    	q = '{} IN ({})'.format(attribute, s)
    return q 

def getByID(idx, table=None):
    if table is None: table = tb_measure # ohdsi.west.measurement
#     # searchByID() 

#     q = SELECT person_id, measurement_date, measurement_time, value_as_number, value_source_value
# FROM ohdsi.west.measurement

    return df

def searchByDiag(codes, **kargs): 
    """
	Given a set of diagnostic codes, find corresponding cohort. 

    Params
    ------
    cohort
    save_intermediate 
    simplify_code 
    exclude_ids: 

    nrows

    - I/O 
      output_dir 

    """
    def get_outputdir():  # <- cohort
        # e.g. cohort=CKD 
        #      => tpheno/data-exp/CKD
        basedir = sys_config.read('DataExpRoot')
        cohort_name = kargs.get('cohort', None)
        if cohort_name: 
            return seqparams.getCohortGlobalDir(cohort, basedir=basedir, create_dir=False)
        return basedir

    import seqparams 
    global DBScope, tb_condition

    # [params]
    id_field = kargs.get('id_field', 'person_id')

    save_intermediate = kargs.get('save_intermediate', True)
    overwrite_intermediate = kargs.get('overwrite_intermediate', True)
    sample_subset = kargs.get('sample_subset', False)

    output_dir = get_outputdir() 
    fsep = '|'

    # [params] db table
    field_source = 'condition_source_value'
    field_date = 'condition_start_date'

    table = kargs.get('table', tb_condition)
    base_only = kargs.get('simplify_code', True)
    exclude_ids = kargs.get('exclude_ids', None)

    # [params] sampling; only relevant if sample_subset is True
    abs_max = 5000000
    ro = kargs.get('ro', 10)
    nrows_base = kargs.get('nrows', 200000)  # maxium number
    nrows = nrows_base * ro
    nrows = min(nrows, abs_max)
    
    # [params] query
    if sample_subset: 
    	select_ = 'top(%s) person_id, condition_start_date, condition_source_value' % nrows
    else: 
    	select_ = 'person_id, condition_start_date, condition_source_value'

    from_ = '%s.%s' % (DBScope, table)  # e.g. ohdsi.west.condition_occurrence

    if exclude_ids: # exclude these IDs
        where_ = constraint_in(exclude_ids, attribute='person_id', negate=True) + ' AND ' + \
                 '(' + make_constraint(codes, constraint=field_source, base_only=base_only) + \
                 ')'
    else: 
        where_ = make_constraint(codes, constraint=field_source, base_only=base_only)

    # make query string
    q = '' 
    if table.startswith('cond'): 
        q = query_condition(select_=select_, from_=from_, where_=where_, by='ID')
        print('query> (top %s):\n%s\n' % (nrows_base if sample_subset else 'n/a', q))

    # sys.exit(0)

    # [query] overwrite query here

    # [params] database
    server = seq_maker_config.Server
    user = seq_maker_config.User
    password = seq_maker_config.Password
    database = 'ohdsi'

    conn = pymssql.connect(server=server, user=user, password=password, database=database)
    cursor = conn.cursor()

    # execute query 
    div('Querying table: %s from server %s, database: %s' % (from_, server, database))
    df = pd.read_sql(q, conn)
    assert not df.empty, "query:\n%s\n does not return any data!" % q
    shape0 = df.shape

    # sample subset
    if sample_subset and (df.shape[0] > nrows_base): 
        df = df.sample(nrows_base, axis=0)
        print('verify> dim (tb=%s): %s > after sampling: %s' % (table, str(shape0), str(df.shape)))

    if save_intermediate: 
        print('searchByDiag> keep intermediate dataframes (group ID=%s) in %s' % (kargs.get('cohort', '?'), output_dir))
        fp = os.path.join(output_dir, '%s-query_codes.csv' % table)
        if not os.path.exists(fp) or (os.path.exists(fp) and overwrite_intermediate): 
            print('output> writing querying result df (dim: %s) to %s' % (str(df.shape), fp))

            # df = normalize(df, target_field=field_source) # very expensive
            df.to_csv(fp, sep=fsep, index=False, header=True, encoding='utf-8')
    
    return df

def filterCandidatesByCodes(codes, dataframe=None, relation='and', **kargs): 
    """
    Find all candidates whose diagnostic codes include the input codes. 


    Use 
    ---
    1. find canidates using searchByDiag() and narraw the candidates down to a smaller set 
       in which each patient has all of the input diagnostic codes
    2. use t_extract_diag() to find desired input codes prior to this call
       see t_select()

    """
    def get_outputdir():  # <- cohort
        # e.g. cohort=CKD 
        #      => tpheno/data-exp/CKD
        basedir = sys_config.read('DataExpRoot')
        cohort_name = kargs.get('cohort', None)
        if cohort_name: 
            return seqparams.getCohortGlobalDir(cohort, basedir=basedir, create_dir=False)
        return basedir
    def get_input():  # <- ifile, outputdir, {field_source, field_date}, fsep
        df_condition = dataframe
        if df_condition is None: 
            ifile = kargs.get('ifile', '%s-query_codes.csv' % table)
            ipath = os.path.join(outputdir, ifile)
            print('info> loading default precomputed cohort dataframe at %s' % ipath)
            dtypes = {field_source: str}
            df_condition = pd.read_csv(ipath, sep=fsep, index_col=False, header=0, error_bad_lines=True, 
                                     parse_dates=[field_date], dtype=dtypes)
        else: 
            print('info> loading user provided input dataframe (dim=%s)' % str(dataframe.shape))
        assert df_condition is not None and not df_condition.empty, "No condition data found!"
        return df_condition
    def test_preconditions(): 
        assert hasattr(codes, '__iter__'), "Input codes are ill-formatted: %s" % codes

        return
    def summary_stats():
        n_unique = len(set(df_condition[field_id].values))
        print('info> df_condition > unique IDs: %d, dim: %s' % (n_unique, str(df_condition.shape))) 

        # get most representative code (e.g. for output file naming)
        counter = collections.Counter(icd9utils.getRootSequence(df_condition[field_source].values))
        repr_code = counter.most_common(1)[0][0].strip() 
        print('info> most representative code: %s' % repr_code)

        return
    def save_candidates(): # <- targetCodes, outputdir, field_id='person_id'
        fpath = os.path.join(outputdir, 'candidates-%s-%s.csv' % (targetCodes[0], len(targetCodes)))
        header = [field_id, 'code']
        adict = {h:[] for h in header}
        for k, vals in candidates.items(): 
            adict[field_id].append(k)  
            adict['code'].append(','.join(str(v).strip() for v in vals))
        df = pd.DataFrame(adict, columns=header)
        print('save> Saving candidate data to %s' % fpath)
        df.to_csv(fpath, sep=fsep, index=False, header=True)
        return 

    import collections

    # [params]
    outputdir = get_outputdir()
    table = kargs.get('table', tb_condition)
    save_intermediate = kargs.get('save_intermediate', True)
    exclude_ids = set(targets.target_set)
    base_only = kargs.get('simplify_code', pmed.containsBaseForm(codes))
    fsep = '|'

    # [params] db table
    field_id = 'person_id'
    field_source = 'condition_source_value'
    field_date = 'condition_start_date'

    df_condition = get_input() # <- dataframe
    test_preconditions()
    summary_stats()  # e.g. most representative codes in the input dataframe

    candidates = {}
    cnt = 0 
    commo_unmatched = set()  # [todo] use min heap + priority queue

    targetCodes = codes
    codeSet = set(targetCodes)
    nCodes = len(codeSet)
    n_nomatch = 0
    if base_only: 
    	# [todo] optimize
        for pid, df in df_condition.groupby('person_id'): 
            # assert not pid in exclude_ids

            refcodes = list(set(df[field_source].values))
            refcodes = normalize_diag(refcodes)
            refcodes_base = icd9utils.getRootSequence(refcodes)

            # [test]
            # if cnt < 10: print('test> pid: %s\n  => %s\n  => %s' % (pid, refcodes[:15], refcodes_base[:15]))
            commonCodes = codeSet.intersection(refcodes_base)
            if relation == 'and': 
                # diff = codeSet - set(refcodes_base)
                # if len(commonCodes) == nCodes: 
                if commonCodes.issubset(refcodes_base): 
                    candidates[pid] = refcodes
                    n_candidates = len(candidates)
                    if n_candidates % 50 == 0: 
                        print('filterCandidatesByCodes> Found %d candidates (out of %d) | op=%s ' % \
                            (n_candidates, cnt, relation))
                        extraCodes = set(refcodes_base) - codeSet
                        if len(extraCodes) > 0: 
                            div(message='pid: %s has extra codes: %s' % (pid, list(extraCodes)), symbol='%')
                else: 
                    # [test]
                    n_nomatch += 1 
            else: # or 
                # n = len(codeSet-set(refcodes_base))
                if len(commonCodes) > 0: 
                    candidates[pid] = refcodes 
                    n_candidates = len(candidates)
                    if n_candidates % 100 == 0: 
                        print('filterCandidatesByCodes> Found %d candidates (out of %d) | op=%s, common codes:\n  + %s\n' % \
                            (n_candidates, cnt, relation, commonCodes))
                else: 
                    n_nomatch += 1

            if n_nomatch % 500 == 0: 
                print('filterCandidatesByCodes> number of unmatched IDs reaches %d ...' % n_nomatch)  # i.e. the patient doesn't have ... 

            cnt += 1 
    else: # full coding e.g. 250.03 
        for pid, df in df_condition.groupby('person_id'): 
            # assert not pid in exclude_ids

            refcodes = list(set(df[field_source].values))
            refcodes = normalize_diag(refcodes)

            # [test]
            # if cnt < 10: print('test> pid: %s => %s' % (pid, refcodes[:15]))
            commonCodes = codeSet.intersection(refcodes)
            if relation == 'and': 
                if commonCodes.issubset(refcodes): 
                    candidates[pid] = refcodes
                    n_candidates = len(candidates)
                    if n_candidates % 50 == 0: 
                        print('filterCandidatesByCodes> Found %d candidates (out of %d) | op=%s...' % (n_candidates, cnt, relation))
                else: 
                    n_nomatch += 1
            else: # or 
                # n = len(set(codes)-set(refcodes))
                if len(commonCodes) > 0: 
                    candidates[pid] = refcodes 
                    n_candidates = len(candidates)
                    if n_candidates % 100 == 0: print('test> Found %d candidates (out of %d) ...' % (n_candidates, cnt))
                else: 
                    n_nomatch += 1

            if n_nomatch % 500 == 0: 
                print('filterCandidatesByCodes> number of unmatched IDs reaches %d ...' % n_nomatch)  # i.e. the patient doesn't have ... 

            cnt += 1 

    print('info> Found %d eligible patients diagnosed with given codes:\n%s\n' % (len(candidates), codes))
    if candidates and save_intermediate: save_candidates()

    return candidates  # person_id -> {code}

def extract_diag(**kargs):
    """
    Given intermediate dataframe (query result from say searchByID), extract relevant diagnostic codes. 

    Use 
    ---
    searchByID() -> extract_diag()
    searchByDiag() 
    """ 
    global DBScope, tb_condition

    # [params]
    basedir = kargs.get('input_dir', sys_config.read('DataExpRoot')) 
    table = kargs.get('table', tb_condition)
    save_intermediate = kargs.get('save_intermediate', False)

    default_ifile = '%s-query_ids.csv' % table
    ifile = kargs.get('input_file', default_ifile)
    df_condition = kargs.get('input_dataframe', None)
    fsep = '|'

    # [params] db table
    field_source = 'condition_source_value'
    field_date = 'condition_start_date'

    # [params] control 
    n_codes = kargs.get('n_codes', 15)  # most common diagnostic codes (in their base forms)
    base_only = kargs.get('simplify_code', True)

    # condition header: person_id|condition_start_date|condition_source_value
    # [load]
    if df_condition is None: 
        ipath = os.path.join(basedir, ifile)
        print('info> loading precomputed cohort dataframe at %s' % ipath)
        dtypes = {field_source: str}
        df_condition = pd.read_csv(ipath, sep=fsep, index_col=False, header=0, error_bad_lines=True, 
                                     parse_dates=[field_date], dtype=dtypes) 
    assert df_condition is not None and not df_condition.empty, "No condition data found!"
    print('info> dim(df_condition): %s' % str(df_condition.shape))

    common_codes = most_common(df_condition, target_field=field_source, 
    	n_codes=n_codes, base_only=base_only, save_intermediate=save_intermediate)    

    msg = 'verify> input %s > most frequent %d diagnostic codes:\n' % (ifile, n_codes)
    msg += indent('%s\n' % common_codes, nfill=8, mode='r')
    print(msg)

    return common_codes

def person_to_diagnoses(df_condition):
    from pattern import medcode as pmed
    # [params] db table
    field_source = 'condition_source_value'
    field_date = 'condition_start_date'

    # df_condition = normalize(df_condition, target_field=field_source)

    ptc = {}  # person_id to codes
    ptc_base = {}  # person_id to codes in base forms
    for pid, df in df_condition.groupby('person_id'): 
        ptc[pid] = list(set(c for c in df[field_source].values if pmed.isICDv2(c))) 
        ptc_base[pid] = list(set(icd9utils.getRootSequence(ptc[pid]))) 

    return ptc, ptc_base

def person_to(df, mode='medication'): 
    """
    Generic version of 
    """
    pass

def intersect(d):
    # return list(set.intersection(*map(set, lists)))
    return list(set(d[0]).intersection(*d[1:]))

def common_codes(df_cond, n_sample=10): 
    import random
    import collections
    # from pattern import medcode as pmed

    ptc, ptc_base = person_to_diagnoses(df_cond)
    idx = random.sample(ptc.keys(), n_sample) 

    # Q1: what are their common diagnostic codes? 
    n_trial = 100
    codeset = [] 
    for i in range(n_trial): 
        dlist = []
        for pid in idx: 
            dlist.append(ptc[pid])
        # if i == 0: 
        #     print('dlist:\n%s\n' % dlist)
        common_codes = intersect(dlist)

        # common_codes = [c for c in common_codes if not str(c).lower().startswith(('v', ))]
        codeset.append(common_codes) 

    div(message='Common codes (n_sample=%d)...' % n_sample)
    for i, codes in enumerate(codeset): 
        cstr = ' '.join(str(c) for c in codes)
        print('[%s] %s\n' % (i, cstr))

    topn = 20
    div(message='Frequent codes (n_sample=%d)...' % len(ptc))
    counter = collections.Counter()
    for i, (pid, codes) in enumerate(ptc.items()): 
        counter.update(codes)
    ranked_codes = counter.most_common(topn)
    print('info> topn %d codes:\n%s\n' % (topn, ranked_codes))

    return

def t_extract_diag(**kargs): 
    """
    Group by IDs and find most commonly occurring diagnostic codes among/between 
    patients. 

    Use 
    ---
    searchByDiag() 

    """
    global DBScope, tb_condition

    # [params]
    basedir = kargs.get('input_dir', sys_config.read('DataExpRoot')) 
    table = kargs.get('table', tb_condition)
    save_intermediate = kargs.get('save_intermediate', False)

    default_ifile = '%s-query_ids.csv' % table
    ifile = kargs.get('input_file', default_ifile)
    df_condition = kargs.get('input_dataframe', kargs.get('df', None))
    fsep = '|'

    # [params] db table
    field_source = 'condition_source_value'
    field_date = 'condition_start_date'

    # [params] control 
    n_codes = kargs.get('n_codes', 15)  # most common diagnostic codes (in their base forms)
    base_only = kargs.get('simplify_code', True)

    # condition header: person_id|condition_start_date|condition_source_value
    if df_condition is None: 
        ipath = os.path.join(basedir, ifile)
        print('info> loading precomputed cohort dataframe at %s' % ipath)
        dtypes = {field_source: str}
        df_condition = pd.read_csv(ipath, sep=fsep, index_col=False, header=0, error_bad_lines=True, 
                                     parse_dates=[field_date], dtype=dtypes) 
    assert df_condition is not None and not df_condition.empty, "No condition data found!"
    print('info> dim(df_condition): %s' % str(df_condition.shape))

    # normalize diagnostic codes 
    df_condition = normalize(df_condition, target_field=field_source)

    ptc = {}  # person_id to codes
    ptc_base = {}  # person_id to codes in base forms
    for pid, df in df_condition.groupby('person_id'): 
        ptc[pid] = set(df[field_source].values)
        ptc_base[pid] = set(icd9utils.getRootSequence(df[field_source].values))


    ####################################################################################
    # just a bunch of test codes below ... 
    ####################################################################################

    # see log file in .../doc/cohort.py.txt

    # Q1: what are their common diagnostic codes? 
    target_set = [2881284, 2807302]   # [hardcode] target_field.target_set
    print('info> ID: %s has %d unique codes' % (target_set[0], len(ptc[2881284])))
    print('info> ID: %s has %d unique codes' % (target_set[1], len(ptc[2807302])))
    common_codes = list(ptc[2881284].intersection(ptc[2807302]))
    common_codes_base = list(ptc_base[2881284].intersection(ptc_base[2807302]))   
    
    print('info> common codes (%d):\n%s\n' % (len(common_codes), common_codes))
    print('info> common base codes (%d):\n%s\n' % (len(common_codes_base), common_codes_base)) 

    # Q2: most frequence yet common codes? 
    n_codes = 20
    n_codes_max = len(common_codes_base)

    for n_codes in (20, 30, 40, n_codes_max): 
        for pset in (ptc, ptc_base, ): 
            counter1 = collections.Counter(pset[2881284])
            counter2 = collections.Counter(pset[2807302])
            set1 = set([e[0] for e in counter1.most_common(n_codes)])
            set2 = set([e[0] for e in counter2.most_common(n_codes)])
            common_codes = list(set1.intersection(set2))

            # exclude E, V codes 
            common_codes = [c for c in common_codes if not str(c).lower().startswith(('v', ))]

            print('\ninfo> common codes among top %d (size=%d):\n%s\n' % (n_codes, len(common_codes), common_codes))

    return common_codes

def normalize_diag(codes): 
    def dot_normalize(code): 
        if len(code)>0 and code[-1] == '.': 
            return code[:-1] 
        return code  

    msg = '' 
    for i, e in enumerate(codes): 
        e = str(e)
        e = e.strip()
        el = e.split()

        if len(el) > 1: 

            # clean incomplete expression e.g. 920. 
            code = pmed.convert(el[0]+el[1], nocatch=True)
            # el[0] = dot_normalize(el[0])
            # el[1] = dot_normalize(el[1])  # e.g. 367.4. 
            # code = el[0] + '.' + el[1]

            # use first element as a constraint: el[0]
            if pmed.isICD(code):   
                codes[i] = code
            else: 
            	# msg += "Invalid diag code: %s\n" % e
                print("Invalid diag code: %s\n" % e)

        elif not e: 
            print("Empty diag code: %s" % e)
        else:  # additional processing
            if pmed.isICD(e):  # extra dot S53. 920. is considered valid 
                if e[-1] == '.': 
                    msg += 'warning> incomplete coding: %s\n' % e
                    codes[i] = e[:-1]
                else: 
                    # noop, don't modeify
                	pass
            else: 
                e = pmed.convert(e, nocatch=True)
                if pmed.isICD(e): 
                    codes[i] = e 
                else: 
                    # msg += "Invalid diag code: %s\n" % e
                    print("Invalid diag code: %s\n" % e)
    # print(msg)

    return codes

def normalize(df, target_field=None, **kargs): 
    def dot_normalize(code): 
        if len(code)>0 and code[-1] == '.': 
            return code[:-1] 
        return code    

    if target_field is None: target_field = 'condition_source_value'

    basedir = kargs.get('output_dir', sys_config.read('DataExpRoot'))
    token_empty = 'empty'
    token_unknown = 'unknown'
    token_error = 'error'
    fsep = '|'
    
    msg = ''
    assert not df.empty

    for r, row in df.iterrows(): 
    	e = row[target_field]
        e = str(e)
        e = e.strip()
        el = e.split()

        if len(el) > 1: 

            # clean incomplete expression e.g. 920. 
            el[0] = dot_normalize(el[0])
            el[1] = dot_normalize(el[1])  # e.g. 367.4. 
            
            code = el[0] + '.' + el[1]

            # use first element as a constraint: el[0]
            if pmed.isICD(code):   
                df.loc[r, target_field] = code 

            elif pmed.isWord(el[0]): # e.g. I9^ 6A
                msg += 'info> found unusal source value: %s\n' % e 
            elif predicate.isDate(e): 
                msg += 'info> found date in source value: %s\n' % e
            else: 
                msg += 'warning> unknown (complex) diagnostic codes: %s\n' % e

        elif not e: 
            pass 
        else:  # additional processing
            if pmed.isICD(e):  # extra dot S53. 920. is considered valid 
                if e[-1] == '.': 
                    msg += 'warning> incomplete coding: %s\n' % e
                    df.loc[r, target_field] = e[:-1]
                else: 
                	df.loc[r, target_field] = e
                # a lot of codes that end with '.'
                # noncoded_condition[e] = dot_normalize(e)

            elif pmed.isMedCode(e): # 6819 as Gambian Trypanosomiasis
                pass  

            else: # typical ICD{9,10} code, no need to map 
                msg += "unknown condition 1-gram: %s\n" % e  # e.g. ADM, I10, V.0.8, ***, E, .22.1
                
        # print(msg) # this doesn't print
    return df

def most_common(df, target_field=None, n_codes=20, base_only=True, save_intermediate=False, **kargs):
    """

    Note
    ----
    1. condition source value in the database may not be well-formatted and as such, 
       it may be better off using SQL like statement to search for patterns rather than 
       searching by exact diagnostic codes. 

    Related 
    -------
    normalize()
    """
    def dot_normalize(code): 
        if len(code)>0 and code[-1] == '.': 
            return code[:-1] 
        return code

    candidates = []
    if target_field is None: target_field = 'condition_source_value'


    basedir = kargs.get('output_dir', sys_config.read('DataExpRoot'))
    token_empty = 'empty'
    token_unknown = 'unknown'
    token_error = 'error'
    fsep = '|'
    
    msg = ''
    assert not df.empty
    for e in df[target_field].values: 
        e = str(e)
        e = e.strip()
        el = e.split()
        if len(el) > 1: 

            # clean incomplete expression e.g. 920. 
            el[0] = dot_normalize(el[0])
            el[1] = dot_normalize(el[1])  # e.g. 367.4. 
            
            code = el[0] + '.' + el[1]

            # use first element as a constraint: el[0]
            if pmed.isICD(code):   
                candidates.append(code)

            elif pmed.isWord(el[0]): # e.g. I9^ 6A
                msg += 'info> found unusal source value: %s' % e 
            elif predicate.isDate(e): 
                msg += 'info> found date in source value: %s' % e
            else: 
                msg += 'warning> unknown (complex) diagnostic codes: %s' % e

        elif not e: 
            pass 
        else:  # additional processing
            if pmed.isICD(e):  # extra dot S53. 920. is considered valid 
                if e[-1] == '.': 
                    msg += 'warning> incomplete coding: %s' % e
                    candidates.append(e[:-1])
                else: 
                	candidates.append(e)
                # a lot of codes that end with '.'
                # noncoded_condition[e] = dot_normalize(e)

            elif pmed.isMedCode(e): # 6819 as Gambian Trypanosomiasis
                pass  

            else: # typical ICD{9,10} code, no need to map 
                msg += "unknown condition 1-gram: %s" % e  # e.g. ADM, I10, V.0.8, ***, E, .22.1
                
        print(msg)

    assert len(candidates) > 0, "No candidate diagnostic codes found."

    # [byproduct]
    if save_intermediate: 
    	counter = collections.Counter(candidates)
    	clist = counter.most_common(10)
    	print('info> most common (complete) code:\n%s\n' % clist)
    	ofile = 'icd_lookup-%s.csv' % (icd9utils.getRootCode2(clist[0][1]))
    	fpath = os.path.join(basedir, ofile)
        header = ['code', 'description']
        adict = {h:[] for h in header}    # {'code': [], 'description': []}
        for c in candidates: 
            adict['code'].append(c)
            adict['description'].append(icd9utils.lookup2(c, default=''))
        df = DataFrame(adict, columns=header)
        df.to_csv(fpath, sep=fsep, index=False, header=True)
        print('byproduct> saving ICD lookup table to %s' % fpath)
        
    print('info> number of candidates (complete codes): %d' % len(candidates))
    if base_only: # [1]
        candidates = icd9utils.getRootSequence(candidates)
        print('verify> number of candidates (base codes): %d' % len(candidates))

    counter = collections.Counter(candidates)
    code_count = counter.most_common(n_codes)  # most_common returns a list of tuples (e.g. [(5, 3), (1, 2), ...])
    print('info> most common code_count:\n%s\n' % code_count)

    return [e[0] for e in code_count]

def query_condition(**kargs):
    by = kargs.get('by', 'codes')

    token_query_end = ';'
    default_tb = '%s.%s' % (DBScope, tb_condition)  # e.g. ohdsi.west.condition_occurrence

    params = {'select_clause': kargs.get('select_', 'person_id, condition_start_date, condition_source_value'), 
              'from_clause': kargs.get('from_', default_tb), 
              'where_clause': kargs.get('where_', make_constraint(codes=cohort_diabetes()[0])), 
             } 
    
    is_nested_select = by.lower().startswith('id')

    if by.lower().startswith('id'):  # nested select
        q = Q_Condition_ID.format(**params)
    else: 
        q = Q_Condition.format(**params)

    if not is_nested_select and q[-3:].find(token_query_end) < 0: 
        q += ';'
    return q  

def query_medication(**kargs): 
    by = kargs.get('by', 'ID')
    token_query_end = ';'
    default_tb = '%s.%s' % (DBScope, tb_drug)  # e.g. ohdsi.west.condition_occurrence

    params = {'select_clause': kargs.get('select_', 'person_id, drug_exposure_start_date, drug_source_value'), 
              'from_clause': kargs.get('from_', default_tb), 
              'where_clause': kargs.get('where_', None), 
             } 

    # [todo] use make_constraint for where clause
    assert params['where_clause'] is not None, "where clause was not provided!"
    
    q = Q_Drug.format(**params)

    return q 

def query_lab(**kargs): 

    id_field = kargs.get('id_field', 'person_id')
    by = kargs.get('by', 'ID')
    token_query_end = ';'
    default_tb = '%s.%s' % (DBScope, tb_measure)  # e.g. ohdsi.west.measurement

    # person_id, measurement_date, measurement_time, value_as_number (lab values), value_source_value (MED code)
    params = {'select_clause': kargs.get('select_', 'person_id, measurement_date, measurement_time, value_as_number, value_source_value'), 
              'from_clause': kargs.get('from_', default_tb), 
              'where_clause': kargs.get('where_', None), 
             } 

    # where clause [todo] use make_constraint 
    # where_clause = params.get('where_clause')
    if params['where_clause'] is None: 
        if by.lower().startswith('id'):
            idx = kargs.get('idx', kargs.get('person_id', None))
            assert idx is not None and len(idx) > 0
            params['where_clause'] = '{} in ({});'.format(id_field, ','.join(str(e) for e in idx))
        else: 
            raise NotImplementedError

    # [todo] use make_constraint for where clause
    assert params['where_clause'] is not None, "where clause was not provided!"
    q = Q_Lab.format(**params)

    return q  

def make_constraint(codes, constraint='condition_source_value', **kargs): 
    """
    Given a set of codes, formulate SQL query to pull out relevant patients. 

    Note
    ----
    1. factored from seqMaker2
    2. can also use constraint_in(codes, attribtue='condition_source_value')
    """
    if kargs.get('simplify_code', False): 
    	return make_constraint_base(codes, constraint, **kargs)
    else: 
        # check codes (base only or full?)
        is_complete = True
        for c in codes: 
            if pmed.isICDBase(c): 
                is_complete = False; break 
        if not is_complete:   
            return make_constraint_base(codes, constraint, **kargs)

    # [params]
    omit_where = kargs.get('omit_where', True)

    if len(codes) == 0: return '' 
    if len(codes) == 1: 
        codestr = "'%s'" % codes[0]
    else: 
        codestr = "'%s'" % codes[0]
        for code in codes[1:]: 
            codestr += ", '%s'" % code

    if omit_where: 
        if constraint is not None: 
    	    q = "%s IN (%s)" % (constraint, codestr)  # to be compatible with query_condition()
        else: 
            q = codestr
    else: 
        q = "WHERE %s IN (%s)" % (constraint, codestr)

    return q
def make_query_constraint(**kargs): 
    return make_constraint(**kargs)


def constraint_like(alist, attribute='condition_source_value', **kargs):
    """

    Related 
    -------
    1. constraint_in()
    """
    omit_where = kargs.get('omit_where', True)

    # make unique
    ucodes = list(set(alist))

    q = ''
    if len(ucodes) == 0: return ''
    if len(ucodes) == 1: 
    	q = "%s LIKE '%s%%'" % (constraint, ucodes[0]) if omit_where else "WHERE %s LIKE '%s%%'" % (constraint, ucodes[0])
    else: 
    	if omit_where: 
            q = "%s LIKE '%s%%'" % (constraint, ucodes[0])
        else: 
            q = "WHERE %s LIKE '%s%%'" % (constraint, ucodes[0])

        for code in ucodes[1:-1]: 
            q += ' ' + 'OR'
            q += ' ' + constraint + " LIKE '%s%%'" % code 
            
        q += ' ' + 'OR' + constraint + " LIKE '%s%%'" % ucodes[-1]
     
    return q

def make_constraint_base(codes, constraint='condition_source_value', **kargs):
    """
    Specialized version of constraint_like() for diagnostic code input. 

    Note
    ----
    1. A more general version is constraint_like()

    """
    omit_where = kargs.get('omit_where', True)

	# ensure that codes are normalized to their base forms
    codes = icd9utils.getRootSequence(codes, sep=kargs.get('sep', ' '))

    ucodes = list(set(codes))
    n_codes, n_ucodes = len(codes), len(ucodes)
    if n_codes != n_ucodes: 
        print('warning> uniq # of codes: %d (< all codes: %d)' % (n_ucodes, n_codes))

    q = ''
    if len(ucodes) == 0: return ''
    if len(ucodes) == 1: 
    	q = "%s LIKE '%s%%'" % (constraint, ucodes[0]) if omit_where else "WHERE %s LIKE '%s%%'" % (constraint, ucodes[0])
    else: 
    	if omit_where: 
            assert constraint is not None, "constraint attribute needs to be specified in the case of approximate match ..."
            q = "%s LIKE '%s%%'" % (constraint, ucodes[0])
        else: 
            q = "WHERE %s LIKE '%s%%'" % (constraint, ucodes[0])

        for code in ucodes[1:-1]: 
            q += ' ' + 'OR'
            q += ' ' + constraint + " LIKE '%s%%'" % code 
            
        q += ' ' + 'OR' + ' ' + constraint + " LIKE '%s%%'" % ucodes[-1]
     
    return q

def cohort_diabetes(**kargs):
    """

    Memo
    ----
    1. CCS codes (see complete listing in ICD9toDisease.txt)
       section 1: Diabetes mellitus without complication
       section 2: Diabetes mellitus with complications
       section 3: Diabetes or abnormal glucose tolerance complicating pregnancy; childbirth; or the puerperium

    2. 249: secondary diabetes 
       250: type I ends with {1, 3}
            type II ends with {0, 2}

    3. 790: Nonspecific findings on examination of blood
    """

    ### define codes here

    # Diabetes mellitus
    code_str = """
            24900 25000 25001 7902 79021 79022 79029 7915 7916 V4585 V5391 V6546 

              24901 24910 24911 24920 24921 24930 24931 24940 24941 24950 24951 24960 24961 24970 24971 24980 24981 24990 24991 25002
        25003 25010 25011 25012 25013 25020 25021 25022 25023 25030 25031 25032 25033 25040 25041 25042 25043 25050 25051 25052
        25053 25060 25061 25062 25063 25070 25071 25072 25073 25080 25081 25082 25083 25090 25091 25092 25093

        64800 64801 64802 64803 64804 64880 64881 64882 64883 64884
    """
    codes = icd9utils.preproc_code(code_str, simplify_code=False)
    print('> n_codes: %d\n> codes:\n%s\n' % (len(codes), codes))

    # [test]
    bcodes = icd9utils.preproc_code(code_str, simplify_code=True, unique=True)
    print('> n_codes: %d\n> codes:\n%s\n' % (len(bcodes), bcodes))
    
    # ### convert codes to where clauses
    # print('> query:\n%s\n' % make_constraint(codes))
    # print('> query similar:\n%s\n' % make_constraint_base(bcodes))

    return (codes, bcodes)


def t_fetch(**kargs): 
    """

    Query
    -----
    1. top 40 + '250'

SELECT person_id, condition_start_date, condition_source_value
FROM ohdsi.west.condition_occurrence
WHERE person_id in (SELECT DISTINCT person_id
                    FROM ohdsi.west.condition_occurrence
                    WHERE person_id NOT IN (2881284,2807302) AND (condition_source_value LIKE '199%' OR condition_source_value LIKE '709%' OR 
                    condition_source_value LIKE '719%' OR condition_source_value LIKE '715%' OR condition_source_value LIKE '272%' OR 
                    condition_source_value LIKE '414%' OR condition_source_value LIKE '535%' OR condition_source_value LIKE '401%' OR 
                    condition_source_value LIKE '530%' OR condition_source_value LIKE '496%' OR condition_source_value LIKE '490%' OR 
                    condition_source_value LIKE '250%' OR condition_source_value LIKE '493%'));

    <log> this gives data dim: (36,953,033, 3)

    """
    # 1. find all the records of the seed examples, from which all relevant diagnostic codes are extracted 
    df = searchByID(**kargs)  
    codes = extract_diag(save_intermediate=False, base_only=True)
    # searchByDiag(codes=codes)

    # searchByDiag prototype
    # q = make_constraint(codes, constraint='condition_source_value', **kargs)
    # div(message='query:\n%s\n' % q, symbol='*')

    # test case 
    print('> find common diag codes between any two patients ...')
    # t_extract_diag(df=df)
    # [log] from top40: ['199', '709', '719', '715', '535', '414', '272', '401', '530', '496', '490', '493']

    # ['719', 'V68', 'V67', '715', 'V70', '401', '496', '490', 'V76', '493'] # [hardcode]
    # comm_top30 = ['719', '715', '401', '496', '490', '493']   # [log] results in df of dim: (3,408,800, 3)
    comm_top40 = ['199', '709', '719', '715', '535', '414', '272', '401', '530', '496', '490', '493'] + ['250']
    searchByDiag(codes=comm_top40, exclude_ids=targets.target_set, base_only=True)

    # candidates = filterCandidatesByCodes(codes=comm_top30, dataframe=None, relation='and')

    return 

def t_select(**kargs):  # given intermediate files
    # ['719', 'V68', 'V67', '715', 'V70', '401', '496', '490', 'V76', '493'] # [hardcode]
    comm_top30 = ['719', '715', '401', '496', '490', '493']   # [log] results in df of dim: (3,408,800, 3)
    # searchByDiag(codes=comm_top30, exclude_ids=targets.target_set, base_only=True)

    # dataframe <- None => load from file
    candidates = filterCandidatesByCodes(codes=comm_top30, dataframe=None, relation='and')

    # [log]
    # info> Found 805 eligible patients diagnosed with given codes (common_top30)

    return	

def make_lab_stats(df, cohort='PTSD', table='measurement', n_sample=1000, **kargs):
    def eval_concept_map(): 
        # [input] df_map
        rglabs = {}  # reverse group map (i.e. member => group)
        for c, cH in zip(df_map['individual_lab'].values, df_map['group'].values): 
            if not rglabs.has_key(c): rglabs[c] = [] 
            rglabs[c].append(cH)

        # [test]
        for c in rglabs.keys(): 
            assert len(rglabs[c]) == 1, "lab %s has more than one leader? %s" % (c, rglabs[c])
            rglabs[c] = rglabs[c][0]

        return rglabs

    # from batchpheno import qrymed2
    # import random
    import outlier

    # header = ['group', 'individual_lab', 'group_description', 'individual_description']
    df_map = kargs.get('concept_map', None) # map labs to their associated groups (see t_map_lab)
    outputdir = kargs.get('output_dir', sys_config.read('DataExpRoot')) # temporary data dir
    tConceptGrouping = False
    
    tFilterOutliers = True
    n_min_outlier_detection = 5  # min number of examples for outlier detection

    print('info> Computing lab statistics with input dim %s ...' % str(df.shape))

    cohort_name = cohort
    if cohort_name is None: cohort_name = 'diabetes'
    # ctrl_cohort_name = '%s-Negative' % cohort_name
    table_name = table

    header_lval = ['code', 'mean', 'median', 'std', 'min', 'max', 'n_sample', ]
    pivots = ['value_source_value', ]  # lab MED codes
    sortbyattr = ['code']

    lval_groups = df.groupby(pivots)   # df from lab table
    ldict = {h:[] for h in header_lval}  # for dataframe; header_lval => value
    vdict = {}  # group code -> values 
            
    missing_val_default = 0 # float('nan')

    if df_map is not None:  
        tConceptGrouping = True  
        div(message='Group similar lab codes according to concept map (dim: %s)' % str(df_map.shape))

        # each lab is mapped to exactly 1 leader
        rglabs = eval_concept_map()  # member MED -> group leader MED
        n_grouped = 0
        for i, (lc_, lg) in enumerate(lval_groups):  # foreach (lab code, values)
            # dfg.sort(sort_fields, ascending=False, inplace=False)
            lc = int(lc_)
            lcg = rglabs.get(lc, lc); assert isinstance(lcg, int) # if no leader, then it's its own leader
            if lc != lcg:
                n_grouped += 1 
                if n_grouped % 10 == 0: print('   + lab code %s => group %s' % (lc, lcg))

            # lg['value_as_number'].replace('', np.nan, inplace=True)
            vals0 = lg['value_as_number'].dropna().values
            if not vdict.has_key(lcg): vdict[lcg] = [] 
            if len(vals0) > 0: 
                # [todo] if number of lab values are suff, do bootstrapping
                # vals = random.sample(vals0, min(n_sample, len(vals0))) # store the values for wilcoxon test (between cohort and control)
                n_vals = len(vals0)
                vdict[lcg].extend(list(vals0))  # lcg may contain 'vals' from many 'lc'
        
            else: 
                print('warning> no values found for lab code: %s (group: %s) (size considering n/a? %d)' % \
                    (lc, lcg, len(lg['value_as_number'].values) )) 
        
        print('verify> found %d effective group' % n_grouped)
        assert n_grouped > 1, "No grouped members?"

        for lcg, vals0 in vdict.items(): 
            # vals_subset = random.sample(vals0, min(n_sample, len(vals0))) # store the values for wilcoxon test (between cohort and control)
            n_vals = n_vals0 = len(vals0)
            if tFilterOutliers and n_vals0 > n_min_outlier_detection: 
                # print('input> np.array(vals0): %s' % np.array(vals0))
                vals0 = outlier.reject_outliers(np.array(vals0), m=2.5)
                n_vals = len(vals0)

            ldict['code'].append(lcg)
            ldict['n_sample'].append(n_vals)

            if n_vals > 0: 
                # ldict['count'].append(len(vals0))  # number of lab values
                ldict['mean'].append(np.mean(vals0))
                ldict['median'].append(np.median(vals0))
                ldict['std'].append(np.std(vals0))
                ldict['max'].append(np.max(vals0))
                ldict['min'].append(np.min(vals0))
            else: 
                ldict['mean'].append(missing_val_default)
                ldict['median'].append(missing_val_default)
                ldict['std'].append(missing_val_default)
                ldict['max'].append(missing_val_default)
                ldict['min'].append(missing_val_default)

    else:    
        for i, (lc_, lg) in enumerate(lval_groups):  # foreach (lab code, values)
            # dfg.sort(sort_fields, ascending=False, inplace=False)
            lc = int(lc_)
            if i < 10: print('   + lab code? %s' % lc)

            # lg['value_as_number'].replace('', np.nan, inplace=True)
            vals0 = lg['value_as_number'].dropna().values

            n_vals0 = len(vals0)
            if tFilterOutliers and n_vals0 > n_min_outlier_detection: 
                vals0 = outlier.reject_outliers(np.array(vals0), m=2.5)
                n_vals0 = len(vals0)

            if n_vals0 > 0: 
                # [todo] if number of lab values are suff, do bootstrapping
                # vals = random.sample(vals0, min(n_sample, len(vals0))) # store the values for wilcoxon test (between cohort and control)
                vdict[lc] = vals
        
                # dataframe 
                ldict['code'].append(lc)

                ldict['mean'].append(np.mean(vals0)) 
                ldict['median'].append(np.median(vals0))
                ldict['std'].append(np.std(vals0))
                ldict['max'].append(np.max(vals0))
                ldict['min'].append(np.min(vals0))
                ldict['n_sample'].append(n_vals0)
            else: 
                print('warning> no values found for lab code: %s (size considering n/a? %d)' % \
                    (lc, len(lg['value_as_number'].values) ))
                vdict[lc] = [] 

                ldict['code'].append(lc)
                ldict['mean'].append(missing_val_default)
                ldict['median'].append(missing_val_default)
                ldict['std'].append(missing_val_default)
                ldict['max'].append(missing_val_default)
                ldict['min'].append(missing_val_default)
                ldict['n_sample'].append(0)
        assert len(ldict) == len(header_lval)

    if len(ldict) > 0: 

        df_lval = DataFrame(ldict, columns=header_lval)
        df_lval.sort_values(sortbyattr, ascending=True, inplace=True)
 
        # [output]
        fname = '%s-lab_values-%s.csv' % (table_name, cohort_name)
        if tConceptGrouping: 
            fname = '%s-grouped_lab_values-%s.csv' % (table_name, cohort_name)

        fpath = os.path.join(outputdir, fname)  
        print('io> make_lab_stats: saving df of dim %s to:\n%s' % (str(df_lval.shape), fpath))   
        df_lval.to_csv(fpath, sep='|', index=False, header=True)
    
        fname = '%s-lab_values-%s.pkl' % (table_name, cohort_name)
        if tConceptGrouping: 
            fname = '%s-grouped_lab_values-%s.pkl' % (table_name, cohort_name)
       
        fpath = os.path.join(outputdir, fname) 
        print('io> saving lab values to %s' % fname)
        pickle.dump(vdict, open(fpath, "wb" ))
    else: 
        print('make_lab_stats> Warning: no lab values found!')

    return vdict 

def query_source_values(df_lab, drop_cols=None, cohort='PTSD', descriptions=None):
    from batchpheno import qrymed2
    # df_lab = pd.read_csv(fpath, sep='|', header=0, index_col=False, error_bad_lines=True)
    dim0 = df_lab.shape
    
    if drop_cols is None: drop_cols = ['value_source_value', 'value_as_number']
    # df.dropna(axis=0, how='any') | df_lab = df_lab[np.isfinite(df_lab['value_source_value'])]
    # df_lab = df_lab[pd.notnull(df_lab['value_source_value'])]
    df_lab = df_lab.dropna(subset=drop_cols)
    dim1 = df_lab.shape
    print('dim> cohort (%s) > lab values > from %s to %s (dropping NaN rows)' % (cohort, str(dim0), str(dim1)))

    lab_tests = df_lab['value_source_value'].unique()
    print('verify> number of unique lab tests: %d' % len(lab_tests)) # [log] number of unique lab tests: 2555

    n_queried = 0
    if descriptions is None: 
        descriptions = {}

    assert isinstance(descriptions, dict)
    for i, lt in enumerate(lab_tests):
        val = 'unknown'
        try: 
            ltest = int(lt)
            if not descriptions.has_key(ltest):
                # val = qrymed2.getName2(ltest, err_default='unknown') 
                descriptions[ltest] = qrymed2.getName2(ltest, err_default=val) 
        except Exception, e:
             print('> could not query %s' % lt) 
             descriptions[ltest] = val
        
        n_queried += 1 

        if n_queried % 20 == 0: 
            print('progress> %dth lab test: %s => %s' % (n_queried, ltest, descriptions[ltest]))

    sdvals = []
    for lt in df_lab['value_source_value'].values: 
        val = 'unknown'
        try: 
            ltest = int(lt)
            val = descriptions.get(ltest, 'unknown')
        except: 
            pass
        sdvals.append(val)

    df_lab['source_description'] = sdvals
    # print('io> saving source description (qrymed returns) to %s' % fpath)
    # df_lab.to_csv(fpath, sep='|', index=False, header=True) 

    return df_lab

def qrymed3(code, default_val='unknown'): 
    return qrymed2.getName2(code, err_default=default_val) 

def create_lab_lookup(**kargs): 

    cohort_name = 'PTSD'
    ctrl_cohort_name = '%s-Negative' % cohort_name
    drop_cols = ['value_source_value', 'value_as_number']
    outputdir = kargs.get('output_dir', sys_config.read('DataExpRoot')) # temporary data dir
    ref_dir = os.path.join(outputdir, 'archive')
    table_name = 'measurement'

    # header: person_id|measurement_date|measurement_time|value_as_number|value_source_value|source_description
    fname = 'measurement-query_ids-%s.csv' % cohort_name
    ref_path = os.path.join(ref_dir, fname)
    df = pd.read_csv(ref_path, sep='|', header=0, index_col=False, error_bad_lines=True)
    assert not df.empty 

    df = df.dropna(subset=drop_cols)
    df = df[ ['value_source_value', 'source_description'] ]
    df = df.drop_duplicates(subset=['value_source_value'])  
    fpath = os.path.join(outputdir, '%s-lab_lookup-%s.csv' % (table_name, cohort_name))   
    print('io> save lab lookup table to:\n%s' % fpath)
    df.to_csv(fpath, sep='|', index=False, header=True)

    return  

def make_tset(mode='binary', **kargs): # binary classification
    def eval_concept_map(): 
        # [input] df_map
        rglabs = {}  # reverse group map (i.e. member => group)
        for c, cH in zip(df_map['individual_lab'].values, df_map['group'].values): 
            if not rglabs.has_key(c): rglabs[c] = [] 
            rglabs[c].append(cH)

        # [test]
        for c in rglabs.keys(): 
            assert len(rglabs[c]) == 1, "lab %s has more than one leader? %s" % (c, rglabs[c])
            rglabs[c] = rglabs[c][0]

        return rglabs

    import sys

    outputdir = basedir = kargs.get('output_dir', sys_config.read('DataExpRoot')) # temporary data dir

    cohort_name = 'PTSD'
    ctrl_cohort_name = '%s-Negative' % cohort_name
    table_name = 'measurement'

    df_map = kargs.get('concept_map', None)
    tConceptGrouping = kargs.get('concept_grouping', False if df_map is None else True)

    # [params] summary statistics
    sl = 0.05  # significance level at 5%
    sl_suffix = str(sl).split('.')[1]
    hd_basics = ['code', 'mean', 'median', 'min', 'max', 'std']
    hd_test = ['pval_wilcoxon', 'sig_%s' % sl_suffix, 'grand_mean', 'grand_median']
    hd_desc = ['n_sample', 'is_active', 'description'] 
    header_ht = hd_basics + hd_test + hd_desc
    
    # [params] drop NaN 
    drop_cols = ['value_source_value', 'value_as_number']

    # lab values
    # measurement-lab_values-PTSD.csv  | measurement-lab_values-PTSD-Negative.csv

    header = ['person_id', 'measurement_date', 'measurement_time', 'value_as_number', 'value_source_value', 'source_description']
    # load cohort 
    fname = '%s-query_ids-%s.csv' % (table_name, cohort_name)
    fpath = os.path.join(outputdir, fname)
    
    print('io> loading lab table for cohort: %s' % cohort_name)
    # dtypes = {'value_source_value': int, } # [error] Integer column has NA values in column 4
    df_lab = pd.read_csv(fpath, sep='|', header=0, index_col=False, error_bad_lines=True)
    df_lab = df_lab.dropna(subset=drop_cols)
    df_lab['value_source_value'] = df_lab['value_source_value'].astype(int)
    
    print('dim> PTSD cohort > lab values (after dropping NaNs)  > %s' % str(df_lab.shape))
    lab_tests = df_lab['value_source_value'].unique()

    # load control 
    fname = '%s-query_ids-%s.csv' % (table_name, ctrl_cohort_name)
    fpath = os.path.join(outputdir, fname)
    assert os.path.exists(fpath), "Nonexistent input %s" % fpath
    print('io> loading control data from %s' % fpath)
    df_lab_ctrl = pd.read_csv(fpath, sep='|', header=0, index_col=False, error_bad_lines=True) # dtype=dtypes
    df_lab_ctrl = df_lab_ctrl.dropna(subset=drop_cols)
    df_lab_ctrl['value_source_value'] = df_lab_ctrl['value_source_value'].astype(int)
    print('dim> PTSD control cohort > lab values (after dropping NaNs)  > %s' % str(df_lab_ctrl.shape))
    lab_tests_ctrl = df_lab_ctrl['value_source_value'].unique()

    print('info> n_labtests: %d vs n_labtests (ctrl): %d' % (len(lab_tests), len(lab_tests_ctrl)))
    # sys.exit(0)
   
    # load summary statistics
    fname = '%s-lab_values-%s.csv' % (table_name, cohort_name)  
    if tConceptGrouping:  # perhaps we don't want to use grouped labs to make tset 
        fname = '%s-grouped_lab_values-%s.csv' % (table_name, cohort_name)

    fpath_stats = os.path.join(outputdir, fname)  
    print('io> loading (cohort) lab summary stats from %s' % fpath_stats)
    df_stats = pd.read_csv(fpath_stats, sep='|', header=0, index_col=False, error_bad_lines=True) 
    df_stats['code'] = df_stats['code'].astype(int)

    pivots = ['person_id', ]  

    # means as default values
    lab_mean = dict(zip(df_stats['code'].values, df_stats['mean'].values))
    # lab_status = dict(zip(df_stats['code'].values, df_stats['is_active'].values))

    # define feature set 
    div(message='Define feature set: Var active in both groups')
    # fset = list(df_stats['code'].values) # + ['target']
    fset = set(lab_tests).intersection(lab_tests_ctrl)
    n_lab_common = len(fset)
    fset_prime = []
    for lt in df_stats['code'].values: 
        if lt in fset: 
            fset_prime.append(lt)
    fset = fset_prime

    # [log] n_commmon: 2206 =?= 2206 >? (n_df_stats: 2532)
    print('verify> n_commmon: %d =?= %d >? (n_df_stats: %d)' % (n_lab_common, len(fset), df_stats.shape[0]))

    # one patient, one training instance
    div(message='Create positive cases (cohort=%s)' % cohort_name)
    n_person = 0  # unique patient
    D = []
    for i, (mrn, dfg) in enumerate(df_lab.groupby(pivots)):  # foreach (lab code, values)
        n_person += 1 

        # poll 
        lvpairs = zip(dfg['value_source_value'], dfg['value_as_number'])  # patent records
        lvals = {}
        for lt, lv in lvpairs: 
            lt = int(lt)  # lab code 
            if not lvals.has_key(lt): lvals[lt] = [] 
            lvals[lt].append(lv)
        pr = {}
        for lt, vals in lvals.items(): 
            pr[lt] = np.mean(vals)

        fvals = []
        for lt in fset:  # foreach lab test 
            lt = int(lt) 
            val = pr.get(lt, lab_mean[lt])  # lab_mean should have a default  
            fvals.append(val)
        D.append(fvals)
    tset = DataFrame(D, columns=fset)
    print('info> Found %d positive cases (dim: %s)' % (n_person, str(tset.shape)))

    # add meta data 
    tset['target'] = 1 # positive cases

    div(message='Create negative cases (cohort=%s)' % ctrl_cohort_name)
    n_person_ctrl = 0  # unique patient
    D = []
    for i, (mrn, dfg) in enumerate(df_lab_ctrl.groupby(pivots)):  # foreach (lab code, values)
        n_person_ctrl += 1  
        
        # poll 
        # pr = dict(zip(dfg['value_source_value'], dfg['value_as_number']))  # patent records
        lvpairs = zip(dfg['value_source_value'], dfg['value_as_number'])  # patent records
        lvals = {}
        for lt, lv in lvpairs: 
            lt = int(lt)
            if not lvals.has_key(lt): lvals[lt] = [] 
            lvals[lt].append(lv)
        pr = {}
        for lt, vals in lvals.items(): 
            pr[lt] = np.mean(vals)        

        fvals = []
        for lt in fset:  # foreach lab test 
            lt = int(lt) 
            val = pr.get(lt, lab_mean[lt])  # lab_mean should have a default  
            fvals.append(val)
        D.append(fvals)
    tset_ctrl = DataFrame(D, columns=fset)
    print('info> Found %d negative cases (dim: %s)' % (n_person_ctrl, str(tset_ctrl.shape)))

    # add meta data 
    tset_ctrl['target'] = 0 # negative cases
    tset_binary = pd.concat([tset, tset_ctrl], ignore_index=True)

    fname = '%s-tset-%s.csv' % (table_name, cohort_name)
    fpath = os.path.join(basedir, fname)
    print('io> saving training set (cohort=%s, dim=%s) to:\n%s' % (cohort_name, str(tset_binary.shape), fpath))
    tset_binary.to_csv(fpath, sep=',', index=False, header=True)
 
    return 

def t_select_features(ts=None, **kargs): 
    """

    Reference
    ---------
    1. lasso CV: http://scikit-learn.org/stable/auto_examples/feature_selection/plot_select_from_model_boston.html

       If “median” (resp. “mean”), then the threshold value is the median (resp. the mean) of the feature importances. 
       A scaling factor (e.g., “1.25*mean”) may also be used. If None and if the estimator has a parameter penalty set to l1, 
       either explicitly or implicitly (e.g, Lasso), the threshold used is 1e-5.

    2. http://scikit-learn.org/stable/auto_examples/exercises/plot_cv_diabetes.html#sphx-glr-auto-examples-exercises-plot-cv-diabetes-py

    Related
    -------
    1. See seqmaker.evaluate for its generic version

    """
    from sklearn.feature_selection import SelectFromModel
    from sklearn.linear_model import LassoCV
    import evaluate

    outputdir = basedir = kargs.get('output_dir', sys_config.read('DataExpRoot')) # temporary data dir

    # params: cohort and table
    cohort_name = 'PTSD'
    ctrl_cohort_name = '%s-Negative' % cohort_name
    table_name = 'measurement'

    df_map = kargs.get('concept_map', None)
    tConceptGrouping = kargs.get('concept_grouping', False if df_map is None else True)

    # default feature set
    fname = '%s-tset-%s.csv' % (table_name, cohort_name)
    fpath_ts = os.path.join(basedir, fname)

    n_iter = kargs.get('n_iter', 1)  # number of iterations
    n_features = kargs.get('n_features', None)
    if n_features is not None and n_iter == 1: 
        n_iter = 10  
    max_iter = kargs.get('max_iter', 3000) # maximum # of iterations for LASSO CV

    if ts is None: 
        ts = pd.read_csv(fpath_ts, sep=',', header=0, index_col=False, error_bad_lines=True) # dtype=dtypes

    ts = ts.reindex(np.random.permutation(ts.index))
    fset = evaluate.get_feature_set(ts, meta_fields=['target', ])

    print('io> Given tset of dim: %s, n_fset: %d' % (str(ts.shape), len(fset))) # [log] tset of dim: (12444, 2207)
    X, y = evaluate.transform(ts, standardize_='minmax')

    # multiple iterations

    # We use the base estimator LassoCV since the L1 norm promotes sparsity of features.
    clf = LassoCV(max_iter=3000) # 1000 by default
    fcount = collections.Counter()
    if n_iter == 1: # no control over the number of features obtained
    
        # Set a minimum threshold of 1e-5 via l1
        sfm = SelectFromModel(clf, threshold=None)
        sfm.fit(X, y)
        n_features = sfm.transform(X).shape[1]
        print('info> number of features (via l1): %d' % n_features)

        # [test]
        Xp = X[:,sfm.get_support()]
        print('info> Xp dim: %s ~? %d' % (str(Xp.shape), n_features))

        # feature selected 
        fset_active = fset[sfm.get_support()]  # feature is of string type
        print('info> n_fset_active: %d =?= %d > example:\n%s\n' % \
           (len(fset_active), Xp.shape[1], random.sample(fset_active, min(n_features, 10))) )
    else: 
        ni = n_iter
        while ni: 
            # shuffle the training data
            ts = ts.reindex(np.random.permutation(ts.index))
            X, y = evaluate.transform(ts, standardize_='minmax')

            sfm = SelectFromModel(clf, threshold=None)  # threshold for coeff
            sfm.fit(X, y)
            nf = sfm.transform(X).shape[1]
            print('info> n_iter=%d > number of features (via l1): %d' % (n_iter-ni, nf))
            Xp = X[:,sfm.get_support()]

            fset_active = fset[sfm.get_support()]  # feature is of string type! 
            fcount.update(fset_active)
            ni -= 1 
        fset_active_cnt = fcount.most_common(n_features) # n_features !=? ni
        fset_active = [f[0] for f in fset_active_cnt]
        print('verify> most frequently selected features (n:%d=?=%d):\n%s' % (n_features, len(fset_active), fset_active_cnt[-20:]))
        print('        < all features ever been selected: %d' % len(fcount))

    # load summary stats to get feature description
    fname = '%s-lab_values-%s.csv' % (table_name, cohort_name)

    # if tConceptGrouping:  # perhaps we don't want to use grouped labs to make tset 
    #     fname = '%s-grouped_lab_values-%s.csv' % (table_name, cohort_name)

    fpath_stats = os.path.join(outputdir, fname)  
    print('io> loading (cohort) lab summary stats from %s' % fpath_stats)
    df_stats = pd.read_csv(fpath_stats, sep='|', header=0, index_col=False, error_bad_lines=True) 
    df_stats['code'] = df_stats['code'].astype(int)

    # assert len(set(fset_active)-set(df_stats['code'])) == 0
    print('dtype> converting lab code to INT ...')
    fset_active = set(int(f) for f in fset_active)
    actives = []

    lookuptb = dict(zip(df_stats['code'].values, df_stats['description'].values))

    header_lasso = ['code', 'description']
    adict = {h:[] for h in header_lasso}
    n_active = 0

    fset_stats = df_stats['code'].values
    for c in fset_stats: 
        if c in fset_active: 
            actives.append(1)
            adict['code'].append(c)
            adict['description'].append(lookuptb[c])
            n_active +=1 
        else: 
            actives.append(0)
    
    assert n_active > 0 
    if n_iter == 1: print('verify> n_active: %d =?= selected: %d' % (n_active, Xp.shape[1]))
    df_stats['lasso_selected'] = actives 
    print('io> saving test statistics (dim: %s) to:\n%s' % (str(df_stats.shape), fpath_stats))
    df_stats.to_csv(fpath_stats, sep='|', index=False, header=True)

    # [related] measurement-sig_labs-PTSD.csv => measurement-wilcoxon-PTSD.csv
    fname = '%s-lasso-%s.csv' % (table_name, cohort_name)
    fpath_lasso = os.path.join(outputdir, fname)
    df_lasso = DataFrame(adict, columns=header_lasso)
    print('io> saving lasso-selected feature df(dim: %s) to:\n%s' % (str(df_lasso.shape), fpath_lasso))
    df_lasso.to_csv(fpath_lasso, sep='|', index=False, header=True)    

    fname = '%s-sig_labs-%s.csv' % (table_name, cohort_name)
    fpath_sig = os.path.join(outputdir, fname)
    df_sig = pd.read_csv(fpath_sig, sep='|', header=0, index_col=False, error_bad_lines=True) 

    # significant by both 
    # [note] lab variables that are active, wilcoxon significant and chosen by lasso
    common_codes = set(df_lasso['code'].astype(int)).intersection(df_sig['code'].astype(int))
    print('verify> number of selected lab tests via ensemble: %d' % len(common_codes))
    header_ensemble = ['code', 'description']
    adict = {h:[] for h in header_ensemble}
    for c in common_codes: 
        adict['code'].append(c)
        adict['description'].append(lookuptb[c])

    fname = '%s-ensemble-%s.csv' % (table_name, cohort_name)
    fpath_ensemble = os.path.join(outputdir, fname)
    df_ensemble = DataFrame(adict, columns=header_ensemble)
    print('io> saving ensemble-selected features df(dim: %s) to:\n%s' % (str(df_ensemble.shape), fpath_ensemble))
    df_ensemble.to_csv(fpath_ensemble, sep='|', index=False, header=True) 

    # sl = 0.05
    # sl_suffix = str(sl).split('.')[1]
    # f_sig = 'sig_%s' % sl_suffix
    # cond_active = df_stats['is_active'] == 1
    # cond_sig = df_stats[f_sig]

    return

def t_lab_ptsd(**kargs):
    """
    Refactored from t_search_ptsd(), this template function focuses on 
    lab tests. 

    Use 
    --- 
    1. grouping 'similar' lab tests 
    2. consolidate lab features and statistics. 


    Chain
    -----
    t_search_ptsd()

    t_lab_ptsd(): use this to prepare data
    t_map_lab(): use this to group

    Output
    ------
    tpheno/data-exp/measurement-code_desc-PTSD.csv

    """ 
    import sys
    from scipy.stats import wilcoxon
    from batchpheno import sampling  # bootstrap resampling
    
    cohort_name = 'PTSD'
    ctrl_cohort_name = '%s-Negative' % cohort_name

    projdir = sys_config.read('ProjDir')  # '/phi/proj/poc7002/tpheno'
    basedir = os.path.join(projdir, cohort_name)  # source 
    outputdir = kargs.get('output_dir', sys_config.read('DataExpRoot')) # temporary (output) data dir

    div('Step 5: Lab value analysis ...')  # refactored from t_search_ptsd()
    # header: person_id, measurement_date, measurement_time, value_as_number, value_source_value
    # augmented: person_id, measurement_date, measurement_time, value_as_number, value_source_value, source_description
    table_name = 'measurement'
    pivots = ['value_source_value', ]  # lab codes
    # df = searchByID(table=table_name, ids=idx, cohort='PTSD') # scope will be added
    # print('dim> PTSD cohort > lab values  > %s' % str(df.shape))
    # print df.head(10)
    # print df.tail(10)

    # [params] lab header: person_id|measurement_date|measurement_time|value_as_number|value_source_value|source_description
    #          source: measurement-query_ids-PTSD.csv | measurement-query_ids-PTSD-Negative.csv

    print('io> output directory: %s' % outputdir)
    fname = '%s-query_ids-%s.csv' % (table_name, cohort_name)
    fpath = os.path.join(outputdir, fname)
    fname_ctrl = '%s-query_ids-%s-Negative.csv' % (table_name, cohort_name)  # PTSD control group
    fpath_ctrl = os.path.join(outputdir, fname_ctrl)
    
    fpaths = [fpath, fpath_ctrl, ]
    columns = ['value_source_value', 'source_description'] # select only MED codes and their descriptions
    ulabs = set(); n_ulabs = []; dfx = []
    for i, fpath in enumerate(fpaths):
        print('io> loading lab table #%d (cohort:%s) from: %s' % (i, cohort_name, os.path.basename(fpath)))
        # dtypes = {'value_source_value': int, } # [error] Integer column has NA values in column 4
        df_lab = pd.read_csv(fpath, sep='|', header=0, index_col=False, error_bad_lines=True)
        print('status> loaded lab table (cohort:%s), dim: %s' % (cohort_name, str(df_lab.shape)))

        print df_lab.head(10)
        lab_tests = df_lab['value_source_value'].unique()
        # descriptions = {}
        print('verify> number of unique lab tests: %d' % len(lab_tests)) # [log] number of unique lab tests: 2555
        
        dfx.append(df_lab[columns])

        n_ulabs.append(len(lab_tests))
        ulabs.update(lab_tests)

    # [log] info> number of total unique lab tests/codes: 2817 (sum of [2532, 2491])
    print('info> number of total unique lab tests/codes: %d (union of %s)' % (len(ulabs), str(n_ulabs)))

    fname = '%s-code_desc-%s.csv' % (table_name, cohort_name)
    fpath = os.path.join(outputdir, fname)

    # [log] tpheno/data-exp/measurement-code_desc-PTSD.csv
    dfp = pd.concat(dfx, ignore_index=True)
    print('output> writing code-to-description mapping to:\n%s\n' % fpath)
    dfp.to_csv(fpath, sep='|', index=False, header=True)

    return 

def get_labs(has_measurements=True, **kargs):
    from batchpheno import qrymed2
    
    return

def t_concept_grouping(**kargs):
    return t_map_lab(**kargs) 
def t_map_lab(**kargs): 
    """

    Note
    ----
    1. Outputs

       data/code_measure_specimen.csv
            header: ['code', 'entity_measured', 'specimen']
                    entity_measured can be a string containing multiple codes
                    specimen can be NULL and a string containing multple values
            among *203853 codes:  15164 have 'entity measured' out of which 11758 also have 'assesses sample'
                ... 06.16.17

    2. Pandas 
       null value? pandas.isnull(obj)

    """
    def normalize(output):
        if isinstance(output, int): 
            return [output] # 
        elif isinstance(output, float):
            if pd.isnull(output): # output could be NaN
                return []
            else:  
                return [int(output)]  
        elif isinstance(output, str): 
            return sorted([int(e.strip()) for e in output.split() if not pd.isnull(e)])
        elif output is None: 
            return []
        
        # then output must be a list
        # assert hasattr(output, "__iter__")
        assert isinstance(output, list)
        if not output: 
            return []
        return sorted(output) # noop (e.g. already a list)
    def is_subset(s1, s2, bidirection=True): # set1 is a subset of set2? or set2 is a subset of set1? 
        if bidirection: 
            return len(set(s1)-set(s2))==0 or len(set(s2)-set(s1))==0
        return len(set(s1)-set(s2))==0

    import sys, re, gc
    from batchpheno import qrymed2
    from pattern import medcode
    # import graph    # use BFS or DFS to find reachable vertices (from one group leader to the others)
    
    cohort_name = 'PTSD'
    ctrl_cohort_name = '%s-Negative' % cohort_name
    projdir = sys_config.read('ProjDir')  # '/phi/proj/poc7002/tpheno'
    basedir = os.path.join(projdir, cohort_name)  # source 
    outputdir = kargs.get('output_dir', sys_config.read('DataExpRoot')) # temporary (output) data dir

    # header: person_id, measurement_date, measurement_time, value_as_number, value_source_value
    # augmented: person_id, measurement_date, measurement_time, value_as_number, value_source_value, source_description

    # [input] tpheno/data-exp/measurement-code_desc-PTSD.csv
    #         this should include CONTROL group as well
    div('step 1> loading lab test source data')
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
    load_data = True # load data for superset

    # preprocessing 
    superset = qrymed2.getDescendent(1) # '70745'
    print('info> size of total MED codes: %d' % len(superset)) # [log] size of total MED codes: 204174
    header = ['code', 'entity_measured', 'specimen']  # [note]
    edict = {h:[] for h in header}
    n_entries = ne = nes = 0

    # [params] concept hierarchy 
    concept_grouping = 'ge' # {'shallow', 'deep', 'ge'}  ge: ME eqaul but specimens greater or equal to    
    
    # [poll]
    df = None
    fpath = os.path.join(basedir, 'code_measure_specimen.csv')
    # temp_path = os.path.join(basedir, 'code_measure_specimen.csv')

    # Keep track of only MEDs with 'entity measured'
    if load_data and (os.path.exists(fpath) and os.path.getsize(fpath) > 0): 
        # mdict = pickle.load(open(temp_path, 'rb'))
        df = pd.read_csv(fpath, sep='|', header=0, index_col=False, error_bad_lines=True)
    else: 
        for code in superset: 
            slot = 16
            cmd = 'qrymed -val %s %s' % (slot, code)  # e.g. 35789: CPMC Laboratory Test: Lym  
            entity_measured = qrymed2.eval(cmd, verbose=False)       # e.g. 32028 - Lymphocytes

            # entity_measured = entity_measured.split()  # can also split '\n' 
            if entity_measured is not None: 
                edict['code'].append(code) 
                edict['entity_measured'].append(entity_measured) # it's possible that the output contains multiple codes

                slot = 14  # assesses sample (e.g. 46125: body fluid cell count specimen)
                cmd = 'qrymed -val %s %s' % (slot, code)
                specimen = qrymed2.eval(cmd, verbose=False) 
                if specimen is not None: 
                    nes += 1
            
                edict['specimen'].append(specimen) # could be None? 
                ne += 1 

                if ne % 100 == 0: print("status> found %d codes with 'entity measured' ..." % ne)
    
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
   
    # first put loaded data in maps; TOTAL sets
    edict0 = dict(zip(df['code'].values, df['entity_measured'].values))
    sdict0 = dict(zip(df['code'].values, df['specimen'].values))   # codes in sdict0 is a subset of those in edict0

    for adict in (edict0, sdict0, ): 
        assert isinstance(adict.keys()[0], int), "code %s is not an integer: dtype=%s" % (adict.keys()[0], type(adict.keys()[0]))

    # standardized edict and sdict
    edict, sdict = {}, {}  # map: entity measured | map: specimen
    multitest, multisample = [], []
    no_meX = []

    # ulabs = edict0.keys() # total set as long as measured entity attribute exists
    ulabs = set(df_lab['value_source_value'].values) 
    print('info> number of unique labs: %d' % len(ulabs)) # [log] number of unique labs: 2817
    for c in ulabs: # foreach code in the target lab codes
        code = int(c)
        entities = edict0.get(code, None)  # can be a string containing multiple codes 
        # assert len(entities) > 0 
        if entities is None: 
            # print('+ code %s does not have entity measured ...' % c)
            no_meX.append(c)

            # does that always mean no sample assessed? Yes!
            samples = sdict0.get(code, None)
            assert samples is None, "info> code %s: has 0 measured entity but assesses sample: %s" % (c, str(samples))
            continue 

        # 'entities' can be either a string (>= 1 codes) or a single code (somehow cast to float type)
        entities = normalize(entities) # [int(e.strip()) for e in entities.split()] # can be a string containing multiple codes 
        if len(entities) > 1: 
            # [log] lab code 72222 measures multiple lab values: [32092, 98156]
            print('info> lab code %s measures multiple lab values: %s' % (code, entities)) 
            multitest.append(code)
        assert isinstance(entities, list) # [debug]
        edict[code] = entities

        samples = sdict0.get(code, None) # code also has specimen involved 
        if samples is not None: 
            samples = normalize(samples) # [int(e.strip()) for e in samples.split()]  # can be a string containing multiple codes 
            if len(samples) > 1: 
                # code 108645 has multiple specimens: [2390, 32415]
                print('info> code %s has multiple specimens: %s' % (code, samples))
                multisample.append(code)
            sdict[code] = samples

    # [log] total entries: df_lab.shape[0]: 3630814  | 2089 have entity measured.
    print('info> among all target lab codes (n:%d < nrows:%d) <? %d have entity measured.' % (len(ulabs), df_lab.shape[0], len(edict)))

    # [log] 104903 codes out of 3630814 do not have measured entity (examples: [100027, 114544, 112478, 114544, 100027, 100027, ])
    print('info> %d codes out of %d do not have measured entity (examples: %s)' % (len(no_meX), df_lab.shape[0], no_meX[:10]))

    # [log] 7672 codes have multiple measurements while 34408 assess multiple samples
    print('info> %d codes have multiple measurements while %d assess multiple samples' % (len(multitest), len(multisample)))

    # [stats]
    n_codes, n_codes_specimen = len(edict), len(sdict)
    # [log] info> n_codes: 2089 (n_codes_specimen: 2089) vs total rows: 3630814
    print('info> n_codes: %d (n_codes_specimen: %d) vs total rows: %d' % (n_codes, n_codes_specimen, df_lab.shape[0]))

    # [poll] find all descendents
    div(message="Pre-compute descendents of all measured entities and specimens.")

    load_data1 = True
    edict_desc0, edict_desc = {}, {}
    temp_path = os.path.join(basedir, 'measured_entity_descendents.pkl')
    if load_data1 and os.path.exists(temp_path): 
        edict_desc0 = pickle.load(open(temp_path, 'rb'))
    # else: 
    #   pass
    # do this incrementally by taking out of 'else' condition
    n_incr = 0
    for i, (c, entities) in enumerate(edict.items()): 
        for me in entities: # for each m.e. of code 'c'
            if me in edict_desc0: 
                edict_desc[me] = edict_desc0[me]
            else: 
                n_incr += 1
                edescx = qrymed2.getDescendent(me, self_=False, to_int=True, verbose=False)  # don't allow identify case (which will be handled by ==)
                edict_desc[me] = normalize(edescx)  # could be None or a list of values
                if edescx is not None: 
                    if i % 20: print('++ ME %s has descendents: %s' % (me, str(edescx)))
    pickle.dump(edict_desc, open(temp_path, "wb" ))
    edict_desc0 = None; gc.collect() 
    print('info> found additional %d codes with entity measured (16)' % n_incr)

    load_data2 = True
    sdict_desc0, sdict_desc = {}, {}
    temp_path = os.path.join(basedir, 'specimen_descendents.pkl')
    if load_data2 and os.path.exists(temp_path): 
        sdict_desc0 = pickle.load(open(temp_path, 'rb'))
    # else: 
    #     pass
    # do this incrementally by taking out of 'else' condition
    n_incr = 0
    for i, (c, samples) in enumerate(sdict.items()): 
        for specimen in samples: 
            if specimen in sdict_desc0: 
                sdict_desc[specimen] = sdict_desc0[specimen]
            else: 
                n_incr += 1
                sdescx = qrymed2.getDescendent(specimen, self_=False, to_int=True, verbose=False)
                sdict_desc[specimen] = normalize(sdescx)
                if sdescx is not None: 
                    if i % 20 == 0: print('++ Assessed sample %s has descendents: %s' % (specimen, str(sdescx)))
    pickle.dump(sdict_desc, open(temp_path, "wb" ))
    sdict_desc0 = None; gc.collect() 
    print('info> found additional %d codes with assessed sample (14)' % n_incr)

    assert len(edict_desc) > 0 and len(sdict_desc) > 0

    # [poll] ancesters and descendents? 
    div(message="Pre-compute ancestors of all measured entities (not used).") # not used for now
    load_data3 = True
    ancmap = {}
    temp_path = os.path.join(basedir, 'code_ancestors.pkl') # os.path.join(basedir, 'code_descendents.pkl')
    if load_data3 and os.path.exists(temp_path): # and os.path.exists(dpath)
        ancmap = pickle.load(open(temp_path, 'rb'))
    else: 
        for i, c0 in enumerate(edict.keys()): # df_lab['value_source_value'].values: 
            # dcx = qrymed2.getDescendent(c0, self_=False) # to_int: True
            # if dcs is not None: 
            #     descmap[c0] = dcx 
            
            # for computing foreach ancestor (of c0) later
            acx = qrymed2.getAncestor(c0, self_=False, verbose=False) # for computing IS-A: isA(c1, c2, self_=True, lookup=None, anc_lookup=None)
            # if acx is not None: 
            #     acmx = []
            #     acmx = [ca for ca in acx if ca in edict] # only ancestors (of c0) with known set of measuremnts are relevant
            #     if len(acmx) > 0: 
            #         ancmap[c0] = acmx  # all ancestors (of 'code') with entity-measured attribtues
                    
            # [use] later need to query each ancestor's entity and specimen
            ancmap[c0] = normalize(acx)
  
        print('info> n_codes with effective ancestors (i.e. with entity measured): %d' % len(ancmap))
        pickle.dump(ancmap, open(temp_path, "wb" ))

    # Policy #1: group labs that point to the same 'entity measured'
    div(message='Policy #1: Grouping lab codes if they have exactly the same M.E. ...')
    ordered_edict = zip(edict.keys(), edict.values())
    grouped = set(); glabs = {}
    n_me_subset = 0
    for i, (c1, vals1) in enumerate(ordered_edict):  
        if c1 in grouped: continue
        
        glabs[c1] = []
        grouped.add(c1)

        for j, (c2, vals2) in enumerate(ordered_edict[i+1:]): 
            if c2 in grouped: continue 
                    
            et1, et2 = edict[c1], edict[c2]
            assert len(et1) > 0 and len(et2) > 0
            if et1 == et2:  # or qrymed2.isA()
                glabs[c1].append(c2)  # use the first code c1 as the group leader
                grouped.add(c2) 
                # print('  + linking lab %d to lab %s | %s == %s' % (c1, glabs[c1], str(et1), str(et2)))
            elif is_subset(et1, et2, bidirection=True): 
                # [log] + MEntity(91765) ~ MEntity(35971): [32056] ~ [32056, 58763, 100776]
                div(message='policy1> MEntity(%s) ~ MEntity(%s): %s ~ %s' % (c1, c2, str(et1), str(et2)), symbol='%')
                n_me_subset += 1
            
    # [test]
    n_linked = n_linked_multiple = 0
    for c, cx in glabs.items(): 
        if cx: 
            n_linked += 1
            # print('glabs> code %s was linked to %s' % (c, str(cx)))
            if len(cx) > 1: 
                # print('verify> lab %s was linked to multiple classes: %s' % (c, cx))
                n_linked_multiple += 1

    # [log] n_linked (codes with assigned classes in glabs): 364 | n_linked::multiclass: 276
    print('policy #1> n_linked (codes with assigned classes in glabs): %d | n_linked::multiple: %d' % (n_linked, n_linked_multiple))
    # [log] policy #1> n_ME_subset: 57
    print('policy #1> n_ME_subset: %d' % n_me_subset) # measured entities is a subset of 

    # convert to dataframe 
    # [input] 
    dflab_map = dict(zip(df_lab['value_source_value'].values, df_lab['source_description'].values))
    # [test]
    for i, (k, v) in enumerate(dflab_map.items()): 
        assert isinstance(k, int)
        if i >= 2: break

    # save the mapping 
    overwrite = True
    stem, identify = ('grouped_labs', 'Psimple-Tb%s-%s' % (table_name, cohort_name))
    fname = 'grouped_labs-Psimple-Tb%s-%s.csv' % (table_name, cohort_name)
    fpath = os.path.join(outputdir, fname)
    header = ['group', 'individual_lab', 'group_description', 'individual_description']
    if overwrite or not os.path.exists(fpath): 
        adict = {h:[] for h in header}
        for cH, cx in glabs.items(): 
            for c in cx:  
                adict['group'].append(cH) # group representative code
                adict['individual_lab'].append(c) # group members
                adict['group_description'].append(dflab_map[cH])
                adict['individual_description'].append(dflab_map[c])
        df = DataFrame(adict, columns=header)
        print('io> saving df[dim: %s] of grouped labs to %s' % (str(df.shape), fpath))
        df.to_csv(fpath, sep='|', index=False, header=True)
    glabs = None; gc.collect()

    # Policy #2: 
    # Rule:      c1 -> c2  i.e. c1 < c2 iff 
    #                                        me1 == me2 && s1 <= s2  (i.e. s1 is-a s2)
    div(message='Policy #2: Grouping lab codes according to concept hierarchy of ME and assessed specimens ...')
    ordered_edict = zip(edict.keys(), edict.values())
    grouped = set()
    n_case1 = n_case2 = n_case3 = n_case3a = n_case3b = n_case3c = n_case3ax = n_case3axx = 0 
    n_me_subset = 0

    glabs2 = glabs = {}  # lab leader -> members 
    rglabs = {}  # reverse map: member to head
    for i, (c1, vals1) in enumerate(ordered_edict):  
        if c1 in grouped: continue
        
        glabs[c1] = []  # c1 can serve as a potential leader of (at least itself)
        
        for j, (c2, vals2) in enumerate(ordered_edict):  # need full search because order is important (c1, c2) <> (c2, c1) 
            
            if c2 == c1 or (c2 in grouped): continue

            rglabs[c2] = []
                    
            et1, et2 = edict[c1], edict[c2]
            assert len(et1) > 0 and len(et2) > 0

            if et1 == et2:  # or qrymed2.isA()
                s1, s2 = sdict.get(c1, []), sdict.get(c2, [])
                if not s1 and not s2: 
                    print(' + case 1> lab {%s, %s} has same M.E. %s but neither has assessed specimens' % (c1, c2, str(et1)))
                    # print('  + linking lab %d to lab %s | %s == %s' % (c1, glabs[c1], str(et1), str(et2)))
                    n_case1 += 1
                elif not s1 or not s2: # one is None but not both
                    print(' + case 2> lab {%s, %s} has same M.E. %s but one of them does not have specimens' % (c1, c2, str(et1)))
                    if not s1: 
                        print("      ++ lab %s has no sample vs lab %s has %s" % (c1, c2, str(s2)))
                    else: 
                        print("      ++ lab %s has no sample vs lab %s has %s" % (c2, c1, str(s1)))
                    n_case2 += 1
                else:  # both have specimens 

                    n_case3 += 1 
                    if s1 == s2: 
                        n_case3a += 1

                        glabs[c1].append(c2)  # use the first code c1 as the group leader, c1 == c2
                        print('  + lab %d == lab %s | ME: %s == %s, S: %s == %s' % (c1, c2, str(et1), str(et2), str(s1), str(s2) ))
                        
                        rglabs[c2].append(c1)  # c2 is assigned to leader c1 (see if it can have multiple leaders?) 
                        grouped.update([c1, c2])  # link establisehd between c1 and c2

                    elif is_subset(s2, s1, bidirection=False): 
                        n_case3b += 1

                        glabs[c1].append(c2)  # c1 > c2 (because c2(s) is a subset of c1(s))
                        print('  + lab %d > lab %s | ME: %s == %s, S: %s :- %s (subset)' % (c1, c2, str(et1), str(et2), str(s1), str(s2) ))
                        
                        rglabs[c2].append(c1)  # c2 is assigned to leader c1 (see if it can have multiple leaders?) 
                        grouped.update([c1, c2])
                        
                    else: 
                        # [test]
                        # example: lab 128418 >? lab 35819 | ME: [31987] == [31987], S: [32415] > [32415, 35713]
                        if len(s1) > 1 or len(s2) > 1: # tricky case: multiple samples yet not identical
                            div(message='   ++ both lab have multiple samples: (%s: %s VS %s: %s)' % (c1, str(s1), c2, str(s2)))
                            n_case3ax += 1 

                            # [log] info> n_case3: 4064 | n_case3ax: 88, n_case3axx: 88
                            if len(s1) != len(s2): 
                                div(message='      +++ multiple samples yet not identical! (%s: %s VS %s: %s)' % (c1, str(s1), c2, str(s2)))
                                n_case3axx += 1 

                        # policy #2a
                        # found_isA = False  # loose if found ANY isA
                        # for s1i in s1:
                        #     for s2j in s2: 
                        #         if s1i in sdict_desc[s2j]: # if exists any pair s.t. s1i < s2j (or s1 is-a s2)     
                        #             glabs2[c1].append(c2)
                        #             print('  + linking lab %d to lab %s | ME: %s == %s, S: %s < %s' % (c1, c2, str(et1), str(et2), str(s1), str(s2) )) 
                        #             found_isA = True 
                                
                        #         if found_isA: break
                        #     if found_isA: break

                        # policy #2b 

                        # policy: can every s2j in s2 (from c2) find a match in s1 such that s2j < s1i (from c1)? 
                        n_found_isA = 0   
                        for s2j in s2:  # want s2j to find a s1i such that s2j is a descendent or equal 
                            matched = False

                            for s1i in s1: 
                                if s2j == s1i or (s2j in sdict_desc[s1i]): 
                                    matched = True  # find at least one match in s1
                                    break   
                            if matched: n_found_isA += 1

                        # [log] (cond=multiple samples) linking lab 32732 to lab 89790 | ME: [31955] == [31955], S: [149] < [2393, 35641] 
                        #       149 isA 2393
                        if n_found_isA == len(s2): # each s1i should at least have a correspdoning s2j, where s1i < s2j
                            n_case3c += 1 
                            
                            glabs[c1].append(c2)  # c1 > c2

                            if len(s1) == len(s2): 
                                print('  + (multiple specimens: same size) lab %d > lab %s | ME: %s == %s, S: %s > %s' % (c1, c2, str(et1), str(et2), str(s1), str(s2) )) 
                            else: 
                                # [log] S: [32415] > [32415, 35713]
                                print('  + (multiple specimens: diff size) lab %d > lab %s | ME: %s == %s, S: %s > %s' % (c1, c2, str(et1), str(et2), str(s1), str(s2) )) 

                            rglabs[c2].append(c1)  # c2 is assigned to leader c1 (see if it can have multiple leaders?) 
                            grouped.update([c1, c2])

            # [todo] it's possible that et1 != et2 and s1 != s2
            # elif is_subset(et1, et2, bidirection=True): 
            #     div(message='  + MEntity(%s) ~ MEntity(%s): %s ~ %s' % (c1, c2, str(et1), str(et2)), symbol='%')
            #     n_me_subset += 1

        ### end foreach lab code c2 
    ### end foreach lab code c1

    # [log] v3 
    # info> n_case1: 98, n_case2: 601, n_case3: 4488
    # info> n_case3: 4488 | n_case3a (s1=s2): 715, n_case3b (s1:-s2): 15, n_case3c (s1>s2): 15
    # info> n_case3: 4488 | n_case3ax: 220, n_case3axx: 216 
    print('info> n_case1: %d, n_case2: %d, n_case3: %d' % (n_case1, n_case2, n_case3))
    print('info> n_case3: %d | n_case3a (s1=s2): %d, n_case3b (s1:-s2): %d, n_case3c (s1>s2): %d' % (n_case3, n_case3a, n_case3b, n_case3b))
    print('info> n_case3: %d | n_case3ax: %d, n_case3axx: %d' % (n_case3, n_case3ax, n_case3axx))

    # [poll]
    # desc_map = {}
    # for _, ancestors in glabls2.items(): 
        
    #     candicate_class = set(ancestors)
    #     for ca in ancestors: 
    #         descx = qrymed2.getDescendent(ca, self_=False, to_int=True)
    n_linked = n_linked_multiple = 0
    conceptToLabs = {}

    # each code 'c' is a potential leader for codes in cx
    n_self_linked = 0
    n_members = n_outsample = 0  
    # ulabs = set(df_lab['value_source_value'].values) 
    print('verify> number of unique labs: %d' % len(ulabs)) # [log] number of unique labs: 2817
    cx = None # [debug]
    for c in ulabs: # foreach lab members
        if not c in glabs: # not linked yet, either a code without 'measured entity' or not connected to members 
            n_outsample += 1
            # conceptToLabs[c] = set([c])  # links to itself 
        else: 
            cx = glabs[c]
            if len(cx) > 0: 
                conceptToLabs[c] = set(cx)
                print(' + code %s is a leader to members (n=%d): %s' % (c, len(cx), str(cx)))
                n_linked += 1 
                n_members += len(cx)

                assert not c in cx
            else:
                print(' + code %s does not have a member' % c)
                n_self_linked += 1 
                # conceptToLabs[c] = set([c])  # add self link 

    # [log] info> n_linked: 452 (avg members: 3.037611), n_self_linked: 632, n_outsample (not in glabs): 1733
    print('info> n_linked: %d (avg members: %f), n_self_linked: %d, n_outsample (not in glabs): %d' % \
        (n_linked, n_members/(n_linked+0.0), n_self_linked, n_outsample))

    # it's possible that a member is linked to multiple leaders, need to resolve to one leader?
    multileaders = [] 
    n_self_linked2 = 0 
    
    unlink = {}
    rglabs2 = {}
    n_multiple = 0

    # want each code to have at least one leader (itself)
    # ulabs = set(df_lab['value_source_value'].values)

    # [note] this blocks never found any code linked to multiple group leaders
    for c, leaders in rglabs.items(): 
        if not leaders: # c never assigned to a leader 
            # print('veriy> code %s has no leader | is it a leader itself? %s' % (c, c in glabs))
            assert c in glabs, "lab %s has no leader but yet it's not a leader itself?" % c

        if len(leaders) > 1: 
            print('info> lab %s is linked to more than one leaders (n=%d): %s' % (c, len(leaders), leaders))
            multileaders.append(c)

            # pick the most generic one? 
            largest = []; n_max = -1
            the_leader = leaders[0]
            for leader in leaders: 
                members = labs[leader]
                if len(members) > n_max: 
                    n_max = len(members)
                    largest = members[:]
                    the_leader = leader
            print('  + reassigning lab %s to leader %s' % (c, the_leader)) # never happened
            rglabs2[c] = the_leader
            # unlink c from the other groups
            unlink[c] = set(leaders)-set(the_leader)  # unlink code 'c' from these leaders
    print('info> new leader assignments after resolving multiple leaders | n=%d' % len(rglabs2))  # [log] n=0
    assert len(multileaders) == 0
    
    if multileaders: 
        for c, ul in unlink.items(): 
            for u in ul: 
                conceptToLabs[u].remove(c)

            # this assertion will be evaluated even if multileades condition is False !!! 
            # assert c in conceptToLabs[rglabs2[c]] # 'c' should be a member of its unique leader
        
    # cH may be a member of some other groups => need to find the ultimate group leader
    if concept_grouping.startswith('d'):  
        div(message='Consolidating all group leaders that are themselves members of other group leaders ...')

        n_nested = 0
        header = ['group', 'member']
        adict = {h:[] for h in header}

        # build a trace chain first
        heads = set(conceptToLabs) # set of keys (group MED codes)
        conceptToLabsPrime = {} # {cH: set() for cH in conceptToLabs.keys()}
        chains = {}
        for cH, codes in conceptToLabs.items(): 
            # conceptToLabsPrime[cH].update(codes)
            others = heads-set([cH])
            is_nested = False
            for cH2 in others: 
                if cH in conceptToLabs[cH2]: # if cH is a member of another leader cH2, then cH < cH2, cH2 should inherit all cH members
                    is_nested = True
                    if not chains.has_key(cH): chains[cH] = set()
                    chains[cH].add(cH2)  # link cH to cH2 (cH < cH2)
            if is_nested: n_nested += 1

        print('info> Found %d (=?=%d) group leaders nested underneath someone else' % (n_nested, len(chains)))

    # [test] if group leaders has multiple members, remove itself
    n_removed = 0 
    for cH, members in conceptToLabs.items(): 
        # members = conceptToLabs[cH].copy()
        if len(members) > 1: 
            if cH in members: 
                print('info> removing self leader %s from %s' % (cH, members))
                conceptToLabs[cH].remove(cH)
                n_removed += 1
    assert n_removed == 0, "there exist group leaders that were linked to not only self but also others ..."
    
    # sort
    n_unlinked = 0
    for c in ulabs: # make sure all lab members are linked
        # assert c in conceptToLabs
        if not c in conceptToLabs: 
            n_unlinked += 1 
            continue 

        members = conceptToLabs[c]
        n_members = len(members)
        if n_members > 1: 
            assert not c in members
            conceptToLabs[c] = sorted(list(members))  # ordering
        else: 
            assert n_members == 1

    # [log] n_linked (codes with assigned classes): 452 | n_linked::multiclass: 0
    print('policy #2> n_linked (codes with assigned classes): %d | n_linked::multiclass: %d' % (n_linked, n_linked_multiple))

    # save the mapping assuming that dflab_map is avail 
    # io> saving df[dim: (3679, 4) (n_codes: 2306)] of grouped labs to tpheno/data-exp/grouped_labs-Pge3-Tbmeasurement-PTSD.csv
    overwrite = True  
    version = 3   
    stem, identifier = ('grouped_labs', 'P%s%s-Tb%s-%s' % (concept_grouping, version, table_name, cohort_name))
    fname = 'grouped_labs-%s.csv' % identifier
    fpath = os.path.join(outputdir, fname)
    header = ['group', 'individual_lab', 'group_description', 'individual_description']
    sortbyattr = ['group', 'individual_lab']
    n_eff_groups = 0
    if overwrite or not os.path.exists(fpath): 
        
        adict = {h:[] for h in header}
        for cH, codes in conceptToLabs.items(): 
            if len(codes) > 1: 
                n_eff_groups += 1
                assert not cH in codes
            for c in codes:  
                adict['group'].append(cH) # group representative code
                adict['individual_lab'].append(c) # group members
                adict['group_description'].append(dflab_map[cH])
                adict['individual_description'].append(dflab_map[c])
        df = DataFrame(adict, columns=header)
        # df = df.sort('group', ascending=True)
        df.sort_values(sortbyattr, ascending=True, inplace=True)
        print('io> saving df[dim: %s (n_codes: %d)] of grouped labs to %s' % (str(df.shape), len(conceptToLabs), fpath))
        df.to_csv(fpath, sep='|', index=False, header=True)    
            
        # [test]
        ugroups = set(df['group'].values)
        umembers = set(df['individual_lab'].values) 
        ulabx = ugroups.union(umembers)
        
        # among all target lab codes (n:2817 < nrows:5708308) <? 2306 have entity measured.
        # test> n_groups: 2306, n_members: 2306, ulabs: (2306 <? 2817 [source])
        print('test> n_groups: %d (n_eff_groups: %d), n_members: %d, ulabs: (%d <? %d [source])' % \
            (len(ugroups), n_eff_groups, len(umembers), len(ulabx), len(ulabs)))

        # [test]
        # each member in 'individual_lab' should be mapped to exactly one group leader (itself of somenoe else)
        nrow = df.shape[0]
        rglabs = {}  # member -> leader
        for i, (r, row) in enumerate(df.iterrows()): 
            ch, c = row['group'], row['individual_lab']
            if not rglabs.has_key(c): rglabs[c] = []
            rglabs[c].append(ch)
        n_multiple = 0
        for i, (c, ch) in enumerate(rglabs.items()): 
            if len(ch) > 1: 
                if i % 10 == 0: print('  ++ lab member %s is mapped to 1+ leaders: %s' % (c, ch)) # => self + others
                n_multiple += 1 
        print('test> number of members mapped to multiple leaders: %d =?= 0' % n_multiple)  # [log] 0

    return

def t_search_ptsd2(**kargs):   # t_search_ptsd() + concept grouping
    # from batchpheno import qrymed2
    import sys
    from scipy.stats import wilcoxon
    from batchpheno import sampling  # bootstrap resampling
    
    cohort_name = 'PTSD'
    ctrl_cohort_name = '%s-Negative' % cohort_name
    basedir = '/phi/proj/poc7002/tpheno/PTSD'

    ### step 0: read lab concept grouping file obtained from t_map_lab()
    # [input] e.g. tpheno/data-exp/grouped_labs-Pge3-Tbmeasurement-PTSD.csv
    div(message='Stage 0: Load lab concept mapping file ...')
    fname = 'grouped_labs-PTSD.csv'  # explicit: grouped_labs-Pge3-Tbmeasurement-PTSD.csv 
    header = ['group', 'individual_lab', 'group_description', 'individual_description']
    fpath = os.path.join(basedir, fname)
    
    df_map = pd.read_csv(fpath, sep='|', header=0, index_col=False, error_bad_lines=True)
    print('io> loaded concept grouping (dim: %s): %s' % (str(df.shape), fpath))

    ### 
    # [input] load PTSD treatmenet group
    fname = '%s-query_ids-PTSD.csv' % table_name
    fpath = os.path.join(outputdir, fname)
    
    print('io> loading lab table for cohort: %s' % cohort_name)
    # dtypes = {'value_source_value': int, } # [error] Integer column has NA values in column 4
    df_lab = pd.read_csv(fpath, sep='|', header=0, index_col=False, error_bad_lines=True)
    print('dim> PTSD cohort > lab values  > %s' % str(df_lab.shape))

    print('step 5a: compute summary statistics of cohort data ...')  # measurement-lab_values-PTSD.csv
    make_lab_stats(df_lab, cohort=cohort_name, table=table_name, n_sample=n_sample_test, output_dir=outputdir)

    return

def t_search_ckd(**kargs): 

    cohort_name = 'CKD'
    ctrl_cohort_name = '%s-Negative' % cohort_name
    tConceptGrouping = True

    # input directory
    # basedir = '/phi/proj/poc7002/tpheno/%s' % cohort_name
    basedir = os.path.join(sys_config.read('DataExpRoot'), cohort_name)
    fname = 'eMerge_NKF_Stage_20170818.csv'
    header = ['patientId', 'Case_Control_Unknown_Status', 'NKF_Stage', ]  
    fpath = os.path.join(sys_config.read('DataExpRoot'), fname)
    df = pd.read_csv(fpath, sep=',', header=0, index_col=False, error_bad_lines=True)

    print('io> loading patient list (cohort=%s) of dim: %s' % (cohort_name, str(df.shape)))
    print df.head(10)
    print df.tail(10)

    person_idx = list(set(df['patientId'].values))
    # create intermediate table (to be used later on to make medical coding sequences)
    tables = ['condition_occurrence', 'drug_exposure', ]
    for tb in tables:  
        print('   + found %d cases in table=%s' % (len(person_idx), tb))
        df = searchByID(table=tb, ids=person_idx, cohort=cohort_name, save_intermediate=True) 
        print('   + %s dataframe dim: %s' % (tb, str(df.shape)))

    return

def readSource(**kargs):
    cohort_name = kargs.get('cohort', 'CKD')
    ctrl_cohort_name = '%s-Negative' % cohort_name
    tConceptGrouping = True

    # input directory
    # basedir = '/phi/proj/poc7002/tpheno/%s' % cohort_name
    basedir = os.path.join(sys_config.read('DataExpRoot'), cohort_name)
    fname = kargs.get('fname', 'eMerge_NKF_Stage_20170818.csv')   
    header = ['patientId', 'Case_Control_Unknown_Status', 'NKF_Stage', ]  
    fpath = os.path.join(basedir, fname)
    df = pd.read_csv(fpath, sep=',', header=0, index_col=False, error_bad_lines=True)   

    return df

def getLabels(**kargs): 
    return t_getLabels(**kargs)

def t_getLabels(**kargs):
    # obtain labels from file using CKD dataset as an example 

    # [todo] configure this 
    srcFiles = {'CKD': 'eMerge_NKF_Stage_20170818.csv'}
    # header: ['patientId', 'Case_Control_Unknown_Status', 'NKF_Stage', ]
    f_label = 'NKF_Stage'
    
    cohort_name = kargs.get('cohort', 'CKD')
    df_src = readSource(cohort=cohort_name, fname=srcFiles[cohort_name])
    print('input> nrow: %d, ncol: %d' % (df_src.shape[0], df_src.shape[1]))
    
    pidx = df_src['patientId'].unique()
    n_persons = len(pidx)
    print('   + nrow: %d >? n_persons: %d' % (df_src.shape[0], n_persons))

    idToLabels = {}
    n_multi = 0
    for pid, df in df_src.groupby('patientId'):
        if not idToLabels.has_key(pid): idToLabels[pid] = set()
        lx = df[f_label].unique()
        nl = len(lx)
        if nl > 1: n_multi += 1 
        idToLabels[pid].update(lx)
    print('info> number of persons with multiple labels: %d' % n_multi)
    return 

def t_search_ptsd(**kargs): 
    """

    Memo
    ---- 
    1. The equivalent of the R pnorm() function is: scipy.stats.norm.cdf() with python 
       The equivalent of the R qnorm() function is: scipy.stats.norm.ppf() with python

    Related
    -------
    t_map_lab | t_concep_grouping 

    """
    # from batchpheno import qrymed2
    import sys
    from scipy.stats import wilcoxon
    from batchpheno import sampling  # bootstrap resampling
    
    cohort_name = 'PTSD'
    ctrl_cohort_name = '%s-Negative' % cohort_name
    tConceptGrouping = True

    basedir = '/phi/proj/poc7002/tpheno/%s' % cohort_name

    ### step 0: read lab concept grouping file obtained from t_map_lab()
    # [input] e.g. tpheno/data-exp/grouped_labs-Pge3-Tbmeasurement-PTSD.csv
    div(message='Stage 0: Load lab concept mapping file ...')
    fname = 'grouped_labs-Pge3-Tbmeasurement-PTSD.csv' # grouped_labs-PTSD.csv | grouped_labs-Pge3-Tbmeasurement-PTSD.csv 
    header = ['group', 'individual_lab', 'group_description', 'individual_description']
    fpath = os.path.join(sys_config.read('DataExpRoot'), fname)
    
    df_map = pd.read_csv(fpath, sep='|', header=0, index_col=False, error_bad_lines=True)
    print('io> loaded concept grouping (dim: %s): %s' % (str(df_map.shape), fpath))

    if tConceptGrouping: assert df_map is not None and not df_map.empty

    ### step 1: Read from source (i.e. selected cohort described by their IDs and other attributes)

    fname = 'PTSD_S1_patient_lists.csv'
    header = ['person_id', 'first_diagnosis', 'ICD_code']
    fpath = os.path.join(basedir, fname)

    df = pd.read_csv(fpath, sep=',', header=0, index_col=False, error_bad_lines=True)
    print('io> loading patient list (cohort=%s) of dim: %s' % (cohort_name, str(df.shape)))
    print df.head(10)
    print df.tail(10)

    ### step 2: 
    field_source = 'condition_source_value'
    field_date = 'condition_start_date'
    idx_cohort = idx = list(set(df['person_id'].values))
    n_cohort = len(idx_cohort)
    print('info> number of target patients: %d' % n_cohort)

    # [note] 'cohort' for file naming; there may be 1+ cohorts, for which default=diabetes (cohort_name: None)
    # df = searchByID(table='condition_occurrence', ids=idx, cohort='PTSD') 
    # print('dim> PTSD cohort: diag > %s' % str(df.shape))
    # print df.head(10)
    # print df.tail(10)

    # /phi/proj/poc7002/tpheno/data-exp
    outputdir = kargs.get('output_dir', sys_config.read('DataExpRoot')) # temporary data dir

    # # [log] writing querying result to /phi/proj/poc7002/tpheno/data-exp/condition_occurrence-query_ids-PTSD.csv
    fname = 'condition_occurrence-query_ids-PTSD.csv'
    fpath = os.path.join(outputdir, fname)
    # df = pd.read_csv(fpath, sep='|', header=0, index_col=False, error_bad_lines=True)
    
    # df = normalize(df, target_field=field_source)  # expensive!
    # df.to_csv(fpath, sep='|', index=False, header=True)
    # print('status> completed normalizing condition table ...')

    idx_diag = list(set(df['person_id'].values))
    print('verify> n_patients: %d, same cohort? (i.e. delta=0) %s' % (len(idx_diag), len(set(idx)-set(idx_diag))) )  # differ by 4 patients

    ### step 3. drug 
    # df_med = searchByID(table='drug_exposure', ids=idx, cohort='PTSD')
    div('Step 3. Create %s medication table' % cohort_name)
    # e.g. /phi/proj/poc7002/tpheno/data-exp/drug_exposure-query_ids-PTSD.csv
    fname = 'drug_exposure-query_ids-PTSD.csv'
    fpath = os.path.join(outputdir, fname)
    df_med = pd.read_csv(fpath, sep='|', header=0, index_col=False, error_bad_lines=True)
    print('io> PTSD cohort: medication > %s' % str(df_med.shape))
    # print df_med.head(10)
    # print df_med.tail(10)

    idx_med = list(set(df_med['person_id'].values))
    # # [log] verify> n_patients: 4387, same cohort? (i.e. delta=0) 1007
    print('verify> n_patients: %d, same cohort? (i.e. delta=0) %s' % (len(idx_med), len(set(idx)-set(idx_med))) )  # differ by 4 patients

    ### step 4. extract common diagnostic codes 
    # t_extract_diag()
    div('Step 4: Extract common diagnostic codes')

    # common diagnositc codes within randomly selected sample patients
    # common_codes(df, n_sample=10)


    # step 5. extract lab values 
    div('Step 5: Lab value analysis ...')
    # header: person_id, measurement_date, measurement_time, value_as_number, value_source_value
    # augmented: person_id, measurement_date, measurement_time, value_as_number, value_source_value, source_description
    table_name = 'measurement'
    pivots = ['value_source_value', ]  # lab codes
    # df = searchByID(table=table_name, ids=idx, cohort='PTSD') # scope will be added
    # print('dim> PTSD cohort > lab values  > %s' % str(df.shape))
    # print df.head(10)
    # print df.tail(10)

    fname = '%s-query_ids-PTSD.csv' % table_name
    fpath = os.path.join(outputdir, fname)
    
    print('io> loading lab table for cohort: %s' % cohort_name)
    # dtypes = {'value_source_value': int, } # [error] Integer column has NA values in column 4
    df_lab = pd.read_csv(fpath, sep='|', header=0, index_col=False, error_bad_lines=True)
    print('io> PTSD cohort > lab values  > %s' % str(df_lab.shape))
    
    # print('io> saving source description (qrymed returns) to %s' % fpath)
    # df_lab['value_source_value'] = df_lab['value_source_value'].astype(int)
    # df_lab.to_csv(fpath, sep='|', index=False, header=True)   

    # drop_cols = ['value_source_value', 'value_as_number']
    # df_lab = df_lab.dropna(subset=drop_cols)

    lab_tests = df_lab['value_source_value'].unique()
    print('info> found %d unique lab test in positive group' % len(lab_tests))
    # query code description
    # descriptions = {}
    # print('verify> number of unique lab tests: %d' % len(lab_tests)) # [log] number of unique lab tests: 2555

    # for ltest in lab_tests:
    #     if not descriptions.has_key(ltest): 
    #         descriptions[ltest] = qrymed2.getName2(ltest, err_default='unknown') 

    ### Step 5a. load lab descriptions 
    div('step 5a> loading lab test source data')
    basedir_desc = sys_config.read('DataExpRoot') # temporary (output) data dir
    table_name = 'measurement'
    header = ['value_source_value', 'source_description'] 
    pivots = ['value_source_value', ]  # lab codes
    fname = '%s-code_desc-%s.csv' % (table_name, cohort_name)  
    fpath = os.path.join(basedir_desc, fname)

    print('io> loading code-to-desc from: %s' % fpath)
    df_desc = pd.read_csv(fpath, sep='|', header=0, index_col=False, error_bad_lines=True)
    print('status> loaded lab table (cohort:%s) > dim: %s' % (cohort_name, str(df_desc.shape)))
    descriptions = dict(zip(df_desc['value_source_value'], df_desc['source_description']))

    # add code descriptions
    # sdvals = []
    # for ltest in df_lab['value_source_value'].values: 
    #     sdvals.append(descriptions[ltest])
    # df_lab['source_description'] = sdvals
    # df_lab.to_csv(fpath, sep='|', index=False, header=True)

    # step 5b statistics
    # [input] measurement-query_ids-PTSD.csv
    n_sample_test = 500  # store the values for wilcoxon test, etc. (between cohort and control)
    # lval_groups = df_lab.groupby(pivots)  # as_index=False # set as_index to False => pivots becomes a separate column instead of an index
    header_lval = ['code', 'mean', 'median', 'std', 'min', 'max', 'n_sample', ]

    print('step 5a: compute summary statistics of cohort data ...')  # measurement-lab_values-PTSD.csv
    make_lab_stats(df_lab, concept_map=df_map, cohort=cohort_name, table=table_name, n_sample=n_sample_test, output_dir=outputdir)

    # step 6. control set (patients without 309.81: PTSD) who 
    #         1. has max lab tests done? 
    #         2. similar but without 309.81?
    # query measurement table and take those not in PTSD cohort (which means they do not have 309.81)

    gamma = 1.5 
    n_sample_control = int(n_cohort * gamma)

    print('step 6: generate control data ...')
    # df = selectFrom(table=table_name, columns=['person_id', ])    
    # idx_lab = df['person_id'].unique()

    # idx_control = list(set(idx_lab)-set(idx_cohort))
    # n_control_total = len(idx_control)

    # idx_control_selected = random.sample(idx_control, min(n_sample_control, n_control_total))
    # n_control_selected = len(idx_control_selected)
    # print('info> number of control candidates: %d >? SELECTED: %d (requested: %d vs n_cohort: %d * gamma: %f)' % \
    #     (n_control_total, n_control_selected, n_sample_control, n_cohort, gamma))

    header = ['person_id', 'measurement_date', 'measurement_time', 'value_as_number', 'value_source_value']
    
    # # step 6.1: Given control cohort, generate its lab table
    # df_lab_ctrl = df = searchByID(table=table_name, ids=idx_control_selected, cohort=ctrl_cohort_name) # scope will be added
    # print('dim> PTSD Control cohort > lab values  > %s' % str(df_lab_ctrl.shape))
    # print df_lab_ctrl.head(10)
    # print df_lab_ctrl.tail(10)

    # statics 


    # lab_ctrl_min = df_lab_ctrl['value_as_number'].min() # series ~ feature
    # lab_ctrl_max = df_lab_ctrl.max()
    # dfq = df_lab.quantile([.25, .5, .75])

    # [load] control data 
    # ctrl_cohort_name = 'PTSD-Negative'
    # dtypes = {'value_source_value': int, }
    fname = '%s-query_ids-%s.csv' % (table_name, ctrl_cohort_name)
    fpath = os.path.join(outputdir, fname)
    assert os.path.exists(fpath), "Nonexistent input %s" % fpath
    print('io> loading control data from %s' % fpath)
    df_lab_ctrl = pd.read_csv(fpath, sep='|', header=0, index_col=False, error_bad_lines=True) # dtype=dtypes

    # given df_lab and df_lab_ctrl 
    print('data> convert df_lab to a binary class training data')
    # make_tset()  # use only the raw features to make tset 

    # sys.exit(0)

    # print('io> saving to %s' % fpath)
    # df_lab_ctrl['value_source_value'] = df_lab_ctrl['value_source_value'].astype(int)
    # df_lab_ctrl.to_csv(fpath, sep='|', index=False, header=True)   

    # drop_cols = ['value_source_value', 'value_as_number']
    # df_lab_ctrl = df_lab_ctrl.dropna(subset=drop_cols)
    # dfmin = df_lab.min()
    # dfq = df_lab.quantile([.25, .5, .75]) # 3 rows add to each feature

    lab_tests_ctrl = df_lab_ctrl['value_source_value'].unique()
    lab_common = set(lab_tests).intersection(lab_tests_ctrl)
    n_lab_common = len(lab_common)
    print('verify> number of unique lab tests (control): %d vs n_commmon: %d' % (len(lab_tests_ctrl), n_lab_common))
    # descriptions = dict(zip(df_lab['value_source_value'], df_lab['source_description']))

    # add code descriptions
    # sdvals = []
    # for lt in df_lab_ctrl['value_source_value'].values: 
    #     sdvals.append(descriptions.get(lt, 'unknown'))
    # df_lab_ctrl['source_description'] = sdvals
    # df_lab_ctrl.to_csv(fpath, sep='|', index=False, header=True)
    
    # Step 6a statistics 
    # header_lval = ['code', 'mean', 'median', 'std', 'min', 'max', 'n_sample', ]
    # pivots = ['value_source_value', ]  # lab codes
    # n_sample_test = 1000
    print('step 6b: compute summary statistics of control data ...')
    make_lab_stats(df_lab_ctrl, concept_map=df_map, cohort=ctrl_cohort_name, table=table_name, n_sample=n_sample_test, output_dir=outputdir)
    
    # sys.exit(0)

    # Step 7. Statistics, rank vars, wo
    print('Step 7: Comparing cohort and control with wilcoxon test, ...')

    # [precond]
    # 1. df_lab's source_description has been specified via qrymed2 
    # 2. lab values obtained
    
    # [params]
    #  first block created by make_lab_stats()
    n_sample_min = 20
    n_resample_min = 5  # bootstrap if at least has 5 examples
    sl = 0.05  # significance level at 5%
    sl_suffix = str(sl).split('.')[1]
    hd_basics = ['code', 'mean', 'median', 'min', 'max', 'std']
    hd_test = ['pval_wilcoxon', 'sig_%s' % sl_suffix, 'grand_mean', 'grand_median']
    hd_desc = ['n_sample', 'is_active', 'description'] 
    header_ht = hd_basics + hd_test + hd_desc

    ht = {}

    # scipy.stats.wilcoxon

    # [note] Wilcoxon signed-rank test
    # The Wilcoxon signed-rank test tests the null hypothesis that two related paired samples come from the same distribution.
    # It is a non-parametric version of the paired T-test.
    # Because the normal approximation is used for the calculations, the samples used should be large. A typical rule is to require that n > 20.

    fname = '%s-lab_values-%s.pkl' % (table_name, cohort_name)
    if tConceptGrouping: 
        fname = '%s-grouped_lab_values-%s.pkl' % (table_name, cohort_name)

    fpath = os.path.join(outputdir, fname)  
    print('io> loading cohort lab values (concept grouped? %s) from %s' % (tConceptGrouping, fpath))  
    vdict = pickle.load(open(fpath, 'rb'))  
    # for lc, vals in vdict.items(): 
    #     pass

    # [cohort]
    fname = '%s-lab_values-%s.pkl' % (table_name, ctrl_cohort_name)
    if tConceptGrouping: 
        fname = '%s-grouped_lab_values-%s.pkl' % (table_name, ctrl_cohort_name)

    fpath = os.path.join(outputdir, fname)    
    print('io> loading control lab values from %s' % fpath)
    vdict_ctrl = pickle.load(open(fpath, 'rb'))  

    lab_common = set(vdict.keys()).intersection(vdict_ctrl.keys())
    print('verify> num of common codes (cohort vs ctrl): %d' % len(lab_common))

    # load summary stats file 
    # header_lval = ['code', 'mean', 'median', 'std', 'min', 'max', 'n_sample', ] < ['is_active', ]
    # 7a. in cohort but not in control? 
    # 7b. hypo tests

    print('Step 7a: Wilcoxon signed-rank test')
    fname = '%s-lab_values-%s.csv' % (table_name, cohort_name)
    if tConceptGrouping: 
        fname = '%s-grouped_lab_values-%s.csv' % (table_name, cohort_name)

    fpath = os.path.join(outputdir, fname)  
    print('io> loading (cohort) lab summary stats from %s (created by make_lab_stats)' % fpath)  # make_lab_stats(
    df_stats = pd.read_csv(fpath, sep='|', header=0, index_col=False, error_bad_lines=True)
    
    n_small_sample = n_small = 0

    # hd_test = ['pval_wilcoxon', 'sig_%s' % sl_suffix , 'grand_mean', 'grand_median']
    adict = {h: [] for h in hd_test}

    # gmeans, gmedians = {}, {}
    # poll attributes: ['pval_wilcoxon', 'sig_05', 'grand_mean', 'grand_median']
    f_sig = 'sig_%s' % sl_suffix

    for lc in df_stats['code'].values:  # lc could be a group leader if tConceptGrouping: True

        run_ht = False
        T, pval = -1, 0.99
        gmean, gmedian = np.nan, np.nan

        assert lc in vdict
        vals = vdict[lc]
        # n_vals, n_vals_ctrl = len(vals), 0 

        if lc in vdict_ctrl:
            vals_ctrl = vdict_ctrl[lc] 
            n_vals_ctrl = len(vals_ctrl)
            
            n_ref, n_ctrl = len(vals), len(vals_ctrl)            
            vals2, vals_ctrl2 = vals, vals_ctrl
            if n_ref < n_sample_min or n_ctrl < n_sample_min: 
                n_small += 1
                print('  + lab code %s (%s) has small sample (%d vs %d ctrl)' % \
                           (lc, descriptions.get(int(lc), 'unknown'), n_ref, n_ctrl))
                
                # try bootstrapping to create more samples
                if n_ref >= n_resample_min and n_ctrl >= n_resample_min:
                    vals2 = sampling.bootstrap_resample(vals, n=n_sample_min)
                    vals_ctrl2 = sampling.bootstrap_resample(vals_ctrl, n=n_sample_min)
                    n_ref, n_ctrl = len(vals2), len(vals_ctrl2)
                    run_ht = True
            else: 
                run_ht = True

            if run_ht: 
                assert n_ref >= n_sample_min and n_ctrl >= n_sample_min, '  + sample still small? (%d vs %d ctrl)' % (n_ref, n_ctrl)
                ns = min(n_ref, n_ctrl)
                vals2 = random.sample(vals2, ns)
                vals_ctrl2 = random.sample(vals_ctrl2, ns)

                T, pval = wilcoxon(vals2, vals_ctrl2)
            else: 
                print('  ++ no testing for lab %s (sample too small [(%d|%d) < %d])' % (lc, n_ref, n_ctrl, n_resample_min))

            # btw, collect grand means and grand medians
            valx = np.hstack((vals2, vals_ctrl2))
            gmean = np.mean(valx)
            gmedian = np.median(valx)

        else: 
            print('    ++ lab code %s (%s) is not active!' % (lc, descriptions.get(int(lc), 'unknown'))) 
        
        adict['pval_wilcoxon'].append(pval)
        adict[f_sig].append( int(pval < sl) )
        adict['grand_mean'].append(gmean)
        adict['grand_median'].append(gmedian)

    assert len(adict['pval_wilcoxon']) == len(vdict.keys())

    # update
    df_stats['pval_wilcoxon'] = adict['pval_wilcoxon']
    df_stats[f_sig] = adict[f_sig]  # significant level at x% 
    df_stats['grand_mean'] = adict['grand_mean']
    df_stats['grand_median'] = adict['grand_median']

    fname = '%s-lab_values-%s.csv' % (table_name, ctrl_cohort_name)
    if tConceptGrouping: 
        fname = '%s-grouped_lab_values-%s.csv' % (table_name, ctrl_cohort_name)

    fpath = os.path.join(outputdir, fname)  
    print('io> loading (control) lab summary stats from:\n%s' % fpath)
    df_stats_ctrl = pd.read_csv(fpath, sep='|', header=0, index_col=False, error_bad_lines=True) 

    print('7a> in cohort but not in control?')
    lt_cohort_only = list(set(df_stats['code']) - set(df_stats_ctrl['code']))

    # [pandas]
    # index: ts[ts[TSet.gold_indicator].isin(gold_index)]
    # update: sdf.loc[cond, ['coeff', 'score', 'description', 'rank']] = [coeff, score, desc, r]  # lowest value, higher rank
    f_active = 'is_active'
    df_stats[f_active] = 1 # assume active
    cond_cohort = df_stats['code'].isin(lt_cohort_only)
    df_stats.loc[cond_cohort, f_active] = 0 # if the lab test (code) present in both, then is_active, o.w. not active

    # add description 
    # descriptions = dict(zip(df_lab['value_source_value'], df_lab['source_description']))
    dvals = []
    for c in df_stats['code'].values: 
        dvals.append(descriptions.get(c, 'unknown'))
    df_stats['description'] = dvals

    header_ht = ['code', 'mean', 'median', 'min', 'max', 'pval_wilcoxon', 'sig_%s' % sl_suffix, 'std', 
                    'n_sample', 'is_active', 'description', 'grand_mean', 'grand_median'] 
    df_stats = df_stats[header_ht]

    fname = '%s-lab_values-%s.csv' % (table_name, cohort_name)
    if tConceptGrouping: 
        fname = '%s-grouped_lab_values-%s.csv' % (table_name, cohort_name)

    fpath = os.path.join(outputdir, fname)  
    print('io> saving test statistics (dim: %s) to:\n%s' % (str(df_stats.shape), fpath))
    df_stats.to_csv(fpath, sep='|', index=False, header=True)

    # statistically sig lab tests based on wilcoxon tests
    cond_sig = df_stats['sig_%s' % sl_suffix] == 1
    codes_sig = df_stats.loc[cond_sig]['code'].values
    
    header_sig = ['code', 'description']
    adict = {h: [] for h in header_sig}
    for c in codes_sig: 
        val = descriptions.get(c, 'unknown')
        adict['code'].append(c)
        adict['description'].append(val)
    
    fname = '%s-sig_labs-%s.csv' % (table_name, cohort_name)
    if tConceptGrouping: 
        fname = '%s-grouped_sig_labs-%s.csv' % (table_name, cohort_name)

    fpath = os.path.join(outputdir, fname)  
    df_sig = DataFrame(adict, columns=header_sig)
    print('io> saving significant lab variables (dim: %s) to:\n%s' % (str(df_sig.shape), fpath))
    df_sig.to_csv(fpath, sep='|', index=False, header=True)

    print("Step 8. Pairwise Pearson correlation (on significantly different vars only?) ...")
    # df_stats.sort()

    

    return 

def loadDF(identifier,  inputdir=None): 
    def read_condition_table(cols=None): 
        dtypes = {'condition_source_value': str}
        fpath1 = os.path.join(basedir, condition_tb)
        assert os.path.exists(fpath1), "Invalid diagnosis input: %s" % fpath1 
        df_condition = pd.read_csv(fpath1, 
                              sep='|', index_col=False, header=0, error_bad_lines=True, 
                                     parse_dates=['condition_start_date'], dtype=dtypes) 
        assert len(set(header_condition)-set(df_condition.columns)) == 0
        if cols is not None: 
            pass
        return df_condition
    def read_drug_table(cols=None): 
        dtypes = {'drug_source_value': str}
        fpath2 = os.path.join(basedir, drug_tb)
        assert os.path.exists(fpath2), "Invalid medication input: %s" % fpath2
        df_drug = pd.read_csv(fpath2, 
                                sep='|', index_col=False, header=0, error_bad_lines=True, 
                                   parse_dates=['drug_exposure_start_date'], dtype=dtypes)
        assert len(set(header_drug)-set(df_drug.columns)) == 0
        return df_drug

    raise NotImplementedError

def loadConditionDF(identifier, inputdir=None, prefix='condition_occurrence', delimit='|'): 
    """
    Load intermediate dataframes (fetched from OHDSI DB)


    Params 
    ------
    identifier: query_codes

    Memo
    ----
    1. example intermediate files
       data-exp/condition_occurrence-query_codes.csv
          => identifier <- query_codes

       ondition_occurrence-query_ids-CKD.csv
          => identifier <- query_ids-CKD

       data-exp/sequencing/condition_occurrence-{0-9}.csv
          => identifier <- 0, 1, ... 9
    """
    def resolve_input(): 
        if inputdir is None: 
            basedir = sys_config.read('DataExpRoot')  # tpheno/data-exp
            inputdir = os.path.join(basedir, 'sequencing')  # tpheno/data-exp/sequencing
        assert os.path.exists(inputdir), "Invalid input dir:\n%s\n" % inputdir

        ifile = '%s-%s.csv' % (prefix, identifier)
        fpath = os.path.join(inputdir, ifile)
        assert os.path.exists(fpath), "Invalid intermediate file: %s" % ifile 
        return fpath
    def verify_output(): 
        assert df_condition is not None and not df_condition.empty, "Null dataframe."
        nrow, ncol = df_condition.shape[0], df_condition.shape[1]
        print('  + intermediate file dim: %d by %d' % (nrow, ncol))
        return 

    # parameters for intermediate files
    fpath = resolve_input()
    col_date, col_source = 'condition_start_date', 'condition_source_value'
    dtypes = {col_source: str}
    df_condition = pd.read_csv(fpath, sep=delimit, index_col=False, header=0, error_bad_lines=True, 
                                     parse_dates=[col_date], dtype=dtypes)

    verify_output()
    return

def loadDrugDF(identifier, inputdir=None, prefix='drug_exposure', delimit='|'): 
    """
    Load intermediate dataframes (fetched from OHDSI DB)


    Params 
    ------
    identifier: query_codes

    Memo
    ----
    1. example intermediate files
       data-exp/condition_occurrence-query_codes.csv
          => identifier <- query_codes

       ondition_occurrence-query_ids-CKD.csv
          => identifier <- query_ids-CKD

       data-exp/sequencing/condition_occurrence-{0-9}.csv
          => identifier <- 0, 1, ... 9
    """
    def resolve_input(): 
        if inputdir is None: 
            basedir = sys_config.read('DataExpRoot')  # tpheno/data-exp
            inputdir = os.path.join(basedir, 'sequencing')  # tpheno/data-exp/sequencing
        assert os.path.exists(inputdir), "Invalid input dir:\n%s\n" % inputdir

        ifile = '%s-%s.csv' % (prefix, identifier)
        fpath = os.path.join(inputdir, ifile)
        assert os.path.exists(fpath), "Invalid intermediate file: %s" % ifile 
        return fpath
    def verify_output(): 
        assert df is not None and not df.empty, "Null dataframe."
        nrow, ncol = df.shape[0], df.shape[1]
        print('  + intermediate file dim: %d by %d' % (nrow, ncol))
        return 

    # parameters for intermediate files
    fpath = resolve_input()
    col_date, col_source = 'drug_exposure_start_date', 'drug_source_value'
    dtypes = {col_source: str}
    df = pd.read_csv(fpath, sep=delimit, index_col=False, header=0, error_bad_lines=True, 
                            parse_dates=[col_date], dtype=dtypes)

    verify_output()
    return

def t_augment_cohort(**kargs):
    """
    A template function for making cohort-specific coding sequences, where
    the cohort is defined via 
    a. a set of diagnostic codes (e.g. those defined in clincal classification software, CCS)
    b. a set of patiet IDs (col=person_id)


    Note
    ----
    1. the resulted docuemnts should be seqClassifer.makeTSet ready
       see seqClassifer.processDocuments 
           seqClassifer.loadDocuments

    Related
    -------
    t_search_ckd(): from tempalte functions t_search_{disease}

    """
    def select_by_ids(df, person_ids, col='person_id'): 
        return df.loc[df[col].isin(idx)]
    def select_by_diag(df, codes):  # <- diagCodes
        # condition_occurrence-0.csv => identifier <- 0 
        # df_condition = loadConditionDF(identifier=identifier, inputdir=basedir_sequencing) 
        assert 'condition_source_value' in df.columns, "Invalid condition dataframe > header=%s" % df.columns.values
        candidates = filterCandidatesByCodes(codes=codes, dataframe=df)  # person_id -> {code}
        uids = set(candidates.keys())  # candidates: person_id -> {code}
        print('  + found %d unique candidates in group %s' % (len(uids), identifier))
        return uids  # candidates: person_id -> {code}
    def load_labeled_candidates(prefix='condition_occurrence', suffix='query_ids-src', col='person_id'): 
        # load the person_ids of the original, labeled dataset 
        ifile = '%s-%s.csv' % (prefix, suffix)  # header: person_id|condition_start_date|condition_source_value
        inputdir = seqparams.getCohortGlobalDir(cohort_name) # basedir: sys_config.read('DataExpRoot')
        df_src = loadConditionDF(identifier=suffix, inputdir=inputdir, prefix=prefix, delimit='|')
        assert col in df_src, "Invalid input dataframe > header=%s" % df_src.columns.values
        uids = set(df_src[col])
        print('  + found %d unique IDs from the labeled candidates' % len(uids))
        return uids
    def save_intermediate(df, prefix, suffix='query_ids-augment', delimit='|'): # df_condition, df_drug
        # save new intermediate files that include the new candiates

        # tpheno/data-exp/<cohort>
        outputdir = seqparams.getCohortGlobalDir(cohort_name) # basedir: sys_config.read('DataExpRoot')
        if not df.empty:
            fp = os.path.join(outputdir, '%s-%s.csv' % (prefix, suffix))
            # if not os.path.exists(fp) or (os.path.exists(fp) and overwrite_intermediate): 
            print('  + writing augmented canidates to %s' % fp)
            df.to_csv(fp, sep=delimit, index=False, header=True, encoding='utf-8') 
        return fp
    def get_global_cohort_dir(): 
        return seqparams.getCohortGlobalDir(cohort_name) # basedir: sys_config.read('DataExpRoot')
        
    import seqMaker2 as smaker
    import seqparams
    from cohort import ccs
    # import cohort.ccs as ccs

    # parameters for intermediate files
    n_parts = 10
    basedir = sys_config.read('DataExpRoot')  # tpheno/data-exp
    basedir_sequencing = os.path.join(basedir, 'sequencing')  # tpheno/data-exp/sequencing

    # use batchpheno.icd9utils.preproc_code() to process CCS ICD9 string
    cohort_name = kargs.get('cohort', 'CKD')
    diagCodes = ccs.getCohort(cohort_name)
    # CKD_codes = ['585', '585.1', '585.2', '585.3', '585.4', '585.5', '585.6', '585.9', '792.5', 'V42.0', 'V45.1', 'V45.11', 
    #              'V45.12', 'V56.0', 'V56.1', 'V56.2', 'V56.31', 'V56.32', 'V56.8']
    
    # labeled candidates  
    labeledPersonIDs = load_labeled_candidates(suffix='query_ids-src')
    tExcludeLabeled = True

    # if intermediate files are not available yet 
    # searchByDiag(codes=CKD_codes, exclude_ids=[], base_only=False)
    groupToIDs = {}
    dfc, dfd = [], [] # condition and drug subsets
    for i in range(0, 10): # condition_occurrence-0.csv, drug_exposure-0.csv 
        df_condition = loadConditionDF(identifier=identifier, inputdir=basedir_sequencing) 
        df_drug = loadDrugDF(identifier=identifier, inputdir=basedir_sequencing)
        idx = select_by_diag(df_condition, codes=diagCodes)

        # exclusion list (don't include those already labeled)
        if tExcludeLabeled: 
            idx = set(idx) - set(labeledPersonIDs)
            print('t_augment_cohort> After excluding labeled cadidates (n=%d) > %d remaining' % (len(labeledPersonIDs), len(idx)))

        # groupToIDs[i] = idx
        if len(idx) > 0: 
            dfci = select_by_ids(df_condition, person_ids=idx)
            dfdi = select_by_ids(df_drug, person_ids=idx)
            assert not dfci.empty()
            dfc.append(dfci)
            if not dfdi.empty(): dfd.append(dfdi)
    
    # create new intermediate dataframes 
    df_condition = pd.concat(dfc, ignore_index=True)
    f_condition = save_intermediate(df_condition, prefix='condition_occurrence')  # 'condition_occurrence-query_ids-augment.csv'
    df_drug = pd.concat(dfd, ignore_index=True)
    f_drug = save_intermediate(df_drug, prefix='drug_exposure') # 'drug_exposure-query_ids-augment.csv'

    # [input] condition_occurrence-query_ids-augment.csv 
    #         drug_exposure-query_ids-augment.csv
    # if intermediate files (dataframe) is available
    for tstamp in (True, ): 
        ret = smaker.make_seq(include_timestamps=tstamp,   # save_intermediate=True
                    include_diag=True, include_med=True, 
                    inputdir=get_global_cohort_dir(),     # tpheno/data-exp/<cohort>
                    condition_table='condition_occurrence-query_ids-augment.csv', 
                    drug_table='drug_exposure-query_ids-augment.csv',  
                    cohort=cohort_name, 
                    save_csv=True, save_id=True)  # save .csv file and ID file 
        
    return 

def t_sequencing(**kargs): 
    """
    Batch population-scale sequence maker. 
    For sequence reader, refer to seqReader module.
    """
    import seqMaker2 as smaker
    ### range query
    # low, hi = (0, 10000)
    # q = Q_Drug_RangeID.format(lower_bound=low, upper_bound=hi)
    # print('query>\n%s\n' % q)
    # query(statement=q, name='test', save_intermediate=True)

    # range query an attribute (e.g. person_id) => 
    # find out all patients and put their intermediate dataframes in special directory (e.g. .../sequencing)
    n_parts = 10
    basedir = sys_config.read('DataExpRoot')
    basedir_sequencing = os.path.join(basedir, 'sequencing')  #
    tRunRQ = False  # run the query that attempts to find out the range of an attribute in the DB (e.g. person_id)

    ### template files (condition and drug)
    # condition_occurrence-0.csv, drug_exposure-0.csv
    if tRunRQ: smaker.rangeQuery(**kargs) # use this to prepare intermediate dataframes

    n_persons = n_persons_eff = 0
    div('step> Include both diagnoses and medications ...')
    # [memo] 3 to 10, then 0 to 3
    for tstamp in (False, True): 
        # for i in range(n_parts): 
        for i in range(0, 10): # condition_occurrence-0.csv, drug_exposure-0.csv 
            cohort_name = 'group-%s' % (i+1)  # starting from 1
            ret = smaker.make_seq(include_timestamps=tstamp,   # save_intermediate=True
                    include_diag=True, include_med=True, 
                    inputdir=basedir_sequencing, 
                    condition_table='condition_occurrence-%s.csv' % i,  # zero-based
                    drug_table='drug_exposure-%s.csv' % i, 
                    cohort=cohort_name, 
                    save_csv=False, save_id=True)  # save .csv file and ID file 
            min_id, max_id = ret['min_id'], ret['max_id']
            n_persons += len(ret['person_id'])
            n_persons_eff += len(ret['person_id_eff'])
            div('status> finished %d sets of documements (n_persons=%d, n_persons_eff=%d, minId=%s, maxId=%s)' % \
                ((i+1), n_persons, n_persons_eff, min_id, max_id))

    div('step> Now include only diagoses ...')

    return

def test(**kargs): 
    # cohort_diabetes(**kargs)

    # print('test> query condition ...')
    # codes, bcodes = cohort_diabetes()
    # q = query_condition(where_=make_constraint(codes=codes, base_only=True))
    # print('query>\n%s\n' % q)

    ### 1. create source doucments (medical coding sequences)
    # print('test> fetch data ...')
    # t_fetch()

    # print('test> select candidate set matching common diag codes ...')
    # t_select(**kargs)

    ### 2. Create augmented source documents (in addition to labeled candidates)
    ###    Use: does extra documents help learning MedVec with higher predictive performance? 
    
    t_augment_cohort(**kargs)

    return

def test2(**kargs): 
    
    ### stage 1. Cohort Definition
    # t_search_ptsd(**kargs)
    # t_search_ckd(**kargs)  # to be used later on in seqMaker2 to make per-patient coding sequences

    # check label
    # t_getLabels(cohort='CKD')  # labeled data

    # make_tset()
    # t_select_features(n_features=100)

    ### stage 2

    # Gropuing lab codes according to MED
    # t_lab_ptsd(**kargs)  # prepare data
    # t_map_lab(**kargs)
    
    # t_consolidate(**kargs) # just use the modified t_search_ptsd() 
    # t_search_ptsd(**kargs)
    # t_select_features(n_features=100)

    return 

if __name__ == "__main__": 
    test()
    # test2()
