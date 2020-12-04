import pymssql
# import mssql_config

import pandas as pd
from pandas import DataFrame, Series
import os, gc, sys
from os import getenv 
import time, re, string, collections

import random
import scipy
import numpy as np

try:
    import cPickle as pickle
except:
    import pickle

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
    save_intermediate = kargs.get('save_intermediate', True)
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
    """
    global DBScope, tb_condition

    # [params]
    id_field = kargs.get('id_field', 'person_id')

    save_intermediate = kargs.get('save_intermediate', True)
    overwrite_intermediate = kargs.get('overwrite_intermediate', True)
    sample_subset = kargs.get('sample_subset', False)

    output_dir = kargs.get('output_dir', sys_config.read('DataExpRoot')) # temporary data dir
    fsep = '|'

    # [params] db table
    field_source = 'condition_source_value'
    field_date = 'condition_start_date'

    table = kargs.get('table', tb_condition)
    base_only = kargs.get('base_only', kargs.get('simplify_code', True))
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

    if exclude_ids: 
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
    df_condition = dataframe 

    # [params]
    basedir = kargs.get('input_dir', sys_config.read('DataExpRoot')) 
    table = kargs.get('table', tb_condition)
    save_intermediate = kargs.get('save_intermediate', True)
    exclude_ids = set(targets.target_set)
    base_only = kargs.get('base_only', pmed.containsBaseForm(codes))
    fsep = '|'

    # [params] db table
    field_id = 'person_id'
    field_source = 'condition_source_value'
    field_date = 'condition_start_date'

    if df_condition is None: 
    	ifile = kargs.get('ifile', '%s-query_codes.csv' % table)
        ipath = os.path.join(basedir, ifile)
        print('info> loading precomputed cohort dataframe at %s' % ipath)
        dtypes = {field_source: str}
        df_condition = pd.read_csv(ipath, sep=fsep, index_col=False, header=0, error_bad_lines=True, 
                                     parse_dates=[field_date], dtype=dtypes) 
    
    assert df_condition is not None and not df_condition.empty, "No condition data found!"

    n_unique = len(set(df_condition[field_id].values))
    print('info> df_condition > unique: %d, dim: %s' % (n_unique, str(df_condition.shape)))

    # get most representative code (e.g. for output file naming)
    counter = collections.Counter(icd9utils.getRootSequence(df_condition[field_source].values))
    repr_code = counter.most_common(1)[0][0].strip() 
    print('info> most representative code: %s' % repr_code)

    candidates = {}
    cnt = 0 
    commo_unmatched = set()  # [todo] use min heap + priority queue

    if base_only: 
    	# [todo] optimize
        for pid, df in df_condition.groupby('person_id'): 
            # assert not pid in exclude_ids

            refcodes = list(set(df[field_source].values))
            refcodes = normalize_diag(refcodes)
            refcodes_base = icd9utils.getRootSequence(refcodes)

            # [test]
            # if cnt < 10: print('test> pid: %s\n  => %s\n  => %s' % (pid, refcodes[:15], refcodes_base[:15]))
        
            diff = set(codes) - set(refcodes_base)
            diff_rev = set(refcodes_base) - set(codes)
            if len(diff) == 0: 
                candidates[pid] = refcodes

                n_candidates = len(candidates)
                if n_candidates % 10 == 0: 
                    print('test> Found %d candidates (out of %d) ...' % (n_candidates, cnt))
                
                if len(diff_rev) > 0: 
                    div(message='pid: %s has extra codes: %s' % (pid, list(diff_rev)), symbol='%')
            else: 
                # [test]
                if cnt < 100 or cnt % 5000 == 0: 
                    print('test> pid: %s > unmatched: %s' % (pid, diff))  # i.e. the patient doesn't have ... 

            cnt += 1 
    else: 
        for pid, df in df_condition.groupby('person_id'): 
            # assert not pid in exclude_ids

            refcodes = list(set(df[field_source].values))
            refcodes = normalize_diag(refcodes)

            # [test]
            # if cnt < 10: print('test> pid: %s => %s' % (pid, refcodes[:15]))
        
            if len(set(codes) - refcodes) == 0: 
                candidates[pid] = refcodes

                n_candidates = len(candidates)
                if n_candidates % 10 == 0: 
                    div(message='test> Found %d candidates (out of %d) ...' % (n_candidates, cnt), symbol='%')

            cnt += 1 

    print('info> Found %d eligible patients diagnosed with given codes:\n%s\n' % (len(candidates), codes))

    if candidates and save_intermediate: 
    	fpath = os.path.join(basedir, 'candidates-%s-%s.csv' % (codes[0], len(codes)))
    	header = [field_id, 'diagnosis']
    	adict = {h:[] for h in header}
        for k, vals in candidates.items(): 
            adict[field_id].append(k)  
            adict['diagnosis'].append(', '.join(str(v).strip() for v in vals))
        df = pd.DataFrame(adict, columns=header)
        print('info> Saving candidate data to %s' % fpath)
        df.to_csv(fpath, sep=fsep, index=False, header=True)

    return candidates

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
    base_only = kargs.get('base_only', kargs.get('simplify_code', True))

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
    base_only = kargs.get('base_only', kargs.get('simplify_code', True))

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
            if pmed.isICDv2(code):   
                codes[i] = code
            else: 
            	# msg += "Invalid diag code: %s\n" % e
                print("Invalid diag code: %s\n" % e)

        elif not e: 
            print("Empty diag code: %s" % e)
        else:  # additional processing
            if pmed.isICDv2(e):  # extra dot S53. 920. is considered valid 
                if e[-1] == '.': 
                    msg += 'warning> incomplete coding: %s\n' % e
                    codes[i] = e[:-1]
                else: 
                	pass
            else: 
                e = pmed.convert(e, nocatch=True)
                if pmed.isICDv2(e): 
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
    if kargs.get('base_only', False): 
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
    	q = "%s IN (%s)" % (constraint, codestr)  # to be compatible with query_condition()
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
            q = "%s LIKE '%s%%'" % (constraint, ucodes[0])
        else: 
            q = "WHERE %s LIKE '%s%%'" % (constraint, ucodes[0])

        for code in ucodes[1:-1]: 
            q += ' ' + 'OR'
            q += ' ' + constraint + " LIKE '%s%%'" % code 
            
        q += ' ' + 'OR' + ' ' + constraint + " LIKE '%s%%'" % ucodes[-1]
     
    return q

def make_lab_stats(df, cohort='PTSD', table='measurement', n_sample=1000, **kargs):
    # from batchpheno import qrymed2
    # import random

    outputdir = kargs.get('output_dir', sys_config.read('DataExpRoot')) # temporary data dir

    print('info> Computing lab statistics with input dim %s ...' % str(df.shape))

    cohort_name = cohort
    if cohort_name is None: cohort_name = 'diabetes'
    # ctrl_cohort_name = '%s-Negative' % cohort_name
    table_name = table

    header_lval = ['code', 'mean', 'median', 'std', 'min', 'max', 'n_sample', ]
    pivots = ['value_source_value', ]  # lab codes

    lval_groups = df.groupby(pivots)
    ldict = {h:[] for h in header_lval}  # for dataframe; header_lval => value
    vdict = {}  # code -> values 
    for i, (lc_, lg) in enumerate(lval_groups):  # foreach (lab code, values)
        # dfg.sort(sort_fields, ascending=False, inplace=False)
        lc = int(lc_)
        if i < 10: print('   + lab code? %s' % lc)

        # lg['value_as_number'].replace('', np.nan, inplace=True)
        vals0 = lg['value_as_number'].dropna().values

        if len(vals0) > 0: 
            # [todo] if number of lab values are suff, do bootstrapping
            vals = random.sample(vals0, min(n_sample, len(vals0))) # store the values for wilcoxon test (between cohort and control)
            n_vals = len(vals)
            vdict[lc] = vals
        
            # dataframe 
            ldict['code'].append(lc)
            ldict['mean'].append(np.mean(vals0)) 
            ldict['median'].append(np.median(vals0))
            ldict['std'].append(np.std(vals0))
            ldict['max'].append(np.max(vals0))
            ldict['min'].append(np.min(vals0))
            ldict['n_sample'].append(n_vals)
        else: 
            print('warning> no values found for lab code: %s (size considering n/a? %d)' % \
                (lc, len(lg['value_as_number'].values) ))

    if len(ldict) > 0: 
        df_lval = DataFrame(ldict, columns=header_lval)

        # [output]
        fname = '%s-lab_values-%s.csv' % (table_name, cohort_name)
        fpath = os.path.join(outputdir, fname)  
        print('make_lab_stats> saving df of dim %s to:\n%s' % (str(df_lval.shape), fpath))   
        df_lval.to_csv(fpath, sep='|', index=False, header=True)
    
        fname = '%s-lab_values-%s.pkl' % (table_name, cohort_name)
        fpath = os.path.join(outputdir, fname) 
        pickle.dump(vdict, open(fpath, "wb" ))
    else: 
        print('make_lab_stats> Warning: no lab values found!')

    return

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

def t_select(**kargs): 
    # ['719', 'V68', 'V67', '715', 'V70', '401', '496', '490', 'V76', '493'] # [hardcode]
    comm_top30 = ['719', '715', '401', '496', '490', '493']   # [log] results in df of dim: (3,408,800, 3)
    # searchByDiag(codes=comm_top30, exclude_ids=targets.target_set, base_only=True)

    # dataframe <- None => load from file
    candidates = filterCandidatesByCodes(codes=comm_top30, dataframe=None, relation='and')

    # [log]
    # info> Found 805 eligible patients diagnosed with given codes (common_top30)

    return	

def t_search_ptsd(**kargs): 
    from batchpheno import qrymed2
    # step 1: 

    basedir = '/phi/proj/poc7002/tpheno/PTSD'
    fname = 'PTSD_S1_patient_lists.csv'
    cohort_name = 'PTSD'
 
    header = ['person_id', 'first_diagnosis', 'ICD_code']
    fpath = os.path.join(basedir, fname)

    df = pd.read_csv(fpath, sep=',', header=0, index_col=False, error_bad_lines=True)

    print('dim> %s' % str(df.shape))
    print df.head(10)
    print df.tail(10)

    # step 2: 
    field_source = 'condition_source_value'
    field_date = 'condition_start_date'

    idx = list(set(df['person_id'].values))
    print('info> number of patients: %d' % len(idx))

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

    # idx_diag = list(set(df['person_id'].values))
    # print('verify> n_patients: %d, same cohort? (i.e. delta=0) %s' % (len(idx_diag), len(set(idx)-set(idx_diag))) )  # differ by 4 patients

    # # step 3. drug 
    # df_med = searchByID(table='drug_exposure', ids=idx, cohort='PTSD')

    # e.g. /phi/proj/poc7002/tpheno/data-exp/drug_exposure-query_ids-PTSD.csv
    fname = 'drug_exposure-query_ids-PTSD.csv'
    fpath = os.path.join(outputdir, fname)
    # df_med = pd.read_csv(fpath, sep='|', header=0, index_col=False, error_bad_lines=True)
    # print('dim> PTSD cohort: medication > %s' % str(df_med.shape))
    # print df_med.head(10)
    # print df_med.tail(10)

    # idx_med = list(set(df_med['person_id'].values))
    # # [log] verify> n_patients: 4387, same cohort? (i.e. delta=0) 1007
    # print('verify> n_patients: %d, same cohort? (i.e. delta=0) %s' % (len(idx_med), len(set(idx)-set(idx_med))) )  # differ by 4 patients

    # step 4. extract common diagnostic codes 
    # t_extract_diag()

    # common diagnositc codes within randomly selected sample patients
    # common_codes(df, n_sample=10)


    # step 5. extract lab values 
    # header: person_id, measurement_date, measurement_time, value_as_number, value_source_value
    # augmented: person_id, measurement_date, measurement_time, value_as_number, value_source_value, source_description
    table_name = 'measurement'
    # df = searchByID(table=table_name, ids=idx, cohort='PTSD') # scope will be added
    # print('dim> PTSD cohort > lab values  > %s' % str(df.shape))
    # print df.head(10)
    # print df.tail(10)

    # cohort 
    fname = '%s-query_ids-PTSD.csv' % table_name
    fpath = os.path.join(outputdir, fname)
    df_lab = pd.read_csv(fpath, sep='|', header=0, index_col=False, error_bad_lines=True)
    print('dim> PTSD cohort > lab values  > %s' % str(df_lab.shape))

    # control


    lab_tests = df_lab['value_source_value'].unique()
    descriptions = {}

    for ltest in lab_tests:
        if not descriptions.has_key(ltest): 
            descriptions[ltest] = qrymed2.getName2(ltest, err_default='unknown') 

    sdvals = []
    for ltest in df_lab['value_source_value'].values: 
        sdvals.append(descriptions[ltest])

    df_lab['source_description'] = sdvals
    df_lab.to_csv(fpath, sep='|', index=False, header=True)

    # step 6. control set (patients without 309.81: PTSD) who 
    #         1. has max lab tests done? 
    #         2. similar but without 309.81?


    return 

def t_search_ptsd(**kargs): 
    # from batchpheno import qrymed2
    import sys
    
    cohort_name = 'PTSD'
    ctrl_cohort_name = '%s-Negative' % cohort_name

    # step 1: 

    basedir = '/phi/proj/poc7002/tpheno/PTSD'
    fname = 'PTSD_S1_patient_lists.csv'
 
    header = ['person_id', 'first_diagnosis', 'ICD_code']
    fpath = os.path.join(basedir, fname)

    df = pd.read_csv(fpath, sep=',', header=0, index_col=False, error_bad_lines=True)

    print('dim> %s' % str(df.shape))
    print df.head(10)
    print df.tail(10)

    # step 2: 
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
    df = pd.read_csv(fpath, sep='|', header=0, index_col=False, error_bad_lines=True)
    
    # df = normalize(df, target_field=field_source)  # expensive!
    # df.to_csv(fpath, sep='|', index=False, header=True)
    # print('status> completed normalizing condition table ...')

    idx_diag = list(set(df['person_id'].values))
    print('verify> n_patients: %d, same cohort? (i.e. delta=0) %s' % (len(idx_diag), len(set(idx)-set(idx_diag))) )  # differ by 4 patients

    # # step 3. drug 
    # df_med = searchByID(table='drug_exposure', ids=idx, cohort='PTSD')

    # e.g. /phi/proj/poc7002/tpheno/data-exp/drug_exposure-query_ids-PTSD.csv
    fname = 'drug_exposure-query_ids-PTSD.csv'
    fpath = os.path.join(outputdir, fname)
    df_med = pd.read_csv(fpath, sep='|', header=0, index_col=False, error_bad_lines=True)
    print('dim> PTSD cohort: medication > %s' % str(df_med.shape))
    # print df_med.head(10)
    # print df_med.tail(10)

    idx_med = list(set(df_med['person_id'].values))
    # # [log] verify> n_patients: 4387, same cohort? (i.e. delta=0) 1007
    print('verify> n_patients: %d, same cohort? (i.e. delta=0) %s' % (len(idx_med), len(set(idx)-set(idx_med))) )  # differ by 4 patients

    # step 4. extract common diagnostic codes 
    # t_extract_diag()

    # common diagnositc codes within randomly selected sample patients
    # common_codes(df, n_sample=10)


    # step 5. extract lab values 
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
    
    # dtypes = {'value_source_value': int, } # [error] Integer column has NA values in column 4
    df_lab = pd.read_csv(fpath, sep='|', header=0, index_col=False, error_bad_lines=True)
    print('dim> PTSD cohort > lab values  > %s' % str(df_lab.shape))

    print('io> saving source description (qrymed returns) to %s' % fpath)
    df_lab['value_source_value'] = df_lab['value_source_value'].astype(int)
    df_lab.to_csv(fpath, sep='|', index=False, header=True)   

    # drop_cols = ['value_source_value', 'value_as_number']
    # df_lab = df_lab.dropna(subset=drop_cols)

    lab_tests = df_lab['value_source_value'].unique()
    descriptions = {}
    print('verify> number of unique lab tests: %d' % len(lab_tests)) # [log] number of unique lab tests: 2555

    # for ltest in lab_tests:
    #     if not descriptions.has_key(ltest): 
    #         descriptions[ltest] = qrymed2.getName2(ltest, err_default='unknown') 

    # sdvals = []
    # for ltest in df_lab['value_source_value'].values: 
    #     sdvals.append(descriptions[ltest])

    # df_lab['source_description'] = sdvals
    # df_lab.to_csv(fpath, sep='|', index=False, header=True)

    # step 5a statistics
    # [input] measurement-query_ids-PTSD.csv
    n_sample_test = 500  # store the values for wilcoxon test, etc. (between cohort and control)
    # lval_groups = df_lab.groupby(pivots)  # as_index=False # set as_index to False => pivots becomes a separate column instead of an index
    header_lval = ['code', 'mean', 'median', 'std', 'min', 'max', 'n_sample', ]

    print('step 5a: compute summary statistics of cohort data ...')  # measurement-lab_values-PTSD.csv
    make_lab_stats(df_lab, cohort=cohort_name, table=table_name, n_sample=n_sample_test, output_dir=outputdir)

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

    # header = ['person_id', 'measurement_date', 'measurement_time', 'value_as_number', 'value_source_value']
    
    # step 6.1: Given control cohort, generate its lab table
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

    print('io> saving to %s' % fpath)
    df_lab_ctrl['value_source_value'] = df_lab_ctrl['value_source_value'].astype(int)
    df_lab_ctrl.to_csv(fpath, sep='|', index=False, header=True)   

    # drop_cols = ['value_source_value', 'value_as_number']
    # df_lab_ctrl = df_lab_ctrl.dropna(subset=drop_cols)
    # dfmin = df_lab.min()
    # dfq = df_lab.quantile([.25, .5, .75]) # 3 rows add to each feature

    lab_tests_ctrl = df_lab_ctrl['value_source_value'].unique()
    lab_common = set(lab_tests).intersection(lab_tests_ctrl)
    n_lab_common = len(lab_common)
    print('verify> number of unique lab tests (control): %d vs n_commmon: %d' % (len(lab_tests_ctrl), n_lab_common))
    # descriptions = dict(zip(df_lab['value_source_value'], df_lab['source_description']))

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
    make_lab_stats(df_lab_ctrl, cohort=ctrl_cohort_name, table=table_name, n_sample=n_sample_test, output_dir=outputdir)
    
    sys.exit(0)
    # Step 7. Statistics, rank vars, wo
    
    # [precond]
    # 1. df_lab's source_description has been specified via qrymed2 
    # 2. lab values obtained
    
    # [cohort]
    header_ht = []
    ht = {}

    # scipy.stats.wilcoxon

    # [note] Wilcoxon signed-rank test
    # The Wilcoxon signed-rank test tests the null hypothesis that two related paired samples come from the same distribution.
    # It is a non-parametric version of the paired T-test.
    # Because the normal approximation is used for the calculations, the samples used should be large. A typical rule is to require that n > 20.
    n_sample_min = 20

    print('step 7: comparing cohort and control with wilcoxon test, etc. ...')
    fname = '%s-lab_values-%s.pkl' % (table_name, cohort_name)
    fpath = os.path.join(outputdir, fname)  
    print('io> loading cohort lab values from %s' % fpath)  
    vdict = pickle.load(open(fpath, 'rb'))  
    # for lc, vals in vdict.items(): 
    #     pass

    # [cohort]
    fname = '%s-lab_values-%s.pkl' % (table_name, ctrl_cohort_name)
    fpath = os.path.join(outputdir, fname)    
    print('io> loading control lab values from %s' % fpath)
    vdict_ctrl = pickle.load(open(fpath, 'rb'))  

    lab_common = set(vdict.keys()).intersection(vdict_ctrl.keys())
    print('verify> num of common codes (cohort vs ctrl): %d' % len(lab_common))
    
    n_small_sample = n_small = 0
    for lc, vals in vdict.items(): 
        if lc in vdict_ctrl:
            vals_ctrl = vdict_ctrl[lc] 
            if len(vals) < n_sample_min or len(vals_ctrl) < n_sample_min: 
                n_small += 1
                print('  + lab code %s (%s) has small sample (%d vs %d ctrl)' % \
                           (lc, descriptions.get(int(lc), 'unknown'), len(vals), len(vals_ctrl)))
            
           

    # header_lval = ['code', 'mean', 'median', 'std', 'min', 'max', 'n_sample', ] < ['is_active', ]
    # 7a. in cohort but not in control? 
    # 7b. hypo tests

    fname = '%s-lab_values-%s.csv' % (table_name, cohort_name)
    fpath = os.path.join(outputdir, fname)  
    print('io> loading (cohort) lab summary stats from %s' % fpath)
    df_stats = pd.read_csv(fpath, sep='|', header=0, index_col=False, error_bad_lines=True)

    fname = '%s-lab_values-%s.csv' % (table_name, ctrl_cohort_name)
    fpath = os.path.join(outputdir, fname)  
    print('io> loading (control) lab summary stats from %s' % fpath)
    df_stats_ctrl = pd.read_csv(fpath, sep='|', header=0, index_col=False, error_bad_lines=True) 

    print('7a> in cohort but not in control?')
    lt_cohort_only = list(set(df_stats['code']) - set(df_stats_ctrl['code']))

    # [pandas]
    # index: ts[ts[TSet.gold_indicator].isin(gold_index)]
    # update: sdf.loc[cond, ['coeff', 'score', 'description', 'rank']] = [coeff, score, desc, r]  # lowest value, higher rank
    f_active = 'is_active'
    df_stats[f_active] = 0  # False
    cond_cohort = df_stats['code'].isin(lt_cohort_only)
    df_stats.loc[cond_cohort, f_active] = 1 # if the lab test (code) present in both, then is_active



    return 

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

    return fpath

def t_ptsd_qrymed(**kargs): 
    from batchpheno import qrymed2
    # import random
    
    cohort_name = 'PTSD'
    ctrl_cohort_name = '%s-Negative' % cohort_name

    # step 1: 

    basedir = '/phi/proj/poc7002/tpheno/PTSD'
    fname = 'PTSD_S1_patient_lists.csv'
 
    header = ['person_id', 'first_diagnosis', 'ICD_code']
    fpath = os.path.join(basedir, fname)

    # cohort source #1
    df = pd.read_csv(fpath, sep=',', header=0, index_col=False, error_bad_lines=True)

    print('dim> %s' % str(df.shape))
    print df.head(10)
    print df.tail(10)

    # step 2: 
    field_source = 'condition_source_value'
    field_date = 'condition_start_date'

    idx_cohort = idx = list(set(df['person_id'].values))
    n_cohort = len(idx_cohort)
    print('info> number of target patients: %d' % n_cohort)

    # /phi/proj/poc7002/tpheno/data-exp
    outputdir = kargs.get('output_dir', sys_config.read('DataExpRoot')) # temporary data dir

    # Cohort conditions: writing querying result to /phi/proj/poc7002/tpheno/data-exp/condition_occurrence-query_ids-PTSD.csv
    fname = 'condition_occurrence-query_ids-%s.csv' % cohort_name
    fpath = os.path.join(outputdir, fname)
    df = pd.read_csv(fpath, sep='|', header=0, index_col=False, error_bad_lines=True)

    idx_diag = list(df['person_id'].unique())
    print('verify> n_patients: %d, same cohort? (i.e. delta=0) %s' % (len(idx_diag), len(set(idx)-set(idx_diag))) )  # differ by 4 patients

    # # step 3. drug 
    # df_med = searchByID(table='drug_exposure', ids=idx, cohort='PTSD')

    # e.g. /phi/proj/poc7002/tpheno/data-exp/drug_exposure-query_ids-PTSD.csv
    fname = 'drug_exposure-query_ids-%s.csv' % cohort_name
    fpath = os.path.join(outputdir, fname)
    df_med = pd.read_csv(fpath, sep='|', header=0, index_col=False, error_bad_lines=True)
    print('dim> PTSD cohort: medication > %s' % str(df_med.shape))
    # print df_med.head(10)
    # print df_med.tail(10)

    idx_med = list(set(df_med['person_id'].values))
    # # [log] verify> n_patients: 4387, same cohort? (i.e. delta=0) 1007
    print('verify> n_patients: %d, same cohort? (i.e. delta=0) %s' % (len(idx_med), len(set(idx)-set(idx_med))) )  # differ by 4 patients

    # step 4. extract common diagnostic codes 
    # t_extract_diag()

    # common diagnositc codes within randomly selected sample patients
    # common_codes(df, n_sample=10)

    div('Step 5: Extract lab values')
    ref_dir = os.path.join(outputdir, 'archive')
    # step 5. extract lab values 
    # header: person_id, measurement_date, measurement_time, value_as_number, value_source_value
    # augmented: person_id, measurement_date, measurement_time, value_as_number, value_source_value, source_description
    table_name = 'measurement'
    pivots = ['value_source_value', ]  # lab codes
    # df = searchByID(table=table_name, ids=idx, cohort='PTSD') # scope will be added
    # print('dim> PTSD cohort > lab values  > %s' % str(df.shape))
    # print df.head(10)
    # print df.tail(10)

    # 
    fpath_lookup = os.path.join(outputdir, '%s-lab_lookup-%s.csv' % (table_name, cohort_name))  
    print('io> load lab lookup file from:\n%s' % fpath_lookup)
    dfl = pd.read_csv(fpath_lookup, sep='|', header=0, index_col=False, error_bad_lines=True) 
    descriptions = dict(zip(dfl['value_source_value'], dfl['source_description']))   

    fname = '%s-query_ids-%s.csv' % (table_name, cohort_name)
    fpath = os.path.join(outputdir, fname)
    
    print('io> loading cohort (%s) data from %s' % (cohort_name, fpath))

    # dtypes = {'value_source_value': int, } # [error] Integer column has NA values in column 4
    df_lab = pd.read_csv(fpath, sep='|', header=0, index_col=False, error_bad_lines=True)

    # dropna + query
    df_lab = query_source_values(df_lab, cohort=cohort_name, descriptions=descriptions)

    print('io> saving source description (qrymed returns) to %s' % fpath)
    df_lab['value_source_value'] = df_lab['value_source_value'].astype(int)
    df_lab.to_csv(fpath, sep='|', index=False, header=True)

    # step 5a statistics
    # [input] measurement-query_ids-PTSD.csv
    n_sample_test = 500  # store the values for wilcoxon test, etc. (between cohort and control)
    # lval_groups = df_lab.groupby(pivots)  # as_index=False # set as_index to False => pivots becomes a separate column instead of an index
    header_lval = ['code', 'mean', 'median', 'std', 'min', 'max', 'n_sample', ]
    # make_lab_stats(df_lab, cohort=cohort_name, table=table_name, n_sample=n_sample_test, output_dir=outputdir)

    # step 6. control set (patients without 309.81: PTSD) who 
    #         1. has max lab tests done? 
    #         2. similar but without 309.81?
    # query measurement table and take those not in PTSD cohort (which means they do not have 309.81)

    gamma = 1.5 
    n_sample_control = int(n_cohort * gamma)

    # df = selectFrom(table=table_name, columns=['person_id', ])    
    # idx_lab = df['person_id'].unique()

    # idx_control = list(set(idx_lab)-set(idx_cohort))
    # n_control_total = len(idx_control)

    # idx_control_selected = random.sample(idx_control, min(n_sample_control, n_control_total))
    # n_control_selected = len(idx_control_selected)
    # print('info> number of control candidates: %d >? SELECTED: %d (requested: %d vs n_cohort: %d * gamma: %f)' % \
    #     (n_control_total, n_control_selected, n_sample_control, n_cohort, gamma))

    # header = ['person_id', 'measurement_date', 'measurement_time', 'value_as_number', 'value_source_value']
    
    # step 6.1: Given control cohort, generate its lab table
    # df_lab_ctrl = df = searchByID(table=table_name, ids=idx_control_selected, cohort=ctrl_cohort_name) # scope will be added
    # print('dim> PTSD Control cohort > lab values  > %s' % str(df_lab_ctrl.shape))
    # print df_lab_ctrl.head(10)
    # print df_lab_ctrl.tail(10)

    # statics 


    # lab_ctrl_min = df_lab_ctrl['value_as_number'].min() # series ~ feature
    # lab_ctrl_max = df_lab_ctrl.max()
    # dfq = df_lab.quantile([.25, .5, .75])

    # [load] cohort data 
    # ctrl_cohort_name = 'PTSD-Negative'
    # dtypes = {'value_source_value': int, }

    descriptions = dict(zip(df_lab['value_source_value'], df_lab['source_description']))
    lab_tests = df_lab['value_source_value'].unique()

    fname = '%s-query_ids-%s.csv' % (table_name, ctrl_cohort_name)
    fpath = os.path.join(outputdir, fname)
    assert os.path.exists(fpath), "Nonexistent input %s" % fpath
    print('io> loading control data (%s) set from %s' % (ctrl_cohort_name, fpath))
    df_lab_ctrl = pd.read_csv(fpath, sep='|', header=0, index_col=False, error_bad_lines=True) # dtype=dtypes

    # dropna + query
    df_lab_ctrl = query_source_values(df_lab_ctrl, cohort=ctrl_cohort_name, descriptions=descriptions)
    # dfmin = df_lab.min()
    # dfq = df_lab.quantile([.25, .5, .75]) # 3 rows add to each feature

    lab_tests_ctrl = df_lab_ctrl['value_source_value'].unique()
    lab_common = set(lab_tests).intersection(lab_tests_ctrl)
    n_lab_common = len(lab_common)
    print('verify> number of unique lab tests (control): %d vs n_commmon: %d' % (len(lab_tests_ctrl), n_lab_common))

    print('io> saving source description (qrymed returns) to %s' % fpath)
    df_lab_ctrl['value_source_value'] = df_lab_ctrl['value_source_value'].astype(int)
    df_lab_ctrl.to_csv(fpath, sep='|', index=False, header=True)    

    # df_lab_ctrl['source_description'] = sdvals
    # df_lab_ctrl.to_csv(fpath, sep='|', index=False, header=True)
    
    # Step 6a statistics 
    # header_lval = ['code', 'mean', 'median', 'std', 'min', 'max', ]
    # pivots = ['value_source_value', ]  # lab codes
    # n_sample_test = 1000
    # make_lab_stats(df_lab_ctrl, cohort=ctrl_cohort_name, table=table_name, n_sample=n_sample_test, output_dir=outputdir)

    return

def t_add_attributes(**kargs):

    cohort_name = 'PTSD'
    ctrl_cohort_name = '%s-Negative' % cohort_name 
    table_name = 'measurement'

    outputdir = kargs.get('output_dir', sys_config.read('DataExpRoot')) # temporary data dir

    fname = '%s-query_ids-%s.csv' % (table_name, cohort_name)
    fpath = os.path.join(outputdir, fname)
    print('io> loading cohort (%s) data from %s' % (cohort_name, fpath))
    # dtypes = {'value_source_value': int, } # [error] Integer column has NA values in column 4
    df_lab = pd.read_csv(fpath, sep='|', header=0, index_col=False, error_bad_lines=True)

    # load summary statistics
    fname = '%s-lab_values-%s.csv' % (table_name, cohort_name)
    fpath_stats = os.path.join(outputdir, fname)  
    print('io> loading (cohort) lab summary stats from %s' % fpath)
    df_stats = pd.read_csv(fpath_stats, sep='|', header=0, index_col=False, error_bad_lines=True) 

    # add description 
    descriptions = dict(zip(df_lab['value_source_value'], df_lab['source_description']))
    dvals = []
    for c in df_stats['code'].values: 
        dvals.append(descriptions.get(c, 'unknown'))
    df_stats['description'] = dvals

    print('io> saving test statistics (dim: %s) to:\n%s' % (str(df_stats.shape), fpath_stats))
    df_stats.to_csv(fpath_stats, sep='|', index=False, header=True)

    return

def test(**kargs): 
    # create_lab_lookup(**kargs)
    # t_search_ptsd(**kargs)
    # t_ptsd_qrymed(**kargs)
    t_add_attributes(**kargs)

    return 

if __name__ == "__main__": 
    test()
