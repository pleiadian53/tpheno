# encoding: utf-8

import pymssql
# import mssql_config
import pandas as pd
import os, gc, sys
from os import getenv 
import time, re, string, random

# local modules 
from batchpheno import icd9utils, utils, predicate, qrymed2
from batchpheno.utils import div
from config import seq_maker_config, sys_config

from pattern import medcode as pmed
import cohort 
import seqparams  # TDoc

from pandas import DataFrame, Series
import pandas as pd


#########################################################################
#
#  Similar to seqMaker, seqMaker2 creates documents consisting of 
#  temporally-ordered diagnostic and treatment sequences, wherein each 
#  seuqence corresponds to a time-series of diagnostic codes, MED codes
#  and other related codings that are considered informative for sequence 
#  learning for the purpose of disease progressive modeling. 
# 
#  Update 
#  ------
#
#  
#  Todo
#  ----
# 
#########################################################################

### System Variables ### 
# [default] condition table 
tb_condition = 'condition_occurrence'
tb_drug = 'drug_exposure'
tb_measure = 'measurement'

fp_condition = GConditionTable = '%s.csv' % tb_condition
fp_drug = GDrugTable = '%s.csv' % tb_drug
fp_measure = GLabTable = '%s.csv' % tb_measure

### Queries 
# diabetes-related conditions
# note that we want to pull out all records associated with the cohort definition 
# defined in the inner select statement; same patient may have other complications 
# other than the diagoses within the cohort. 
Q_Condition = """
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
Q_Condition_Diabetes = Q_Condition

# diabetes cohort
Q_Cohort = """
SELECT person_id, condition_start_date, condition_source_value
FROM ohdsi.west.condition_occurrence
WHERE condition_source_value like '250%' or
      condition_source_value like '249%' or 
      condition_source_value like '791%' or
      condition_source_value like '790%' or 
      condition_source_value like '648%' or
      condition_source_value like 'V65%' or 
      condition_source_value like 'V45%' or 
      condition_source_value like 'V53%';  
"""

# get all the patient IDs

Q_Drug = """
DECLARE @Expo table (person_id int, start_date date, source_value varchar(50))

INSERT into @Expo (person_id, start_date, source_value)
{cohort}

SELECT person_id, drug_exposure_start_date, drug_source_value
FROM ohdsi.west.drug_exposure
WHERE person_id in (select DISTINCT person_id from @Expo); 
"""

Q_Range = """
SELECT person_id, drug_exposure_start_date, drug_source_value, drug_concept_id
FROM {table}
WHERE {target_attribute} between {lower_bound} and {upper_bound}; 
"""

# example {target_attribute}: person_id
Q_Drug_RangeID = """
SELECT person_id, drug_exposure_start_date, drug_source_value, drug_concept_id
FROM ohdsi.west.drug_exposure
WHERE person_id between {lower_bound} and {upper_bound}; 
"""

Q_Condition_RangeID = """
SELECT person_id, condition_start_date, condition_source_value, condition_concept_id
FROM ohdsi.west.condition_occurrence
WHERE person_id between {lower_bound} and {upper_bound}; 
"""

Q_Drug_PersonID = """
SELECT person_id, drug_exposure_start_date, drug_source_value, drug_concept_id
FROM ohdsi.west.drug_exposure
WHERE person_id in ({id_set}); 
"""

Q_Condition_PersonID = """
SELECT person_id, condition_start_date, condition_source_value, condition_concept_id
FROM ohdsi.west.condition_occurrence
WHERE person_id in ({id_set}); 
"""

# [note]
# value_source_value: MED code for measurement_source_value
# measurement_source_value: description

Q_Measure = """
DECLARE @Expo table (person_id int, start_date date, source_value varchar(50))

INSERT into @Expo (person_id, start_date, source_value)
{cohort}

SELECT person_id, measurement_date, measurement_time, value_as_number, value_source_value
FROM ohdsi.west.measurement
WHERE person_id in (select DISTINCT person_id from @Expo); 
"""

### query ICD-10 codes
Q_ICD_template = """
    SELECT DISTINCT concept_code, concept_name
  FROM [ohdsi_west_pending_20161027].[dbo].[concept]
  WHERE (concept_code IN ('D69.0', 'H10.45', 'L50.0', 'Z91.010', 'Z91.012', 'Z91.048', 'Z84.89')
OR concept_code LIKE 'J30.%'
OR concept_code LIKE 'J67.%'
OR concept_code LIKE 'L20.%'
OR concept_code LIKE 'L22%'
OR concept_code LIKE 'Z88.%'
)
"""

Q_ICD = """
SELECT DISTINCT concept_code, concept_name
FROM ohdsi_west_pending_20161027.dbo.concept
WHERE concept_code IN ({code_set}); 
"""


def lookup(**kargs): 
    global Q_ICD

    server = seq_maker_config.Server
    user = seq_maker_config.User
    password = seq_maker_config.Password
    database = 'ohdsi'

    conn = pymssql.connect(server=server, user=user, password=password, database=database)
    cursor = conn.cursor()

    # [input]
    tb = 'ohdsi_west_pending_20161027.dbo.concept'
    codes = kargs.get('codes', [])
    query = Q_ICD.format(code_set=codes)

    print('> querying table: %s ...' % tb)
    div(message=query, symbol='%')
    df = pd.read_sql(query, conn)
    
    print df

    return

def rangeQuery(statement=None, **kargs):
    """

    Params
    ------
    key_attribute: the attribute whose scope is the range of search
    attributes: the columns of the table in the select statement

    Memo
    ----
    1. table='ohdsi.west.drug_exposure' & attribute='person_id'
       person_id | min: 4, max: 9999806 (10000000) ~ 10M

    2. tabe='ohdsi.west.condition_occurrence'
       person_id | min: 0, max: 9999999 (10000000) ~ 10M

    """
    def get_attributes(): 
        attributes = kargs.get('attributes', ['person_id', 'condition_start_date', 'condition_source_value', 'condition_concept_id'])
        return ', '.join(attributes)

    import math 

    q0 = statement 
    if q0 is None:     
        q0 = """
              SELECT {attribute}
              FROM {table}
             """  
    tFindRange = False  

    # interval = 1000000  
    attribute = kargs.get('key_attribute', 'person_id')
    rmin, rmax = 0, 10000000   # (0, 10M)
    if tFindRange:              
        tables = ['ohdsi.west.condition_occurrence', ] # 'ohdsi.west.drug_exposure'
        minx, maxx = [], []
        for tb in tables: 
            q = q0.format(attribute=attribute, table=tb) # query 'person_id' in table 'condition_occurrence'
            df = query(statement=q)
    
            rmin, rmax = min(df[attribute].values), max(df[attribute].values)
            minx.append(rmin)
            maxx.append(rmax)
            df = None; gc.collect()
            print('info> table=%s query attribute: %s | min: %s, max: %s' % (tb, attribute, rmin, rmax))
        rmin, rmax = min(minx), max(maxx)
    else: 
        assert rmax > rmin
    
    n_datasets = 10 # partition data into this many sets/files
    interval = int(math.ceil((rmax-rmin)/n_datasets))  
    # [params]
    tb_prefx = 'ohdsi.west'

    # tables = ['condition_occurrence', 'drug_exposure', ] 
    # idx = [0, 1000000, 2000000, 3000000,4000000, 5000000, 6000000, 7000000, 8000000, 9000000]

    # [params]
    # tmap: table -> query template
    # tables: ohdsi.west.condition_occurrence, ohdsi.west.drug_exposure
    tmap = {'condition_occurrence': Q_Condition_RangeID,  # table known
            'drug_exposure': Q_Drug_RangeID, 
            }

    for i in range(n_datasets): # (0, ):  # set index set here
        low, hi = rmin+i*interval, rmin+(i+1)*interval
        div(message='Set #%d > attribute: %s | value %s ~ %s' % (i, attribute, low, hi))

        for tb, qtemp in tmap.items():  # foreach table and query template
            tablename = '%s.%s' % (tb_prefx, tb)  # table is already written in query template
            filename = tb

            # Q_Range.format(table=table, target_attribute='person_id', lower_bound=low, upper_bound=hi)
            q = qtemp.format(lower_bound=low, upper_bound=hi)  # query filled with min and max allowable values 

            print('query>\n%s\n' % q)

            # e.g. {condition_occurrence-8.csv, condition_occurrence-9.csv ...}
            query(statement=q, name=filename, save_intermediate=True, index=i)  # this saves intermediate data for make_seq()

    return

def query(statement, **kargs):  
    """
    Execute a query statement in the DB under the following configuration: 

    server = seq_maker_config.Server
    user = seq_maker_config.User
    password = seq_maker_config.Password
    database = 'ohdsi'
    
    """
    def save_part(df, tb, index=0): 
        fp = os.path.join(temp_dir, '%s-%d.csv' % (tb, index))  # e.g. measurement-153.csv
        if not os.path.exists(fp) or (os.path.exists(fp) and overwrite_intermediate): 
            print('info> writing querying result to %s' % fp)
            df.to_csv(fp, sep=fsep, index=False, header=True, encoding='utf-8')
        return

    # [params] DB
    server = seq_maker_config.Server
    user = seq_maker_config.User
    password = seq_maker_config.Password
    database = 'ohdsi'

    conn = pymssql.connect(server=server, user=user, password=password, database=database)
    cursor = conn.cursor()

    # [params] 
    outputdir = kargs.get('outputdir', os.path.join(sys_config.read('DataExpRoot'), 'sequencing'))
    if not os.path.exists(outputdir): os.mkdir(outputdir)

    save_intermediate = kargs.get('save_intermediate', False)
    overwrite_intermediate = kargs.get('overwrite_intermediate', False) # overwrite even if existed
    fsep = '|'
    
    ### query
    df = pd.read_sql(statement, conn)  # query indexed by table name
    
    if save_intermediate: 
        index = kargs.get('index', None)
        basename = kargs.get('name', "sequencing")
        if index is not None: basename = "%s-%s" % (basename, index)
        fpath = os.path.join(outputdir, "%s.csv" % basename)
        df.to_csv(fpath, sep=fsep, index=False, header=True, encoding='utf-8')
        print('io> Saved result set (dim=%s) to %s' % (str(df.shape), fpath))     

    return df

def t_search_diabetes(**kargs): 
    return fetch(**kargs)

def fetch(**kargs):
    """
    Fetch desire data according to the cohort definition (specified via the 
    global queries or by cohort.py for more general cohort definitions). 

    Memo
    ----
    1. no prob. fetching chunked query result set
    """
    def save_part(df, tb, index=0): 
        fp = os.path.join(temp_dir, '%s-%d.csv' % (tb, index))  # e.g. measurement-153.csv
        if not os.path.exists(fp) or (os.path.exists(fp) and overwrite_intermediate): 
            print('info> writing querying result to %s' % fp)
            df.to_csv(fp, sep=fsep, index=False, header=True, encoding='utf-8')
        return

    # import seq_maker_config
    global Q_Condition, Q_Drug, Q_Measure  # queries for diabetes

    # [params] DB
    server = seq_maker_config.Server
    user = seq_maker_config.User
    password = seq_maker_config.Password
    database = 'ohdsi'

    conn = pymssql.connect(server=server, user=user, password=password, database=database)
    cursor = conn.cursor()

    # [params] 
    cohort_name = 'diabetes'
    save_intermediate = kargs.get('save_intermediate', False)
    overwrite_intermediate = kargs.get('overwrite_intermediate', False) # overwrite even if existed

    temp_dir = kargs.get('temp_dir', sys_config.read('DataExpRoot')) # temporary data dir
    fsep = '|'

    # [params] control 
    tFetchMeasurements = False

    ### Define queries
 
    # [I/O] cohort 
    # Q_Cohort uses diagnostic codes to define the target cohort
    queries = { tb_condition: Q_Condition, 
                tb_drug: Q_Drug.format(cohort=Q_Cohort), 
                tb_measure: Q_Measure.format(cohort=Q_Cohort), 
               }

    # smaller tables
    for tb in (tb_condition, tb_drug, ):
        print('> querying table: %s ...' % tb)
        q = queries[tb]
        df = pd.read_sql(q, conn)  # query indexed by table name
        print('> dim (table=%s): %d by %d' % (tb, df.shape[0], df.shape[1]))

        if save_intermediate: 
            fname = '%s-query_ids-%s.csv' % (tb, cohort_name) # to be compatible with seqmaker.cohort
            fp = os.path.join(temp_dir, fname)
            if not os.path.exists(fp) or (os.path.exists(fp) and overwrite_intermediate): 
                print('info> writing querying result to %s' % fp)
                df.to_csv(fp, sep=fsep, index=False, header=True, encoding='utf-8')

    # larger tables
    if tFetchMeasurements: 
        measure_chunksize = 1000000
        for tb in (tb_measure, ): 
            print('> querying large table: %s ...' % tb)
                
            dfs = []
            cnt = 0 
            q = queries[tb]
            for df in pd.read_sql(q, conn, chunksize=measure_chunksize): 
                print('> fetching chunk #%d (dim=%s) ...' % ((cnt+1), str(df.shape)))  # [m1]
                # df = pd.concat(dfs, ignore_index=True) 
                if save_intermediate: 
                    save_part(df, tb, index=cnt)
                cnt += 1 
            print('status> Found %d parts associated with %s-table.' % (cnt, tb_measure))
            
    conn.close()

    return

def load_noncoded(**kargs): 
    header = ['source_value', 'internal_code']
    sep = '|'
    input_dir = sys_config.read('DataExpRoot')

    cohort_name = kargs.get('cohort', 'diabetes')
    tdoc_prefix = seq_compo = kargs.get('composition', 'condition_drug')
    # doc_basename = '%s-%s' % (seq_compo, cohort_name) if cohort_name is not None else seq_compo

    # example: condition_noncoded-PTSD.cs
    f_noncoded = noncoded_name = kargs.get('name', None)
    if noncoded_name is None: 
        f_noncoded = '%s_noncoded-%s.csv' % (tdoc_prefix, cohort_name)
    else: 
        if noncoded_name.startswith('cond'): 
            f_noncoded = 'condition_noncoded-%s.csv' % cohort_name
        elif noncoded_name.startswith(('pres', 'drug')):
            f_noncoded = 'drug_noncoded-%s.csv' % cohort_name
        else: 
            f_noncoded = 'lab_noncoded-%s.csv' % cohort_name

    ipath = os.path.join(input_dir, f_noncoded)

    assert os.path.exists(ipath), "lookup table does not exist in:\n%s\n" % ipath
    df = pd.read_csv(ipath, sep=sep, header=0, index_col=False, error_bad_lines=True)
    print('info> loaded %d noncoded entries' % df.shape[0])

    if not df.empty: 
        return dict(zip(df['internal_code'].values, df['source_value'].values))
    return {}

def load_noncoded_by(name='condition', cohort='diabetes'):
    kargs = {} 
    kargs['name'] = name
    kargs['cohort'] = cohort
    return load_noncoded(**kargs)

# [todo]
def codifySeq(dfv, noncoded, include_timestamps=False, drug_source_value_sep=':', remove_source_dups=False, pad_empty_code=True): 
    token_empty = 'empty'
    token_unknown = 'unknown'
    token_error = 'error'
    t_col, d_col = 'start_date', 'data'
    time_sep = '|'  # time separator (from its corresponding codes)

    coded_seq = []
    sep = drug_source_value_sep    # ':'
    n_skipped = 0
    nrow = dfv.shape[0]

    assert isinstance(noncoded, dict)
    if remove_source_dups: # remove duplicate entries? 
            
        seen = set()
        # for i, e in enumerate(dfv[col].values): 
        n_unmapped = 0
        for i, row in dfv.iterrows(): 
            e = str(row[d_col]) 
            e = e.strip()
            el = e.split()

            # first, map code/symbol to 'standardized' representation
            if not e: # empty 
                if not pad_empty_code: # consider empty code? 
                    n_skipped += 1 
                    print('data> no content, skipping (n_skipped=%d) ...' % n_skipped)
                    continue
                else: 
                    e = noncoded.get(e, token_empty)
            else: 
                if len(el) > 1: # complex
                    effval = noncoded.get(e, None)
                    if effval is not None: 
                        e = effval
                    else: 
                        n_unmapped += 1
                        print('WARNING> unmapped complex token: %s => %s' % (e, token_unknown))  # [log] 2038-01-01 00:00:00 => unknown
                        e = token_unknown
                else: # singleton 
                    # if there's a mapped value, use it o.w. leave it as it is
                    try: 
                        e = noncoded[e] # find it's mapping (e.g. may have removed drug prefix)
                    except: 
                        pass 

            if not e in seen: # then add to the sequence
                # assert e[-1] != token_end_history, "symbol ended with dot: %s" % e  # [exception] symbol ended with dot: 920.
                
                if include_timestamps: # attach timestamps?  
                    t = str(row[t_col].to_datetime().strftime("%Y-%m-%d"))  # don't need specific time in the day
                    coded_seq.append(t+time_sep+e) # time_sep is defined in outer scope
                else: 
                    coded_seq.append(e) 
            seen.add(e)
    else: 
        #for e in dfv[col].values: 
        for i, row in dfv.iterrows(): 
            e = str(row[d_col])
            e = e.strip() 
            el = e.split()

            # first, map code/symbol to 'standardized' representation
            if not e: 
                if not pad_empty_code: 
                    n_skipped += 1 
                    print('data> no content, skipping (n_skipped=%d) ...' % n_skipped)
                    continue
                e = noncoded.get(e, token_empty)
            else: 
                if len(el) > 1: 
                    effval = noncoded.get(e, None)
                    if effval is not None: 
                        e = effval
                    else: 
                        print('warning> found unknown token: %s => %s' % (e, token_unknown))  # [log] 2038-01-01 00:00:00 => unknown
                        e = token_unknown
                else: # singleton 
                    # if there's a mapped value, use it o.w. leave it as it is
                    try: 
                        e = noncoded[e]  # find it's mapping (e.g. may have removed drug prefix)
                    except: 
                        pass 
                
            if include_timestamps: 
                t = str(row[t_col].to_datetime().strftime("%Y-%m-%d"))
                coded_seq.append(t+time_sep+e) # time_sep is defined in outer scope
            else: 
                coded_seq.append(e) # as is 

    return coded_seq

def transform(seq, **kargs):
    """
    Transform the input string (a sequence of medical codes [and possibly other tokens])
    to another representation (e.g. simplified by taking base part of diagnostic codes) easier 
    to derive distributed vector representation. 

    Input
    -----
    seq: a string of medical code sequence

    """
    op_simplify = kargs.get('simplify_code', False) or kargs.get('base_only', False)
    op_split = kargs.get('split', False)
    token_sep = kargs.get('token_sep', ',')

    if op_simplify: 
        seq = simplify(seq, sep=token_sep)

    if op_split: 
        if isinstance(seq, str): 
            seq = seq.split(token_sep)

    return seq

def simplify(seq, sep=' '): 
    """
    Simplify diagnostic codes. 

    No-op for non-diagnostic codes. 

    Input
    -----
    seq: a string of space-separated diagnostic codes (icd9, icd10, etc.)


    """
    return icd9utils.getRootSequence(seq, sep=sep)  # sep is only relevant when seq is a string

def separate_time(seq, delimit='|', verify_=False): 
    """

    Memo
    ----
    1. when the 'if e' conditional statement weren't added, an error showed up (upon execusting seqAnalyzer)
       that indicated the possibility of e being empty but e has ever been detected as being empty ever since
       the condtiional statement is added in. 
    """
    times, codes = [], []
    N = len(seq)
    n_empty = 0  
    for i, e in enumerate(seq): 
        if e:    # memo [1]
            try: 
                t, c = e.split(delimit)  # assumeing that the format is strictly followed (timestamp|code, )
            except: 
                msg = "time> Error: could not separate time for e='%s' ..." % e
                raise ValueError, msg 
            times.append(t)
            codes.append(c)
        else: 
            n_empty += 1 
            before = seq[i-1] if i >= 1 else 'n/a'
            after = seq[i+1] if i < N-1 else 'n/a'
            print('warning> found %d emtpy string element within seq: (%s) -> ((%s)) -> (%s)' % (n_empty, before, seq[i], after))
            

    return (codes, times)

def parse(coding_seq, time_sep='|', token_sep=',', token_visit=';', token_record='$', check_time=False):
    """
    Given a coding sequence (which consists of all visits), separate the codes from 
    their corresponding timestamps. 

    coding_seq: sequence string consists of the coding sequence of a single patient 
    e.g. 2008-04-19|E849.8,2008-04-19|E888.8;2008-04-19|920;2011-04-23|E000.8,2011-04-23|E849.0$ 

    Memo  
    ----    
    token_sep: separator for each token (e.g. codes) within a single visit (values of the same date)
    token_visit:  separator for each visit, which can consists of tokens representing diagnoses and medications
    token_record: end token for a patient's entire medical record 

    Use
    ---
    parse() + assemble()
    1. store .csv version of the coding sequences
       example format: 
          header = ['person_id', 'sequence', 'timestamp',]
    2. check data integrity of the coding sequences 
    """ 
    import seqparams 
    coding_seq = coding_seq.strip()
    assert coding_seq[-1] == token_record, "ill-formatted coding sequence (token_record='%s') vs last few:\n%s\n" % \
        (token_record, coding_seq[-20:])
    seq = coding_seq[:-1]
    
    codeSeq, timeSeq = [], []
    visits = [visit.strip() for visit in seq.split(token_visit) if len(visit) > 0] # remove empty strings 
    n_visits = len(visits)

    # syntax 
    token_unknown = seqparams.TDoc.token_unknown # 'unknown'

    has_time = True
    if n_visits > 0: 

        # find out if timestamps exist
        if check_time: 
            for i, visit in enumerate(visits): 
                tokens = visit.split(token_sep)
                if len(tokens) > 0: 
                    ct = tokens[0].split(time_sep)
                    if ct == 2: 
                        has_time = True 
                    else: 
                        has_time = False
                if has_time or i > 10: break

        if has_time: 
            for i, visit in enumerate(visits): 
                vseq, tseq = [], []
                tokens = visit.split(token_sep)  # each visit consists a set of codes 

                for token in tokens: 
                    token = token.strip()
                    parse_err = False
                    try: 
                        # [format] <timestamp>|<code>
                        t, c = token.split(time_sep)  # format: time followed by code
                        if len(t)==0:  # no time stamp
                            parse_err = True
                        else: 
                            if len(c) == 0: 
                                c = token_unknown # 'unknown'   # [todo] this should not happen though
                    except: 
                        msg = 'parse> Error: failed to split token %s (time_sep: %s)\nverify> visit: %s' % ((token, time_sep, visit))
                        # raise ValueError, msg   # e.g. verify> visit: 2011-03-10|,,2011-03-10|MED:62184
                        print msg
                        parse_err = True
                    if not parse_err: 
                        vseq.append(c.strip())
                        tseq.append(t.strip())
                
                if len(vseq) > 0: 
                    # assert len(vseq) == len(tseq)
                    codeSeq.append(vseq)
                    timeSeq.append(tseq)
            assert len(codeSeq) == len(timeSeq)
        else: 
            for i, visit in enumerate(visits):
                tokens = []
                try:  
                    tokens = visit.split(token_sep)  # each visit consists a set of codes 
                except: 
                    msg = 'parse> Error: failed to parse visit (sep=%s): %s' % ((token_sep, visit))
                    continue # skip this visit

                if len(tokens) > 0: 
                    codeSeq.append([t.strip() for t in tokens])

    else: 
        print('warning: input does not have information content (n_visits=0)')

    return (codeSeq, timeSeq)

def assemble(codeset, timeset=None, token_sep=',', token_visit=';', token_record='$', for_csv=True):
    """

    Params
    ------
    for_csv: if True, all tokens will be separated only by 'token_sep' 
             => token_visit will not be used 
    """
    if timeset is not None: assert len(timeset) == len(codeset), "verify>\ncodeset: %s\ntimeset: %s\n" % (codeset[:10], codeset[:10])
    if for_csv: return assemble_csv_format(codeset, timeset=timeset, token_sep=token_sep)

    cstr, tstr = '', ''
    if not codeset: 
        print('warning: input does not have information content (n_visits=0)')
        return ('', '')

    for visit in codeset[:-1]: 
        cstr += token_sep.join(visit) + token_visit

    # add last visit
    cstr += token_sep.join(codeset[-1]) + token_record # no need to add this token in structured format

    if len(timeset) > 0: 
        for visit in timeset[:-1]: 
            tstr += token_sep.join(visit) + token_visit
        # add last visit
        tstr += token_sep.join(timeset[-1]) + token_record        

    return (cstr, tstr)

def assemble_csv_format(codeset, timeset=None, token_sep=','):
    cstr, tstr = '', ''
    if not codeset: 
        print('warning: input does not have information content (n_visits=0)')
        return ('', '')

    for visit in codeset[:-1]: 
        cstr += token_sep.join(visit) + token_sep   # no need to distinguish visits (since dates are available)

    # add last visit
    cstr += token_sep.join(codeset[-1]) # no need to add record end token in structured format

    if len(timeset) > 0: 
        for visit in timeset[:-1]: 
            tstr += token_sep.join(visit) + token_sep
        # add last visit
        tstr += token_sep.join(timeset[-1])  # no need to add record end token in structured format
    return (cstr, tstr)

def separate_time2(seq, time_sep='|', token_sep=',', token_visit=';', token_record='$', check_time=False, for_csv=True): 
    cseq, tseq = parse(seq, time_sep=time_sep, token_sep=token_sep, 
                            token_visit=token_visit, token_record=token_record, check_time=check_time)
    
    # reassemble list-of-token format to string format using appropriate separators
    cstr, tstr = assemble(cseq, tseq, token_sep=token_sep, token_visit=token_visit, token_record=token_record, for_csv=for_csv)
    return (cstr, tstr)

def makeSeqFromDataFrames(**kargs): 
    raise NotImplementedError, "Use seqmaker.seqReader for now."

def makeSeqGeneric(**kargs):
    """
    Convert dataframes obtained from querying DBs (fetch, query, rangeQuery, etc.), 
    to medical coding sequences. 

    Memo
    ----
    (*) header: ['person_id', 'sequence']



    (*) table='ohdsi.west.drug_exposure' & attribute='person_id'
            person_id | min: 4, max: 9999806 (10000000) ~ 10M

        tabe='ohdsi.west.condition_occurrence'
            person_id | min: 0, max: 9999999

    """ 
    # from pattern import medcode as pmed

    assert kargs.has_key('cohort') # 'diabetes'
    
    # [params] input dataframes (use fetch() or cohort.py to generate)
    kargs['condition_table'] = kargs.get('condition_table', fp_condition)
    kargs['drug_table'] = kargs.get('drug_table', fp_drug)
    # kargs['lab_table'] = kargs.get('lab_table', fp_lab)

    return make_seq(**kargs)

def makeSourceToConceptMap(**kargs): 
    """
    
    Memo
    ----
    1. This table is available on OHDSI database
       dbo.source_to_concept_map
    """
    return makeConceptMap(**kargs)
def makeConceptMap(**kargs):
    """
    Map concept IDs to source values
    """ 
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
    def test_input_table(df): 
        div(message='Cohort: %s' % kargs.get('cohort', 'generic'))
        print('   + df dim: %s' % str(df.shape))
        print('   + top 5 rows:\n%s\n' % df.head(5))
        assert not df.empty 
        return

    ret = {}  # return value

    cohort_name = kargs.get('cohort', 'generic') # or 'diabetes'
    basedir = inputdir = kargs.get('inputdir', sys_config.read('DataExpRoot')) # document source directory
    assert os.path.exists(basedir), "Invalid input directory: %s" % basedir
    
    # [params] input dataframes (use fetch() or cohort.py to generate)
    condition_tb = kargs.get('condition_table', fp_condition)
    drug_tb = kargs.get('drug_table', fp_drug)

    # necessary columns
    header_condition = ['person_id', 'condition_start_date', 'condition_source_value', 'condition_concept_id', ]
    header_drug = ['person_id', 'drug_exposure_start_date', 'drug_source_value', 'drug_concept_id', ]
    header_lab = ['person_id', 'measurement_date', 'measurement_time', 'value_as_number', 'value_source_value']
    
    # condition header: person_id|condition_start_date|condition_source_value
    df_condition = read_condition_table(); test_input_table(df_condition)

    # drug_exposure header: 
    df_drug = read_drug_table(); test_input_table(df_drug)

    header = ['source_value', 'concept_id']
    df1[header] = df_condition[['condition_source_value', 'condition_concept_id']]
    df2 = DataFrame()
    df2[header] = df_drug[['drug_source_value', 'drug_concept_id']]
    df[header] = pd.concat([df1, df2], ignore_index=True)


    return

def makeSeq(**kargs): 
    return make_seq(**kargs)
def make_seq(**kargs): 
    """


    Params
    ------
    tdoc_overwrite = True # [IO]
    include_tdoc_csv = kargs.get('gen_csv', False)
       - generates sorted person_ids 
       - generate sequences attached with their corresponding person_ids (delegate this to seqReader)
    save_ids

    token_empty = 'empty'
    token_unknown = 'unknown'
    token_error = 'error'
    pad_empty_code = True 

    (*) settings
        basedir

    (*) control 
        cohort
 
        remove_source_prefix # this applies to both drug (MED:12345 => 12345) and condition (I9, I0)
        include_diag: True by default, include diagnostic codes
        include_med: True by default
        include_timestamps: 
        include_tdoc_csv: save structured file for the coding sequences (.csv)


    Input
    -----
    condition_table 
    drug_table


    Note 
    ----
    1. Sorting prior to Groupby 

    Memo
    ----
    1. key fields: 
       condition_start_date, condition_source_value
       drug_exposure_start_date, drug_source_value
       measurement_date, measurement_time

    2. datetime format 
       e.g. 2057-11-09  12:38:00    (beginning of a new ice age)

       in codify_seq
          str(row[t_col]) includes padding time info e.g. 2010-07-29 00:00:00

    3. Noncoded data element 
       represent 'lasix 20mg by mouth once a day' as an internal code 
          but doing reverse lookup is a bit troublesome 
          => 
          just "compress" the sentence and retain all the (important) tokens in the document. 

    Todo
    ----
    1. use f_tdoc_csv (.csv) instead of f_tdoc (.dat) with tentative header: 
       ['person_id', 'sequence', 'timestamp', ]

    2. use seqReader's facility transform (string -> list of tokens) and separate_time() 
       to separate codes and their corresponding timestamps

    Inner Functions 
    ---------------
    1. codify_seq(dfv): 
            dfv comes from apply groupby to (standardized) dates
                ... for d, dfv in dfx.groupby(t_col, sort=False)
            where each dfv in the loop corresponds to the subset of dataframe of unique date


    Todo
    ----
    1. 2D representation: timestamps vs medical codings

    """
    def codify_seq(dfv): # data from the same visit (the same date)
        coded_seq = []
        sep = drug_source_value_sep    # ':'
        n_skipped = 0
        nrow = dfv.shape[0]
        if remove_source_dups: # remove duplicate entries? 
            seen = set()
            # for i, e in enumerate(dfv[col].values): 
            n_unmapped = 0 
            for i, row in dfv.iterrows(): # dfv combines (condition, mediacation, etc.)
                e = str(row[d_col]) 
                e = e.strip()
                el = e.split()

                tSkip = False
                # first, map code/symbol to 'standardized' representation
                if len(el) == 0: #  # IF e = '' or e = '  ' THEN e.split() => []
                    if not pad_empty_code: # consider empty code? True by default
                        n_skipped += 1 
                        print('data> no content, skipping (n_skipped=%d) ...' % n_skipped)
                        tSkip = True
                    else: 
                        e = noncoded.get(e, token_empty)  # lookup the internal representation 
                else: 
                    if len(el) > 1: # complex
                        effval = noncoded.get(e, None)
                        if effval is not None: 
                            e = effval
                        else: 
                            n_unmapped += 1 

                            # [note] this seems to be only dates
                            # print('WARNING> unmapped complex token: %s => %s' % (e, token_unknown))  # [log] 2038-01-01 00:00:00 => unknown
                            e = token_unknown
                            # e.g. 2038-01-01 00:00:00 => unknown
                            # [note] usually a date
                            
                    else: # singleton 
                        # if there's a mapped value, use it o.w. leave it as it is
                        try: 
                            e = noncoded[e] # find it's mapping (e.g. may have removed drug prefix)
                        except: 
                            pass 
                
                    # # this also remove condition_source_prefix 
                    # if e.find(sep) > 0: 
                    #     e0, e1 = e.split(sep)
                    #     if e0 in drug_prefix: # try to preserve prefix
                    #         if remove_drug_source_prefix:  # remove source value prefix? e.g. NDC:66336060940
                    #             e = e1
                    #     else: 
                    #         if remove_source_prefix: 
                    #             e = e1
                    
                    # code block above is pre-processed during table read 
                if tSkip: continue 
                if not e in seen: 
                    # assert e[-1] != token_end_history, "symbol ended with dot: %s" % e  # [exception] symbol ended with dot: 920.

                    # e.g. '***   '
                    assert len(e) > 0, "warning: containing null string (%s) in a visit:\n%s\n#" % (e, dfv[d_col].values)

                    if include_timestamps: # attach timestamps?  
                        t = str(row[t_col].to_datetime().strftime("%Y-%m-%d"))  # don't need specific time in the day
                        if len(t) > 0: 
                            coded_seq.append(t+time_sep+e) # time_sep is defined in outer scope
                            # ensure that e does not contain time_sep
                        else: 
                            print('warning> no timestamps given! <action> ignore this code: %s' % e) 
                    else: 
                        coded_seq.append(e) 
                seen.add(e)
        else: 
            #for e in dfv[col].values: 
            for i, row in dfv.iterrows(): 
                e = str(row[d_col])
                e = e.strip() 
                el = e.split()

                # first, map code/symbol to 'standardized' representation
                if len(el) == 0:  # IF e = '' or e = '  ' THEN e.split() => []
                    if not pad_empty_code: 
                        n_skipped += 1 
                        print('data> no content, skipping (n_skipped=%d) ...' % n_skipped)
                        continue
                    e = noncoded.get(e, token_empty)
                else: 
                    if len(el) > 1: 
                        effval = noncoded.get(e, None)
                        if effval is not None: 
                            e = effval
                        else: 
                            print('warning> found unknown token: %s => %s' % (e, token_unknown))  # [log] 2038-01-01 00:00:00 => unknown
                            e = token_unknown
                    else: # singleton 
                        # if there's a mapped value, use it o.w. leave it as it is
                        try: 
                            e = noncoded[e]  # find it's mapping (e.g. may have removed drug prefix)
                        except: 
                            pass 
                
                    # if e.find(sep) > 0: 
                    #     e0, e1 = e.split(sep)
                    #     if e0 in drug_prefix: # try to preserve prefix
                    #         if remove_drug_source_prefix: 
                    #             e = e1
                    #     else: 
                    #         if remove_source_prefix: 
                    #             e = e1
                
                if include_timestamps: 
                    t = str(row[t_col].to_datetime().strftime("%Y-%m-%d"))
                    if len(t) > 0: 
                        coded_seq.append(t+time_sep+e) # time_sep is defined in outer scope
                    else: 
                        print('warning> no timestamps given! <action> ignore this code: %s' % e) 
                else: 
                    coded_seq.append(e) # as is 

        return coded_seq

    def dot_normalize(code): 
        if len(code)>0 and code[-1] == '.': 
            return code[:-1] 
        return code

    def condense(x, index=0, sep='_'):  # 1 cervical soft collar => 1CervicalSoftCollar
        x = x.translate(None, string.punctuation)  # remove punctuations '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
        xp = sep.join([e.lower().capitalize() for e in x.split()])
        if len(xp) == 0: 
            # e.g. '***   ' => x: '  ' => xp: ''
            print('warning> empty after condense: %s => %s' % (x, xp))
            return token_empty
        return xp

    def report_params(): 
        print('params> mode: %s' % 'test' if test_mode else 'regular')
        print('params> include timestamps? %s' % 'yes' if include_timestamps else 'no')
        print('params> remove drug source prefix? (e.g. MED:1234)? %s' % 'yes' if remove_source_prefix else 'no')
        print('params> remove duplicate entries? %s' % 'yes' if remove_source_dups else 'no')

        print('params> for each patient, each visit is considered a sentence? %s' % 'yes' if group_by_visit else 'no')

        print('params> padding empty data element into the sequence? %s' % 'yes'  if pad_empty_code else 'no')
        return  

    def track(**adict): # record desired return values into 'ret'  
        for k, v in adict.items(): 
            ret[k] = v  # this persists after the funciton call
        return ret

    # from pattern import medcode as pmed
    # from pandas import DataFrame, Series
    # import seqparams as sp

    # group by patient ID > sort records according to timestamps
    # df2 = caller.join(other.set_index('key'), on='key')

    # [params] global vars on diabetes cohort 
    # fp_condition = '%s.csv' % tb_condition  # condition_occurrence.csv
    # fp_drug = '%s.csv' % tb_drug  # drug_exposure.csv
    # fp_measure = '%s.csv' % tb_measure

    # [params] PTSD cohort 
    # condition_occurrence-query_ids-PTSD.csv, drug_exposure-query_ids-PTSD.csv
    
    ret = {}  # return value

    cohort_name = kargs.get('cohort', 'generic') # or 'diabetes'

    basedir = inputdir = kargs.get('inputdir', sys_config.read('DataExpRoot')) # document source directory
    assert os.path.exists(basedir), "Invalid input directory: %s" % basedir
    
    # [params] input dataframes (use fetch() or cohort.py to generate)
    condition_tb = kargs.get('condition_table', fp_condition)
    drug_tb = kargs.get('drug_table', fp_drug)

    # necessary columns
    header_condition = ['person_id', 'condition_start_date', 'condition_source_value', ]
    header_drug = ['person_id', 'drug_exposure_start_date', 'drug_source_value', ]
    header_lab = ['person_id', 'measurement_date', 'measurement_time', 'value_as_number', 'value_source_value']
    
    # condition header: person_id|condition_start_date|condition_source_value
    dtypes = {'condition_source_value': str}
    fpath1 = os.path.join(basedir, condition_tb)
    assert os.path.exists(fpath1), "Invalid diagnosis input: %s" % fpath1 
    df_condition = pd.read_csv(fpath1, 
                          sep='|', index_col=False, header=0, error_bad_lines=True, 
                                 parse_dates=['condition_start_date'], dtype=dtypes) 

    assert len(set(header_condition)-set(df_condition.columns)) == 0
    # drug_exposure header: 
    dtypes = {'drug_source_value': str}
    fpath2 = os.path.join(basedir, drug_tb)
    assert os.path.exists(fpath2), "Invalid medication input: %s" % fpath2
    df_drug = pd.read_csv(fpath2, 
                           sep='|', index_col=False, header=0, error_bad_lines=True, 
                                parse_dates=['drug_exposure_start_date'], dtype=dtypes)
    assert len(set(header_drug)-set(df_drug.columns)) == 0

    div(message='Cohort: %s (diabetes by default if None)' % cohort_name)
    print('data> df_condition dim: %s | df_drug dim: %s' % (str(df_condition.shape), str(df_drug.shape)))
    print('data> df_condition(top 5):\n%s\n' % df_condition.head(5))
    print('data> df_drug(top5):\n%s\n' % df_drug.head(5))

    assert not df_condition.empty and not df_drug.empty
    ### join, group by, sort, 'merge times'
    # fkey = 'person_id'
    # condition_drug = df_condition.join(df_drug.set_index(fkey), on=fkey, how='outer')
    
    # sort out which pid has records in all dataframes? 
    # pids = set(df_condition['person_id'].values)
    
    # [params]
    test_mode, skip_tdoc, side_effects = kargs.get('test_', False), False, False
    # source_values = set() 
    # literal_values = set()  # non-coded source values e.g. little green pill
    # f_srcvals = 'source_values.dat' # [output]

    # [params] test conditions 
    include_timestamps = kargs.get('include_timestamps', False)
    include_diag = kargs.get('include_diag', True)  # include diagnostic codes 
    include_med = kargs.get('include_med', True)    # include med_codes
    time_sep = '|'  # time separator (from its corresponding codes)

    # [params] doc format and experimental settings 
    group_by_visit = True
    merge_patients = False  # if True, then every visit is separated by newline and end-of-doc-per-patient separator is also just a newline

    # [params] protocal and grammar
    # [todo] use TDoc class
    token_sep = ','
    token_end_visit = ';' # each visit can consists of multiple diagnoses and medications
    token_end_history = '$'  # end token of entire medical history (of a patient)
    patient_sep = '\n'  # this is to facilitate io

    if merge_patients: 
        token_end_visit = token_end_history = '\n'

    # [params] time series document settings
    seq_compo = seq_composition = kargs.get('seq_compo', 'condition_drug')
    tdoc_prefix = seq_compo

    # [todo]
    # output_files = seqparams.TDoc.setName(cohort=cohort_name, seq_compo=seq_compo, 
    #     include_timestamps=include_timestamps)

    # [params] output file 
    # [note] default cohort if None is diabetes; other cohorts e.g. PDSD cohort: condition_drug_seq-PTSD.dat
    f_base = 'timed_seq' if include_timestamps else 'seq'  # this corresponds to doctype in TDoc
    meta = kargs.get('meta', None)
    if meta is not None: f_base = '%s-%s' % (f_base, meta)  # e.g.  timed_seq-med-PTSD => medication codes only
    if cohort_name is not None: f_base = '%s-%s' % (f_base, cohort_name)

    f_tdoc = '%s_%s.dat' % (tdoc_prefix, f_base)  # [output]
    
    # [todo] use this format instead of .dat 
    # [params] output file 2 (.csv which includes person_ids and other meta data)
    f_tdoc_csv = '%s_%s.csv' % (tdoc_prefix, f_base)  # example: condition_drug_timed_seq-CKD.csv
    
    # only one kind of id file
    f_tdoc_id = '%s_%s.id' % (tdoc_prefix, f_base) # [output]
    print('io> .dat file name: %s, .csv: %s, .csv (ID): %s' % (f_tdoc, f_tdoc_csv, f_tdoc_id))
   
    # [params] control 
    tdoc_overwrite = True # [IO]
    include_tdoc_csv = kargs.get('save_csv', True) and include_timestamps # for now, only save .csv when include_timestamp is true
    save_ids = kargs.get('save_id', True)
    save_intermediate = kargs.get('save_intermediate', False) # if True, then save noncoded symbol mapping

    condition_source_value_sep = ':'  # I9:309.81
    drug_source_value_sep = ':'  # e.g. MED:63408, NDC:00121072716

    remove_drug_source_prefix = False
    drug_prefix = pmed.DrugPrefix  # ['']
    remove_source_prefix = False # this applies to both drug and condition (I9, I0)
    remove_source_dups = True # 

    # note: noncoded data element is probably from drug table
    noncoded_header = ['source_value', 'internal_code']
    prefix_prescription = 'prescription'   # [todo] medcode.prefix_prescription
    prefix_condition = 'condition'  # [todo] medcode.prefix_condition
    prefix_date = 'date'

    token_empty = 'empty'
    token_unknown = 'unknown'
    token_error = 'error'
    pad_empty_code = True 

    # [output]
    f_noncoded = '%s-noncoded.csv' % tdoc_prefix
    f_noncoded_drug = 'drug-noncoded.csv'  # [output]
    f_noncoded_condition = 'condition-noncoded.csv' # [output]
    if cohort_name is not None:
        f_noncoded = '%s_noncoded-%s.csv' % (tdoc_prefix, cohort_name)
        f_noncoded_drug = 'drug_noncoded-%s.csv' % cohort_name # [output]
        f_noncoded_condition = 'condition_noncoded-%s.csv' % cohort_name # [output]         

    dfnc_sep = '|'  # dataframe-noncoded seperator

    # [output]
    for fdoc in (f_tdoc, f_tdoc_csv, f_tdoc_id): 
        tdoc_path = os.path.join(basedir, fdoc)
        if os.path.exists(tdoc_path) and tdoc_overwrite:
            print('io> deleting existing time series at %s' % tdoc_path) 
            os.unlink(tdoc_path)
    # srcvals_path = os.path.join(basedir, f_srcvals) # factored to seqAnalyzer

    report_params()

    ### Preprocessing (dealing with non-coded data element e.g. lasix 20mg by mouth once a day)
    div(message='Identify non-coded values and assign appropriate internal tokens (e.g. concatenation) ...')
    
    noncoded = {}  # this generalizes to anything that requires a mapping for a canonical form
    noncoded_condensed = {}

    # drug/medication codes
    assert not remove_drug_source_prefix, "Perhaps prefix is too important to drop (MED, NDC, MULTUM)"
    for e in df_drug['drug_source_value'].values:  # or use 'drug_concept_id'
        e = str(e)
        e = e.strip()
        el = e.split()

        if len(el) > 1:
            # ebase = el[0]
            
            if pmed.isPrefixedDrugSourceVal(el[0]):
                print('verify> multi-tokens where first one is a medication code: %s' % el[0])

            # [policy]
            # 1. replace complex tokens by simplier ones, masking them 
            # 2. just compress multiple tokens into one single token
            # 3. keyword extraction 

            # this is probably not coded 
            # n = len(noncoded_condensed) # numbering
            # if pmed.isSourceVal(el[0]):
            #     print('warning> complex prescription code: %s => %s' % (e, el[0]))  # 1 cervical soft collar => 1
            #     noncoded[e] = el[0] # reduce it to only the first sensible source value 
            # elif pmed.isWord(el[0]):  # only encode element like 'lasix 20mg by mouth once a day'
            #     noncoded[e] = '%s%s' % (prefix_prescription, n)  # after this, we'll have n+1 elements 

            # e.g. 2042-01-01 00:00:00
            # if predicate.isDate(e): 
            #     ep = token_unknown
            #     noncoded[e] = ep # don't keep track of meaningless timestamp values
            #     print('info> date in (drug) source value: %s => %s' % (e, ep)) # e.g. 3 hour GTT
            # else:
                
            ep = condense(e)  # ... policy #2 
            # ep = '%s%s' % (prefix_prescription, n)  # internal represetation

            # noncoded_condensed[ce] = ep 
            print('verify> complex medication code: %s => %s' % (e, ep))  
            # e.g. calcium + D 1000-400 => Calcium_D_1000400
            #      Coumadin INR goal 2-2.5 for 4 weeks => Coumadin_Inr_Goal_225_For_4_Weeks
            # [todo] strip off numbers? 

            noncoded[e] = ep 
            
            # [log] e.g. _please review all unapproved meds written by data

        elif len(el) == 0:  # if e is an empty string 
            noncoded[e] = token_empty
        else: # normal singleton token
            # if pmed.isMedCode(e):  # MED:, NDC:, MULTUM: 
            if pmed.isPrefixedDrugSourceVal(e):     # MED:, NDC:, MULTUM:
                # if remove_drug_source_prefix:
                #     e0, e1 = e.split(sep)

                #     ep = e1 
                #     noncoded[e] = ep 
                # else: 
                #     pass # leave it as it is
                pass # noop, just take the expression 'e' as it is
            # if not pmed.isSourceVal(e): 
            else: 
                # code not qualified by valid prefixes 
                
                if pmed.isSourceVal(e): 
                    # pass # no-op
                    print("verify> medication code not prefixed: %s" % e)
                else: 
                    print("verify> noncoded medication unigram: %s" % e)  # e.g. Test; for now, don't map to internal coding
                    # e.g. m/vit, glucometer
                    # ep = e
                    # noncoded[e] = e
                # code it? 

    r = len(noncoded)/(df_drug['drug_source_value'].shape[0]+0.0)
    print('info> # of non-coded data elements in drug table: %d (noncoded ratio: %f)' % (len(noncoded), r))

    # dfnc_drug = pd.DataFrame(noncoded.items(), columns=noncoded_header)
    # print('io> saving noncoded drug table to %s' % os.path.join(basedir, f_noncoded_drug))
    # dfnc_drug.to_csv(os.path.join(basedir, f_noncoded_drug), sep=dfnc_sep, index=False, header=True, encoding='utf-8') 
    
    # diagnostic codes
    noncoded_condition = {}
    for e in df_condition['condition_source_value'].values:  # or use 'condition_concept_id'
        e = str(e)
        e = e.strip()
        # if e.startswith('I9'): 
        el = e.split()
        if len(el) > 1: 
            # this is probably not coded 
            n = len(noncoded_condition)

            # clean incomplete expression e.g. 920. 
            el[0] = dot_normalize(el[0])
            el[1] = dot_normalize(el[1])  # e.g. 367.4. 
            
            code = el[0] + '.' + el[1]

            # use first element as a constraint: el[0]
            if pmed.isICD(code):   
                ep = code 
                # noncoded_condition[e] = ep

            # if pmed.isWord(el[0]): # e.g. I9^ 6A
            else: 
                
                ep = condense(e)  # or '%s%s' % (prefix_condition, n) 
                # noncoded_condition[e] = ep # '%s%s' % (prefix_condition, n)  # after this, we'll have n+1 elements
                print('verify> non-diagnostic codes in (condition) source value: %s => %s' % (e, ep))
            # elif predicate.isDate(e): 
            #     ep = token_unknown
            #     # noncoded_condition[e] = ep  # don't keep track of meaningless timestamp values
            #     print('info> date in (condition) source value: %s => %s' % (e, ep))
            # else: 
            #     ep = token_unknown
            #     print('warning> unknown diagnostic codes: %s => %s' % (e, ep))

            noncoded_condition[e] = ep 
        elif len(el) == 0: 
            noncoded_condition[e] = token_empty
        else:  # additional processing
            if pmed.isICD(e):  # extra dot S53. 920. is considered valid 
                if e[-1] == '.': 
                    print('verify> incomplete coding: %s' % e)
                    noncoded_condition[e] = e[:-1]
                    # e.g. V42.  E888.

                # a lot of codes that end with '.'
                # noncoded_condition[e] = dot_normalize(e)
            else: 
                is_resolved = False
                ep = pmed.convert(e, nocatch=True) # try to convert it to valid code
                if pmed.isICD(ep): 
                    noncoded_condition[e] = ep 
                    is_resolved = True
                else: 
                    if pmed.isMedCode(e): # 6819 as Gambian Trypanosomiasis
                        # convert to ICD-9? 
                        # pass  # leave it as it is
                        ep0 = qrymed2.medToICD(e)
                        if ep0 and pmed.isICD(ep0):
                            ep = ep0 
                            noncoded_condition[e] = ep
                            is_resolved = True
                
                if is_resolved: 
                    print('verify> resolved diagnostic codes: %s => %s' % (e, ep)) 
                    # e.g. 78650 => 786.50
                else: 
                    ep = condense(e)
                    print("verify> unresolved potentially non-diagnistic codes: %s => %s" % (e, ep))  # e.g. ADM, I10, V.0.8, ***, E, .22.1
                    noncoded_condition[e] = ep # token_unknown
                    # e.g. 999999 => 999999

                # 1. leave it as it is
                # 2. map it
    
    r = len(noncoded_condition)/(df_condition.shape[0]+0.0)
    print('info> # of non-coded or (improperly code) data elements in condition table: %d (noncoded ratio: %f)' % \
        (len(noncoded_condition), r))

    # if save_intermediate: 
    # dfnc_condition = pd.DataFrame(noncoded_condition.items(), columns=noncoded_header)
    # # print('io> saving noncoded condition table to %s' % os.path.join(basedir, f_noncoded_condition))
    # # dfnc_condition.to_csv(os.path.join(basedir, f_noncoded_condition), sep=dfnc_sep, index=False, header=True, encoding='utf-8')  
    
    # dfnc = pd.concat([dfnc_drug, dfnc_condition], ignore_index=True)
    # print('io> saving noncoded condition+drub table to %s' % os.path.join(basedir, f_noncoded))
    # dfnc.to_csv(os.path.join(basedir, f_noncoded), sep=dfnc_sep, index=False, header=True, encoding='utf-8')
    # print('info> total # of non-coded entries: %d' % dfnc.shape[0])

    # consolidate dictionary 
    # [note] codify*() only looks up noncoded
    for k, v in noncoded_condition.items(): 
        if not noncoded.has_key(k): 
            noncoded[k] = v 
        else: 
            print('warning> mixed entry? (%s, %s)' % (k, v))  # data elements in condition table coincide with those in drug table
    noncoded_condensed = noncoded_condition = None; gc.collect()
    
    div(message='stage> Completed preprocessing.', symbol='#')
    print('  + found %d non-coded or inappropriately coded values' % len(noncoded))
    
    # now start to build (raw) coding sequence (.dat)
    tdoc_path = os.path.join(basedir, f_tdoc) 
    fp = open(tdoc_path, 'a')
    div(message='Consolidate sources (condition_source_value) and create med coding sequences', symbol='%')
    # [note]
    #  condition_tkey = 'condition_start_date' ... drug_tkey = 'drug_exposure_start_date'
    #  standardized columns: t_col, d_col = 'start_date', 'data'

    # all person_ids 
    id_col = 'person_id'

    person_ids = set(df_condition[id_col].values) # turning to set does not guarantee ordering
    person_ids.update(df_drug[id_col].values)
    ret['person_id'] = person_ids = sorted(person_ids) # ascending order

    n_persons = len(person_ids)
    min_id, max_id = min(person_ids), max(person_ids)  # if char, then implicitly apply ord()
    track(min_id=min_id, max_id=max_id) # store the value in output 'ret'

    n_doc_total = 0
    print('  + Found %d unique IDs (%s, min=%s, max=%s)' % (n_persons, id_col, min_id, max_id))

    #for pid, df in df_condition.groupby('person_id'):  # foreach candidate found in the condition table
    test_ids = set(random.sample(person_ids, min(len(person_ids), 20)))

    # [params] saving .csv via f_tdoc_csv
    header = ['person_id', 'sequence', 'timestamp', ]  # 'timestamp': use seqReader to process time stamps
    seqdict = {h:[] for h in header}
    personIdEff = []  # persons with data

    print('test> df_condition: %s' % list(df_condition.columns.values))
    print('test> df_drug: %s' % list(df_drug.columns.values))

    # [params] control
    t_col, d_col = 'start_date', 'data'
    condition_tkey = 'condition_start_date'
    drug_tkey = 'drug_exposure_start_date'
    # include_diag = kargs.get('include_diag', True); include_med = kargs.get('include_med', True)
    assert include_diag or include_med, "Diagnostic and medication codes cannot be both absent."

    update_cycle = 1000
    cnt = n_null = n_update_cycle = 0
    tdoc = unit_tdoc = ''  # unit_tdoc: holds time series for single patient only 
    for ith, pid in enumerate(person_ids):  # foreach person ids (sorted in ascending order)
        # df = df_condition.loc[df_condition[id_col]==pid]
        # df2 = df_drug.loc[df_drug[id_col]==pid]

        ### stardardize date & time and unify data formats from different dataframes 
        df = df2 = None
        has_diag = has_med = False
        if include_diag: 
            df = df_condition.loc[df_condition[id_col]==pid]
            if not df.empty: 
                df[t_col] = df[condition_tkey]  # e.g. condition_start_date
                df[d_col] = df['condition_source_value']
                df = df[[t_col, d_col]]
                has_diag = True
        if not has_diag: # dummy data
            # if ith == 0: assert meta is not None, "Need meta field to distinguish from regular doc"
            df = DataFrame(columns=[t_col, d_col])

        if include_med: 
            df2 = df_drug.loc[df_drug[id_col]==pid]
            if not df2.empty: 
                df2[t_col] = df2[drug_tkey] # e.g. drug_exposure_start_date
                df2[d_col] = df2['drug_source_value']
                df2 = df2[[t_col, d_col]]
                has_med = True
        if not has_med: # dummy data
            df2 = DataFrame(columns=[t_col, d_col])

        ### other dataframes (e.g. lab)

        if ith == 0:  # [test]
            assert all(df.columns == df2.columns)
            # print('test> df: %s' % list(df.columns.values))
            # print('test> df2: %s' % list(df2.columns.values))
        
        ### combine and sort 
        # dfx = pd.concat([df, df2, df3, ...], ignore_index=True)  # ok. 
        dfx = pd.concat([df, df2], ignore_index=True)  # ok. 
        assert not (include_diag and include_med) or not dfx.empty
        if dfx.empty:
            n_null += 1 
            if n_null % 100 == 0: print('  + person %s has no data (n_null=%d so far)' % (pid, n_null)) 
            continue

        dfx.sort_values(t_col, ascending=True, inplace=True, kind='mergesort') # [1] assume that groupby preserves order

        # [test]
        if test_mode and (pid in test_ids): 
            print('test> consolidated dataframe (condition+drug):\n%s\n' % dfx.head(50))
            # dfx.groupby('start_date', sort=False).sort_index(ascending=True)

            sv0 = '%s:' % pid
            for i, (d, dfv) in enumerate(dfx.groupby(t_col, sort=False)): # has been sorted 
                sv0 += token_sep.join(codify_seq(dfv))  # x,y,z
                sv0 += token_end_visit
                print('\n    + i=%d > patient #%d, PID: %s > date: %s, dfv => \n%s\n' % (i, cnt, pid, d, dfv.head(20)))
            if len(sv) > 0: sv = sv[:-1]
            if len(sv) > 0: s = sv + token_end_history

            print('    + %s: %s' % (pid, sv0))

            if skip_tdoc: 
                if cnt < 100: 
                    continue 
                else: 
                    print('test complete.')
                    break 

        # if side_effects: # collect meta data and summary statistics (across all patients)
        #     source_values.update(dfx.values) # later on, tease out literal values using seqAnalyzer
       
        ### form document 
        assert group_by_visit, "seqReader now assumes that codes are separated by visits/dates."

        s = ''
        if group_by_visit: # [2] 1. group by date prior to 2.; each visit is separated by 'token_end_visit' 
            n_visits = 0
            sv = ''

            # [todo] group-by seems to change data type 
            for d, dfv in dfx.groupby(t_col, sort=False): # foreach date and its associated records (foreach visit)

                # canonicalize coded time series 
                vseq = codify_seq(dfv)  # this mixes both code and time stamps 
                assert len(vseq) > 0, "empty content on date %s" % str(d)

                sv += token_sep.join(vseq)  # x,y,z  token_sep:','
                # sv += token_sep.join(str(e).split(drug_source_value_sep)[-1] for e in dfv['data'].values)
                sv += token_end_visit  # e.g. ';'
                
                n_visits += 1 

            # [debug]
            # if cnt < 100: print('data> PID: %s, n_visits: %d > sv (group by date):\n%s\n' % (pid, n_visits, sv))
            if cnt < 30:  
                print('data> PID: %s > dfx:\n%s\n' % (pid, dfx.head(100)))
                print('data> visit oriented series:\n%s\n' % sv)

            # sv[-1] = token_end_history  => string is immutable in python 
            if len(sv) > 0: 
                assert sv[-1] == token_end_visit
                sv = sv[:-1]  # clip off the last token, which is a token_sep (e.g. ',')
            # if cnt % 5 == 0: assert sv_temp[-1] == token_end_visit  # ok.

            if len(sv) > 0:  # token_end_history: $
                # a patient document is the coding sequence of multiple visists + end codon
                s = sv + token_end_history  # end token for the entire doc (medical history) of a patient
                # s = ''.join(sv_temp)

        else: # 2. combine data from all dates into one single doc
            s = token_sep.join(codify_seq(dfx))
            
            if cnt < 100: print('data> PID: %s > sv (group by date):\n%s\n' % (pid, sv))
            if len(s) > 0: 
                s += token_end_history
        
        # [test]
        # if test_mode and cnt % 100 == 0: 
        #     dfx_head, dfx_tail = dfx.head(10), dfx.tail(10)
        #     print('test> (patient #%d, PID: %d)' % (cnt, pid))
        #     print('df_head: %s' % ' '.join(str(e) for e in dfx_head.values))
        #     print('    seq: %s' % s[:100])
        #     print('df_tail: %s' % ' '.join(str(e) for e in dfx_tail.values))
        #     print('    seq: %s' % s[-100:])

        # final time series doc
        if len(s) > 0: 
            assert s[-1] == token_end_history
            unit_tdoc = s + patient_sep # patient separater is a new line (collection of all visits)
            tdoc += unit_tdoc
            cnt += 1 

            # test
            if test_mode and cnt <= 100:
                print('test> n_persons: %d > current person ID: %s, record dim: %s' % (cnt, pid, str(dfx.shape))) 
                print('data>\n%s\n' % dfx.head(20))  # df is not sorted by date
                print('data> unit tdoc > (n_persons=%d)>\n%s\n' % (cnt, unit_tdoc))

            personIdEff.append(pid) # only keep non-empty sequences

            if include_tdoc_csv: 
                assert include_timestamps, "only save .csv when include_timestamps is set to True as well ..."
                # parse the string and separate codes from their timestamps
                cstr, tstr = separate_time2(s, time_sep=time_sep, token_sep=token_sep, 
                                    token_visit=token_end_visit, token_record=token_end_history, check_time=False)
                
                if len(cstr) > 0: 
                    assert len(tstr) > 0
                    seqdict['person_id'].append(pid)
                    seqdict['sequence'].append(cstr)
                    seqdict['timestamp'].append(tstr)
                else: 
                    print('verify> resulted empty string from input sequence: %s' % str(s))
                # else: 
                #     seqdict['sequence'].append(s)

            # [test]
            if cnt % 250 == 0:  # cnt incre when coding sequence (s) is non-empty 
                print('info> (patient #%d, PID: %d) > df: %s, df2: %s, combo records: %s' % (cnt, pid, str(df.shape), str(df2.shape), str(dfx.shape)))
         
        # incremental updates (e.g. every 1000 patients)
        if cnt % update_cycle == 0: 
            n_update_cycle += 1
            # print('info> writing tdoc to %s (processed %d docs)' % (tdoc_path, cnt)) 
            fp.write(tdoc) # only write to buffer

            time.sleep(1)
            fp.flush()  # write doc to file immediately; important for 'streaming' operations

            ndoc_cycle = len(tdoc.split('\n'))
            print('info> writing the %d set (n_doc:%d, total:%d) to:\n%s\n' % (n_update_cycle, ndoc_cycle, cnt, tdoc_path))
            
            tdoc = ''  # reset for the next write cycle

    ### end foreach person_ids
    
    # need to write the last time 
    if len(tdoc) > 0: 
        fp.write(tdoc) # only write to buffer
        time.sleep(1)
        fp.flush()
        nd_last = len(tdoc.split('\n'))
        print('info> writing the last set (n_doc:%d, total:%d) to:\n%s\n' % (nd_last, cnt, tdoc_path))

    # close file 
    fp.close()
    div(message='io> Saved doc (n_docs=%d =?= n_persons=%d) to %s\n' % (len(personIdEff), n_persons, tdoc_path), symbol='#')
    ret['person_id_eff'] = personIdEff  # track(person_id_eff=personIdEff)

    if include_tdoc_csv and len(seqdict) > 0: 
        header = ['person_id', 'sequence', 'timestamp', ]  # 'label'
        df = DataFrame(seqdict, columns=header)
        assert df.shape[0] == len(personIdEff), "One person should have one single document (%d docs vs %d IDs)" % (df.shape[0], len(personIdEff))
        tdoc_path2 = os.path.join(basedir, f_tdoc_csv) 
        df.to_csv(tdoc_path2, sep='|', index=False, header=True)
        div(message='io> Saved .csv (id+seq) to %s\n' % tdoc_path2, symbol='#')
        
        # save person_ids 
    if save_ids and n_persons > 0: 
        header = ['person_id', ]
        adict = {'person_id': personIdEff}
        df = DataFrame(adict, columns=header)
        fpath = os.path.join(basedir, f_tdoc_id)
        df.to_csv(fpath, sep='|', index=False, header=True)
        div(message='io> Saved person IDs to %s\n' % fpath, symbol='#')
        
    print('\n--- Completed sequencing cohort %s ---\n' % cohort_name)

    return ret

def makeSeqFromDF(**kargs):
    """
    Make MCSs from input dataframes (intermmediate data obtained, for instance,from cohort.searchByID())
    """
    return make_seq2(**kargs) 
def make_seq2(**kargs): 
    def codify_seq(dfv): # data from the same visit (the same date)
        coded_seq = []
        sep = drug_source_value_sep    # ':'
        n_skipped = 0
        nrow = dfv.shape[0]
        if remove_source_dups: # remove duplicate entries? 
            seen = set()
            # for i, e in enumerate(dfv[col].values): 
            n_unmapped = 0 
            for i, row in dfv.iterrows(): # dfv combines (condition, mediacation, etc.)
                e = str(row[d_col]) 
                e = e.strip()
                el = e.split()

                tSkip = False
                # first, map code/symbol to 'standardized' representation
                if len(el) == 0: #  # IF e = '' or e = '  ' THEN e.split() => []
                    if not pad_empty_code: # consider empty code? True by default
                        n_skipped += 1 
                        print('data> no content, skipping (n_skipped=%d) ...' % n_skipped)
                        tSkip = True
                    else: 
                        e = noncoded.get(e, token_empty)  # lookup the internal representation 
                else: 
                    if len(el) > 1: # complex
                        effval = noncoded.get(e, None)
                        if effval is not None: 
                            e = effval
                        else: 
                            n_unmapped += 1 

                            # [note] this seems to be only dates
                            # print('WARNING> unmapped complex token: %s => %s' % (e, token_unknown))  # [log] 2038-01-01 00:00:00 => unknown
                            e = token_unknown
                            # e.g. 2038-01-01 00:00:00 => unknown
                            # [note] usually a date
                            
                    else: # singleton 
                        # if there's a mapped value, use it o.w. leave it as it is
                        try: 
                            e = noncoded[e] # find it's mapping (e.g. may have removed drug prefix)
                        except: 
                            pass 
                    
                    # code block above is pre-processed during table read 
                if tSkip: continue 
                if not e in seen: 
                    # assert e[-1] != token_end_history, "symbol ended with dot: %s" % e  # [exception] symbol ended with dot: 920.

                    # e.g. '***   '
                    assert len(e) > 0, "warning: containing null string (%s) in a visit:\n%s\n#" % (e, dfv[d_col].values)

                    if include_timestamps: # attach timestamps?  
                        t = str(row[t_col].to_datetime().strftime("%Y-%m-%d"))  # don't need specific time in the day
                        if len(t) > 0: 
                            coded_seq.append(t+time_sep+e) # time_sep is defined in outer scope
                            # ensure that e does not contain time_sep
                        else: 
                            print('warning> no timestamps given! <action> ignore this code: %s' % e) 
                    else: 
                        coded_seq.append(e) 
                seen.add(e)
        else: 
            #for e in dfv[col].values: 
            for i, row in dfv.iterrows(): 
                e = str(row[d_col])
                e = e.strip() 
                el = e.split()

                # first, map code/symbol to 'standardized' representation
                if len(el) == 0:  # IF e = '' or e = '  ' THEN e.split() => []
                    if not pad_empty_code: 
                        n_skipped += 1 
                        print('data> no content, skipping (n_skipped=%d) ...' % n_skipped)
                        continue
                    e = noncoded.get(e, token_empty)
                else: 
                    if len(el) > 1: 
                        effval = noncoded.get(e, None)
                        if effval is not None: 
                            e = effval
                        else: 
                            print('warning> found unknown token: %s => %s' % (e, token_unknown))  # [log] 2038-01-01 00:00:00 => unknown
                            e = token_unknown
                    else: # singleton 
                        # if there's a mapped value, use it o.w. leave it as it is
                        try: 
                            e = noncoded[e]  # find it's mapping (e.g. may have removed drug prefix)
                        except: 
                            pass 
                
                if include_timestamps: 
                    t = str(row[t_col].to_datetime().strftime("%Y-%m-%d"))
                    if len(t) > 0: 
                        coded_seq.append(t+time_sep+e) # time_sep is defined in outer scope
                    else: 
                        print('warning> no timestamps given! <action> ignore this code: %s' % e) 
                else: 
                    coded_seq.append(e) # as is 

        return coded_seq

    def dot_normalize(code): 
        if len(code)>0 and code[-1] == '.': 
            return code[:-1] 
        return code

    def condense(x, index=0, sep='_'):  # 1 cervical soft collar => 1CervicalSoftCollar
        x = x.translate(None, string.punctuation)  # remove punctuations '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
        xp = sep.join([e.lower().capitalize() for e in x.split()])
        if len(xp) == 0: 
            # e.g. '***   ' => x: '  ' => xp: ''
            print('warning> empty after condense: %s => %s' % (x, xp))
            return token_empty
        return xp

    def report_params(): 
        print('params> mode: %s' % 'test' if test_mode else 'regular')
        print('params> include timestamps? %s' % 'yes' if include_timestamps else 'no')
        print('params> remove drug source prefix? (e.g. MED:1234)? %s' % 'yes' if remove_source_prefix else 'no')
        print('params> remove duplicate entries? %s' % 'yes' if remove_source_dups else 'no')

        print('params> for each patient, each visit is considered a sentence? %s' % 'yes' if group_by_visit else 'no')

        print('params> padding empty data element into the sequence? %s' % 'yes'  if pad_empty_code else 'no')
        return  

    def track(**adict): # record desired return values into 'ret'  
        for k, v in adict.items(): 
            ret[k] = v  # this persists after the funciton call
        return ret
    def test_input():  # <- cohort_name, df_condition, df_drug
        # condition header: person_id|condition_start_date|condition_source_value

        assert not df_condition.empty, "condition dataframe is empty"
        assert not df_drug.empty, "medication dataframe is empty"

        dtypes = {'condition_source_value': str}
        assert len(set(header_condition)-set(df_condition.columns)) == 0
        # drug_exposure header: 
        dtypes = {'drug_source_value': str}
        assert len(set(header_drug)-set(df_drug.columns)) == 0

        div(message='Cohort: %s (diabetes by default if None)' % cohort_name)
        print('  + df_condition dim: %s | df_drug dim: %s' % (str(df_condition.shape), str(df_drug.shape)))
        print('  + df_condition(top 5):\n%s\n' % df_condition.head(5))
        print('  + df_drug(top5):\n%s\n' % df_drug.head(5))

        return 

    # necessary columns
    header_condition = ['person_id', 'condition_start_date', 'condition_source_value', ]
    header_drug = ['person_id', 'drug_exposure_start_date', 'drug_source_value', ]
    header_lab = ['person_id', 'measurement_date', 'measurement_time', 'value_as_number', 'value_source_value']
    ret = {}  # return value

    cohort_name = kargs.get('cohort', 'generic') # or 'diabetes'
    basedir = inputdir = kargs.get('inputdir', sys_config.read('DataExpRoot')) # document source directory
    assert os.path.exists(basedir), "Invalid input directory: %s" % basedir
    
    # [params] input dataframes (use fetch() or cohort.py to generate)
    df_condition = kargs.get('condition_table', DataFrame())
    df_drug = kargs.get('drug_table', DataFrame())
    df_lab = kargs.get('lab_table', DataFrame())
    
    test_input()

    # [params]
    test_mode, skip_tdoc, side_effects = kargs.get('test_', False), False, False

    # [params] test conditions 
    include_timestamps = kargs.get('include_timestamps', False)
    include_diag = kargs.get('include_diag', True)  # include diagnostic codes 
    include_med = kargs.get('include_med', True)    # include med_codes
    time_sep = '|'  # time separator (from its corresponding codes)

    # [params] doc format and experimental settings 
    group_by_visit = True
    merge_patients = False  # if True, then every visit is separated by newline and end-of-doc-per-patient separator is also just a newline

    # [params] protocal and grammar
    # [todo] use TDoc class
    token_sep = ','
    token_end_visit = ';' # each visit can consists of multiple diagnoses and medications
    token_end_history = '$'  # end token of entire medical history (of a patient)
    patient_sep = '\n'  # this is to facilitate io

    if merge_patients: token_end_visit = token_end_history = '\n'

    # [params] time series document settings
    seq_compo = seq_composition = kargs.get('seq_compo', 'condition_drug')
    tdoc_prefix = seq_compo

    # [todo]
    # output_files = seqparams.TDoc.setName(cohort=cohort_name, seq_compo=seq_compo, 
    #     include_timestamps=include_timestamps)

    # [params] output file 
    # [note] default cohort if None is diabetes; other cohorts e.g. PDSD cohort: condition_drug_seq-PTSD.dat
    f_base = 'timed_seq' if include_timestamps else 'seq'  # this corresponds to doctype in TDoc
    meta = kargs.get('meta', '')
    if meta: f_base = '%s-%s' % (f_base, meta)  # e.g.  timed_seq-med-PTSD => medication codes only
    if cohort_name is not None: f_base = '%s-%s' % (f_base, cohort_name)

    f_tdoc = '%s_%s.dat' % (tdoc_prefix, f_base)  # [output]
    
    # [todo] use this format instead of .dat 
    # [params] output file 2 (.csv which includes person_ids and other meta data)
    f_tdoc_csv = '%s_%s.csv' % (tdoc_prefix, f_base)  # example: condition_drug_timed_seq-CKD.csv
    
    # only one kind of id file
    f_tdoc_id = '%s_%s.id' % (tdoc_prefix, f_base) # [output]
    print('io> .dat file name: %s, .csv: %s, .csv (ID): %s' % (f_tdoc, f_tdoc_csv, f_tdoc_id))
   
    # [params] control 
    tdoc_overwrite = True # [IO]
    include_tdoc_csv = kargs.get('save_csv', True) and include_timestamps # for now, only save .csv when include_timestamp is true
    save_ids = kargs.get('save_id', True)
    save_intermediate = kargs.get('save_intermediate', False) # if True, then save noncoded symbol mapping

    condition_source_value_sep = ':'  # I9:309.81
    drug_source_value_sep = ':'  # e.g. MED:63408, NDC:00121072716

    remove_drug_source_prefix = False
    drug_prefix = pmed.DrugPrefix  # ['']
    remove_source_prefix = False # this applies to both drug and condition (I9, I0)
    remove_source_dups = True # 

    # note: noncoded data element is probably from drug table
    noncoded_header = ['source_value', 'internal_code']
    prefix_prescription = 'prescription'   # [todo] medcode.prefix_prescription
    prefix_condition = 'condition'  # [todo] medcode.prefix_condition
    prefix_date = 'date'

    token_empty = 'empty'
    token_unknown = 'unknown'
    token_error = 'error'
    pad_empty_code = True 

    # [output]
    f_noncoded = '%s-noncoded.csv' % tdoc_prefix
    f_noncoded_drug = 'drug-noncoded.csv'  # [output]
    f_noncoded_condition = 'condition-noncoded.csv' # [output]
    if cohort_name is not None:
        f_noncoded = '%s_noncoded-%s.csv' % (tdoc_prefix, cohort_name)
        f_noncoded_drug = 'drug_noncoded-%s.csv' % cohort_name # [output]
        f_noncoded_condition = 'condition_noncoded-%s.csv' % cohort_name # [output]         

    dfnc_sep = '|'  # dataframe-noncoded seperator

    # [output]
    for fdoc in (f_tdoc, f_tdoc_csv, f_tdoc_id): 
        tdoc_path = os.path.join(basedir, fdoc)
        if os.path.exists(tdoc_path) and tdoc_overwrite:
            print('io> deleting existing time series at %s' % tdoc_path) 
            os.unlink(tdoc_path)
    # srcvals_path = os.path.join(basedir, f_srcvals) # factored to seqAnalyzer

    report_params()

    ### Preprocessing (dealing with non-coded data element e.g. lasix 20mg by mouth once a day)
    div(message='Identify non-coded values and assign appropriate internal tokens (e.g. concatenation) ...')
    
    noncoded = {}  # this generalizes to anything that requires a mapping for a canonical form
    noncoded_condensed = {}

    # drug/medication codes
    assert not remove_drug_source_prefix, "Perhaps prefix is too important to drop (MED, NDC, MULTUM)"
    for e in df_drug['drug_source_value'].values:  # or use 'drug_concept_id'
        e = str(e)
        e = e.strip()
        el = e.split()

        if len(el) > 1:
            # ebase = el[0]
            
            if pmed.isPrefixedDrugSourceVal(el[0]):
                print('verify> multi-tokens where first one is a medication code: %s' % el[0])

            ep = condense(e)  # ... policy #2 
            # ep = '%s%s' % (prefix_prescription, n)  # internal represetation

            # noncoded_condensed[ce] = ep 
            print('verify> complex medication code: %s => %s' % (e, ep))  
            # e.g. calcium + D 1000-400 => Calcium_D_1000400
            #      Coumadin INR goal 2-2.5 for 4 weeks => Coumadin_Inr_Goal_225_For_4_Weeks
            # [todo] strip off numbers? 

            noncoded[e] = ep 
            
            # [log] e.g. _please review all unapproved meds written by data

        elif len(el) == 0:  # if e is an empty string 
            noncoded[e] = token_empty
        else: # normal singleton token
            # if pmed.isMedCode(e):  # MED:, NDC:, MULTUM: 
            if pmed.isPrefixedDrugSourceVal(e):     # MED:, NDC:, MULTUM:
                pass # noop, just take the expression 'e' as it is
            # if not pmed.isSourceVal(e): 
            else: 
                # code not qualified by valid prefixes 
                
                if pmed.isSourceVal(e): 
                    # pass # no-op
                    print("verify> medication code not prefixed: %s" % e)
                else: 
                    print("verify> noncoded medication unigram: %s" % e)  # e.g. Test; for now, don't map to internal coding
                    # e.g. m/vit, glucometer
                    # ep = e
                    # noncoded[e] = e
                # code it? 

    r = len(noncoded)/(df_drug['drug_source_value'].shape[0]+0.0)
    print('info> # of non-coded data elements in drug table: %d (noncoded ratio: %f)' % (len(noncoded), r))
    
    # diagnostic codes
    noncoded_condition = {}
    for e in df_condition['condition_source_value'].values:  # or use 'condition_concept_id'
        e = str(e)
        e = e.strip()
        # if e.startswith('I9'): 
        el = e.split()
        if len(el) > 1: 
            # this is probably not coded 
            n = len(noncoded_condition)

            # clean incomplete expression e.g. 920. 
            el[0] = dot_normalize(el[0])
            el[1] = dot_normalize(el[1])  # e.g. 367.4. 
            
            code = el[0] + '.' + el[1]

            # use first element as a constraint: el[0]
            if pmed.isICD(code):   
                ep = code 
                # noncoded_condition[e] = ep

            # if pmed.isWord(el[0]): # e.g. I9^ 6A
            else: 
                
                ep = condense(e)  # or '%s%s' % (prefix_condition, n) 
                # noncoded_condition[e] = ep # '%s%s' % (prefix_condition, n)  # after this, we'll have n+1 elements
                print('verify> non-diagnostic codes in (condition) source value: %s => %s' % (e, ep))

            noncoded_condition[e] = ep 
        elif len(el) == 0: 
            noncoded_condition[e] = token_empty
        else:  # additional processing
            if pmed.isICD(e):  # extra dot S53. 920. is considered valid 
                if e[-1] == '.': 
                    print('verify> incomplete coding: %s' % e)
                    noncoded_condition[e] = e[:-1]
                    # e.g. V42.  E888.

                # a lot of codes that end with '.'
                # noncoded_condition[e] = dot_normalize(e)
            else: 
                is_resolved = False
                ep = pmed.convert(e, nocatch=True) # try to convert it to valid code
                if pmed.isICD(ep): 
                    noncoded_condition[e] = ep 
                    is_resolved = True
                else: 
                    if pmed.isMedCode(e): # 6819 as Gambian Trypanosomiasis
                        # convert to ICD-9? 
                        # pass  # leave it as it is
                        ep0 = qrymed2.medToICD(e)
                        if ep0 and pmed.isICD(ep0):
                            ep = ep0 
                            noncoded_condition[e] = ep
                            is_resolved = True
                
                if is_resolved: 
                    print('verify> resolved diagnostic codes: %s => %s' % (e, ep)) 
                    # e.g. 78650 => 786.50
                else: 
                    ep = condense(e)
                    print("verify> unresolved potentially non-diagnistic codes: %s => %s" % (e, ep))  # e.g. ADM, I10, V.0.8, ***, E, .22.1
                    noncoded_condition[e] = ep # token_unknown
                    # e.g. 999999 => 999999

                # 1. leave it as it is
                # 2. map it
    
    r = len(noncoded_condition)/(df_condition.shape[0]+0.0)
    print('info> # of non-coded or (improperly code) data elements in condition table: %d (noncoded ratio: %f)' % \
        (len(noncoded_condition), r))

    # consolidate dictionary 
    # [note] codify*() only looks up noncoded
    for k, v in noncoded_condition.items(): 
        if not noncoded.has_key(k): 
            noncoded[k] = v 
        else: 
            print('warning> mixed entry? (%s, %s)' % (k, v))  # data elements in condition table coincide with those in drug table
    noncoded_condensed = noncoded_condition = None; gc.collect()
    
    div(message='stage> Completed preprocessing.', symbol='#')
    print('  + found %d non-coded or inappropriately coded values' % len(noncoded))
    
    # now start to build (raw) coding sequence (.dat)
    tdoc_path = os.path.join(basedir, f_tdoc) 
    fp = open(tdoc_path, 'a')
    div(message='Consolidate sources (condition_source_value) and create med coding sequences', symbol='%')

    # all person_ids 
    id_col = 'person_id'

    person_ids = set(df_condition[id_col].values) # turning to set does not guarantee ordering
    person_ids.update(df_drug[id_col].values)
    ret['person_id'] = person_ids = sorted(person_ids) # ascending order

    n_persons = len(person_ids)
    min_id, max_id = min(person_ids), max(person_ids)  # if char, then implicitly apply ord()
    track(min_id=min_id, max_id=max_id) # store the value in output 'ret'

    n_doc_total = 0
    print('  + Found %d unique IDs (%s, min=%s, max=%s)' % (n_persons, id_col, min_id, max_id))

    #for pid, df in df_condition.groupby('person_id'):  # foreach candidate found in the condition table
    test_ids = set(random.sample(person_ids, min(len(person_ids), 20)))

    # [params] saving .csv via f_tdoc_csv
    header = ['person_id', 'sequence', 'timestamp', ]  # 'timestamp': use seqReader to process time stamps
    seqdict = {h:[] for h in header}
    personIdEff = []  # persons with data

    print('test> df_condition: %s' % list(df_condition.columns.values))
    print('test> df_drug: %s' % list(df_drug.columns.values))

    # [params] control
    t_col, d_col = 'start_date', 'data'
    condition_tkey = 'condition_start_date'
    drug_tkey = 'drug_exposure_start_date'
    # include_diag = kargs.get('include_diag', True); include_med = kargs.get('include_med', True)
    assert include_diag or include_med, "Diagnostic and medication codes cannot be both absent."

    update_cycle = 1000
    cnt = n_null = n_update_cycle = 0
    tdoc = unit_tdoc = ''  # unit_tdoc: holds time series for single patient only 
    for ith, pid in enumerate(person_ids):  # foreach person ids (sorted in ascending order)

        ### stardardize date & time and unify data formats from different dataframes 
        df = df2 = None
        has_diag = has_med = False
        if include_diag: 
            df = df_condition.loc[df_condition[id_col]==pid]
            if not df.empty: 
                df[t_col] = df[condition_tkey]  # e.g. condition_start_date
                df[d_col] = df['condition_source_value']
                df = df[[t_col, d_col]]
                has_diag = True
        if not has_diag: # dummy data
            # if ith == 0: assert meta is not None, "Need meta field to distinguish from regular doc"
            df = DataFrame(columns=[t_col, d_col])

        if include_med: 
            df2 = df_drug.loc[df_drug[id_col]==pid]
            if not df2.empty: 
                df2[t_col] = df2[drug_tkey] # e.g. drug_exposure_start_date
                df2[d_col] = df2['drug_source_value']
                df2 = df2[[t_col, d_col]]
                has_med = True
        if not has_med: # dummy data
            df2 = DataFrame(columns=[t_col, d_col])

        ### other dataframes (e.g. lab)

        if ith == 0:  # [test]
            assert all(df.columns == df2.columns)
            # print('test> df: %s' % list(df.columns.values))
            # print('test> df2: %s' % list(df2.columns.values))
        
        ### combine and sort 
        # dfx = pd.concat([df, df2, df3, ...], ignore_index=True)  # ok. 
        dfx = pd.concat([df, df2], ignore_index=True)  # ok. 
        assert not (include_diag and include_med) or not dfx.empty
        if dfx.empty:
            n_null += 1 
            if n_null % 100 == 0: print('  + person %s has no data (n_null=%d so far)' % (pid, n_null)) 
            continue

        dfx.sort_values(t_col, ascending=True, inplace=True, kind='mergesort') # [1] assume that groupby preserves order

        # [test]
        if test_mode and (pid in test_ids): 
            print('test> consolidated dataframe (condition+drug):\n%s\n' % dfx.head(50))
            # dfx.groupby('start_date', sort=False).sort_index(ascending=True)

            sv0 = '%s:' % pid
            for i, (d, dfv) in enumerate(dfx.groupby(t_col, sort=False)): # has been sorted 
                sv0 += token_sep.join(codify_seq(dfv))  # x,y,z
                sv0 += token_end_visit
                print('\n    + i=%d > patient #%d, PID: %s > date: %s, dfv => \n%s\n' % (i, cnt, pid, d, dfv.head(20)))
            if len(sv) > 0: sv = sv[:-1]
            if len(sv) > 0: s = sv + token_end_history

            print('    + %s: %s' % (pid, sv0))

            if skip_tdoc: 
                if cnt < 100: 
                    continue 
                else: 
                    print('test complete.')
                    break 

        # if side_effects: # collect meta data and summary statistics (across all patients)
        #     source_values.update(dfx.values) # later on, tease out literal values using seqAnalyzer
       
        ### form document 
        assert group_by_visit, "seqReader now assumes that codes are separated by visits/dates."

        s = ''
        if group_by_visit: # [2] 1. group by date prior to 2.; each visit is separated by 'token_end_visit' 
            n_visits = 0
            sv = ''

            # [todo] group-by seems to change data type 
            for d, dfv in dfx.groupby(t_col, sort=False): # foreach date and its associated records (foreach visit)

                # canonicalize coded time series 
                vseq = codify_seq(dfv)  # this mixes both code and time stamps 
                assert len(vseq) > 0, "empty content on date %s" % str(d)

                sv += token_sep.join(vseq)  # x,y,z  token_sep:','
                # sv += token_sep.join(str(e).split(drug_source_value_sep)[-1] for e in dfv['data'].values)
                sv += token_end_visit  # e.g. ';'
                
                n_visits += 1 

            # [debug]
            # if cnt < 100: print('data> PID: %s, n_visits: %d > sv (group by date):\n%s\n' % (pid, n_visits, sv))
            if cnt < 30:  
                print('data> PID: %s > dfx:\n%s\n' % (pid, dfx.head(100)))
                print('data> visit oriented series:\n%s\n' % sv)

            # sv[-1] = token_end_history  => string is immutable in python 
            if len(sv) > 0: 
                assert sv[-1] == token_end_visit
                sv = sv[:-1]  # clip off the last token, which is a token_sep (e.g. ',')
            # if cnt % 5 == 0: assert sv_temp[-1] == token_end_visit  # ok.

            if len(sv) > 0:  # token_end_history: $
                # a patient document is the coding sequence of multiple visists + end codon
                s = sv + token_end_history  # end token for the entire doc (medical history) of a patient
                # s = ''.join(sv_temp)

        else: # 2. combine data from all dates into one single doc
            s = token_sep.join(codify_seq(dfx))
            
            if cnt < 100: print('data> PID: %s > sv (group by date):\n%s\n' % (pid, sv))
            if len(s) > 0: 
                s += token_end_history
        
        # final time series doc
        if len(s) > 0: 
            assert s[-1] == token_end_history
            unit_tdoc = s + patient_sep # patient separater is a new line (collection of all visits)
            tdoc += unit_tdoc
            cnt += 1 

            # test
            if test_mode and cnt <= 100:
                print('test> n_persons: %d > current person ID: %s, record dim: %s' % (cnt, pid, str(dfx.shape))) 
                print('data>\n%s\n' % dfx.head(20))  # df is not sorted by date
                print('data> unit tdoc > (n_persons=%d)>\n%s\n' % (cnt, unit_tdoc))

            personIdEff.append(pid) # only keep non-empty sequences

            if include_tdoc_csv: 
                assert include_timestamps, "only save .csv when include_timestamps is set to True as well ..."
                # parse the string and separate codes from their timestamps
                cstr, tstr = separate_time2(s, time_sep=time_sep, token_sep=token_sep, 
                                    token_visit=token_end_visit, token_record=token_end_history, check_time=False)
                
                if len(cstr) > 0: 
                    assert len(tstr) > 0
                    seqdict['person_id'].append(pid)
                    seqdict['sequence'].append(cstr)
                    seqdict['timestamp'].append(tstr)
                else: 
                    print('verify> resulted empty string from input sequence: %s' % str(s))
                # else: 
                #     seqdict['sequence'].append(s)

            # [test]
            if cnt % 250 == 0:  # cnt incre when coding sequence (s) is non-empty 
                print('info> (patient #%d, PID: %d) > df: %s, df2: %s, combo records: %s' % (cnt, pid, str(df.shape), str(df2.shape), str(dfx.shape)))
         
        # incremental updates (e.g. every 1000 patients)
        if cnt % update_cycle == 0: 
            n_update_cycle += 1
            # print('info> writing tdoc to %s (processed %d docs)' % (tdoc_path, cnt)) 
            fp.write(tdoc) # only write to buffer

            time.sleep(1)
            fp.flush()  # write doc to file immediately; important for 'streaming' operations

            ndoc_cycle = len(tdoc.split('\n'))
            print('info> writing the %d set (n_doc:%d, total:%d) to:\n%s\n' % (n_update_cycle, ndoc_cycle, cnt, tdoc_path))
            
            tdoc = ''  # reset for the next write cycle

    ### end foreach person_ids
    
    # need to write the last time 
    if len(tdoc) > 0: 
        fp.write(tdoc) # only write to buffer
        time.sleep(1)
        fp.flush()
        nd_last = len(tdoc.split('\n'))
        print('info> writing the last set (n_doc:%d, total:%d) to:\n%s\n' % (nd_last, cnt, tdoc_path))

    # close file 
    fp.close()
    div(message='io> Saved doc (n_docs=%d =?= n_persons=%d) to %s\n' % (len(personIdEff), n_persons, tdoc_path), symbol='#')
    ret['person_id_eff'] = personIdEff  # track(person_id_eff=personIdEff)

    if include_tdoc_csv and len(seqdict) > 0: 
        header = ['person_id', 'sequence', 'timestamp', ]  # 'label'
        df = DataFrame(seqdict, columns=header)
        assert df.shape[0] == len(personIdEff), "One person should have one single document (%d docs vs %d IDs)" % (df.shape[0], len(personIdEff))
        tdoc_path2 = os.path.join(basedir, f_tdoc_csv) 
        df.to_csv(tdoc_path2, sep='|', index=False, header=True)
        div(message='io> Saved .csv (id+seq) to %s\n' % tdoc_path2, symbol='#')
        
        # save person_ids 
    if save_ids and n_persons > 0: 
        header = ['person_id', ]
        adict = {'person_id': personIdEff}
        df = DataFrame(adict, columns=header)
        fpath = os.path.join(basedir, f_tdoc_id)
        df.to_csv(fpath, sep='|', index=False, header=True)
        div(message='io> Saved person IDs to %s\n' % fpath, symbol='#')
        
    print('\n--- Completed sequencing cohort %s ---\n' % cohort_name)

    return ret 

# [refactor] cohort.py
def make_constraint(codes, constraint='condition_source_value', **kargs): 
    """
    Given a set of codes, formulate SQL query to pull out relevant patients. 
    """
    # import select_cohort
    return cohort.make_constraint(codes, constraint=constraint, **kargs)
def make_query_constraint(**kargs): 
    return make_constraint(**kargs)

# [refactor] cohort.py 
def make_constraint_base(codes, constraint='condition_source_value', **kargs):
    # import select_cohort
    return cohort.make_constraint_base(codes, constraint=constraint, **kargs)

def t_preproc(**kargs):

    ### define codes here

    # Diabetes mellitus without complication
    code_str = '24900 25000 25001 7902 79021 79022 79029 7915 7916 V4585 V5391 V6546'

    # Diabetes mellitus with complications
    code_str += ' ' + """24901 24910 24911 24920 24921 24930 24931 24940 24941 24950 24951 24960 24961 24970 24971 24980 24981 24990 24991 25002
        25003 25010 25011 25012 25013 25020 25021 25022 25023 25030 25031 25032 25033 25040 25041 25042 25043 25050 25051 25052
        25053 25060 25061 25062 25063 25070 25071 25072 25073 25080 25081 25082 25083 25090 25091 25092 25093"""

    # Diabetes or abnormal glucose tolerance complicating pregnancy; childbirth; or the puerperium
    code_str += ' ' + "64800 64801 64802 64803 64804 64880 64881 64882 64883 64884" 
       
    codes = icd9utils.preproc_code(code_str)
    print('> n_codes: %d\n> codes:\n%s\n' % (len(codes), codes))

    print('\nNow, do base only\n')
    bcodes = icd9utils.preproc_code(code_str, base_only=True)
    print('> n_codes: %d\n> codes:\n%s\n' % (len(bcodes), bcodes))
    print('> codeset:\n%s\n' % set(bcodes))

    ### convert codes to where clauses
    print('> raw codes:\n%s\n' % make_constraint(codes))
    print('> base codes:\n%s\n' % make_constraint_base(bcodes))

    return 

def t_make_seq(**kargs): 

    # fetch(**kargs)  # the cohort changes, uncomment this
    # [note] also use cohort.py to make more general query statements
    n_persons = makeSeq(**kargs)

    return 

def t_map_concept_id(**kargs): 
    """
    Create a map between concept IDs and source values 
    """
    n_parts = 10
    basedir = sys_config.read('DataExpRoot')
    basedir_sequencing = os.path.join(basedir, 'sequencing')  #
    tRunRQ = True  # run the query that attempts to find out the range of an attribute in the DB (e.g. person_id)

    ### template files (condition and drug)
    # e.g. condition_occurrence-0.csv, drug_exposure-0.csv
    if tRunRQ: 
        rangeQuery(**kargs) # use this to prepare intermediate dataframes

    n_persons = n_persons_eff = 0
    div('step> Include both diagnoses and medications ...')
    # [memo] 3 to 10, then 0 to 3

    for i in range(0, 10): # condition_occurrence-0.csv, drug_exposure-0.csv 
        cohort_name = 'group-%s' % (i+1)  # starting from 1
        ret = makeConceptMap(include_timestamps=tstamp,   # save_intermediate=True
                    include_diag=True, include_med=True, 
                    inputdir=basedir_sequencing, 
                    condition_table='condition_occurrence-%s.csv' % i,  # zero indexed
                    drug_table='drug_exposure-%s.csv' % i, # zero indexed
                    cohort=cohort_name, 
                    save_csv=False, save_id=True)  # save .csv file and ID file 
        min_id, max_id = ret['min_id'], ret['max_id']
        n_persons += len(ret['person_id'])
        n_persons_eff += len(ret['person_id_eff'])
        div('status> finished %d sets of documements (n_persons=%d, n_persons_eff=%d, minId=%s, maxId=%s)' % \
                ((i+1), n_persons, n_persons_eff, min_id, max_id))

    return

def t_sequencing(**kargs): 
    """
    Batch population-scale sequence maker. 
    For sequence reader, refer to seqReader module.
    """

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
    if tRunRQ: 
        rangeQuery(**kargs) # use this to prepare intermediate dataframes

    n_persons = n_persons_eff = 0
    div('step> Include both diagnoses and medications ...')
    # [memo] 3 to 10, then 0 to 3
    for tstamp in (False, True): 
        # for i in range(n_parts): 
        for i in range(0, 10): # condition_occurrence-0.csv, drug_exposure-0.csv 
            cohort_name = 'group-%s' % (i+1)  # starting from 1
            ret = make_seq(include_timestamps=tstamp,   # save_intermediate=True
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
    # diag code only
    # n_persons = n_persons_eff = 0
    # for tstamp in (False, True): 
    #     # for i in range(n_parts): 
    #     for i in range(0, 10): # condition_occurrence-0.csv, drug_exposure-0.csv 
    #         cohort_name = 'group-%s' % (i+1)  # starting from 1
    #         ret = make_seq(include_timestamps=tstamp,   # save_intermediate=True
    #                 include_diag=True, include_med=False, 
    #                 meta='diag', 
    #                 inputdir=basedir_sequencing, 
    #                 condition_table='condition_occurrence-%s.csv' % i,  # zero-based
    #                 drug_table='drug_exposure-%s.csv' % i, 
    #                 cohort=cohort_name, 
    #                 save_csv=False, save_id=True)  # save .csv file and .id file 
    #         n_persons += len(ret['person_id'])
    #         n_persons_eff += len(ret['person_id_eff'])
    #         div('status> finished %d sets of documements (n_persons=%d, n_persons_eff=%d)' % ((i+1), n_persons, n_persons_eff))

    return

def t_sequencing_specialized(**kargs):
    """
    Similar to the template function t_sequence but this method creates 
    specialized sequences such as those with only diagnostic codes and those 
    subject to constraints. 

    """

    n_parts = 10
    basedir = sys_config.read('DataExpRoot')
    basedir_sequencing = os.path.join(basedir, 'sequencing')  #
    tRunRQ = False  # run the query that attempts to find out the range of an attribute in the DB (e.g. person_id)

    # template files (condition and drug)
    # condition_occurrence-0.csv, drug_exposure-0.csv
    if tRunRQ: 
        rangeQuery(**kargs) # use this to prepare intermediate dataframes

    # diag code only
    n_persons = n_persons_eff = 0
    for tstamp in (True, ): 
        # for i in range(n_parts): 
        for i in range(0, 10): # condition_occurrence-0.csv, drug_exposure-0.csv 
            cohort_name = 'group-%s' % (i+1)  # starting from 1
            ret = make_seq(include_timestamps=tstamp,   # save_intermediate=True
                    include_diag=True, include_med=False, 
                    meta='diag', 
                    inputdir=basedir_sequencing, 
                    condition_table='condition_occurrence-%s.csv' % i,  # zero-based
                    drug_table='drug_exposure-%s.csv' % i, 
                    cohort=cohort_name, 
                    save_csv=False, save_id=True, 
                    test_mode=False)  # save .csv file and .id file
            min_id, max_id = ret['min_id'], ret['max_id'] 
            n_persons += len(ret['person_id'])
            n_persons_eff += len(ret['person_id_eff'])
            div('status> finished %d sets of documements (n_persons=%d, n_persons_eff=%d, min_id=%s, max_id=%s)' % \
                ((i+1), n_persons, n_persons_eff, min_id, max_id))

    print('info> n_persons=%d, n_persons_eff=%d' % (n_persons, n_persons_eff))
    return

def t_lookup(**kargs): 
    codes = ['D69.0', 'H10.45', 'L50.0', 'Z91.010', 'Z91.012', 'Z91.048', 'Z84.89']

    # make where clause
    cstr = make_constraint(codes, constraint=None) # [note] simply stating ', '.join(codes) does NOT work
    
    print 'info> looking up %s' % cstr
    print('info> looking up %s' % cstr)

    lookup(codes=cstr)

    return 

def searchByID(person_ids, **kargs): 
    """

    Input: person IDs (which defines a cohort of interest)
    Output: dataframes containing columns associated with medical codes, values, timestamps

    """

    import cohort 

    cohort_name = kargs.get('cohort', 'generic')

    # [options] lab: 'measurement' i.e. ohdsi.west.measurement
    tables = ['condition_occurrence', 'drug_exposure', ]  # ODHSI table names
    for tb in tables:  
        print('   + found %d cases in table=%s' % (len(person_idx), tb))
        df = cohort.searchByID(table=tb, ids=person_ids, cohort=cohort_name, save_intermediate=True) 
        print('   + %s dataframe dim: %s' % (tb, str(df.shape)))

    return 

def test(**kargs): 
    """

    Use
    ---
    1. To generate cohort-specific coding sequences 
       step 1: use cohort.t_search_<cohort>() to read relevant DB data into intermmediate files (in the form of dataframes)
               
               Or, if IDs/MRNs are known, then use searchByID() in this module

               !!! attention
               For diabetes cohort, use fetch() for now ... 10.20.17 

       step 2: use t_make_seq() to formulate coding sequences (given the dataframes generated in step 1)

               makeSeqGeneric(**kargs)
               makeSeq(**kargs)

    """
    def get_input_dir(): # where to expect the input table/dataframe? 
        cohort_name = kargs.get('cohort', 'CKD')
        prefix_dir = kargs.get('inputdir', seqparams.getCohortGlobalDir(cohort_name)) # sys_config.read('DataExpRoot')/<cohort>
        assert os.path.exists(prefix_dir)
        return prefix_dir

    import cohort
    # cohort_name = 'diabetes' # default

    # div('Cohort> See the following queries ...')
    # print('+ drug query ...')
    # print Q_Drug.format(cohort=Q_Cohort)
   
    # print('\n+ measurement query ...')
    # print Q_Measure.format(cohort=Q_Cohort)

    # print('\n')
    # t_preproc()

    ### make time series of medical sequence 
    # cohort 1: diabetes
    #    basedir default: data-exp (i.e. sys_config.read('DataExpRoot'))
    # t_make_seq(save_intermediate=True, include_timestamps=False, cohort='diabetes')
    # t_make_seq(save_intermediate=True, include_timestamps=True, cohort='diabetes')    # tagged with timestamps
    
    # cohort 2: PTSD
    #     basedir default: data-exp (i.e. sys_config.read('DataExpRoot'))
    #     condition table: condition_occurrence-query_ids-PTSD.csv
    #        + header: ['person_id', 'condition_start_date', 'condition_source_value']
    #     drug table: drug_exposure-query_ids-PTSD.csv
    #        + header: ['person_id', 'drug_exposure_start_date', 'drug_source_value']

    # [note] condition_table is created via cohort.searchByID
    # cohort_name = 'PTSD'
    # t_make_seq(save_intermediate=True, include_timestamps=True, 
    #     condition_table='condition_occurrence-query_ids-PTSD.csv', 
    #     drug_table='drug_exposure-query_ids-PTSD.csv', cohort='PTSD')

    ### Arbitrary Cohorts: use seqmaker.cohort to obtain the intermediate dataframe first 
    # 1. cohort definition via diagnostic codes
    # 2. cohort definition via person IDs 

    # General Cohort: 'diabetes', 'PTSD', 'CKD', ... 
    # cohort.t_search_ckd()
    # condition_table: condition_occurrence-query_ids-CKD.csv
    # drug_table: drug_exposure-query_ids-CKD.csv
    
    cohort_name = 'diabetes' # examples: 'diabetes', 'PTSD', 'CKD'
    condition_df = 'condition_occurrence-query_ids-%s.csv' % cohort_name
    medication_df = 'drug_exposure-query_ids-%s.csv' % cohort_name
    inputdir = sys_config.read('DataExpRoot')

    for tstamp in (True, ): # include timestamp ( ), not include timestamp (done)
        t_make_seq(save_intermediate=True, 
            include_timestamps=tstamp, 
            inputdir=inputdir, 
            condition_table=condition_df, 
            drug_table=medication_df, 
            save_id=True,            # True by default
            save_csv=True,  # include the structured version of the sequences (header: person_id, sequence, timestamp)
            cohort=cohort_name)    
    
    # use seqAnalyzer.t_labeling to insert labels

    ### lookup descriptions of medical codes
    # t_lookup(**kargs)

    ### Todo 
    # make_seq_generic()

    return 

def t_parse(**kargs):
    print('t_parse> parsing coding sequence(s) ... ')

    seq = """3008-06-06|MED:72702;2108-07-15|NDC:58980010817,2208-07-15|NDC:55513053010;2238-08-13|MED:62934,
    2108-08-13|NDC:6811501x090,2108-08-13|NDC:552x9062730,2008-08-13|MED:62x39;2008-08-17|NDC:6818x051503,2108-08-17|NDC:5801602x360,
    2108-08-17|NDC:68387036530,2108-08-17|MULTUM:2036,2108-08-17|NDC:00169008283,2228-08-17|MULTUM:467;2118-08-18|NDC:68180051403,
    2208-08-18|NDC:58016086490;2108-09-07|MED:106x80;2108-09-15|MED:106x93,2108-09-15|MED:106392,2108-09-15|MED:106394;2108-09-18|MED:62525,
    2108-09-18|MED:102417,2308-09-18|MED:66042,3008-09-18|MED:61190,2300-09-18|MED:61326,2300-09-18|MED:63509,2300-09-18|MED:81149,
    2108-09-18|MED:72y00,2228-09-18|MED:71169,3008-09-18|MED:62z73,2118-09-18|MED:62714,2118-09-18|MED:62643,3111-09-18|MED:61171,
    2108-09-18|MED:61802,2118-09-18|MED:61509,3088-09-18|MED:61x31$
    """ 
    s, t = parse(seq)
    print('  + codes:\n%s\n' % s)
    print('  + times:\n%s\n' % t)
    cstr, tstr = separate_time2(seq, time_sep='|', token_sep=',', token_visit=';', token_record='$', check_time=False, for_csv=False)
    print('  + codes:\n%s\n' % cstr)
    print('  + times:\n%s\n' % tstr)

    return

def test2(**kargs): 
    """
    Test general sequencing related operations. 
    """

    ### Parsing coding sequences 
    t_parse(**kargs)

    ### basic sequencing operations
    # t_sequencing(**kargs)

    # t_sequencing_specialized(**kargs)  # e.g. diag only 


    return

if __name__ == "__main__": 
    test()
    # test2()  # general facilities for sequencing
