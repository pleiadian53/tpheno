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
import seqparams

from pandas import DataFrame, Series
import pandas as pd


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
SELECT person_id, drug_exposure_start_date, drug_source_value
FROM {table}
WHERE {target_attribute} between {lower_bound} and {upper_bound}; 
"""

# example {target_attribute}: person_id
Q_Drug_RangeID = """
SELECT person_id, drug_exposure_start_date, drug_source_value
FROM ohdsi.west.drug_exposure
WHERE person_id between {lower_bound} and {upper_bound}; 
"""

Q_Condition_RangeID = """
SELECT person_id, condition_start_date, condition_source_value
FROM ohdsi.west.condition_occurrence
WHERE person_id between {lower_bound} and {upper_bound}; 
"""

Q_Drug_PersonID = """
SELECT person_id, drug_exposure_start_date, drug_source_value
FROM ohdsi.west.drug_exposure
WHERE person_id in ({id_set}); 
"""

Q_Condition_PersonID = """
SELECT person_id, condition_start_date, condition_source_value
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

    # [I/O] cohort 
    # Q_Cohort uses diagnostic codes to define the target cohort
    # Change query here
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

def test(): 
    pass

if __name__ == "__main__": 
    test()