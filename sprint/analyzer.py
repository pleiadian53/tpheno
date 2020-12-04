# encoding: utf-8

from pandas import DataFrame
import pandas as pd 

# needed imports
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

def source(**kargs): 
    # input data sources
    ipath = kargs.get('ipath', DataDir) # input path
    ext = 'csv'
    dfiles = [name for name in glob.glob('%s/*.%s' % (ipath, ext))]
    print('io> all data files:\n%s\n' % dfiles) # full paths
    
    return dfiles

def filter_source(**kargs):
    return filter() 

# [todo]
def merge(**kargs): 

    jattr = kargs.get('on', 'MASKID') # join attribute
    jtype = kargs.get('how', 'inner') # join type; options {‘left’, ‘right’, ‘outer’, ‘inner’}, default 'inner'

    # load data
    ifiles = source(**kargs)
    bases = [base(f) for f in ifiles]
    adict = {b:{} for b in bases}

    dfx = []
    for ifile in ifiles: 
        df = pd.read_csv(ifile, sep=',', header=0, index_col=False, error_bad_lines=True)
        dfx.append(df)

    # join tables 
    for df in dfx: 
        pd.merge(df_a, df_b, on=jattr, how='inner')

    return

def query(**kargs): 
    # load data
    ifiles = source(**kargs)
    bases = [base(f) for f in ifiles]
    adict = {b:{} for b in bases}
    for ifile in ifiles: 
        df = pd.read_csv(ifile, sep=',', header=0, index_col=False, error_bad_lines=True)


def analyze_values(**kargs): 
    def base(fpath): 
        bf = os.path.basename(fpath)
        return bf.split('.')[0]

    mode = kargs.get('mode', 'ordered')
    
    ifiles = source(**kargs)
    bases = [base(f) for f in ifiles]
    adict = {b:{} for b in bases}

    if mode.startswith('ord'): 
        for ifile in ifiles: 
            df = pd.read_csv(ifile, sep=',', header=0, index_col=False, error_bad_lines=True)
            header = df.columns.values

            # adict[base(ifile)]
            counts = dict(df.apply(pd.Series.nunique))

            hc_map = []
            for h in header: 
                hc_map.append( (h, counts[h]) )

            adict[base(ifile)] = hc_map  # header-count in the order as the column appears in the dataframe
    else: 
        for ifile in ifiles: 
            df = pd.read_csv(ifile, sep=',', header=0, index_col=False, error_bad_lines=True)
            header = df.columns.values
            adict[base(ifile)] = dict(df.apply(pd.Series.nunique))  # header-count in the order as the column appears in the dataframe

    return adict

def ordered_display(adict, names): 
    pass 

def dumpclean(obj, n_sep=0, show_index=False, i=0):
    if type(obj) == dict:
        for i, (k, v) in enumerate(obj.items()):
            if hasattr(v, '__iter__'):
                if n_sep:
                    blank_lines = '\n' * n_sep
                    print "%s%s" % (blank_lines, k)
                else: 
                    print k
                dumpclean(v, n_sep, show_index, i=i)
            else:
                if show_index:
                    print '[%d] %s : %s' % (i, k, v) 
                else: 
                    print '%s : %s' % (k, v)
    elif type(obj) == list:
        for i, v in enumerate(obj):
            if hasattr(v, '__iter__'):
                dumpclean(v, n_sep, show_index, i=i)
            else:
                if show_index:
                    print('[%d] %s' % (i, v)) 
                else: 
                    print v
    else: # including tuples
        if show_index: 
            print('[%d] %s' % (i, str(obj))) 
        else: 
            print str(obj)

def n_unique_values(df): 
    """
    Find number of unique values associated with each column. 
    """

    return

def columns_by_dtype(df):
    
    return

def t_join(**kargs):
    """


    References
    ----------
    1. https://chrisalbon.com/python/pandas_join_merge_dataframe.html
    """ 
    raw_data = {
        'subject_id': ['1', '2', '3', '4', '5'],
        'first_name': ['Alex', 'Amy', 'Allen', 'Alice', 'Ayoung'],
        'last_name': ['Anderson', 'Ackerman', 'Ali', 'Aoni', 'Atiches']}
    df_a = pd.DataFrame(raw_data, columns = ['subject_id', 'first_name', 'last_name'])
    print df_a
    
    raw_data = {
        'subject_id': ['4', '5', '6', '7', '8'],
        'first_name': ['Billy', 'Brian', 'Bran', 'Bryce', 'Betty'],
        'last_name': ['Bonder', 'Black', 'Balwner', 'Brice', 'Btisan']}
    df_b = pd.DataFrame(raw_data, columns = ['subject_id', 'first_name', 'last_name'])
    print df_b

    raw_data = {
        'subject_id': ['1', '2', '3', '4', '5', '7', '8', '9', '10', '11'],
        'test_id': [51, 15, 15, 61, 16, 14, 15, 1, 61, 16]}
    df_n = pd.DataFrame(raw_data, columns = ['subject_id','test_id'])
    print df_n

    # Join the two dataframes along columns
    pd.concat([df_a, df_b], axis=1)

    # merge 
    # "Full outer join produces the set of all records in Table A and Table B, with matching records from both sides where available. 
    # If there is no match, the missing side will contain null."
    pd.merge(df_new, df_n, on='subject_id')

    # outer join 
    # "Inner join produces only the set of records that match in both Table A and Table B."
    pd.merge(df_a, df_b, on='subject_id', how='outer')

    # left outer join 
    # "Left outer join produces a complete set of records from Table A, with the matching records (where available) in Table B. 
    # If there is no match, the right side will contain null."
    pd.merge(df_a, df_b, on='subject_id', how='left')

    # right outer join 
    pd.merge(df_a, df_b, on='subject_id', how='right')


    # left outer + suffix 
    pd.merge(df_a, df_b, on='subject_id', how='left', suffixes=('_left', '_right'))


    # inner join: "Inner join produces only the set of records that match in both Table A and Table B."  
    pd.merge(df_a, df_b, on='subject_id', how='inner')

    return


def t_unique(**kargs): 
    data = {'name': ['Jason', 'Molly', 'Tina', 'Jake', 'Amy'],
        'year': [2012, 2012, 2013, 2014, 2014],
        'reports': [4, 24, 31, 2, 3]}
    df = pd.DataFrame(data, index = ['Cochice', 'Pima', 'Santa Cruz', 'Maricopa', 'Yuma'])

    #List unique values in the df['name'] column
    col = 'name'
    # print df.name.unique()  # a Series 
    print('info> uniq values of %s: %s' % (col, df[col].unique() )) # df.name.unique()
    # print(' + df.name > type: %s, value:\n%s\n' % (type(df.name), df.name))

    # count 
    counts = dict(df.apply(pd.Series.nunique))  # apply this function per column
    print('info> counts (dtype: %s):\n%s\n' % (type(counts), counts))  # df.apply(pd.Series.nunique) => Series

    col = 'reports'
    print('info> n_unique(%s): %d' % (col, counts[col]))

    return

def t_apply(**kargs): 
    """

    Reference
    ---------
    1. https://chrisalbon.com/python/pandas_apply_operations_to_dataframes.html
    """
    def times100(x): # create a function called times100
        # that, if x is a string,
        if type(x) is str:
            # just returns it untouched
            return x
        # but, if not, return it multiplied by 100
        elif x:
            return 100 * x
        # and leave everything else
        else:
            return

    data = {'name': ['Jason', 'Molly', 'Tina', 'Jake', 'Amy'],
        'year': [2012, 2012, 2013, 2014, 2014],
        'reports': [4, 24, 31, 2, 3],
        'coverage': [25, 94, 57, 62, 70]}
    df = pd.DataFrame(data, index = ['Cochice', 'Pima', 'Santa Cruz', 'Maricopa', 'Yuma'])

    capitalizer = lambda x: x.upper()

    # apply() can apply a function along any axis of the dataframe
    x = df['name'].apply(capitalizer)
    print x 

    # map() applies an operation over each element of a series
    x = df['name'].map(capitalizer)
    print x

    # Drop the string variable so that applymap() can run
    df = df.drop('name', axis=1)
    print('input> df:\n%s\n' % df)

    # # Return the square root of every cell in the dataframe
    # s = df.applymap(np.sqrt)
    # print s

    # s1 = df.apply(np.sqrt)
    # print s1  # result same as above

    sum_rows = df.apply(sum, axis=0) # axis 0 => apply function across rows
    print df.append(sum_rows, ignore_index=True) # sum_rows is a Series, can only be appended if ignore_index <- True

    df['sum_columns'] = df.apply(sum, axis=1)
    print df

    m100 = df.applymap(times100)
    print m100

    return



def t_dtype(**kargs):

    ### use groupby to find all columns ~ dtypes  
    df = pd.DataFrame([[1, 2.3456, 'c', 'd', 78]], columns=list("ABCDE")) 

    print df.dtypes

    # group by dtype 
    g = df.columns.to_series().groupby(df.dtypes).groups

    adict = {k.name: v for k, v in g.items()}
    
    print adict 


    ### select_dtypes
    df = pd.DataFrame({'NAME': list('abcdef'),
                       'On_Time': [True, False] * 3,
                       'On_Budget': [False, True] * 3})

    df.select_dtypes(include=['bool'])
    bool_columns = list(df.select_dtypes(include=['bool']).columns)
    print(bool_columns)


    return

def test(**kargs): 

    # gropu by datatypes 
    # t_dtype()

    ### Analyze column values => separate training data sets 
    # source()

    # unique values 
    # t_unique()

    # apply/map function 
    # t_apply()

    ### Analyze SPRINT datasets 
    dumpclean(analyze_values(), n_sep=2, show_index=True)
    
    return 

if __name__ == "__main__":
    test()