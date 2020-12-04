from pandas import DataFrame
import pandas as pd 

import os, sys, re, collections
import numpy as np

from batchpheno.utils import div

# needed imports
import matplotlib
matplotlib.use('Agg') # use a non-interactive backend such as Agg (for PNGs), PDF, SVG or PS.
from matplotlib import pyplot as plt


def t_concat(**kargs):

    df1 = pd.DataFrame({"x":[1, 2, 3, 4, 5], 
                    "y":[3, 4, 5, 6, 7]}, 
                   index=['a', 'b', 'c', 'd', 'e'])


    df2 = pd.DataFrame({"y":[1, 3, 5, 7, 9], 
                    "z":[9, 8, 7, 6, 5]}, 
                   index=['b', 'c', 'd', 'e', 'f'])

    # join = 'inner' => using index elements found in both DFs (aligned by indexes for joining)
    pd.concat([df1, df2], join='inner') # by default axis=0, i.e. concatenating vertically (or along the rows)

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


    # Merge multiple dataframes
    df1 = pd.DataFrame(np.array([
        ['a', 5, 9],
        ['b', 4, 61],
        ['c', 24, 9]]),
         columns=['name', 'attr11', 'attr12'])
    df2 = pd.DataFrame(np.array([
        ['a', 5, 19],
        ['b', 14, 16],
        ['c', 4, 9]]),
        columns=['name', 'attr21', 'attr22'])
    df3 = pd.DataFrame(np.array([
        ['a', 15, 49],
        ['b', 4, 36],
        ['c', 14, 9]]),
        columns=['name', 'attr31', 'attr32'])

    pd.merge(pd.merge(df1,df2,on='name'),df3,on='name')

    # alternatively, 
    df1.merge(df2,on='name').merge(df3,on='name')

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
    def count_letters(r):
        return len(r) 

    data = {'name': ['Jason', 'Molly', 'Tina', 'Jake', 'Amy'],
        'year': [2012, 2012, 2013, 2014, 2014],
        'reports': [4, 24, 31, 2, 3],
        'coverage': [25, 94, 57, 62, 70]}
    df = pd.DataFrame(data, index = ['Cochice', 'Pima', 'Santa Cruz', 'Maricopa', 'Yuma'])

    capitalizer = lambda x: x.upper()

    df['n_letters'] = df['name'].apply(count_letters)
    print df
    print("\n")

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

    ### arithmetics on columns 
    print('\narithmetics on rows and columns ...\n')
    df['ratio'] = df['reports']/df['coverage'].astype(float)
    print df

    return

def idf(ndf, n): 
    import math
    
    # n: total number of docs 
    # ndf: |d in D and t in d|, number of docs that contain word t
    # t: word
    return math.log[ (1 + n) / 1 +  ndf] + 1

def t_groupby(**kargs):
    """


    Reference
    ---------
    1. http://pandas.pydata.org/pandas-docs/stable/groupby.html
    2. http://nbviewer.jupyter.org/urls/bitbucket.org/hrojas/learn-pandas/raw/master/lessons/06%20-%20Lesson.ipynb

    """

    basedir = '/Users/pleiades/Documents/work/tpheno/seqmaker/data' # '/phi/proj/poc7002/tpheno/seqmaker/data'
    fname = 'motifs-Cdiagnosis-L1-CID16-kmeans-Sregular-D2Vtfidf.csv'

    fpath = os.path.join(basedir, fname) 
    df = pd.read_csv(fpath, sep='|', header=0, index_col=False, error_bad_lines=True)
    nrow = df.shape[0]
    print('io> loaded df of dim: %s' % str(df.shape))
    print('verify> top:\n%s\n' % df.head(20))

    header = ['length', 'ngram', 'count', 'global_count', 'ratio']
    pivots = ['length', ] # groupby attributes
    sort_fields = ['ratio', ]

    # [note] By default the group keys are sorted during the groupby operation
    groups = df.groupby(pivots)  # as_index=False # set as_index to False => pivots becomes a separate column instead of an index
    
    dfx = []
    for n, dfg in groups: 
        # df.apply(lambda x: x.order(ascending=False)) # take top 3, .head(3)
        dfg_sorted = dfg.sort(sort_fields, ascending=False, inplace=False)

        # doc-length-adjusted tf 
        dfg_sorted['tf'] = dfg_sorted['count']/nrow

        # idf 
        dfx.append(dfg_sorted)
        # print('> length: %d:\n%s\n' % (n, df_group))

    df = pd.concat(dfx, ignore_index=True)
    print('verify> top:\n%s\n' % df.head(20))
    
    return

def t_groupby2(**kargs): 
    """

    Note
    ----
    1. groupby by default sort the group key

    Reference 
    ---------
    1. notebook 
       http://nbviewer.jupyter.org/urls/bitbucket.org/hrojas/learn-pandas/raw/master/lessons/06%20-%20Lesson.ipynb


    """
    div(message='Example #1')
    df1 = DataFrame( { 
         "Name" : ["Alice", "Bob", "Mallory", "Mallory", "Bob" , "Mallory"] , 
         "City" : ["Seattle", "Seattle", "Portland", "Seattle", "Seattle", "Portland"] } )

    g1 = df1.groupby( [ "Name", "City"] ) # this results in hierarchical index 

    dfx = []
    for g, df in g1: 
        print '%s =>\n%s' % (g, df)
        dfx.append(df)
    df = pd.concat(dfx, ignore_index=True)[['Name', 'City']]
    print('io> final grouped+combined df:\n%s\n' % df)

    # g1 = df1.groupby( [ "Name", "City"] )
   
    # [log] Cannot access callable attribute 'reset_index' of 'DataFrameGroupBy' objects, try using the 'apply' method
    # print('io> g1.reset_index: %s' % g1.reset_index())  # this doesn't work


    div(message='Example #2')
    np.random.seed(53)
    n=10
    df = pd.DataFrame({'mygroups' : np.random.choice(['dogs','cats','hyenas','zebras'], size=n), 
                      'data' : np.random.randint(1000, size=n)})

    df['cid'] = np.random.randint(2, size=n)
    print('data> \n%s\n' % df)

    g2 = df.groupby('mygroups', sort=True, as_index=False)
    # g2.set_index(['mygroups'])
    counts = []   # entropy
    for g, df in g2: 
        count = len(df['cid'].values)
        counts.append(count)
    

    print g2



    print('\napply aggregate function ...')
    g3 = df.groupby('mygroups', sort=False).sum()
    # print g2.reset_index()
    print g3  # this uses group key mygroups as indices
    print('\nVS\n')
    print g3.reset_index() 

    print('\nnow sort the group index ...\n')
    g3.sort_index(ascending=False)
    print g3

    return


def t_sort(**kargs): 
    """

    Note
    ----
    1. sort_values is meant to sort by the values of columns
       sort_index is meant to sort by the index labels (or a specific level of the index, or the column labels when axis=1)
           <ref> https://stackoverflow.com/questions/19332171/difference-between-sort-values-and-sort-index



    """
    # from pandas import DataFrame
    # import pandas as pd

    # customize sort: assign-sort-drop 
    d = {'one':[2,3,1,4,5],
          'two':[5,4,3,2,1],
          'letter':['a','a','b','b','c']}
    df = DataFrame(d)
    df.assign(f = df['one']**2 + df['two']**2).sort_values('f').drop('f', axis=1)

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
    # t_groupby(**kargs)

    # apply function to rows and columns 
    t_apply(**kargs)

    ### sort according to a column within groupby-results
    # e.g. groupby length of n-grams, and then sort within each group according to term frequency and global term frequency 
    #      seqmaker.seqCluster 
    # t_groupby2(**kargs)

    return 

if __name__ == "__main__":
    test()