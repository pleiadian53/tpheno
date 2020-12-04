from pandas import DataFrame
import pandas as pd 

# needed imports
import matplotlib
matplotlib.use('Agg') # use a non-interactive backend such as Agg (for PNGs), PDF, SVG or PS.
from matplotlib import pyplot as plt


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