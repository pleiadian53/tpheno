import pandas as pd
from pandas import Series, DataFrame, TimeSeries
import numpy as np

def t_precision_control(): 
    """

    Topic 
    -----
    1. generate random numbers in a dataframe 
    2. floating point numbers and their precisions
    3. data type
    """

    import random
    #generate a dataframe with two columns, both floating point
    df = pd.DataFrame({'A':[random.random()*10 for i in range(10)],
                     'B':[random.random()*20 for i in range(10)]})
    df2 = df.copy() #make a copy to preserve your original


    df2.loc[:, 'A'] = df2['A'].apply(int) #convert A to an int
    df2.loc[:, 'B'] = df2['B'].round(20) #round B to 20 points of precision

    df2.to_csv('test.csv', header = None, index = False)

    return 

def t_time_series(**kargs): 

	# convert TimeSeries into a DataFrame
    ts = pd.TimeSeries({'a':[1,2,3,4,5], 'b':[6,7,8,9,10]})
    pandas.DataFrame(list(ts.values), index=ts.index)

def t_subsetting(**kargs): 
    df = DataFrame(np.random.randn(5,2),index=range(0,10,2),columns=list('AB'))
    print df

    pos = [2, 4]
    print('> index by position: %s' % pos)
    print df.iloc[pos]  # by position 

    print("> index by label (i.e. %s is considered as a label but not absolute position." % pos)
    print df.loc[pos]

    return


def test(): 
	t_subsetting()

if __name__ == "__main__":
    test() 