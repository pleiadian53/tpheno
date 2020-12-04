import pandas as pd
from pandas import DataFrame, Series
from IPython.display import display
from IPython.display import Image

import numpy as np

# mapping function across rows or columns

def t_apply(): 
    df = DataFrame(np.random.randn(5,2),index=range(0,10,2),columns=list('AB'))
    df.apply(numpy.sum, axis=0) # equiv to df.sum(0)

    return 


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