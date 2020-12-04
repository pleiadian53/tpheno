import pandas as pd
from pandas import DataFrame, Series
from IPython.display import display
from IPython.display import Image

# raw_data = {
#         'subject_id': ['1', '2', '3', '4', '5'],
#         'first_name': ['Alex', 'Amy', 'Allen', 'Alice', 'Ayoung'], 
#         'last_name': ['Anderson', 'Ackerman', 'Ali', 'Aoni', 'Atiches']}
# df_a = pd.DataFrame(raw_data, columns = ['subject_id', 'first_name', 'last_name'])

# print df_a

def t_join(**kargs): 
    """

    Memo
    ----
    1. notice the key K2

  key   A    B
0  K0  A0   B0
1  K1  A1   B1
2  K2  A2   B2
2  K2  A2   B3
3  K3  A3  NaN
4  K4  A4  NaN
5  K5  A5  NaN

    """

    caller = pd.DataFrame({'key': ['K0', 'K1', 'K2', 'K3', 'K4', 'K5', 'K2', 'K2'],
                          'A': ['A0', 'A1', 'A2', 'A3', 'A4', 'A5', 'A11', 'A12']})

    other = pd.DataFrame({'key': ['K0', 'K1', 'K2', 'K2'],
                         'B': ['B0', 'B1', 'B2', 'B3']})

    # Join DataFrames using their indexes.
    df1 = caller.join(other, lsuffix='_caller', rsuffix='_other')

    print df1
    print '--------\n'

    # DataFrame.join always uses other's index but we can use any column in the caller. This method preserves the original caller's index in the result.
    df2 = caller.join(other.set_index('key'), on='key')  # [m1]
    df2 = df2[['key', 'A', 'B']]

    print df2

    return 

def test(): 
    t_join()

if __name__ == "__main__": 
	test()