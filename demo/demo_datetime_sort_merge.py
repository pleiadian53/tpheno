import datetime
from pandas import DataFrame, Series 
import pandas as pd 

start_dates = [
    datetime.date(2009, 11, 5), datetime.date(2009, 11, 13), 
    datetime.date(2009, 11, 25), datetime.date(2009, 11, 26), 
    datetime.date(2009, 12, 4), datetime.date(2009, 12, 7), 
    datetime.date(2009, 12, 29), datetime.date(2009, 12, 30)]

end_dates = [
    datetime.date(2009, 10, 1), datetime.date(2009, 10, 2), 
    datetime.date(2009, 10, 9), datetime.date(2009, 10, 12), 
    datetime.date(2009, 11, 4), datetime.date(2009, 12, 14), 
    datetime.date(2009, 12, 15)]




def t_consolidate(**kargs): 
    
    d1 = {'start_date': ['2012-04-12',  '2012-04-11'], 'condition_value': ['110.25', '117.5']}
    d2 = {'start_date': [ '2012-04-13', '2012-04-12' ], 'drug_value': ['11111', '22222']}

    df = DataFrame(d1)
    df2 = DataFrame(d2)

    dfx = pd.concat([df, df2], ignore_index=True)

    print('data>\n%s\n' % dfx)
    
    print('info> now try to consolidate them ...\n')

    df['data'] = df['condition_value']
    df = df[['start_date', 'data']]

    df2['data'] = df2['drug_value']
    df2 = df2[['start_date', 'data']]

    dfx = pd.concat([df, df2], ignore_index=True)

    print('data>\n%s\n' % dfx)

    dfx.sort_values(['start_date', 'data'], ascending=True, inplace=True, kind='mergesort')

    print('sort>\n%s\n' % dfx)

    return 

def t_rep(**kargs): 

    # 1a. convert string to datetime object 
    string_date = "2013-09-28 20:30:55.78200"
    print datetime.datetime.strptime(string_date, "%Y-%m-%d %H:%M:%S.%f")

    # 1b. convert from datetime to string 
    t = datetime.datetime(2012, 2, 23, 0, 0)
    print t.strftime('%m/%d/%Y') 

    return

def test(**kargs): 
	t_consolidate(**kargs)

	return

if __name__ == "__main__":
    test()

