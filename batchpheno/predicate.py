import numpy as np
import pandas as pd
import re
from dateutil.parser import parse


class P(object):
    """

    Reference 
    ---------
    http://stackoverflow.com/questions/9184632/pointfree-function-combination-in-python
    """
    def __init__(self, predicate):
        self.pred = predicate

    def __call__(self, obj):
        return self.pred(obj)

    def __copy_pred(self):
        return copy.copy(self.pred)

    def __and__(self, predicate):  # predicate composition
        def func(obj):
            return self.pred(obj) and predicate(obj)
        return P(func)

    def __or__(self, predicate):
        def func(obj):
            return self.pred(obj) or predicate(obj)
        return P(func)

def predicate(func):
    """Decorator that constructs a predicate (P) instance from
    the given function."""
    from functools import update_wrapper
    result = P(func)
    update_wrapper(result, func)
    return result

# [test]
@predicate
def is_divisible_by(number, x=19):
    return number % x == 0

@predicate
def is_palindrome(number):
    return str(number) == str(number)[::-1]



def hasValue(v): 
    return not isNull(v)

def isNullStr(v, allow_na_str=True):
    if not v or v.isspace(): 
        return True 
    else: 
        if allow_na_str: 
            if v.lower() in ('nan', 'null', 'none', 'n/a', 'na', ): 
                return True
    return False

def isNull(v):  # [todo]
    if isinstance(v, str): 
        return isNullStr(v)

    try: 
        if np.isnan(v): 
            return True
    except: 
        try:  
            if pd.notnull(v): 
                # subsume np.isfinite(obj)
                return False 
            else: 
                return True 
        except: 
            print('isNull> Warning: both np.isnan() and pd.notnull() failed.')

    return False 

def isNaN(e): 
    return e != e # only works for python 2.5+

def isInf(e): 
    # np.isfinite(obj) 
    try: 
        return np.isinf(e)
    except: 
        pass 
    return False

def isNumber(obj, allow_str=False): 
    """

    Memo
    ---- 
    1. str('130.0').strip().isdigit() ~> False
    2. isinstance('130.0', Number) ~> False
    """
    def is_special(e):
        return isNull(e) or isNaN(e) or isInf(e)

    from numbers import Number
    from decimal import Decimal
    # tval = isinstance(obj, Number) or str(obj).strip().isdigit()
    if not allow_str: 
        # if is_special(obj): return False
        if isNumber2(obj): return True
    else: 
        if isinstance(obj, str): 
            # deal with the string
            # string 'inf', 'nan' will be considered as number!
            try: 
                x = int(float(obj))  
            except: 
                return False
            # if is_special(obj): return False
            if isNumber2(x): return True 
    return False 
    # if obj in ('inf', 'nan', ): return False
    # n = float('nan') 
    # try: 
    #     n = float(obj)
    # except: 
    #     return False 
    # return isinstance(n, Number)

def isNumber2(obj): 
    from decimal import Decimal
    if isinstance(obj, float): 
        # note float is not necessarily a "number" e.g. nan, np.inf 
        try: 
            x = int(obj)
        except: 
            return False
        return True 
    if isinstance(obj, (int, long, complex)): return True
    if isinstance(obj, Decimal): return True
    return False

def isMedDir(_dir): 
    try: 
        int(_dir)
        # check if valid med code <todo>
    except: 
         return False
    return True
    
def isYearZipFile(_file):
    p = re.compile(r'([0-9]+)\.(gz|tar\.gz)')
    if p.match(_file): 
        return True
    return False  

def isDiagnosticCode(x): 
    return isNumber(x, allow_str=True)

def isMedCode(x): 
    try: 
        x = int(float(x)) # x may be 'ill-formatted' as a string of floating number
    except: 
        return False

    n_digits = len(str(x))
    return n_digits <= 6 and n_digits >= 1

def in_subdir(d1, d2):  # is d1 a subdir of d2? 
    """

    Related 
    -------
    os.path.commonprefix([d1, d2]) 
    os.pardir 
    os.path.realpath(dir)

    """
    import os
    #make both absolute    
    d1 = os.path.join(os.path.realpath(d1), '')
    d2 = os.path.realpath(d2)

    #return true, if the common prefix of both is equal to directory
    #e.g. /a/b/c/d.rst and directory is /a/b, the common prefix is /a/b
    return os.path.commonprefix([d2, d1]) == d2

def is_outlier(points, thresh=3.5):
    """
    Returns a boolean array with True if points are outliers and False 
    otherwise.

    Parameters:
    -----------
        points : An numobservations by numdimensions array of observations
        thresh : The modified z-score to use as a threshold. Observations with
            a modified z-score (based on the median absolute deviation) greater
            than this value will be classified as outliers.

    Returns:
    --------
        mask : A numobservations-length boolean array.

    References:
    ----------
        Boris Iglewicz and David Hoaglin (1993), "Volume 16: How to Detect and
        Handle Outliers", The ASQC Basic References in Quality Control:
        Statistical Techniques, Edward F. Mykytka, Ph.D., Editor. 
    """
    if len(points.shape) == 1:
        points = points[:,None]
    median = np.median(points, axis=0)
    diff = np.sum((points - median)**2, axis=-1)
    diff = np.sqrt(diff)
    print('info> input:\n%s\n  + diff:\n%s\n' % (points, diff))
    med_abs_deviation = np.median(diff)

    modified_z_score = 0.6745 * diff / med_abs_deviation

    return modified_z_score > thresh

def isDate(x): 
    from dateutil.parser import parse

    x = str(x)
    parse_able = False
    try: 
        parse(x)
        parse_able = True
    except ValueError:
        return False

    if parse_able: 
        x2 = x.split()  # 1979-01-01 00:00:00  => ['1979-01-01', '00:00:00', ]
        if len(x2) >= 1: 
            dx = x2[0].split('-')
            # if max digit one is a year
            if len(dx) == 3: 
                try: 
                    int(dx[0])
                    int(dx[1])
                    int(dx[2])
                    return True
                except: 
                    return False
    return False

def t_predicate(): 
    pred = (is_divisible_by & is_palindrome)
    print [x for x in xrange(1, 1000) if pred(x)]   

def t_dir(): 
    import os
    d1 = '/fakedir/cdr'
    d2 = '/fakedir/cdr/lab/cerner/feature'
    print "%s is a subdir of %s? %s" % (d1, d2, in_subdir(d1, d2))

    d1, d2 = d2, d1
    print "%s is a subdir of %s? %s" % (d1, d2, in_subdir(d1, d2))

    d1 = '/fakedir/data-feature'
    d2 = '/fakedir/data-learner'
    print "%s is a subdir of %s? %s" % (d1, d2, in_subdir(d1, d2))

    return 

def t_outlier(): 
    points = "0.0, 1.0, 11.4, 7.3, 2.2, 33.5, 19.099, 10.4, 12.0, 2.9, 5.3, 9.1, 173.4, 16.9, 16.4, 13.2, 13.7, 23.5"
    points = [e.strip() for e in points.split(',')]
    
    px = []
    for p in points: 
        try: 
            px.append(float(p))
        except: 
            pass
    px = np.array(px)
    ox = []
    if len(px) > 0: 
        ox = is_outlier(px, thresh=3.5)
        print ox
    else: 
        print('warning> no floats found in %s' % points)

    return ox    

def t_date(): 
    dates = ['2038-01-01 00:00:00', '2038-01-01', '2038/01/01', '2038.01.01', '2038.13.01', '2038-2']
    for d in dates: 
        print('> date: %s, valid? %s' % (d, isDate(d))) 
    
    return   

def test(): 
    
    # t_dir()
    # functional predicate expression
    # t_predicate()

    # code = '00.0'
    # print("> a diag code? %s" % isDiagnosticCode(code))

    # t_outlier()

    t_date()

    return

if __name__ == "__main__": 
    test()
        
