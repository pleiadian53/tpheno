# encoding: utf-8

import collections
import re
import sys
import time
from batchpheno import utils


######################################################
#
#  Process n-grams (standardization, transformation, etc.)
# 
#
# 

def normalize(motif, sep=' '):
    if isinstance(motif, str): 
    	return motif # noop 
    try: 
        return sep.join(motif)  # assuming that motif is a sequence i.e. iterable 
    except: 
    	pass 
    raise ValueError, "Invalid motif format: %s" % str(motif)

def lcs(a, b, sep=' '):
    """
    Find the longest common subsequence between a and b. 

    Input: 
       a: 
       b: 

    """
    lengths = [[0 for j in range(len(b)+1)] for i in range(len(a)+1)]  # 2-by-2 table 

    # row 0 and column 0 are initialized to 0 already
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            if x == y:
                lengths[i+1][j+1] = lengths[i][j] + 1
            else:
                lengths[i+1][j+1] = max(lengths[i+1][j], lengths[i][j+1])
    # read the substring out from the matrix
    result = ""
    x, y = len(a), len(b)

    # backtracking
    if sep is None: 
        # [input] a, b: strings
        while x != 0 and y != 0:
            if lengths[x][y] == lengths[x-1][y]:
                x -= 1
            elif lengths[x][y] == lengths[x][y-1]:
                y -= 1
            else:
                assert a[x-1] == b[y-1]
                result = a[x-1] + result
                x -= 1
                y -= 1
    else: 
        # a, b <- list 
        assert hasattr(a, '__iter__') and hasattr(b, '__iter__'), "inputs must be lists (not strings)"

        while x != 0 and y != 0:
            if lengths[x][y] == lengths[x-1][y]:
                x -= 1
            elif lengths[x][y] == lengths[x][y-1]:
                y -= 1
            else:
                assert a[x-1] == b[y-1]
                result = a[x-1] + sep + result
                x -= 1
                y -= 1

        result = result.strip()
        result = result.split(sep)

    return result # output: list 

def test(**kargs): 
    from itertools import groupby

    s1 = '496 492.8 555 496 496 496 492.8 777 496'
    s2 = '496 492.8 496 777 496 492.8 496'

    s1 = '496 555 496 496 496 492.8 777 496'
    s2 = '496 492.8 496 777 496 492.8 496'

    sep = ' '
    res = lcs(s1.split(sep), s2.split(sep))
    print 'LCS of s1 and s2 = %s' % res

    res2 = lcs(s1, s2, sep=None)
    print 'LCS of s1 and s2 (strings) = %s' % res

    print('> remove duplicates ...')
    s1u = [x[0] for x in groupby(s1.split(sep))]
    print s1
    print ' '.join(s1u)

    return 

if __name__ == "__main__": 
    test()
