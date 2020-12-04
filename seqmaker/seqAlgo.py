# encoding: utf-8

import collections
import re
import sys
import time
from batchpheno import utils

import numpy as np

######################################################################################
#
#  A spin-off module of algorithms that focus on sequence data. 
#
#  Note
#  ---- 
#  1. specialized string and sequence processing methods may be found in separate modules: 
#     a. diagnostic codes (ICD-9, ICD-10) => batchpheno.icd9utils
#
######################################################################################


def longestCommonPrefix(s1, s2): 
    i = 0 
    while i < len(s1) and i < len(s2) and s1[i] == s2[i]: 
        i += 1
    return s1[:1]

def tokenize(string):
    """Convert string to lowercase and split into words (ignoring
    punctuation), returning list of words.
    """
    # '\w+' does not work well for codes with special chars such as '.' as part of the 'word'
    return re.findall(r'([-0-9a-zA-Z_:.]+)', string.lower())  

def simplify(docuemnts): 
    """
    Simply the coding sequences after reading from the source (e.g. .csv, .dat). 
    Note that the simply operation can also be applied during the read while this function 
    is applied after the fact. 

    Related
    -------
    seqReader.simplify

    """
    from batchpheno import icd9utils
    
    if not documents: 
        print('simply> warning: Empty set')
        return documents  # noop 

    assert isinstance(docuemnts[-1], list)
    for i, docuemnt in enumerate(docuemnts): 
        documents[i] = icd9utils.getRootSequence(docuemnt)  # this will not affect medication code e.g. MED:12345

    return documents

# a type of 'simplify' operations
def mergeVisits(documents): 
    return

def simlifyTime(timestamps):
    # todo
    return timestamps 

def find_ngrams(input_list, n=3):
    """
    Example
    -------
    input_list = ['all', 'this', 'happened', 'more', 'or', 'less']

    """
    return zip(*[input_list[i:] for i in range(n)])

def count_ngrams(lines, min_length=1, max_length=4): 
    """
    Iterate through given lines iterator (file object or list of
    lines) and return n-gram frequencies. The return value is a dict
    mapping the length of the n-gram to a collections.Counter
    object of n-gram tuple and number of times that n-gram occurred.
    Returned dict includes n-grams of length min_length to max_length.

    Use this only when (strict) ordering is important; otherwise, use count_ngrams2()

    Input
    -----
    lines: [['x', 'y', 'z'], ['y', 'x', 'z', 'u'], ... ]
    """
    def add_queue():
        # Helper function to add n-grams at start of current queue to dict
        current = tuple(queue)
        for length in lengths:
            if len(current) >= length:  # count n-grams up to length those in queue
                ngrams[length][current[:length]] += 1  # ngrams[length] => counter
    def eval_sequence_dtype(): 
        if not lines: 
            return False # no-op
        if isinstance(lines[0], str): 
            return False
        elif hasattr(lines[0], '__iter__'): 
            return True
        return False

    lengths = range(min_length, max_length + 1)
    ngrams = {length: collections.Counter() for length in lengths}
    queue = collections.deque(maxlen=max_length)

    # tokenized or not? 
    is_tokenized = eval_sequence_dtype()
    # print('> tokenized? %s' % is_tokenized)

    # Loop through all lines and words and add n-grams to dict
    if is_tokenized: 
        # print('input> lines: %s' % lines)
        for line in lines:
            for word in line:
                queue.append(word)
                if len(queue) >= max_length:
                    add_queue()  # this does the counting
            # print('+ line: %s\n+ngrams: %s' % (line, ngrams))
    else: 
        for line in lines:
            for word in tokenize(line):
                queue.append(word)
                if len(queue) >= max_length:
                    add_queue()

    # Make sure we get the n-grams at the tail end of the queue
    while len(queue) > min_length:
        queue.popleft()
        add_queue()
        # print('+ line: %s\n+ngrams: %s' % (line, ngrams))

    return ngrams

def count_ngrams2(lines, min_length=2, max_length=4, **kargs): 
    def eval_sequence_dtype(): 
        if not lines: 
            return False # no-op
        if isinstance(lines[0], str): # ['a b c d', 'e f', ]
            return False
        elif hasattr(lines[0], '__iter__'): # [['a', 'b'], ['c', 'd', 'e'], ]
            return True
        return False

    is_partial_order = kargs.get('partial_order', True)
    lengths = range(min_length, max_length + 1)    
    
    # is_tokenized = eval_sequence_dtype()
    seqx = []
    for line in lines: 
        if isinstance(line, str): # not tokenized  
            seqx.append([word for word in tokenize(line)])
        else: 
            seqx.append(line)
    
    # print('count_ngrams2> debug | seqx: %s' % seqx[:5]) # list of (list of codes)
    if not is_partial_order:  # i.e. total order 
        # ordering is important

        # this includes ngrams that CROSS line boundaries 
        # return count_ngrams(seqx, min_length=min_length, max_length=max_length) # n -> counter (of n-grams)

        # this counts ngrams in each line independently 
        counts = count_ngrams_per_seq(seqx, min_length=min_length, max_length=max_length) # n -> counter (of n-grams)
        return {length: counts[length] for length in lengths}

    # print('> seqx:\n%s\n' % seqx)
    # print('status> ordering NOT important ...')
    
    counts = {}
    for length in lengths: 
        counts[length] = collections.Counter()
        # ngrams = find_ngrams(seqx, n=length)  # list of n-grams in tuples
        if length == 1: 
            for seq in seqx: 
                counts[length].update([(ugram, ) for ugram in seq])
        else: 
            for seq in seqx:  # use sorted n-gram to standardize its entry since ordering is not important here
                counts[length].update( tuple(sorted(ngram)) for ngram in find_ngrams(seq, n=length) ) 

    return counts

def count_ngrams_per_line(**kargs): 
    return count_ngrams_per_seq(**kargs)
def count_ngrams_per_seq(lines, min_length=1, max_length=4): # non boundary crossing  
    """
    Similar to count_ngrams but consider each line as a separate document. 

    """
    def update(ngrams):
        # print('> line = %s' % single_doc)
        for n, counts in ngrams.items(): 
            # print('  ++ ngrams_total: %s' % ngrams_total)
            # print('      +++ ngrams new: %s' % counts)
            ngrams_total[n].update(counts)
            # print('      +++ ngrams_total new: %s' % ngrams_total)

    lengths = range(min_length, max_length + 1)
    ngrams_total = {length: collections.Counter() for length in lengths}

    doc_boundary_crossing = False  # do not allow boundary crossing
    if not doc_boundary_crossing: # don't count n-grams that straddles two documents
        for line in lines: 
            nT = len(line)
            # print(' + line=%s, nT=%d' % (line, nT))
            single_doc = [line]

            # if the line length, nT, is smaller than max_length, will miscount
            ngrams = count_ngrams(single_doc, min_length=1, max_length=min(max_length, nT))
            update(ngrams) # update total counts
    else: 
        raise NotImplementedError

    return ngrams_total

def count_doc_freq(ngram, D, sep=' '): # expensive
    nD = len(D)
    n_matched = 0
    for doc in D:
        ngs = sep.join(ng) 
        ds = sep.join(doc)
        if ds.find(ngs) >= 0: 
            n_matched += 1  
    return n_matched/(nD+0.0)       

def iterative_levenshtein(s, t, costs=(1, 1, 1)):
    """ 
        iterative_levenshtein(s, t) -> ldist
        ldist is the Levenshtein distance between the strings 
        s and t.
        For all i and j, dist[i,j] will contain the Levenshtein 
        distance between the first i characters of s and the 
        first j characters of t
        
        costs: a tuple or a list with three integers (d, i, s)
               where d defines the costs for a deletion
                     i defines the costs for an insertion and
                     s defines the costs for a substitution
    """
    rows = len(s)+1
    cols = len(t)+1
    deletes, inserts, substitutes = costs
    
    dist = [[0 for x in range(cols)] for x in range(rows)]
    # source prefixes can be transformed into empty strings 
    # by deletions:
    for row in range(1, rows):
        dist[row][0] = row * deletes
    # target prefixes can be created from an empty source string
    # by inserting the characters
    for col in range(1, cols):
        dist[0][col] = col * inserts
        
    for col in range(1, cols):
        for row in range(1, rows):
            if s[row-1] == t[col-1]:
                cost = 0
            else:
                cost = substitutes
            dist[row][col] = min(dist[row-1][col] + deletes,
                                 dist[row][col-1] + inserts,
                                 dist[row-1][col-1] + cost) # substitution
    for r in range(rows):
        print(dist[r])
    
 
    return dist[row][col]

def is_subseq(query, base): 
    return isSubsequence(query, base)
def isSubsequence(query, base):
    """
    is the query sequence a subsequence of base, where 
    subsequence can be either contiguous or not (but order-preserved).
    
    e.g. 'indonesia' contains 'india': INDonesIA

    """ 
    it = iter(base)

    # it is an iterator therefore each visited elment will not be re-visited again (i.e. consumed)
    return all(e in it for e in query)  # each element in query string must also occur (in orderly fashion) in base string

def t_isSubsequence(str1,str2,m,n):  # iterative 
    """
    Returns true if str1 is a subsequence of str2
    m is length of str1, n is length of str2
    """
    j = 0   # Index of str1
    i = 0   # Index of str2
     
    # Traverse both str1 and str2
    # Compare current character of str2 with 
    # first unmatched character of str1
    # If matched, then move ahead in str1
    while j<m and i<n:
        if str1[j] == str2[i]:  
            j = j+1
        i = i + 1
         
    # If all characters of str1 matched, then j is equal to m
    return j==m

def t_isSubsequence2(string1, string2, m, n): # recursive
    # Base Cases
    if m == 0:    return True
    if n == 0:    return False
 
    # If last characters of two strings are matching
    if string1[m-1] == string2[n-1]:
        return t_isSubsequence2(string1, string2, m-1, n-1)
 
    # If last characters are not matching
    return t_isSubsequence2(string1, string2, m, n-1)

def isSubsequence2(s1, s2): # have to 'look up' or compute lengths in every call
    m, n = len(s1), len(s2)
    if m == 0: return True 
    if n == 0: return False 

    if s1[m-1] == s2[n-1]: 
    	return isSubsequence2(s1[:m-1], s2[:n-1])
    return isSubsequence2(s1[:m], s2[:n-1])

def traceSubsequence(s1, s2, allow_partial=False): 
    
    # lengths 
    m, n = len(s1), len(s2) 

    j = 0   # Index of s1
    i = 0   # Index of s2
    
    positions = []
    while j<m and i<n:
        if s1[j] == s2[i]:  
            positions.append(i)  # store the matching positions of the target
            j = j+1
        i = i + 1
         
    # If all characters of str1 matched, then j is equal to m
    if j==m: 
    	return positions
    else: 
        if allow_partial: 
            return positions
    return None

def traceSubsequence2(s1, s2, allow_partial=False, no_match_index=-1): 
    matched_indices = traceSubsequence(s1, s2, allow_partial=allow_partial)
    tSub = len(matched_indices) == len(s1) # a subsequence? i.e. all matched? 
    if not tSub: 
        # for those not matched, return -1 as matching indices 
        matched_indices += [no_match_index for _ in range(len(s1)-len(matched_indices))]
    
    return (tSub, matched_indices)

def traceSubsequence3(s1, s2): 
    """
    Find all positions in the whole sequence (s2) that match the given subsequence (s1). 

    Unlike traceSubsequence2, all elements in s1 have to find their matches to count (i.e. no partial match)

    Use
    ---
    1. Find corresponding timestamps where 's1' as an LCS occur in the entire sequence (s2)

    """
    matched_indices = traceSubsequence(s1, s2, allow_partial=False)
    # tSub = len(matched_indices) == len(s1) # a subsequence? i.e. all matched? 
    
    matched_idx = []  # the entire set of matching indices; format: [[], []]
    while matched_indices is not None: 
        matched_idx.append(matched_indices)
        offset = matched_indices[0] 
        prefix_len = offset+1 
        matched_indices = traceSubsequence(s1, s2[prefix_len:], allow_partial=False)
        if matched_indices is not None: 
            matched_indices = list(np.array(matched_indices) + prefix_len)      
    return matched_idx

def sortedSequence(s, sep=' ', reverse=False): 
    """
    Input
    -----
    a string of symbols (usu. codes) delimited by 'sep'

    Output
    ------
    a string of sorted symobol

    Use
    ---
    Find the equivalence string up to permutation
    """
    slist = sorted(s.split(sep), reverse=reverse)
    return sep.join(slist)

### Template/Test Functions ### 

def t_preprocessing(): 
    line = '496 492.8 496 492.8 496 492.8 496 CDC:123459000 MED:7015 MULTUM:127 unknown poison-drug WRONGCODE:172.5'
    tokens = tokenize(line)
    print('string: %s' % line)
    print('tokens: %s' % tokens)
 
    return

def t_count_ngrams(): 
    from itertools import chain 
    lines = [['a', 'x', 'y', 'z'], ['x', 'y', 'z'], ['z', 'y', 'x', 'u'], ['x', 'y'], ['z', 'y', 'u', 'x'], ['x', 'a', 'x', 'y', 'b']]
    ngrams = count_ngrams(lines, min_length=1, max_length=5)
    print('> ngrams frequency:\n%s\n' % ngrams)

    ngrams = count_ngrams2(lines, min_length=1, max_length=5, partial_order=True)
    print('> ngrams frequency (unordered):\n%s\n' % ngrams)


    tokens = list(chain.from_iterable(lines))
    print('> tokens:\n%s\n' % tokens)

    seq = ['A', 'C', 'G', 'T', 'A', 'C', 'A', 'T', 'C', 'G', 'C', 'T']
    n = 3
    print('> sequence of %d-gram:\n%s\n' % (n, find_ngrams(seq, n=n)))

    ngrams = [('x', 'y'), ('u', 'x'), ('u', 'v'), ('x', 'c'), ('a', 'x'), 'x', ('a', 'x', 'x', 'y'), ('u', 'x', 'y', 'z'), ('x', 'y', 'z', 'u'), ('z', 'y', 'x'), ]
    counts = count_given_ngrams2(lines, ngrams, partial_order=True)
    print counts

    counts = count_given_ngrams2(lines, ngrams, partial_order=False)    
    print counts

    # test the summing of frequencies
    seqx1 = [['a', 'x', 'y', 'z'], ['x', 'y', 'z'], ['z', 'y', 'x', 'u']]
    seqx2 = [['x', 'y'], ['z', 'y', 'u', 'x'], ['y', 'y'], ['x', 'a', 'x', 'y', 'b']]
    ngrams = [('x', 'y'), ('u', 'v'), 'x', ('a', 'x', 'x', 'y'), ('u', 'x', 'y', 'z'), ('z', 'y', 'x'), ]
    counts1 = count_given_ngrams2(seqx1, ngrams, partial_order=True)
    counts2 = count_given_ngrams2(seqx2, ngrams, partial_order=True)

    print('> counts1: %s' % counts1)
    print('> counts2: %s' % counts2)

    print "\n"

    seqx1 = [ ['a', 'x', 'y', 'z', ], ['x', 'y', 'z'], ['x', 'y', 'z'], ['x', 'y', 'z'], ['z', 'y', 'x', 'u'], ['z', 'x'], ['y', 'y'], ] 
    counts1 = count_ngrams2(seqx1, min_length=1, max_length=10, partial_order=False)
    print counts1 
    # [log]
    # partial ordering or ordering not important 
    # bigrams: {('x', 'y'): 3, ('y', 'z'): 3, ('u', 'x'): 1, ('a', 'x'): 1, ('y', 'y'): 1, ('x', 'z'): 1}
    # 4-grams: {('u', 'x', 'y', 'z'): 1, ('a', 'x', 'y', 'z'): 1}}
    # crossing boundary? no

    # strict ordering 
    
    # use count_ngram()
    # 4-grams {('y', 'z', 'z', 'y'): 1, ('x', 'y', 'z', 'z'): 1, ('y', 'z', 'x', 'y'): 1, ('z', 'x', 'y', 'y'): 1, ('a', 'x', 'y', 'z'): 1, ('x', 'u', 'z', 'x'): 1, ('z', 'x', 'y', 'z'): 1, ('z', 'z', 'y', 'x'): 1, ('u', 'z', 'x', 'y'): 1, ('y', 'x', 'u', 'z'): 1, ('x', 'y', 'z', 'x'): 1, ('z', 'y', 'x', 'u'): 1})
    # crossing boundary? yes

    # use count_ngram_per_doc()  
    # bigrams: {('x', 'y'): 2, ('y', 'z'): 2, ('a', 'x'): 1, ('z', 'x'): 1, ('y', 'x'): 1, ('z', 'y'): 1, ('y', 'y'): 1, ('x', 'u'): 1}
    # 4-grams: {('a', 'x', 'y', 'z'): 1, ('z', 'y', 'x', 'u'): 1})

    # {('u', 'x', 'y', 'z'): 1, ('a', 'x', 'y', 'z'): 1}
    return

def t_subsequence(**kargs): 

    ### is xy a subsequence of xaaazzzzyy? 
    t_predicate(**kargs)

    return

def t_predicate(**kargs): 
    from batchpheno.utils import div
    q = 'india'
    r = 'indonesia'

    q2 = ['123.5', '374.7', 'J23', ]
    r2 = ['123.5', 'X27.1', '374.7', '334.7', '111', 'J23', '223.4']

    q3 = ['123.5', '374.7', 'J23', ]
    r3 = ['123.5', 'X27.1', '334.7', '111', 'J23', '223.4', '374.7',]

    q4 = ['123.5', '374.7', 'J23', ]
    r4 = ['374.7', 'J23', '123.5',]

    q5 = ['123.5', '374.7', 'J23', ]
    r5 = ['123.5', '374.7',]

    q6 = ['123.5', '374.7',]
    r6 = ['123.5', '374.7', 'J23', ]

    q7 = ['J24', '123.5', '374.7',]
    r7 = ['123.5', '374.7', 'J23', ]

    q8 = ['720.0', ]
    r8 = ['123.5', 'MED:12345', '720.0', '491.0', '720.0', '720.0', '111.1']

    allow_partial_match = True
    for (q, r) in [(q, r), (q2, r2), (q3, r3), (q4, r4), (q5, r5), (q6, r6), (q7, r7), (q8, r8) ]: 
        print "> %s is a subseq of %s? %s" % (q, r, isSubsequence(q, r))

        tMatched0 = isSubsequence(q, r)
        assert tMatched0 == isSubsequence2(q, r), "recursive version not consistent"

        # indices = traceSubsequence(q, r, allow_partial=allow_partial_match)
        tMatched, indices = traceSubsequence2(q, r, allow_partial=allow_partial_match)
        assert tMatched == tMatched0
        print("  + all matched? %s > matched positions: %s" % (tMatched, indices))

        if indices is not None: 
            if isinstance(r, str): 
                rp = ''.join([str(r[i]) for i in indices])
            else: 
        	    rp = [r[i] for i in indices]

            if not allow_partial_match: 
                assert rp == q, "%s != %s" % (rp, q)
    
    # but we need to know which indices are matches (e.g. in order to cross reference their times)
    div(message='What if we need all matching positions?')     

    q9 = ['720.0', '123.5']
    r9 = ['720.0', '123.5', 'x', 'x', 'y', '720.0', 'z', 'z', '123.5', 'x', '720.0', '123.5', 'y', '720.0']

    # [[0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15], [16, 17, 19-21,22,23,25, ]]
    q10 = r9
    r10 = ['720.0', '123.5', 'x', 'x', 'y', 'new', 
           '720.0', 'z', 'z', '123.5', 'x', 'new', 
           '720.0', '123.5', 'y', '720.0', 
           'x', 'x', 'x', 
           '720.0', '123.5', 'new', 'x', 'x', 'y', '720.0', 'z', 'new', 
           'z', '123.5', 'x', '720.0', '123.5','y', 'y', '720.0', 'z'] 


    ### with repeated patterns (including overlap)
    matched_idx = []
    for (q, r) in [(q2, r2), (q8, r8), (q9, r9), (q7, r7), (q10, r10), ]: # is-in, is-in, is-in, not 
        matched_idx = traceSubsequence3(q, r)
        if len(matched_idx) > 0: 
            assert isSubsequence(q, r)
            print('  + query seq: %s is in ref seq: %s' % (q, r))
            print('      + matched idx: %s' % matched_idx)
            for midx in matched_idx: 
               print('    > %s' % list(np.array(r10)[midx]))
        else: 
            print('  + query seq: %s is NOT in ref seq: %s' % (q, r))
            print('      + matched idx: %s' % matched_idx)

    print('-------------')

    q = ['x', 'y', 'z']
    r = ['x', 'a', 'x', 'a', 'y', 'b', 'z']
    matched_idx = traceSubsequence3(q, r)
    print('  + q: %s' % q)
    print('  + r: %s' % r)
    print('  => %s' % matched_idx)

    return

def t_predicate2(): 
    q = 'MED:62439 MED:60926 MED:62934 MED:122364 MED:62934'
    qs = q.split()
    rs = ['784.0', '788.20', '789.09', '788.63', '600.01', '788.41', '788.43', 'NDC:00093075201', 
         'MED:61112', 'MED:60884', 'MED:61124', 'MED:61165', 'MED:61968', 'MED:99143', 'MED:62439', 
         'MED:60926', 'MED:61505', 'MED:62787', 'NDC:00078049115', 'NDC:00247070500', 'NDC:00597005801', 
         'MED:63517', 'MED:61895', 'MED:60592', 'MED:60884', 'MED:62787', 'MED:100198', 'MED:62934', 'MED:62564', 
         'NDC:00247070500', 'NDC:00078049115', 'NDC:00093075201', 'MED:61124', 'MED:61165', 'MED:99143', 'MED:61785', 
         'MED:61939', 'MED:158045', 'MED:61968', 'MED:62934', 'MED:103229', 'MED:167651', 'NDC:00597005801', 'MED:122364', 
         'MED:61319', 'MED:62363', 'MED:62363', 'MED:63356', 'MULTUM:2877', 'MULTUM:2200', 'NDC:00069154041', 'MED:62934', 
         'MULTUM:7247', 'MULTUM:5625', 'MULTUM:5922']
    r = ' '.join(rs)

    for (q, r) in [(qs, rs), ]: 
        print('> (q): %s' % q)
        print('> (r):\n   %s\n' % r) 
        print('  matched? %s' % isSubsequence(qs, r))
    return

def t_edit_distance(): 
    s = 'V70.0 401.9 311 789.00 300.00 729.5 300.00 531.90 276.1 311 789.00 401.9 715.90 401.9'.split()
    t = '311 789.00 300.00 311 V15.82 311 401.9 305.1 401.9'.split()
    iterative_levenshtein(s, t)

    return 

def t_peptide(): 
    from toposort import toposort, toposort_flatten
    import graph

    # construct DAG 

    pdag = {}
    pdag['I0'] = ['ep', ]
    pdag['I1'] = ['I0', ]   # self transition 
    pdag['C1'] = ['I1', ]
    pdag['V1'] = ['I2', 'D1', 'I4', 'V2', 'I3', 'C1', 'I1', ]
    pdag['I2'] = ['D1', 'I4', 'V2', 'I3', 'C1', 'I1', ]
    pdag['I3'] = ['C1', 'I1']
    pdag['D1'] = ['I4', 'V2', 'I3', 'C1', 'I1', ]
    pdag['I4'] = ['V2', 'I3', 'C1', 'I1']
    pdag['L1'] = ['V1', 'I2', 'D1', 'I4', 'V2', 'I3', 'C1', 'I1']
    pdag['V2'] = ['I3', 'C1', 'I1'] 
    pdag['F1'] = ['L1', 'V1', 'I2', 'D1', 'I4', 'V2', 'I3', 'C1', 'I1']
    pdag['N1'] = ['I5','V1', 'I2', 'D1', 'I4', 'V2', 'I3', 'C1', 'I1',]
    pdag['P1'] = ['L2', 'I4', 'V2', 'I3', 'C1', 'I1']
    pdag['V3'] = ['I2', 'D1', 'I4', 'V2', 'I3', 'C1', 'I1', ]
    pdag['V4'] = ['P1', 'L2', 'I4', 'V2', 'I3', 'C1', 'I1', ]
    pdag['P2'] = ['L2', 'I4', 'V2', 'I3', 'C1', 'I1', ]
    pdag['L2'] = ['I4', 'V2', 'I3', 'C1', 'I1', ]
    pdag['I5'] = ['V1', 'I2', 'D1', 'I4', 'V2', 'I3', 'C1', 'I1', ]

    print('info> size(G): %d' % len(pdag))

    return 

def t_NCATS(): 

    # len = 64
    peptide = 'ICVIDLAVSISFNCLIRLPVVEGGIEDPVTCLKAGAICHIVFCPRRYKQIGVCGLPGTKCCKKP'  # target peptide with 18 residules replaced 
    rs = [(1, 'I'), (2, 'CI'), (3, 'VIDIVICI'), 
          (4, 'ICI'), (5, 'DIVICI'), 

          (6, 'LVIDIVICI'), 
          (12, 'FLVIDIVICI'), (13, 'NIVIDIVICI'), 

          (20, 'VIDIVICI'), (21, 'VPLIVICI'), 

          (64, 'PLIVICI'), 


          ]  # recognition signatures

    # 59 ~ 63, no recognition sigature  => K is a blocker? 

    assert peptide[20:] == 'VEGGIEDPVTCLKAGAICHIVFCPRRYKQIGVCGLPGTKCCKKP'
    startpos = True 

    for (si, s) in rs: 
        
        # pepsub = peptide
        # print("%s is a sub of %s? ... %s" % (s, pepsub, isSubsequence(s, pepsub)))

        if startpos: 
            if si < 6: 

                pepsub = peptide[si-1:] 
                print("%s is a sub of %s? ... %s" % (s, pepsub, isSubsequence(s, pepsub)))
            else: 
                # print('+ 1. preserving order > ')
                pepsub = peptide[:si]  # including the last  
                # print("%s is a sub of %s? ... %s" % (s, pepsub, isSubsequence(s, pepsub)))

                # pepsub = peptide[:si] 
                pepsub2 = ''.join([e for e in reversed(pepsub)])
                # print('+ 2. reversing order >')
                print("%s is a sub of %s? ... %s" % (s, pepsub2, isSubsequence(s, pepsub2)))

                if si == 6:   
                    s2 = 'VIDIVICI' # remove L   
                    print("  + %s is a sub of %s? ... %s" % (s2, pepsub2, isSubsequence(s2, pepsub2)))

                if si == 12: 
                    s2 = 'VIDIVICI'  # remove FL
                    print("  + %s is a sub of %s? ... %s" % (s2, pepsub2, isSubsequence(s2, pepsub2)))

                if si == 13: 
                    s2 = 'VIDIVICI'  # remove NI
                    print("  + %s is a sub of %s? ... %s" % (s2, pepsub2, isSubsequence(s2, pepsub2)))

    return 

def t_multiway_lcs(**kargs): 
    import lcs # another module (supposedly) underneath seqmaker

    # template function 
    lcs.t_lcs(**kargs)

    return

def test(): 

    ### preprocessing documents, texts
    # t_preprocessing()

    ### predicates 
    # t_predicate()
    t_predicate2()

    ### enumerate all possible n-grams 
    # t_count_ngrams()

    # string matching applications 
    # t_NCATS()
    # t_peptide()

    return

if __name__ == "__main__": 
    test()

