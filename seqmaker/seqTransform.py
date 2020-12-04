import os, sys, re, random, collections

# install tsne via pip
# from tsne import bh_sne
import numpy as np

import pandas as pd 
from pandas import DataFrame

# temporal sequence modules
from config import seq_maker_config, sys_config
from batchpheno.utils import div
from batchpheno import icd9utils

import seqparams # global control parameters for sequence creation and analysis

try:
    import cPickle as pickle
except:
    import pickle

# disease-specific module
from pattern import diabetes, ptsd, ckd
from pattern import medcode as pmed

from datetime import datetime
import random, os, re

# for impl (approximate) phenotyping algorithm based on 
# from seqmaker import seqAnalyzer as sa
# from seqmaker import seqReader as sr

p_diabetes = re.compile(r"(?P<base>250)\.(?P<subclass>[0-9]{1,2})")
p_secondary_diabetes = re.compile(r"(?P<base>249)\.(?P<subclass>[0-9]{1,2})")
p_gestational_diabetes = re.compile(r"(?P<base>648)\.(?P<subclass>[0-9]{1,2})")

def is_PTSD(code): # [old]
    scode = str(code)
    if scode in ('309.81', ): 
        return True 
    if scode[:3] in ('309', ): # 309.81
    	return True 
    return False	
def is_diabetes(code): 
    scode = str(code)
    if scode[:3] in ('249', '250', '648', ): 
    	return True 
    return False	

def is_CKD(code): 
    """
    585 5851 5852 5853 5854 5855 5856 5859 7925 V420 V451 V4511 V4512 V560 V561 V562 V5631 V5632 V568 
    """
    raise NotImplementedError
    # return False 

def getDiseasePredicate(cohort='unknown'):
    # from pattern import diabetes, ptsd  # and possibly other diseases 
    if not isinstance(cohort, str): # None 
        return None

    cohort_name = cohort.lower()
    if cohort_name.startswith('pt'):
        return ptsd.isCase
    elif cohort_name.startswith('diab'):
        return diabetes.is_diabetes 
    elif cohort_name.startswith('ckd'):
        return ckd.isCase 
    # else: 
    #     raise NotImplementedError, "Predicate for cohort=%s is not yet supported." % cohort_name.capitalize()
    else: 
        print("Predicate for cohort=%s is not yet supported." % cohort_name) # cohort_name.capitalize()
    return None

def removePrefix(D, regex='^I(10|9):', inplace=True): 
    """

    Memo
    ----
    1. to remove both med and diag prefixes, set regex to:  
           '^(I(10|9):)|(^(MED|MULTUM|NDC):)'

    """
    from functools import partial
    # newer format for diagnosic codes in odhsi DB admits prefixes such as I9, I10
    if inplace: 
        for i, doc in enumerate(D):  # D: a list of list of tokens 
            # D[i] = map(removeDiagPrefix(x, test_=True), doc) # 
            D[i] = map(partial(remove_prefix, regex=regex), doc)  # inplace
    else: 
        Dp = []
        for i, doc in enumerate(D):  # D: a list of list of tokens 
            # D[i] = map(removeDiagPrefix(x, test_=True), doc) # 
            Dp.append(map(partial(remove_prefix, regex=regex), doc))  # inplace
        return Dp
    return D
def remove_prefix(x, regex='^I(10|9):'):
    return re.sub(regex, '', x)

# def slice(docs, items=None, **kargs): 
#     return 

def modify(docs, seq_ptype='diag', predicate=None):  # [params] labels
    """
    Transform documents: 

    1. simply coding 
    2. preserve only needed contents (e.g. diagnostic codes only)

    Params
    ------
    simplify_code 
    filter_code

    """ 
    # import seqAlgo

    # [note] the 'analyze' step can be performed independently
    seq_ptype = seqparams.normalize_ctype(seq_ptype) # sequence pattern type: regular, random, diag, med, lab
    # tSimplifyCode = kargs.get('simplify_code', False)
    tFilterContent = False if seq_ptype == 'regular' else True
    tRemoveEmpty = True

    # if tSimplifyCode: 
    #     print('    + simply the codes')
    #     docs = seqAlgo.simplify(docs)  # this will not affect medication code e.g. MED:12345
    # no effect on labels

    ### [control] train specific subset of codes (e.g. diagnostic codes only)
    print('modify> select code according to ctype or predicate | ctype=%s' % seq_ptype)
    # tseqx = kargs.get('timestamps', [])
    if tFilterContent: # really means filter the "content" of each documents
        # precedence: predicate > seq_ptype
        pos_map = filterCodes(docs, seq_ptype=seq_ptype, predicate=predicate)  # if predicate is given, seq_ptype is ignored
        docs = indexToDoc(docs, pos_map) # pos_map: a dictoinary > for i-th document, preserve only positions in 'idx' 
        # condition: docs in 2D np.array

    return docs  # output made consistent with loadDocuments? 
def parallellModify(docs, items, seq_ptype='regular', predicate=None): 
    """
    Similar to the content filtering functionality in transform() but filter 
    both coding sequences and their corresponding timestamps in parallel. 

    Input
    -----
    docs: 2D array of documents (where each doc is a list of tokens/strings)
    items: any objects (e.g. timestamps) in parallel (and having a 1-to-1 relationship) with the input documents
           e.g. suppose items <- timestamps
                and only diagnostic codes are preserved, 
                then will fitler all positions (within each document) that point to medicinal codes (e.g. MED:12345)
                => also remove their timestamps 

    Params
    ------
    seq_ptype
    predicate

    """
    if len(items) == 0: 
        print('parallellModify> No items given! Operation reduced to modify()')
        return modify(docs, **kargs)

    seq_ptype = seqparams.normalize_ctype(seq_ptype) # sequence pattern type: regular, random, diag, med, lab
    pos_map = filterCodes(docs, seq_ptype=seq_ptype, predicate=predicate)  # if predicate is given, seq_ptype is ignored
    assert len(pos_map) == len(docs), "number of documents is not preserved: %d => %d" % (len(docs), len(pos_map))
    new_docs = indexToDoc(docs, pos_map) # pos_map: a dictoinary > for i-th document, preserve only positions in 'idx' 
    new_items = indexToDoc(items, pos_map)
    # condition: data type of new_docs, new_items is preserved

    assert len(new_items) == len(new_docs), "Inconsistent sequence lengths: n_docs: %d <> n_items: %d" % (len(new_docs), len(new_items))
    return (new_docs, new_items)  # condition: docs in 2D np.array
def filterDocuments(D, **kargs):
    """
    Filter unwanted documents through a specified 'policy' (e.g. policy=empty_doc => filter empty documents)

    Params
    ------
    policy
       'empty' or 'empty_doc': remove empty documents
       'unknown': replace empty documents by default unknown token.

    """

    # for now, really just filtering out empty documents and their labels (which reduces sample size)
    policy = kargs.get('policy', 'empty_doc')  # {'empty_doc', 'minimal_evidence', 'unknown_padding'}

    minNCodes = 1
    if policy.startswith('min'): # then only retain documents with minimum supporting evidence 
        minNCodes = kargs.get('min_ncodes', 10)

    L = kargs.get('L', [])
    T = kargs.get('T', [])  # 

    nD0 = len(D)
    hasLabel = True if (L is not None) and len(L) > 0 else False 
    hasTime = True if (T is not None) and len(T) > 0 else False   # timestamps
    if hasLabel: assert nD0 == len(L)
    if hasTime: assert nD0 == len(T)

    D2, L2, T2 = [], [], []
    if policy.startswith(('empty', 'min')): 
        print('filterDocuments> Policy: Remove empty documents ...')
        for i, doc in enumerate(D): 
            if len(doc) >= minNCodes: 
                D2.append(doc)
                if hasLabel: L2.append(L[i])
                if hasTime: T2.append(T[i])
    elif policy.startswith('unknown'):  
        # used this to keep the equality: nrow(document source) == nrow(training set) 
        print('filterDocuments> Policy: Replace empty documents by default unknown token.')
        for i, doc in enumerate(D): 
            if len(doc) >= minNCodes: 
                D2.append(doc)
                if hasLabel: L2.append(L[i]) 
                if hasTime: T2.append(T[i])
            else: 
                D2.append([TDoc.token_unknown, ]) # pad token 'unknown'
                if hasLabel: L2.append(TDoc.label_unknown) # don't use original label 
                if hasTime: T2.append([TDoc.time_unknown, ])      

    else: 
        raise NotImplementedError, "Unknown doc filtering policy: %s" % policy 

    if hasLabel: assert len(D2) == len(L2)
    if hasTime: assert len(D2) == len(T2)
    return (D2, L2, T2) 

def sliceByTime(seq, tseq, timestamp, policy='prior', inclusive=True, t_format="%Y-%m-%d"): 
    """
    Similar to cut, sliceDocuments but the cut point here is determined by timestamp.  

    Input
    -----
    policy: {'prior', 'posterior', 'regular', }

    """
    t0 = datetime.strptime(timestamp, t_format) 
    
    S = T = None
    if policy == 'prior': 
        endpoint = -1
        for i, t in enumerate(tseq): # find max t s.t. t comes before t0
            if inclusive: 
                if datetime.strptime(t, t_format) <= t0: 
                    endpoint = i
            else: 
                if datetime.strptime(t, t_format) < t0: 
                    endpoint = i

        S = seq[:endpoint+1] # include endopint index
        T = tseq[:endpoint+1]

    elif policy == 'posterior': 
        startpoint = len(tseq)
        for i, t in enumerate(tseq): # find min t s.t. t comes after t0
            if inclusive: 
                if datetime.strptime(t, t_format) >= t0: 
                    startpoint = i
                    break # need the first one
            else: 
                if datetime.strptime(t, t_format) > t0: 
                    startpoint = i
                    break

        S = seq[startpoint:] # include endopint index
        T = tseq[startpoint:]
    else: # noop, keep complete sequence
        # noop 
        S = seq 
        T = tseq
    
    return (S, T)

def segmentDocumentByTime(D, L, T, timestamp, **kargs):
    """
    A variation of segmentDocuments that focus on segmenting input documents by 
    a timestamp. 
    """
    def summary():  # D, D2, policy, (n_active)
        tEarlist, tLatest = min(tMin), max(tMax)
        print('segmentDocumentByTime> Latest time stamp: %s' % tLatest)
        if policy in ('prior', 'posterior', ):
            r_active = len(D2)/(len(D)+0.0)  # n_active: has cutpoints and nS > 0 
            print('  + number of active documents: %d (/%d), ratio: %f' % (len(D2), len(D), r_active)) 

            ETimeSpan = sum(tHistory)/(len(tHistory)+0.0)
            StdTimeSpan = np.std(tHistory)
            print('segmentDocuments> Average time to first diagnosis (in days): %f (~ %f yrs)' % (ETimeSpan, (ETimeSpan/365.0)))
            print('                  + std: %f (days) or %f (years)' % (StdTimeSpan, (StdTimeSpan/365.0)))
            print('                  + num of empty docs: %d' % len(nullIDs))     
        return
    def add(i, seg, tseg): # D2, L2, T2, docIDs 
        docIDs.append(i)  # [memo] this works and does retain new elements in the outer scope
        D2.append(seg)
        T2.append(tseg)
        try: 
            L2.append(L[i])
        except: 
            pass
        return  
    def getTimeSpan(tseg):
        # [assumption]: tseg is sorted in ascending order
        t_min = datetime.strptime(tseg[0],  '%Y-%m-%d')  # string parse time 
        t_max = datetime.strptime(tseg[-1], '%Y-%m-%d')
        delta = (t_max-t_min).days
        return delta 
    def display(doc, segments, t_segments=None, show_doc=True, show_segments=True):  # <- policy
        nS = len(segments)
        if nS == 1: 
            if policy.startswith(('pri', 'post')): 
                print("  + Doc referecning a '%s' segment." % policy)
            else: 
                print('  + Doc referecning a single segment (policy=%s)' % policy)
        else: 
            print('  + Doc referecning multiple segments (n=%d, policy=%s)' % (nS, poilcy))
        length = len(doc)

        if show_doc: 
            print('    + [original]:\n%s\n' % abridge(doc, max_len=200))
        else: 
            # suppress the document display
            pass

        if show_segments: 
            for i, segment in enumerate(segments):
                print('       + [s #%d]: %s' % (i, abridge(segment, max_len=600)))
                if t_segments is not None: 
                    print('       + [t #%d]: %s' % (i, abridge(t_segments[i], max_len=600)))
        return
    def abridge(doc, max_len=200, sep=' '): 
        length = len(doc) 
        dp = doc
        midpoint = max_len/2
        if length > max_len: 
            # ', '.join(e for e in doc[:100])
            front = sep.join(e for e in doc[:midpoint])
            back = sep.join(e for e in doc[-midpoint:])
            dp = '[' + front + '...' + back + ']'
        return dp

    import numpy as np
    policy = kargs.get('policy', 'prior')
    
    docIDs, nullIDs = [], []  # new documents (segmented)
    D2, L2, T2 = [], [], []  # new timestamps (segmented)
    
    # policy values: {'regular', 'two', 'prior', 'posterior', 'complete', }
    tHistory = []
    tMax, tMin = [], []
    nullSegments = []

    n_active, n_null = len(D), 0
    timeToFirstDiag = []
    nullSegments = []  # documents without valid segments (even though cutpoints were found)
    noCutpoints = []  # documents without any cutpoints 
    
    tDropNullCut = kargs.get('drop_nullcut', True) # if True, include only documents with active cutpionts (with diagnosis info)
    tHasLabel = True if len(L) > 0 else False
    if tHasLabel: assert len(D) == len(L), "nD: %d <> nL: %d" % (len(D), len(L))

    if policy.startswith(('reg', 'no')):  # regular, noop 
        docIDs = range(0, len(D))
        print('segmentDocuments> Noop (policy=%s) > early return ...' % policy)
        return (docIDs, D, L, T)
    elif policy in ('prior', 'posterior', ): # prior or posterior
        keepNonActive = True  # keep the doc in which none of the position fires the predicate (e.g. no principle diagnosis present) 
        
        n_active = 0
        include_endpoint = kargs.get('inclusive', True) 
        for i, doc in enumerate(D):  
            # use the timestamp of the first occurrence 
            t_ref = timestamp  

            # [test]
            # if i < 100: test_segmenting(i, t_ref, policy)
            tMax.append(T[i][-1])
            tMin.append(T[i][0])

            seg, tseg = sliceByTime(seq=doc, tseq=T[i], timestamp=t_ref, policy=policy, inclusive=include_endpoint, t_format="%Y-%m-%d")
            assert len(seg) == len(tseg), "document segment length: %d but time segment length: %d" % (len(seg), len(tseg))

            nS = len(seg)  # non-empty segment
            if nS > 0: 
                add(i, seg, tseg)  # add new data to D2, L2, T2
                tHistory.append(getTimeSpan(tseg))
                if len(D2) <= 10:   
                    display(doc, [seg,], [tseg, ])
            else: # empty segment
                # keep the doc anyway
                if tDropNullCut: 
                    pass 
                else: 
                    add(i, doc, T[i])
                    nullIDs.append(i)
                    tHistory.append(getTimeSpan(T[i]))
    else: 
        raise NotImplementedError

    summary()
    return (docIDs, D2, L2, T2)

def segmentDocuments2(D, L, T, predicate, **kargs): 
    docIDs, Dp, Tp = segmentDocuments(D, T, predicate=predicate, **kargs)

    # organize labels
    Lp = [] 
    for i in docIDs: 
        Lp.append(L[i])

    return (docIDs, Dp, Lp, Tp)

def segmentDocuments(D, T, predicate, **kargs):
    """
    Segment input documents, D, according to policy on timestamps, T

    Params
    ------
    predicate 
       e.g. pattern.ckd.isCase(code)

    policy 
       a. two, two-halves, halves: segment the entire sequence into two halves with a split point determined by predicate
       b. prior, posterior
       c. regular
       d. complete (complete sets of segments, each of which is a subsequence leading up to the position that fires the predicate)

    inclusive: works only when policy in {'prior', 'posterior'}


    """
    def summary():  # D, D2, policy, (n_active)
        # among all documents (D), how many of them have mutliple cutpoints (e.g. multiple episodes of principle diagnosis)? 
        print('segmentDocuments> policy=%s' % policy)
        nD = len(D)
        n_multi_active = sum(1 for cutpoints in cutpointx if len(cutpoints) > 1)
        r = nD/(n_multi_active+0.0)
        print('  + number of documents with multiple active positions: %d, ratio: %f' % (n_multi_active, r))

        nD2 = len(D2)
        # assert nD2 >= nD # this is not necessarily true, a subset of the D may not have valid segments
        print('  + segment policy (%s) changes nD from %d to %d' % (policy, nD, nD2))

        if policy in ('prior', 'posterior', ):
            r_active = n_active/(nD+0.0)  # n_active: has cutpoints and nS > 0 
            print('  + number of active documents with non-empty segments: %d, ratio: %f' % (n_active, r_active)) 

            ETimeSpan = sum(timeToFirstDiag)/(len(timeToFirstDiag)+0.0)
            StdTimeSpan = np.std(timeToFirstDiag)
            print('segmentDocuments> Average time to first diagnosis (in days): %f (~ %f yrs)' % (ETimeSpan, (ETimeSpan/365.0)))
            print('                  + std: %f (days) or %f (years)' % (StdTimeSpan, (StdTimeSpan/365.0)))
            print('                  + num of docs without valid segments even WITH cutpoints: %d' % len(nullSegments))

        nNullCut = len(noCutpoints)
        if nNullCut: 
            print('  + n_docs without any cutopints: %d (containing no diagnosis info)' % nNullCut)
            print('  + examples:\n')
            Sc = random.sample(noCutpoints, min(nNullCut, 5))
            for i in Sc: 
                doc = D[i]
                length = len(doc)
                if length < 201: 
                    print('    + [dodID=%d]:\n%s\n' % (i, doc))
                else: 
                    print('    + [docID=%d]:\n%s\n' % (i, str(doc[:100]) + '...' + str(doc[-100:])))
                    
        return
    def display(doc, segments, t_segments=None, show_doc=True, show_segments=True):  # <- policy
        nS = len(segments)
        if nS == 1: 
            if policy.startswith(('pri', 'post')): 
                print("\n  >>> Doc referecning a '%s' segment." % policy)
            else: 
                print('\n  + Doc referecning a single segment (policy=%s)' % policy)
        else: 
            print('\n  >>> Doc referecning multiple segments (n=%d, policy=%s)' % (nS, poilcy))
        length = len(doc)

        if show_doc: 
            print('    ... [original]:\n%s\n' % abridge(doc, max_len=200))
        else: 
            # suppress the document display
            pass

        if show_segments: 
            for i, segment in enumerate(segments):
                print('       ... [s #%d]: %s' % (i, abridge(segment, max_len=600)))
                if t_segments is not None: 
                    print('       ... [t #%d]: %s' % (i, abridge(t_segments[i], max_len=600)))
        return
    def abridge(doc, max_len=200, sep=' '): 
        length = len(doc) 
        dp = doc
        midpoint = max_len/2
        if length > max_len: 
            # ', '.join(e for e in doc[:100])
            front = sep.join(e for e in doc[:midpoint])
            back = sep.join(e for e in doc[-midpoint:])
            dp = '[' + front + '...' + back + ']'
        return dp
    def test_segmenting(i, t_prior, policy): 
        print('\n  *** (test) *** testing document #%d ...' % i)
        seg, tseg = sliceByTime(seq=D[i], tseq=T[i], timestamp=t_prior, policy=policy, inclusive=True, t_format="%Y-%m-%d")
        nSegInclude = len(seg)
        
        print('-' * 80)
        display(D[i], [seg,], [tseg, ])
        print('         ... including the end point: length=%d' % nSegInclude)
        
        seg, tseg = sliceByTime(seq=D[i], tseq=T[i], timestamp=t_prior, policy=policy, inclusive=False, t_format="%Y-%m-%d")
        nSegExclude = len(seg)
        
        print('-' * 80)
        display(D[i], [seg,], [tseg, ], show_doc=False)
        print('         ...... excluding the end point: length=%d' % nSegExclude)

        assert nSegInclude >= nSegExclude
        print('\n  *** (test) *** test document #%d\n' % i)
        return
    def addDoc(i, seg, tseg=None): # D2, docIDs 
        docIDs.append(i)  # [memo] this works and does retain new elements in the outer scope
        D2.append(seg)
        if tseg is not None: 
            T2.append(tseg)
        return 
    def addTime(seg):
        T2.append(seg) 
        return 
    def getTimeSpan(tseg):
        # [assumption]: tseg is sorted in ascending order
        t_min = datetime.strptime(tseg[0],  '%Y-%m-%d')  # string parse time 
        t_max = datetime.strptime(tseg[-1], '%Y-%m-%d')
        delta = (t_max-t_min).days
        return delta 
    def hasActive(doc): 
        for j, tok in enumerate(doc):
            if predicate(tok):
                return True 
        return False

    import numpy as np
    assert hasattr(predicate, '__call__'), "Invalid predicate."
    policy = kargs.get('policy', 'two')
    
    # policy values: {'regular', 'two', 'prior', 'posterior', 'complete', }
    if policy.startswith(('reg', 'no')):  # regular, noop 
        docIDs = range(0, len(D))
        print('segmentDocuments> Noop (policy=%s) > early return ...' % policy)
        return (docIDs, D, T)

    D2, docIDs = [], []  # new documents (segmented)
    T2 = []  # new timestamps (segmented)
    cutpointx = [] # same dim as D, D2 (D')
    for i, doc in enumerate(D):
        idx = []
        for j, tok in enumerate(doc):
            if predicate(tok):  # the predicate takes a token in the sequence as an argument
                idx.append(j)
        cutpointx.append(idx)  # diagnosis active positions

    n_active = len(D)
    timeToFirstDiag = []
    nullSegments = []  # documents without valid segments (even though cutpoints were found)
    noCutpoints = []  # documents without any cutpoints 
    tDropNullCut = kargs.get('drop_nullcut', True) # if True, include only documents with active cutpionts (with diagnosis info)
    
    if policy.startswith(('two', 'hal', )):  
        # separate the input document into two halves, one spanning from the start to the index (i) at which the predicate is true (e.g. a diagnosis)
        # and the other spanning from index i+1 until the end
        
        minNPoints = 1  # segment the documents in halves only when there exists more than N cut points
        nMulti = 0 
        for i, doc in enumerate(D): 
            if len(cutpointx[i]) >= minNPoints:
                timestamps = [T[i][cutpoint] for cutpoint in cutpointx[i]]

                # use the timestamp of the first occurrence 
                timestamps = timestamps[:1]

                segments, tSegments = segmentByTime(seq=doc, tseq=T[i], timestamps=timestamps, t_format="%Y-%m-%d")
                assert len(segments) <= len(timestamps), \
                    "n_segments(%d) cannot exceed n_times(%d), some of which may be invalid" % (len(segments), len(timestamps))

                # input document (doc) is broken down into multiple segments
                for j, segment in enumerate(segments): 
                    addDoc(i, segment, tSegments[j])  # i: document index; each segment shares the same document index

                # [test]
                nS = len(segments)
                if nS > 1: 
                    nMulti += 1
                    if nMulti <= 10: 
                        display(doc, segments, tSegments)
                        
            else: 
                if tDropNullCut: 
                    pass 
                else: 
                    # no cutting, just keep the original document
                    addDoc(i, doc, T[i])
                noCutpoints.append(i)

    elif policy in ('prior', 'posterior', ): # take only prediagnosis segment, excluding the visit where predicate holds
        keepNonActive = True  # keep the doc in which none of the position fires the predicate (e.g. no principle diagnosis present) 
        n_active = 0
        include_endpoint = kargs.get('inclusive', False) 
        for i, doc in enumerate(D): 
            if len(cutpointx[i]) > 0: 
                timestamps = [T[i][cutpoint] for cutpoint in cutpointx[i]]

                # use the timestamp of the first occurrence 
                t_prior = timestamps[0] 

                # [test]
                if i < 100: test_segmenting(i, t_prior, policy)
                seg, tseg = sliceByTime(seq=doc, tseq=T[i], timestamp=t_prior, policy=policy, inclusive=include_endpoint, t_format="%Y-%m-%d")
                assert len(seg) == len(tseg), "document segment length: %d but time segment length: %d" % (len(seg), len(tseg))

                # input document (doc) is broken down into multiple segments 
                nS = len(seg)  # non-empty segment
                if nS > 0: 
                    addDoc(i, seg, tseg)
                    timeToFirstDiag.append(getTimeSpan(tseg))
                    n_active += 1   # has non-empty pre or post-diagnostic segment

                    # [test]
                    if n_active <= 10:   
                        display(doc, [seg,], [tseg, ])
                else: 
                    nullSegments.append(i)

            else: 
                # keep the doc anyway
                if tDropNullCut: 
                    pass 
                else: 
                    addDoc(i, doc, T[i])
                timeToFirstDiag.append(getTimeSpan(T[i]))
                noCutpoints.append(i)
               
    elif policy.startswith('compl'): # complete segmentations on cutpoints (usu. defined via diagnostic codes)
        minNPoints = 1  # segment the documents in halves only when there exists more than N cut points
        nMulti = 0 
        for i, doc in enumerate(D): 
            if len(cutpointx[i]) > minNPoints:
                timestamps = [T[i][cutpoint] for cutpoint in cutpointx[i]]
                segments, tSegments = segmentByTime(seq=doc, tseq=T[i], timestamps=timestamps, t_format="%Y-%m-%d")
                assert len(segments) <= len(timestamps), \
                    "n_segments(%d) cannot exceed n_times(%d), some of which may be invalid" % (len(segments), len(timestamps))

                # input document (doc) is broken down into multiple segments
                for j, segment in enumerate(segments): 
                    addDoc(i, segment, tSegments[j])  # i: document index

                # [test]
                nS = len(segments)
                if nS > 1: 
                    nMulti += 1
                    if nMulti <= 10: 
                        display(doc, segments, tSegments)
                        
            else: 
                # no cutting, just keep the original document
                if not tDropNullCut: addDoc(i, doc, T[i])
                noCutpoints.append(i)
                         
    else: 
        raise NotImplementedError, "Unknown segmenting policy: %s" % policy
                
    summary()
    return (docIDs, D2, T2)

def segmentByTime(seq, tseq, timestamps, t_format="%Y-%m-%d"):
    """
    Use a list of timestamps to segment the input sequence

    Input
    -----
    timestamps: a list of timestamps serving as cutpoints
    inclusive: include the position where predicate holds? True by default

    Memo
    ----
    1. for the moment, n(timestamps) = 1


    """
    # from datetime import datetime
    assert len(seq) == len(tseq), "sequence and timestamp have inconsistent lengths: %d vs %d" % (len(seq), len(tseq))
    
    cutpoints = {0, len(tseq)}  # minimal two cutpoints
    for tstamp in timestamps: 
        t0 = datetime.strptime(tstamp, t_format) 
        # find the first t >= tstamp

        cutpoint = len(tseq)
        for i, t in enumerate(tseq): 
            if datetime.strptime(t, t_format) > t0: 
                cutpoint = i; break 
        cutpoints.add(cutpoint) # use the index of the first timestamp > t0

    cutpoints = sorted(cutpoints)
    assert cutpoints[0] == 0 and cutpoints[-1] == len(tseq)
    if len(cutpoints) == 2: 
        return [seq, ], [tseq, ]  # the segment is itself; to be used in an iterator so it has to be wrapped in a list
    else: 
        assert len(cutpoints) >= 3 and cutpoints[1] > 0

        segments, tSegments = [], []
        for i, cutpoint in enumerate(cutpoints): 
            if i == 0: continue 

            prev = cutpoints[i-1]
            segment = seq[prev:cutpoint]
            tSegment = tseq[prev:cutpoint]

            if len(segment) > 0: 
                segments.append(segment)
                tSegments.append(tSegment)

        assert len(segments) == len(cutpoints)-1, "%d valid cutpoints but only got %d segments" % (len(cutpoints), len(segments))

    return (segments, tSegments)  # document segments and time segments

def segmentByVisits(D, T, docIDs, **kargs):  # seqTransform
    """
    D: [[ ... d1 ... ], []] 

       [ [d1: [v1, v2, v3], 
          d2: [v1, v2]
          ...

          dn: [] ]

    L: 
    T: 

    """
    def test_input(): 
        L = kargs.get('L', []) # labels may not be necessary here
        assert len(D) == len(T) == docIDs
        if len(L) > 0: assert len(D) == len(L)
        return 
    def get_stats(docToVisit): 
        nLGrand = 0  # total length across all documents
        nVGrand = 0  # grand total number of visits across all documents
        maxL = -1 # the maximum length of a visit (number of codes) across all visist and all documents
        maxV = -1 # maximum number of visits across all documents
        for docid, vseq  in docToVisit.items(): # e.g. [[v1, v2], [v3, v4, v5], [] ]
            nVDoc = len(vseq)  # number of visits for this document
            if maxV < nVDoc: maxV = nVDoc
            for v in vseq: # foreach visit
                nV = len(v)  # length of each visit; e.g. v1: [c1, c2, c3]
                if maxL < nV: maxL = nV
                nLGrand += nV  # to get total length of visits 
            nVDocTotal += nVDoc

        avgL = nVGrand/(len(nVGrand)+0.0) # averag length of a visit
        avgV = nVDocTotal/(len(docToVisit)+0.0)  # average number of visits per document

        # [test]
        print('  + avgL: %f, maxL: %d' % (avgL, maxL))
        print('  + avgV: %f, maxV: %d' % (avgV, maxV))

        return (avgL, maxL, avgV, maxV)
    # from sampler.sampling import sample_wr

    test_input()
    docToVisit = {}
    for i, doc in enumerate(D): # 'i' is not necessary identical to document ID
        docid, tdoc = docIDs[i], T[i]
        nW = len(doc)

        docToVisit[docid] = []
        # bypass empty documents
        if nW == 0: 
            print('Warning: %d-th document (docID=%d) is empty!' % (i, docid))
            continue

        indices = [0, ]  # indicate i-th visit
        if nW > 1: 
            for j in range(1, nW):
                t = tdoc[j]
                t0 = tdoc[j-1]

                id_prev = id_cur = indices[j-1]
                if t != t0: # tdoc was already sorted
                    id_cur += 1
                indices.append(id_cur)
        id_max = indices[-1]
        n_visits = id_max + 1

        # indices e.g. [0, 0, 1, 1, 1, 2] contains i-sequence representing sequence of ith-visits

        # knowing n_visits, now we can init docToVisit[docId]
        for ith in range(n_visits):
            docToVisit[docid].append([])

        # [test]
        if i < 10: 
            assert len(set(tdoc)) == n_visits

        for j in range(nW):
            ithv = indices[j] # ith visit
            docToVisit[docid][ithv].append(doc[j]) # j-th token assigned to ith-visit
        
        # [test]
        if i < 10: 
            for ith in range(n_visits):
                assert len(docToVisit[docid][ith]) > 0

    avgL, maxL, avgV, maxV = get_stats(docToVisit)
        
    return docToVisit  # docId -> visists [[c1, c2], [c3, c4, c5], [c6, c7], ... [c124]]

def transformDocuments(D, L=[], T=[], policy='empty_doc', seq_ptype='regular', predicate=None, simplify_code=False, **kargs): 
    """
    Modify input documents including simplifying and filtering document content (codes). 
    Input timestamps (T) will be modified accordingly. For instance, if a subset of 
    codes are removed (say only diagnostic codes are preserved), then their timestamps are 
    removed in parallel. 

    Params
    ------
    1) modifying sequence contents: 
       seq_ptype
       predicate

    2) filtering out unwanted documents: 
       policy: policy for filtering documents 
       L: labels will also be removed from the set if their documents are removed 

    3) simply coding (e.g. 250.01 => 250)


    **kargs (i.e. implicit parameters)
    -------
    min_ncodes

    """
    def show_stats(): 
        print('transformDocuments> nD: %d (<- %d), nT: %d, nL: %d' % (len(D), nD0, len(T), len(L)))
        # cohort_name = kargs.get('cohort', '?')
        if len(L) > 0: 
            n_classes = len(set(L))  # seqparams.arg(['n_classes', ], default=1, **kargs) 
            print('  + Stats: n_docs: %d, n_classes:%d' % (len(D), n_classes))
        else: 
            print('  + Stats: n_docs: %d, n_classes:? (no labeling info)' % len(D)) 
        return
    def show_examples(n=10): # D, D2
        while n > 0: 
            r = random.randint(0, len(D2)-1) # select test doc 
            assert len(D2) == len(D), "Filtering content should not change the size of the document set size(D): %d => %d" % (len(D), len(D2))
            print('  + original doc (%s):\n%s\n =>\n   + modified doc (%s):\n%s\n' % (seq_ptype, D[r], seq_ptype, D2[r]))
            n -= 1
        return
    def test_diff(D, D2, n=10, maxtok=100): 
        nD = len(D)

        # N = sum(1 for d in D if len(d) > 0)
        # N2 = sum(1 for d2 in D2 if len(d2) > 0)
        # r = N2/(N+0.0)
        # print('(test_diff) ratio of documents containing disease codes: %f | N=%d, N2=%d' % (r, N, N2))

        n_tests = n
        for i, d2 in enumerate(D2): 
            if len(d2) > 0 and (D[i] != d2): 
                print('(test_diff) original:    %s' % D[i][:maxtok])  
                print('(test_diff) transformed: %s' % d2[:maxtok])  

        # for _ in range(n): 
        #     r = random.randint(0, nD-1)
        #     if D[r] != D2[r]: 
        #         print('(test_diff) original:    %s' % D[r][:maxtok])  
        #         print('(test_diff) transformed: %s' % D2[r][:maxtok])
        return

    seq_ptype = seqparams.normalize_ctype(seq_ptype)
    tSimplifyCode = simplify_code
    tFilterContent = not seq_ptype.startswith('reg') or predicate is not None
    tFilterDocSet = True # policy.startswith( ('empty', 'min'))
    tDebug = True

    # docs
    nD0 = len(D)
    if tSimplifyCode: 
        print('transform> simplifying the codes ...')
        D = seqAlgo.simplify(D)  # this will not affect medication code e.g. MED:12345
        # condition: length of each doc remains identical

    D2, T2 = D, T 
    if tFilterContent:  # e.g. diag only, medication only
        ## precedence: filter by predicate if given => filter by ctype: {'diag', 'med'}

        print('transform> modifying sequence contents based on content type (%s) or a predicate (%s)' % (seq_ptype, str(predicate)))
        if len(T) == 0 or T is None: 
            D2 = modify(D, seq_ptype=seq_ptype, predicate=predicate)
        else: 
            D2, T2 = parallellModify(D, T, seq_ptype=seq_ptype, predicate=predicate)
            assert len(T2) == len(D2)

        if tDebug: show_examples(n=10)
        assert len(D2) == nD0, "Content filtering should not result in reduction of documents (%d -> %d)" % (nD0, len(D2))
    
    # [test] condition: so far size(corpus) should not change! 
    if kargs.get('test_', True): test_diff(D, D2, n=10)  # test n=10 (random) cases

    # docs, labels
    L2 = L 
    if tFilterDocSet: 
        D2, L2, T2 = filterDocuments(D2, L=L2, T=T2, policy=policy, min_ncodes=kargs.get('min_ncodes', 1))
        # condition: size of document sets may become different (e.g. empty documents removed)
    show_stats()
    return (D2, L2, T2)  # (D, L, T)
def transformDocuments2(D, L=[], T=[], **kargs):  # this is not the same as seqTransform.transform()
    """
    Transform the document (as does transformDocuments() but also save a copy of the transformed document)

    Params
    ------
    1) modifying sequence contents: 
       seq_ptype
       predicate

    2) filtering out unwanted documents: 
       policy: policy for filtering documents 
       L: labels will also be removed from the set if their documents are removed 

    3) simply coding (e.g. 250.01 => 250)

    4) splicing: noop, prior (or prediagnostic), posterior 
       slice_policy
       cohort
       predicate 
       cutpoint
       n_active: 1 by default
       inclusive: True by default
    """
    def get_cohort(): # only needed if save_ <- True
        try: 
            return kargs['cohort']
        except: 
            pass 
        raise ValueError, "cohort info is mandatory for saving a new copy of labeled document."
    def determined_doctype(): 
        pass 
    def show_stats(): 
        print('transformDocuments2> nD: %d, nT: %d, nL: %d' % (len(D), len(T), len(L)))
        cohort_name = kargs.get('cohort', '?')
        if len(L) > 0: 
            n_classes = len(set(L))  # seqparams.arg(['n_classes', ], default=1, **kargs) 
            print('  + Stats: n_docs: %d, n_classes:%d | cohort: %s' % (len(D), n_classes, cohort_name))
        else: 
            print('  + Stats: n_docs: %d, n_classes:? (no labeling info) | cohort: %s' % (len(D), cohort_name)) 
        return
    def do_slice(): 
        if not kargs.has_key('slice_policy'): return False
        if kargs['slice_policy'].startswith(('noop', 'reg', 'compl', )):  # effective operations 
            return False 
        return True

    from tdoc import TDoc  # base class defined in seqparams
    import seqReader as sr

    seq_ptype = kargs.pop('seq_ptype', 'regular')
    D, L, T = transformDocuments(D, L=L, T=T, 
            
            # document-wise filter
            policy=kargs.pop('policy', 'empty'),  # 'empty': remove empty document, 'unknown'
            min_ncodes=kargs.pop('min_ncodes', 3), 
            
            # content filter
            seq_ptype=seq_ptype, 
            predicate=kargs.pop('predicate', None), 
            simplify_code=kargs.pop('simplify_code', False))  # T contains typically just timestamps
    # ret = {}
    # ret['sequence'] = docs 
    # if kargs.has_key('labels') and len(labels) > 0: ret['label'] = labels
    # if kargs.has_key('items') and len(items) > 0: ret['item'] = items 
    
    ### save the transformed documents but this would require extra parameters not relevant to transformation operaiton itself (e.g. corhot)
    # becuase we need it to name the new file

    # cut transform (predignosis, postdiagnosis, up-until a set of code, regular, etc.)
    # if do_slice(): 

    #     # inplace operation by default
    #     # [note] slice_predicate is not the same as the predicate for transformDocuments
    #     nD0 = len(D)
    #     D, T = sliceDocuments(D, T=T, 
    #                     policy=kargs.get('slice_policy', 'noop'), 
    #                     cohort=get_cohort(), 
    #                     predicate=kargs.get('slice_predicate', None), # infer predicate from cohort if possible
    #                     cutpoint=kargs.get('cutpoint', None), 
    #                     n_active=1, 
    #                     inclusive=kargs.get('inclusive', True))
    #     assert len(D) == len(T)
    #     assert len(D) == nD0, "size of document set should not be different after splicing nD: %d -> %d" % (len(D), nD0)

    if kargs.get('save_', False): 
        overwrite = True
        fpath = TDoc.getPath(cohort=get_cohort(), seq_ptype=seq_ptype, doc_type='labeled', ext='csv')  # usually there is one file per cohort  
        if not os.path.exists(fpath) and overwrite: 
            # path params: cohort=get_cohort(), seq_ptype=seq_ptype, doctype=doctype, ext='csv', baesdir=basedir
            header = ['sequence', 'timestamp', 'label', ]
            adict = {h: [] for h in header}
            adict['sequence'], adict['timestamp'], adict['label'] = D, T, L
            df.to_csv(fpath, sep='|', index=False, header=True)
            print('transformDocuments2> Saving the transformed document (seq_ptype=%s) to:\n%s\n' % (seq_ptype, fpath))
            # sr.readDocToCSV(cohort=get_cohort(), sequences=D, timestamps=T, labels=L, seq_ptype=seq_ptype)
    show_stats()
    return (D, L, T)

def indexToDoc(seqx, pos_map, inplace=False): 
    seqx = np.array(seqx) # in order to use a list as indices
    
    # inplace
    # if inplace:  
    #     for i, seq in enumerate(seqx): 
    #         seqx[i] = list(np.array(seq[pos_map[i]]))  # preserve only positions in pos_map[i]
    
    new_seqx = []
    if True: 
        # pos_map[i] is a list of positions; seq needs to be np.array() 
       
        for i, seq in enumerate(seqx): 
            try: 
                new_seqx.append(list(np.array(seq)[pos_map[i]]))
            except Exception, e: 
                print('  + document:\n%s\n' % seq[:20]) 
                print('  + position index:\n%s\n' % pos_map[i][:20])            
                raise ValueError, e
        if isinstance(seqx, np.ndarray): 
            new_seqx = np.array(new_seqx)

    return new_seqx # list of lists

def filterCodes(sequences, **kargs): 
    """
    Given a set of complete coding sequences, filter out unwanted 
    codes thereby preserving only those codes of interest (e.g. diagnostic codes only). 
    Filter timestamps and labels accordingly if available. 

    Input: documents (seqx), each of which is a list of tokens
    Output: filtered documents in terms of positions/indices of the tokens within each document

            a dictionary 'ret' with following key-value pairs: 
            'sequences': filtered documents
            'timestamps'
            'labels': 

        then use indexToDoc() to reconstruct input sequences of preserved codes (e.g. documents, timestamps, etc)

    Memo
    ----
    Sequence content types: 
    'regular'
    'random'
    'overlap_ngram' where 'n' depends on input 'ngram'
    'diag'
    'med'
    'lab'

    """
    def to_str(seq, sep=','): 
        return sep.join(str(e) for e in seq)
    def code_str(seq, sep='-'): # convert codes to file naming friendly format 
        s = to_str(seq, sep=sep) 
        # alternative: s = s.translate(None, string.punctuation)  # remove punctuations '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
        return s.replace('.', '')
    # from pattern import medcode as pmed
    n_doc = len(sequences)
    seq_ptype = seqparams.normalize_ctype(kargs.get('seq_ptype', 'regular'))
    # timestamps, labels = kargs.get('timestamps', []), kargs.get('labels', [])
    
    # if len(timestamps) > 0: assert len(timestamps) == n_doc 
    # if len(labels) > 0: assert len(labels) == n_doc 
    # condition: dimension consistent

    # [output]
    # keys = ['sequences', 'timestamps', 'labels', ]
    # ret = {k: [] for k in keys}  # [output] sequences, timestamps, labels

    # user-provided predicate 
    predicate = kargs.get('predicate', None)  # [todo] namespace conflict: batchpheno.predicate
    n_anomaly = 0
    # [todo] just assign appropriate predicate funciton
    # if seq_ptype == 'diag': 
    #     predicate = pmed.isICD
    # elif seq_ptype == 'med': 
    #     predicate == pmed.isMedCode

    candidateIdx = [None] * n_doc
    if hasattr(predicate, '__call__'): 
        for i, sequence in enumerate(sequences):
            iloc = qualify(sequence, predicate=predicate) 
            if len(iloc) == 0: 
                # print('warning> No codes found (predicate=%s) in %d-th sequence/doc:\n%s\n' % \
                #     (predicate.__name__, i, to_str(sequence)))
                n_anomaly += 1 
            candidateIdx[i] = iloc # for i-th document, preserve only positions in 'idx' 
    elif seq_ptype.startswith(('reg', 'noop', 'compl')):
        # effectively noop  
        for i, sequence in enumerate(sequences):
            # idx = qualify(sequence, predicate=predicate) 
            iloc = range(0, len(sequence))  # preserve ALL
            if len(iloc) == 0: n_anomaly += 1 
            candidateIdx[i] = iloc # for i-th document, preserve only positions in 'idx'         

    elif seq_ptype in ('diag', 'med', 'random', 'lab', ):  
        for i, sequence in enumerate(sequences): 
            # repr_seqx[i] = transform_by_ctype(sequence, seq_ptype=seq_ptype) # this doesn't return indices 
            iloc = qualify(sequence, seq_ptype=seq_ptype)
            if len(iloc) == 0: 
                # print('warning> No codes (of type %s) found in %d-th sequence/doc:\n%s\n' % (seq_ptype, i, to_str(sequence)))
                n_anomaly += 1 
            candidateIdx[i] = iloc  # preserve only these positional indices
                
        div(message='A total of %d out of %d documents have valid diag codes.' % (n_doc-n_anomaly, n_doc), symbol='%')
    elif seq_ptype.startswith('overlap'): # this is not really a filter
        raise ValueError, "error> seq_ptype=%s does not apply here being not a filter" % seq_ptype
    #     for i, sequence in enumerate(sequences): 
    #         repr_seqx[i] = transform_by_ngram_overlap(sequence, length=kargs.get('length', 3))
    else: 
        # 'lab', 
        raise NotImplementedError, "Unsupported sequence content type: %s" % seq_ptype  
        
    assert not None in candidateIdx

    return candidateIdx  # map: ith doc -> positions (i.e. position indices preserved)

def execute(seq, **kargs): 
    """
    Given a coding sequence, take only a segment of the sequence according to a 
    given criterion. 
    """
    predicate = kargs.get('predicate', None)
    cutpoint = kargs.get('cutpoint', None)   
    if predicate is not None or cutpoint is not None: 
        return cut(seq, **kargs)
    
    if kargs.get('simplify_code', False): 
        return simplify(seq) 

    # seq_ptype is given? 
    return transform_by_ctype(seq, **kargs)

def transform(seq, **kargs): # cut + filter (e.g. diagnostic only or med only)
    """
    Cut the seq according to cut_policy followed by 
    content transformation (e.g. preserving only diag codes, randomized sequence, etc.)

    This is an older version of transformDocuments() 

    Params 
    ------
    policy: 'noop', 'prior', 'posterior'

    Memo
    ----
    a. transform(doc, cut_policy=cut_policy, inclusive=True, 
                        seq_ptype=seq_ptype, cohort=cohort_name, predicate=condition_predicate)
    """
    policy = kargs.get('cut_policy', 'noop')  # 'prior'
    # assert policy is not None, "perhaps wrong argument given? use cut_policy to specify policy type"
    # predicate = kargs.get('predicate', None) # if None, will use appropriate predicate governed by cohort

    seq_subset = cut(seq, policy=policy, 
                        cohort=kargs.get('cohort', 'unknown'), 
                        predicate=kargs.get('predicate', None), 
                        cutpoint=kargs.get('cutpoint', None), 
                        n_active=1, 
                        inverse=kargs.get('inverse', False), inclusive=kargs.get('inclusive', True)) 
    
    # [test]
    if policy.startswith('prior'): 
        print('transform> after cut (policy=%s):\n%s\n' % (kargs.get('policy', '?'), seq_subset[-20:]))
    else: 
    	print('transform> after cut (policy=%s):\n%s\n' % (kargs.get('policy', '?'), seq_subset[:20]))

    if len(seq_subset) > 0 and kargs.has_key('seq_ptype'): 
        return transform_by_ctype(seq_subset, **kargs) # will check seq_ptype 
    return seq_subset

def cut(seq, **kargs):
    """
    Given a coding sequence, take only a segment of the sequence according to a 
    given criterion. 

    Params 
    ------
    policy: 'noop', 'prior', 'posterior' 

    cohort: use this to find the right predicate from disease-specific modules defined in pattern 
    cutpoint: use this to specify explicit cut points 
              accept a list or single code (of string type)

              e.g. PTSD
                   ['309.81', 'F43.1', ]
                   or 
                   '309.81' if ICD-9 codes only

    inverse 
    inclusive: end point/code is included by default 
    n_active: number of times that a predicate or presense of cutpoints have to be True 
              in order to be considered as a match


    Related
    -------
    sliceDocuments()   

    Design 
    ------ 
    1. Cut point 
       first mention of a diagnostic code associated with a disease (e.g. 250.00)
           predicate: diabetes.is_diabetes_type1

    2. policy 
       prior to cut point (e.g. before the diagnosis)
       or 
       after (posterior) to the cut point 

       what happens if no cut point found (i.e. the sequence doesn't contain the target diag code)? 
       => return the entire sequence

    Note
    ----
    1. Assumption 
       (FFFFFF)T => get sub-sequence up until the predicate becomes True i.e. (FFFFFF)

       but sometimes we want 
       (TTTTTTT)F => get sub-sequence prior to the predicting becoming False
       
    """
    # import medcode as pmed

    policy = kargs.get('policy', 'noop')  # prior, posterior
    if policy in ('noop', 'regular', 'complete', ): 
        return seq # no op

    # if kargs.get('inverse', False): 
    # 	return cut_inverse(seq, **kargs)

    # op_split = kargs.get('split', False)
    token_sep = kargs.get('token_sep', ',')    
    if isinstance(seq, str): seq = seq.split(token_sep)

    # condition: seq IS-A list

    # pattern = kargs.get('pattern', None)
    inverse = kargs.get('inverse', False)   # inverse: True => TTTTTF,  inverse: False => FFFFFT 
    inclusive = kargs.get('inclusive', False if inverse else True) # include cut point?

    # [params] either predicate or cutpoint has to be specified
    predicate = kargs.get('predicate', None)

    cutpoint = kargs.get('cutpoint', None)  # could a single value or a set/list
    n_activated = kargs.get('n_active', 1) # n: subsequence derived from the nth time the predicate is True

    # print('>>> prior to predicate ... ')
    cohort_name = kargs.get('cohort', 'unknown')
    if predicate is None: predicate = getDiseasePredicate(cohort=cohort_name)
    
    if predicate is not None and hasattr(predicate, '__call__'): 
        # print('>>> has predicate: %s' % predicate_)

        # [assumption] default: (FFFFFF)T seuqnece type (e.g. first diagnosis of a target disease)
        if kargs.get('inverse', False):
            # print('info> calling inverse ...')
            return cut_inverse(seq, **kargs)
        # predicate = lambda e: not predicate_(e) if inverse else predicate_  # doesn't work 
        
        n_cur = 0 
        if policy.startswith('pri'): # prior => takes the sequence prior to the cut point (determined by the predicate)
            upper = len(seq) # technically, len(seq)-1 
            for i, e in enumerate(seq): 
                if predicate(e):
                    n_cur += 1 
                    # print('   + e: %s => True' % e)
                if n_cur == n_activated: 
                    upper = i
                    break 
            # print('verify> (inclusive? %s, inverse? %s, n_act: %d) upper: %d' % (inclusive, inverse, n_activated, upper))
            if inclusive: 
                return seq[:upper+1] # want the cutpoint as well
            else: 
            	return seq[:upper]
            
        else: 
            lower = 0 # technically, len(seq)-1 
            for i, e in enumerate(seq): 
                if predicate(e): 
                	n_cur += 1
                if n_cur == n_activated:  
                    lower = i
                    break 

            if inclusive: 
                return seq[lower:]
            else: 
            	return seq[lower+1:]

    elif cutpoint is not None:  # absolute match by default
        pset = cutpoints = set()
        if not hasattr(cutpoint, '__iter__'):
            pset.add(cutpoint)
        else: 
            pset = set(cutpoint)

        n_cur = 0
        if policy.startswith('pri'): # prior => takes the sequence prior to the cut point 
            upper = len(seq) # technically, len(seq)-1 

            # [todo] 
            for i, e in enumerate(seq): 
                if e in pset:  # loose match or strict match? 
                    n_cur += 1 
                if n_cur == n_activated: 
                    upper = i
                    break 

            if inclusive: 
                return seq[:upper+1] # want the cutpoints as well

            else: 
            	return seq[:upper]
        else: 
            lower = 0 # technically, len(seq)-1 
            for i, e in enumerate(seq): 
                if e in pset:
                    n_cur += 1
                if n_cur == n_activated:  
                    lower = i
                    break 

            if inclusive:             
                return seq[lower:]
            else: 
            	return seq[lower+1:]
    else: 
        raise ValueError, "cut> No valid predicate or cut point(s) provided."
   
    return seq # no-op

# def cut_inverse(seq, **kargs): # (TTTTTTT)F 
#     predicate = kargs.get('predicate', None)
#     if predicate is not None and hasattr(predicate, '__call__'): 
#         kargs['predicate'] = lambda e: not predicate(e)  # [todo] what if predicate expects multiple arguments? 
#     else: 
#     	print("warning> No valid predicate is given => just a regular cut()")
    
#     kargs['inverse'] = False
#     return cut(seq, **kargs)
def cut_inverse(seq, **kargs): # (TTTTTTT)F 
    policy = kargs.get('policy', 'noop')  # prior, posterior
    if policy == 'noop': 
        return seq # no op

    # op_split = kargs.get('split', False)
    token_sep = kargs.get('token_sep', ',')
    inclusive = kargs.get('inclusive', True) # include cut point?
    
    if isinstance(seq, str): seq = seq.split(token_sep)

    # [params] either predicate or cutpoint has to be specified
    predicate = kargs.get('predicate', None)
    cutpoint = kargs.get('cutpoint', None)
    n_activated = kargs.get('n_activated', 1) # n: subsequence derived from the nth time the predicate is True

    if predicate is not None and hasattr(predicate, '__call__'): 
        n_cur = 0 
        if policy.startswith('pri'): # prior => takes the sequence prior to the cut point (determined by the predicate)
            upper = len(seq) # technically, len(seq)-1 
            for i, e in enumerate(seq): 
                if not predicate(e):
                    n_cur += 1 
                if n_cur == n_activated: 
                    upper = i
                    break 

            if inclusive: 
                return seq[:upper+1] # want the cutpoint as well
            else: 
            	return seq[:upper]
            
        else: 
            lower = 0 # technically, len(seq)-1 
            for i, e in enumerate(seq): 
                if not predicate(e): 
                	n_cur += 1
                if n_cur == n_activated:  
                    lower = i
                    break 

            if inclusive: 
                return seq[lower:]
            else: 
            	return seq[lower+1:]
    else: 
    	raise ValueError, "No valid predicate provided."
    
    return seq

def segmentByCode(D, T, **kargs):
    def test_dtype(): 
        assert len(D) > 0, "sliceDocuments> Empty input document set"
        assert len(D) == len(T), "length(seq):%d <> length(seq2):%d" % (len(seq), len(seq2))
        
        r = random.randint(0, len(D)-1)
        assert hasattr(D[r], '__iter__') and hasattr(T[r], '__iter__'), "Ill-formated input:\n%s\n" % D[r]
    def has_operation(): 
        has_predicate = True if predicate is not None else False 
        has_cutpoint = True if cutpoint is not None else False
        # assert has_predicate or has_cutpoint, "sliceDocuments> Either a predicate or a set of cutopints have to be given."
        if not has_predicate and not has_cutpoint: 
            print('sliceDocuments> Neither a predicate nor a set of cutopints were given => No-op.')
        return has_predicate or has_cutpoint
    def show_params(): 
        print("sliceDocuments> policy: %s, cohort: %s -> predicate: %s" % (policy, cohort_name, predicate))
        if predicate is None: 
            print('  + cohort: %s ~? cutpoint (set): %s' % (cohort_name, cutpoint))
        return 
    def verify_result(n=10):
        if inplace: 
            # do nothing
            assert len(D) == nD, "size of document set, before and after, is not consistant (nD: %d -> %d)" % (nD, len(D))
            assert len(T) == nT
            return

        print("sliceDocuments> before and after ...")  

        assert len(D2) > 0 
        assert len(D2) == nD, "size of document set, before and after, is not consistant (nD: %d -> %d)" % (nD, len(D2))
        assert len(T2) == nT 
        assert len(D2) == len(T2)

        idx = random.sample(range(0, nD), n)
        for i in idx: 
            print('  + (b) %s' % D[i])
            print('  + (a) %s' % D2[i])
        return
    import random
    # import medcode as pmed

    test_dtype()
    # condition: seq IS-A list

    nD, nT = len(D), len(T)
    cutpoints = kargs.get('cutpoints', [])  # prior, posterior
    if len(cutpoints) == 0: # noop, regular, complete
        return (D, T) # no op

    # pattern = kargs.get('pattern', None)
    inverse = False   # inverse: True => TTTTTF,  inverse: False => FFFFFT 
    inclusive = kargs.get('inclusive', True) # include cut point?
    n_activated = kargs.get('n_active', 1) # n: subsequence derived from the nth time the predicate is True

    # [params] either predicate or cutpoint has to be specified
    cohort_name = kargs.get('cohort', 'unknown')
    predicate = kargs.get('predicate', None)
    if predicate is None: predicate = getDiseasePredicate(cohort=cohort_name)

    # used only when predicate is None; predicate and cutpoint cannot be both None
    cutpoint = kargs.get('cutpoint', None)  # could a single value or a set/list
    if not has_operation(): 
        return (D, T)  # no op
    
    show_params()
    inplace = False  # assuming inplace operation by default
    D2, T2 = [], [] 
    if predicate is not None and hasattr(predicate, '__call__'): 
        for i, seq in enumerate(D): 
            n_cur = 0 

            tseq = T[i]

            # prior or prediagnostic => takes the sequence prior to the cut point (determined by the predicate)
            if policy.startswith( ('pri', 'pre', )): 
                upper = len(seq) # technically, len(seq)-1 
                
                for j, e in enumerate(seq): 
                    if predicate(e):
                        n_cur += 1 
                        
                    if n_cur == n_activated: 
                        upper = j
                        break 
       
                if inclusive: 
                    seq = seq[:upper+1] # want the cutpoint as well
                    tseq = tseq[:upper+1]
                else: 
                    seq = seq[:upper]
                    tseq = tseq[:upper]

            else:  # post- 
                lower = 0 # technically, len(seq)-1 
                for j, e in enumerate(seq): 
                    if predicate(e): 
                        n_cur += 1
                    if n_cur == n_activated:  
                        lower = j 
                        break 

                if inclusive: 
                    seq = seq[lower:]
                    tseq = tseq[lower:]
                else: 
                    seq = seq[lower+1:]
                    tseq = tseq[lower+1:]

            if inplace: 
                D[i], T[i] = seq, tseq
            else: 
                D2.append(seq)
                T2.append(tseq)  

# [todo]
def sliceDocuments2(D, T, **kargs):
    """
    Given a coding sequence, take only a segment of the sequence according to a 
    given criterion. Similar to sliceDocuments() but allows for end points. 

    e.g. preserve the segment between the first occurrence of a target code (e.g. PTSD, 309.11)
         and the last occurrence of the same code

    A variation of cut() and operates on seq2 (e.g. timestamps) as well. 

    Params 
    ------
    policy: 'noop', 'prior', 'posterior' 

    cohort: use this to find the right predicate from disease-specific modules defined in pattern 
    cutpoint: use this to specify explicit cut points 
              accept a list or single code (of string type)

              e.g. PTSD
                   ['309.81', 'F43.1', ]
                   or 
                   '309.81' if ICD-9 codes only

    inverse 
    inclusive: end point/code is included by default 
    n_active: number of times that a predicate or presense of cutpoints have to be True 
              in order to be considered as a match

    """
    def test_dtype(): 
        assert len(D) > 0, "sliceDocuments> Empty input document set"
        assert len(D) == len(T), "length(seq):%d <> length(seq2):%d" % (len(seq), len(seq2))
        
        r = random.randint(0, len(D)-1)
        assert hasattr(D[r], '__iter__') and hasattr(T[r], '__iter__'), "Ill-formated input:\n%s\n" % D[r]
    def has_operation(): 
        has_predicate = True if predicate is not None else False 
        has_cutpoint = True if cutpoint is not None else False
        # assert has_predicate or has_cutpoint, "sliceDocuments> Either a predicate or a set of cutopints have to be given."
        if not has_predicate and not has_cutpoint: 
            print('sliceDocuments> Neither a predicate nor a set of cutopints were given => No-op.')
        return has_predicate or has_cutpoint
    def show_params(): 
        print("sliceDocuments> policy: %s, cohort: %s -> predicate: %s" % (policy, cohort_name, predicate))
        if predicate is None: 
            print('  + cohort: %s ~? cutpoint (set): %s' % (cohort_name, cutpoint))
        return 
    def verify_result(n=10):
        if inplace: 
            # do nothing
            assert len(D) == nD, "size of document set, before and after, is not consistant (nD: %d -> %d)" % (nD, len(D))
            assert len(T) == nT
            return

        print("sliceDocuments> before and after ...")  

        assert len(D2) > 0 
        assert len(D2) == nD, "size of document set, before and after, is not consistant (nD: %d -> %d)" % (nD, len(D2))
        assert len(T2) == nT 
        assert len(D2) == len(T2)

        idx = random.sample(range(0, nD), n)
        for i in idx: 
            print('  + (b) %s' % D[i])
            print('  + (a) %s' % D2[i])
        return
    import random
    # import medcode as pmed

    test_dtype()
    # condition: seq IS-A list

    nD, nT = len(D), len(T)
    policy = kargs.get('policy', 'noop')  # prior, posterior
    if policy.startswith(('noop', 'reg', 'compl', )): # noop, regular, complete
        return (D, T) # no op

    # pattern = kargs.get('pattern', None)
    inverse = False   # inverse: True => TTTTTF,  inverse: False => FFFFFT 
    inclusive = kargs.get('inclusive', True) # include cut point?
    n_activated = kargs.get('n_active', 1) # n: subsequence derived from the nth time the predicate is True

    # [params] either predicate or cutpoint has to be specified
    cohort_name = kargs.get('cohort', 'unknown')
    predicate = kargs.get('predicate', None)
    if predicate is None: predicate = getDiseasePredicate(cohort=cohort_name)

    # used only when predicate is None; cannot be both None
    cutpoint = kargs.get('cutpoint', None)  # could a single value or a set/list
    if not has_operation(): 
        return (D, T)  # no op
    
    show_params()
    inplace = False  # assuming inplace operation by default
    D2, T2 = [], [] 
    if predicate is not None and hasattr(predicate, '__call__'): 
        for i, seq in enumerate(D): 
            n_cur = 0 

            tseq = T[i]

            # prior or prediagnostic => takes the sequence prior to the cut point (determined by the predicate)
            if policy.startswith( ('pri', 'pre', )): 
                upper = len(seq) # technically, len(seq)-1 
                
                for j, e in enumerate(seq): 
                    if predicate(e):
                        n_cur += 1 
                        
                    if n_cur == n_activated: 
                        upper = j
                        break 
       
                if inclusive: 
                    seq = seq[:upper+1] # want the cutpoint as well
                    tseq = tseq[:upper+1]
                else: 
                    seq = seq[:upper]
                    tseq = tseq[:upper]

            else:  # post- 
                lower = 0 # technically, len(seq)-1 
                for j, e in enumerate(seq): 
                    if predicate(e): 
                        n_cur += 1
                    if n_cur == n_activated:  
                        lower = j 
                        break 

                if inclusive: 
                    seq = seq[lower:]
                    tseq = tseq[lower:]
                else: 
                    seq = seq[lower+1:]
                    tseq = tseq[lower+1:]

            if inplace: 
                D[i], T[i] = seq, tseq
            else: 
                D2.append(seq)
                T2.append(tseq) 

    elif cutpoint is not None:  # absolute match by default; can be single value, or multiple values
        pset = cutpoints = set()
        if not hasattr(cutpoint, '__iter__'):
            pset.add(cutpoint)
        else: 
            pset = set(cutpoint)

        for i, seq in enumerate(D): 
            n_cur = 0
            tseq = T[i]

            if policy.startswith('pri'): # prior => takes the sequence prior to the cut point 
                upper = len(seq) # technically, len(seq)-1 

                # [todo] 
                for j, e in enumerate(seq): 
                    if e in pset:  # loose match or strict match? 
                        n_cur += 1 
                    if n_cur == n_activated: 
                        upper = j 
                        break 

                if inclusive: 
                    seq = seq[:upper+1] # want the cutpoint as well
                    tseq = tseq[:upper+1]            
                else: 
                    seq = seq[:upper]
                    tseq = tseq[:upper]

            else: # post

                lower = 0 # technically, len(seq)-1 
                for i, e in enumerate(seq): 
                    if e in pset:
                        n_cur += 1
                    if n_cur == n_activated:  
                        lower = i
                        break 

                if inclusive: 
                    seq = seq[lower:]
                    tseq = tseq[lower:]
                else: 
                    seq = seq[lower+1:]
                    tseq = tseq[lower+1:]

            if inplace: 
                D[i], T[i] = seq, tseq
            else: 
                D2.append(seq)
                T2.append(tseq) 
    else: 
        raise ValueError, "sliceDocuments> No valid predicate or cut point(s) provided."
   
    verify_result()
    if not inplace: 
        return (D2, T2)

    return (D, T) # inplace operation for now

def sliceDocuments(D, T, **kargs):
    """
    Given a coding sequence, take only a segment of the sequence according to a 
    given criterion. 

    A variation of cut() and operates on T (e.g. timestamps) as well. 

    Params 
    ------
    policy: 'noop', 'prior', 'posterior' 

    cohort: use this to find the right predicate from disease-specific modules defined in pattern 
    cutpoint: use this to specify explicit cut points 
              accept a list or single code (of string type)

              e.g. PTSD
                   ['309.81', 'F43.1', ]
                   or 
                   '309.81' if ICD-9 codes only

    inverse 
    inclusive: end point/code is included by default 
    n_active: number of times that a predicate or presense of cutpoints have to be True 
              in order to be considered as a match

    """
    def test_dtype(): 
        assert len(D) > 0, "sliceDocuments> Empty input document set"
        assert len(D) == len(T), "length(seq):%d <> length(seq2):%d" % (len(seq), len(seq2))
        
        r = random.randint(0, len(D)-1)
        assert hasattr(D[r], '__iter__') and hasattr(T[r], '__iter__'), "Ill-formated input:\n%s\n" % D[r]
    def has_operation(): 
        has_predicate = True if predicate is not None else False 
        has_cutpoint = True if cutpoint is not None else False
        # assert has_predicate or has_cutpoint, "sliceDocuments> Either a predicate or a set of cutopints have to be given."
        if not has_predicate and not has_cutpoint: 
            print('sliceDocuments> Neither a predicate nor a set of cutopints were given => No-op.')
        return has_predicate or has_cutpoint
    def show_params(): 
        print("sliceDocuments> policy: %s, cohort: %s -> predicate: %s" % (policy, cohort_name, predicate))
        if predicate is None: 
            print('  + cohort: %s ~? cutpoint (set): %s' % (cohort_name, cutpoint))
        return 
    def verify_result(n=10):
        if inplace: 
            # do nothing
            assert len(D) == nD, "size of document set, before and after, is not consistant (nD: %d -> %d)" % (nD, len(D))
            assert len(T) == nT
            return

        print("sliceDocuments> before and after ...")  

        assert len(D2) > 0 
        assert len(D2) == nD, "size of document set, before and after, is not consistant (nD: %d -> %d)" % (nD, len(D2))
        assert len(T2) == nT 
        assert len(D2) == len(T2)

        idx = random.sample(range(0, nD), n)
        for i in idx: 
            print('  + (b) %s' % D[i])
            print('  + (a) %s' % D2[i])
        return
    import random
    # import medcode as pmed

    test_dtype()
    # condition: seq IS-A list

    nD, nT = len(D), len(T)
    policy = kargs.get('policy', 'noop')  # prior, posterior
    if policy.startswith(('noop', 'reg', 'compl', )): # noop, regular, complete
        return (D, T) # no op

    # pattern = kargs.get('pattern', None)
    inverse = False   # inverse: True => TTTTTF,  inverse: False => FFFFFT 
    inclusive = kargs.get('inclusive', True) # include cut point?
    n_activated = kargs.get('n_active', 1) # n: subsequence derived from the nth time the predicate is True

    # [params] either predicate or cutpoint has to be specified
    cohort_name = kargs.get('cohort', 'unknown')
    predicate = kargs.get('predicate', None)
    if predicate is None: predicate = getDiseasePredicate(cohort=cohort_name)

    # used only when predicate is None; cannot be both None
    cutpoint = kargs.get('cutpoint', None)  # could a single value or a set/list
    if not has_operation(): 
        return (D, T)  # no op
    
    show_params()
    inplace = False  # assuming inplace operation by default
    D2, T2 = [], [] 
    if predicate is not None and hasattr(predicate, '__call__'): 
        for i, seq in enumerate(D): 
            n_cur = 0 

            tseq = T[i]

            # prior or prediagnostic => takes the sequence prior to the cut point (determined by the predicate)
            if policy.startswith( ('pri', 'pre', )): 
                upper = len(seq) # technically, len(seq)-1 
                
                for j, e in enumerate(seq): 
                    if predicate(e):
                        n_cur += 1 
                        
                    if n_cur == n_activated: 
                        upper = j
                        break 
       
                if inclusive: 
                    seq = seq[:upper+1] # want the cutpoint as well
                    tseq = tseq[:upper+1]
                else: 
                    seq = seq[:upper]
                    tseq = tseq[:upper]

            else:  # post- 
                lower = 0 # technically, len(seq)-1 
                for j, e in enumerate(seq): 
                    if predicate(e): 
                        n_cur += 1
                    if n_cur == n_activated:  
                        lower = j 
                        break 

                if inclusive: 
                    seq = seq[lower:]
                    tseq = tseq[lower:]
                else: 
                    seq = seq[lower+1:]
                    tseq = tseq[lower+1:]

            if inplace: 
                D[i], T[i] = seq, tseq
            else: 
                D2.append(seq)
                T2.append(tseq) 

    elif cutpoint is not None:  # absolute match by default
        pset = cutpoints = set()
        if not hasattr(cutpoint, '__iter__'):
            pset.add(cutpoint)
        else: 
            pset = set(cutpoint)

        for i, seq in enumerate(D): 
            n_cur = 0
            tseq = T[i]

            if policy.startswith('pri'): # prior => takes the sequence prior to the cut point 
                upper = len(seq) # technically, len(seq)-1 

                # [todo] 
                for j, e in enumerate(seq): 
                    if e in pset:  # loose match or strict match? 
                        n_cur += 1 
                    if n_cur == n_activated: 
                        upper = j 
                        break 

                if inclusive: 
                    seq = seq[:upper+1] # want the cutpoint as well
                    tseq = tseq[:upper+1]            
                else: 
                    seq = seq[:upper]
                    tseq = tseq[:upper]

            else: # post

                lower = 0 # technically, len(seq)-1 
                for i, e in enumerate(seq): 
                    if e in pset:
                        n_cur += 1
                    if n_cur == n_activated:  
                        lower = i
                        break 

                if inclusive: 
                    seq = seq[lower:]
                    tseq = tseq[lower:]
                else: 
                    seq = seq[lower+1:]
                    tseq = tseq[lower+1:]

            if inplace: 
                D[i], T[i] = seq, tseq
            else: 
                D2.append(seq)
                T2.append(tseq) 
    else: 
        raise ValueError, "sliceDocuments> No valid predicate or cut point(s) provided."
   
    verify_result()
    if not inplace: 
        return (D2, T2)

    return (D, T) # inplace operation for now

def simplify(seq, sep=' '): 
    """
    Simplify diagnostic codes. 

    No-op for non-diagnostic codes. 

    Input
    -----
    seq: a string of space-separated diagnostic codes (icd9, icd10, etc.)

    Version 
    -------
    seqAlgo.simplify()

    """
    return icd9utils.getRootSequence(seq, sep=sep)

def normalize(seq, **kargs):
    """
    Transform the input string (a sequence of medical codes [and possibly other tokens])
    to another representation (e.g. simplified by taking base part of diagnostic codes) easier 
    to derive distributed vector representation. 

    Input
    -----
    seq: a string of medical code sequence

    """
    op_simplify, op_split = kargs.get('simplify_code', True), kargs.get('split', False)
    token_sep = kargs.get('token_sep', ',')

    if op_simplify: 
        seq = simplify(seq, sep=token_sep)

    if op_split: 
        if isinstance(seq, str): 
            seq = seq.split(token_sep)

    return seq

def transformBySeqContentType(seq, **kargs):  # refactored from seqReader
    return transform_by_ctype(seq, **kargs)
def transform_by_ctype(seq, **kargs): # ptype: (sequence) pattern type
    """
    Transform the input sequence (a string or a list/tuple/sequence) according to *sequence pattern type* 

    where sequence pattern types include:  
        1. regular (ordering perserved) (use 'regular')
        2. shuffled (removing ordering) (use 'random')
        3. diagnostic codes only (use 'diag')
        4. medications only (use 'med')
        5. lab tests/values only (use 'lab')
        6. overlap ngrams (use 'overlap_%dgram')

    Input 
    -----
    'seq' can be: 
      1. a list of tokens 
      2. a string of tokens 

    Related 
    -------

    (*) each disease-specific module should support transform()

    1. pattern.diabetes.segment()
       pattern.diabetes.transform()

    Memo
    ----
    Sequence content types (seq_ptype or ctype): 
    'regular'
    'random'
    'overlap_ngram' where 'n' depends on input 'ngram'
    'diag'
    'med'
    'lab'

    """
    seq_ptype = kargs.get('seq_ptype', 'regular')

    if seq_ptype == 'regular': 
        return seq # no op

    op_split = kargs.get('split', False)
    token_sep = kargs.get('token_sep', ',')
    
    # enforce list-of-tokens format
    if isinstance(seq, str): seq = seq.split(token_sep)

    if seq_ptype == 'random': 
        random.shuffle(seq) # inplace op 
    elif seq_ptype == 'diag': 

        # [note] pmed.isCondition(token): old, taking into account of encoded token with prefix 'condition'
        # seq = [token for token in seq if pmed.isICD(token)]  
        seq = filter(pmed.isICD, seq) 
    elif seq_ptype == 'med': 
        # seq = [token for token in seq if pmed.isMedCode(token)]
        seq = filter(pmed.isMedCode, seq)
    elif seq_ptype.startswith('overlap'): 
        assert kargs.has_key('length') 
        seq = transform_by_ngram_overlap(seq, **kargs)
    else: 
        raise NotImplementedError, "unrecognized seq_ptype: %s" % seq_ptype

    return seq

def qualify(seq, seq_ptype=None, predicate=None):
    """
    Given a sequence in list-of-tokens format, identify 
    the codes that satisify a given criterion (either through 
    seq_ptype or through a predicate) and return their indices. 

    Use
    ---
    1. Filter sequences and possibly their timestamps and labels simultaneously. 
    """  
    def select(): 
        return [i for i, token in enumerate(seq) if predicate(token)]

    # filter(predicate, seq) 
    wantedIdx = []
    if seq_ptype is not None: 
        if seq_ptype == 'random': 
            wantedIdx = random.shuffle(range(len(seq)))
        # elif seq_ptype == 'regular':
        #     wantedIdx = [i for i, token in enumerate(seq) if pmed.isICD(token) or pmed.isMedCode(token)]
        elif seq_ptype == 'diag': 
            # print('qualify> use predicate=pmed.isICD ...')
            wantedIdx = [i for i, token in enumerate(seq) if pmed.isICD(token)]
        elif seq_ptype == 'med': 
            # print('qualify> use predicate=pmed.isMedCode ...')
            wantedIdx = [i for i, token in enumerate(seq) if pmed.isMedCode(token)]
        # elif seq_ptype == 'lab': 
        #     pass 
        else: 
            raise NotImplementedError, "unrecognized seq_ptype: %s" % seq_ptype
    else: 
        assert hasattr(predicate, '__call__'), "No predicate is provided while seq_ptype is None."
        wantedIdx = select()
        
    return wantedIdx

def transform_by_ngram_overlap(seq, **kargs):
    """

    Related
    -------
    1. seqmaker.algorithms
    2. pattern.diabetes


    """
    n = kargs.get('length', 3)
    ol_ngram = zip(*[seq[i:] for i in range(n)])  # overlapping n-gram
    # this returns a list of tuples e.g. [(1, 2, 3), (2, 3, 4), (3, 4, 5) ...]
    if kargs.get('flat', False): # if True, return the overalpping n-gram directly without separating them
        return ol_ngram
    
    # want [0, 3, 6, ...], [1, 4, 7, ...], [2, 5, 8, ...] to be the sequences, where numbers are indices of the sequence
    seqx = []
    for i in range(n): 
        seqx.append(ol_ngram[i::n])  # 0, 3
    return seqx

def t_transform(**kargs): 
    def compare(D, D2): 
        for i, doc in enumerate(D):
            print("  + (before) %s" % doc)
            print("  + (after)  %s" % D2[i])
        return 
    def test_modify(D, predicate):
        assert hasattr(predicate, "__call__")
        D2 = []
        # e.g. predicate <- pmed.isICD(code) 
        for i, doc in enumerate(D):
            D2.append(filter(predicate, doc))
        return D2
    def filter_by_predicate(seq, predicate): 
        for tok in seq: 
            pmed.isICD(tok)

    def test_parallel_modify(D, predicate=None, ctype='diag'):
        pos_map = filterCodes(D, predicate=predicate, seq_ptype=ctype)  # if predicate is given, seq_ptype is ignored
        docs = indexToDoc(D, pos_map) # pos_map: a dictoinary > for i-th document, preserve only positions in 'idx'  
        return docs

    from pattern import diabetes as diab 
    from pattern import medcode as pmed
    seq = ['786.50', '786.50', '67894', '413.9', '250.00', '17.2']
    subseq = transform(seq, policy='prior', inclusive=True, seq_ptype='diag', predicate=diab.is_diabetes)
    print('> diag seq: %s' % subseq)

    ### new transform operation implementation
    D = [['789.0', '786.50', 'V67.9', 'MED:102225', 'MED:62934', 'NDC:00074518211', 
          '558.9', '414.9', '727.9', '414.9', '414.9', '715.96', '717.7', 
          '414.9', '414.9', '414.9', '414.9', '414.9', '999999', '715.90', '401.9', 
          '625.9', '611.72', 'V72.9'], 
          ['MED:102235', '250.00', 'MED:62334', '414.9', '727.9', '414.9', '414.9', ]]

    D2 = modify(D, seq_ptype='diag', predicate=None)
    # D2 = test_modify(D, predicate=pmed.isICD)
    # D2 = test_parallel_modify(D, ctype='diag')
    compare(D, D2)

    return 

def test(**kargs): 
    # import random 
    seq = ['A', 'C', 'G', 'T', 'A', 'C', 'A', 'T', 'C', 'G', 'C', 'T']
    seq = ['%s%s' % (e, random.sample(['x', 'y'], 1)[0]) for e in range(200)]

    n = 3
  
    # use following methods to find N-grams 
    s1 = transform_by_ngram_overlap(seq, ng=n, flat=True)
    print('> %d-gram:\n%s\n' % (n, s1))
    s2 = transform(seq, policy='noop', seq_ptype='overlap', ng=3, flat=True)  # policy: noop => bypass cut
    print('> %d-gram:\n%s\n' % (n, s2))

    # the above returns overlapping n-gram
    # but if we want to separate each n-gram sequence 
    print(s2[0::n])
    print('> non-overlapping:\n%s\n\n%s\n\n%s\n' % (s2[0::n], s2[1::n], s2[2::n]))

    seq = range(200)

    # inverse: True => TTTTF
    s3 = transform(seq, policy='prior', predicate=lambda x: x < 15, inverse=True, seq_ptype='overlap', ng=3, flat=True)
    for i, s in enumerate(s3): 
    	print('seq #%d => %s' % (i, s))

    return 

def test2(**kargs):

    ### new transformation operations 
    t_transform()

    return

if __name__ == "__main__": 
    test2()
