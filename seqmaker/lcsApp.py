# encoding: utf-8

import numpy as np
import multiprocessing

import os, sys, re, random, time, math
import collections, gc
from datetime import datetime 

# pandas 
from pandas import DataFrame
import pandas as pd 

# gensim 
from gensim.models import Doc2Vec
import gensim.models.doc2vec

# tpheno
import seqReader as sr
import seqparams, seqAlgo
import labeling 
from batchpheno.utils import div

gHasConfig = True
try: 
    from config import seq_maker_config, sys_config
except: 
    print("tdoc> Could not find default configuration package.")
    gHasConfig = False

from tdoc import TDoc



def calc_cache_pos(strings, indexes):
    factor = 1
    pos = 0
    for s, i in zip(strings, indexes):  # iterate over each string
        pos += i * factor
        factor *= len(s)
    return pos

def lcs_back(strings, indexes, cache):
    if -1 in indexes:
        return ""
    match = all(strings[0][indexes[0]] == s[i]
                for s, i in zip(strings, indexes))
    if match:
        new_indexes = [i - 1 for i in indexes]
        result = lcs_back(strings, new_indexes, cache) + strings[0][indexes[0]]
    else:
        substrings = [""] * len(strings)
        for n in range(len(strings)):
            if indexes[n] > 0:
                new_indexes = indexes[:]
                new_indexes[n] -= 1
                cache_pos = calc_cache_pos(strings, new_indexes)
                if cache[cache_pos] is None:
                    substrings[n] = lcs_back(strings, new_indexes, cache)
                else:
                    substrings[n] = cache[cache_pos]
        result = max(substrings, key=len)
    cache[calc_cache_pos(strings, indexes)] = result
    return result

def lcs_back2(strings, indexes, cache):
    if -1 in indexes:
        return []
    match = all(strings[0][indexes[0]] == s[i]
                for s, i in zip(strings, indexes))
    if match:
        new_indexes = [i - 1 for i in indexes]
        result = lcs_back2(strings, new_indexes, cache)
        result.append(strings[0][indexes[0]])
    else:
        substrings = [[] for i in range(len(strings))] 
        for n in range(len(strings)):
            if indexes[n] > 0:
                new_indexes = indexes[:]
                new_indexes[n] -= 1
                cache_pos = calc_cache_pos(strings, new_indexes)
                if cache[cache_pos] is None:
                    substrings[n] = lcs_back2(strings, new_indexes, cache)
                else:
                    substrings[n] = cache[cache_pos]
        result = max(substrings, key=len)
    cache[calc_cache_pos(strings, indexes)] = result
    return result

def lcs(strings):
    """
    >>> lcs(['666222054263314443712', '5432127413542377777', '6664664565464057425'])
    '54442'
    >>> lcs(['abacbdab', 'bdcaba', 'cbacaa'])
    'baa'
    """
    import random
    isListOfTokens = False
    N = len(strings)
    if N >= 1: 
        sample_str = random.sample(strings, 1)[0]
        if isinstance(sample_str, list):
            isListOfTokens = True 
        else: 
            assert isinstance(sample_str, str)

    if len(strings) == 0:
        return [] if isListOfTokens else ""
    elif len(strings) == 1:
        return strings[0]
    else:
        cache_size = 1
        # result_seq = ""
        for s in strings:  # for each string 
            cache_size *= len(s)  # size(string) ~ size(list of tokens)
        cache = [None] * cache_size
        indexes = [len(s) - 1 for s in strings]

        if isListOfTokens: 
            return lcs_back2(strings, indexes, cache)
        else: 
            return lcs_back(strings, indexes, cache)
    return [] if isListOfTokens else ""

def deriveLCS(D, **kargs):
    """
    Given a set of documents (coding sequences), find their LCSs and rank them 
    according to frequencies (where frequency refers to the number of documents having the LCS). 

    Params
    ------
    D: corpus, list of lists of strings/tokens
    document_clusters

    max_n_pairs 
    remove_duplicates
    min_length
    max_length
    topn_lcs: preserve only this many LCSs (ranked by frequency)

    consolidate_lcs: if True, LCSs up to permutations are considered as identicial 

    lcs_policy

    Use
    ---
    1. load document first 
        D, T, L = processDocuments(cohort=cohort_name, seq_ptype=seq_ptype, 
            predicate=kargs.get('predicate', None),
            simplify_code=kargs.get('simplify_code', False), 
            ifiles=kargs.get('ifiles', []))

    2. find top N LCS by global frequency
       train classifier by treating these LCS either as binary labels (e.g. is_lcs_0 or not)
       or by treating them as multiple class labels 
       so for instance, if topn_lcs = 20, there are effectively 20+1=21 labels {lcs1, lcs2, ..., lcs20, none_of_the_above}
    """
    def filter_by_length(lcs_counts=None): # lcsMinLength, lcsMaxLength, lcsFMapGlobal (could use lcsFMap)
        if lcs_counts is None: lcs_counts = lcsFMapGlobal # or use local/cluster-level map: lcsFMap

        sep = lcs_sep # ' '
        if lcsMinLength is not None and lcsMaxLength is not None: # screen shorter LCS (not as useful for observing disease progression)
            ngr_cnts = []  # adpated from LCS analysis within the space of common n-grams (seqmaker.pathwayAnalyzer)
            for s in lcs_counts.keys(): 
                ntok = len(s.split(sep))  # number of tokens
                if ntok >= lcsMinLength and ntok <= lcsMaxLength: 
                    ngr_cnts.append((s, lcs_counts[s]))  # number of times the LCS (s) was matched
        else: 
            ngr_cnts = [(s, cnt) for s, cnt in lcs_counts.items()]
        return ngr_cnts
    def filter_by_length2(): # [params] lcsmapEst
        # screen shorter LCSs (which are not as useful for observing disease progression)
        hasLenConstraint = True if (lcsMinLength is not None) and (lcsMaxLength is not None) else False 

        # choose 
        if hasLenConstraint: 
            for s, docIds in lcsmapEst.items(): 
                ntok = len(s.split(lcs_sep))  # number of tokens           
                if ntok >= lcsMinLength and ntok <= lcsMaxLength: 
                    ngr_cnts.append((s, len(docIds)))  # number of documents the LCS (s) appear in the document set
        else: 
            ngr_cnts = [(s, len(docIds)) for s, docIds in lcsmapEst.items()]
        return ngr_cnts
    def select_lcs_by_length(): # [params] lcsMap
        # get all eligible LCSs that satisfy the length constraint 
        hasLenConstraint = True if (lcsMinLength is not None) and (lcsMaxLength is not None) else False 
        S = set()
        if hasLenConstraint:
            for cid, lcs_set in lcsMap.items(): # lcsMap: cluster ID -> LCSs set but here, we focus on only 'one cluster'
                for s in lcs_set: 
                    ntok = len(s.split(lcs_sep))  # number of tokens 
                    if ntok >= lcsMinLength and ntok <= lcsMaxLength: 
                        S.add(s)
        else: 
            for cid, lcs_set in lcsMap.items(): 
                S.update(lcs_set)
        return S 
    def stratify_by_length():
        assert lcsMinLength is not None and lcsMaxLength is not None
        
        ldict = {l: {} for l in range(lcsMinLength, lcsMinLength+1)}  
        for s, docIds in lcsmapEst.items(): 
            ntok = len(s.split(lcs_sep))  # number of tokens           
            if ntok >= lcsMinLength and ntok <= lcsMaxLength: 
                ldict[ntok][s] = docIds
        return ldict
    def peek_sep(): 
        # peek data type 
        doc = DCluster[random.sample(cIDs, 1)[0]]
        if isinstance(random.sample(doc, 1)[0], str): 
            print('test> find the doc data type and its delimiter ...')
            
            # infer seperator? 
            sep_candidates = [',', ' ']
            found_sep = False
            the_sep = ' '
            for cid, documents in DCluster.items(): 
                assert len(documents) > 0
                longest_doc = sorted([(i, len(doc)) for i, doc in enumerate(documents)], key=lambda x:x[1], reverse=True)[0][1]

                # try splitting it
                for tok in sep_candidates: 
                    # select hte longest one 
                    if longest_doc.find(tok) > 0:  # e.g. '300.01 317 311' and tok = ' '
                        if len(longest_doc.split(tok)) > 1: 
                            the_sep = tok
                            found_sep = True
                            break 
                if found_sep: break 
            print("status> determined cluster_ngram seperator to be '%s' =?= '%s' (default) ... " % (the_sep, sep))
            sep = the_sep
        return sep
    def merge_cluster(clusters, cids=None): 
        assert isinstance(clusters, dict)
        member = clusters.itervalues().next()
        assert hasattr(member, '__iter__')

        data = []
        if cids is None: # merge all 
           for cid, members in clusters.items(): # members can be a list or list of lists
               data.extend(members)
        else: 
            for cid in cids: 
                try: 
                    members = clusters[cid]
                    data.extend(members)
                except: 
                    raise ValueError, "Unknown cluster ID: %s" % cid
        if not data:
            assert cids is not None 
            print('warning> No data selected given cids:\n%s\n' % cids)
        return data
    def normalize_docs(): # convert input (D) into a canonical form
        # D: either document clusters (cid -> docs) or documents (a list of list of strings/tokens)
        if isinstance(D, dict): 
            # do nothing
            raise ValueError, "use deriveLCS2() to consider multiple clusters"
            # member = D.itervalues().next()
            # assert hasattr(member, '__iter__')
            # DCluster = D
        else: 
            # [test] D must be a list of lists of tokens
            x = random.randint(0, len(D)-1)  # nDoc
            assert len(D) > 0 and hasattr(D[x], '__iter__'), "Invalid input D[%d]:\n%s\n" % (x, D[x])
            print('  + Aggregate %d documents as one single cluster' % len(D))
            DCluster = {0: D}  # one cluster
        return DCluster
    def make_pairs(D_subset, cid=0, policy='random'): # [params] maxNPairs, policy={'longest_first', 'random'}
        # select pairs of documents from which to derive LCSs 
        n = len(D_subset)
        # pairwise lcs   
        n_total_pairs = (n*(n-1))/2.  # approx.
        pairs = [(i, j) for i in range(n) for j in range(i+1, n)]  # likely data source: condition_drug_timed_seq-PTSD-regular.csv
        if maxNPairs is None: return pairs

        # condition: maxNPairs is a number
        if policy.startswith('rand'): # randomized
            npt = len(pairs)
            pairs = random.sample(pairs, min(maxNPairs, npt))  # use lengths 
        elif policy.startswith('long'): # longest first 
            # priority list
            plist = sorted([(i, len(d)) for i, d in enumerate(D_subset)], key=lambda x: x[1], reverse=True) # descending order, longest first
            n = (math.sqrt(8 * maxNPairs)+1)/2  # number of documents needed to make max pairs
            n = int(math.ceil(n)) # overesimate a bit 
            pairs = list(itertools.combinations([i for i, _ in plist[:n]], 2))
            # pairs = itertools.combinations([i for i in plist[:n]], 2)  # return an iterator
        else: 
            raise NotImplementedError, "make_pairs> unrecognized document pairing policy: %s" % policy

        cid = normalize_cluster_id(cid)
        print('  + [cluster %s] choose %d out of %d possible document pairs' % (cid, len(pairs), n_total_pairs))
        return pairs
    def verify_lcs(cid): 
        div(message='[cluster %s] Finished computing pairwise LCSs (%d non-empty pairs out of total %d)' % (cid, n_pairs, n_total_pairs))
        print("  + Found %d void pairs (at least one doc is empty)" % n_void)
        
        n_sample = 5
        print("  + example LCSs found (n_sample=%d) ......" % n_sample)
        for lcs in random.sample(lcsMap[cid], min(n_sample, len(lcsMap[cid]))): 
            print('    : %s' % str(lcs))  # a list of strings (of ' '-separated codes)
        return
    def consolidate_permutations(lcsmap): # lcsmap: doc -> {lcs}
        """
        sequences that consist of the same codes are considered as identical. 
        """
        if not tAggLCS: 
            # noop 
            return lcsmap
        
        sep = Pathway.lcs_sep
        N0 = len(lcsmap)
        eqvmap = {}  # equivalence map: (re-ordered) lcs -> docId 
        sortedlcsmap = {}  # lcs -> (re-ordered) lcs
        for lcs, docIds in lcsmap.items(): 
            sorted_lcs = sortedSequence(lcs, sep=sep)  # Pathway.sep is typically a space
            sortedlcsmap[lcs] = sorted_lcs
            # lcs_list = sorted(lcs.split(lcs_sep))
            # lcs_str = lcs_sep.join(lcs_list)
            if not eqvmap.has_key(sorted_lcs): eqvmap[sorted_lcs] = set(docIds)
            eqvmap[sorted_lcs].update(docIds)  # document IDs with LCSs up to permutations are consolidated

        # [test]
        N = len(eqvmap)        
        for slcs in eqvmap.keys(): 
            eqvmap[slcs] = sorted(eqvmap[slcs])  # sorted() return a list, which is what we want
        print('consolidate_permutations> numeber of LCS entries: %d -> %d (smaller?)' % (N0, N))
        assert N <= N0

        # use the original lcs with the right ordering as entries
        lcsmap2 = {}
        mappedSet = set()  # keep only one entry for {lcs} that share the same document IDs 
        for lcs, docIds in lcsmap.items(): 
            # if not sortedlcsmap[lcs] in mappedSet: # the sorted/standardized lcs has been mapped (to docIDs) yet? 

            # LCSs up to permutations will reference the same document IDs (and get updated multiple times)
            lcsmap2[lcs] = eqvmap[sortedlcsmap[lcs]] # lcs -> sorted lcs -> (consolidated) docIDs (use the equivalent document IDs) 
            # mappedSet.add(sortedlcsmap[lcs]) 

        # [log] lcsmap2 (n=54631) should have the same size as eqvmap (n=53410)
        assert len(lcsmap2) >= N, "lcsmap2 (n=%d) should have the same size as eqvmap (n=%d)" % (len(lcsmap2), N)

        return lcsmap2  # condition: many entries (equivalent seq) refer to the same docIDs
    def filter_lcs_by_docfreq(topn=20, min_df=50): # params: lcsmap, minNDoc
        # preserve the lcs that is present in at least a minimum number of documents
        candidates = []
        for lcs, dx in lcsmap.items():  # lcsmap is the result of comparing to the entire document set
            nd = len(dx)
            if nd >= min_df: 
                candidates.append(lcs)
        nl = len(candidates)
        print('  + Found %d LCSs present in >= %d documents' % (nl, min_df))

        lcsmap2 = {}
        if nl > topn: 
            print('  + sample a subset of LCSs (target=%d)' % topn)

            ### policy #1 select those that occur most often document wide
            ranked = sorted([(c, len(lcsmap[c])) for c in candidates], key=lambda x: x[1], reverse=True)
            maxLCS, maxRankScore = ranked[0][0], ranked[0][1]
            print("  + maximum DF: %d, LCS=\n%s\n" % (maxRankScore, maxLCS))

            ### policy #2 random subsampling
            # candidates = random.sample(candidates, topn)
            
            candidates = [lcs for lcs, cnt in ranked[:topn]]
            # [code] try to break the tie
            
            for lcs in candidates: 
                lcsmap2[lcs] = lcsmap[lcs]  # lcs -> {doc}

        else: 
            if nl > 0: # at least some LCSs exist with DF > min_df
                print('  + warning: candidates (n=%d) less than target (topn=%d) => use all LCSs' % (nl, topn))
                for lcs in candidates: 
                    lcsmap2[lcs] = lcsmap[lcs]  # lcs -> {doc}
            else: 
                print('  + Warning: No LCS found with DFreq >= %d' % min_df)
                # lcsmap2 empty
        
        print('  + concluded with n:%d <=? topn:%d' % (len(lcsmap2), topn))
        return lcsmap2
    def filter_by_anchor(): # [params] lcsmap: lcs -> docIDs
        # eligible LCSs that contain the target diag code (works for cases where diag codes are accurate)
        # e.g. 309.81 Posttraumatic stress disorder (PTSD)
        # cohort dependent 
        cohort_name = kargs.get('cohort', '?')
        print('filter_by_anchor> cohort=%s' % cohort_name)
        N0 = len(lcsmap)
        lcsmap2 = {}
        target = kargs.get('anchor', '309.81') 
        if cohort_name == 'PTSD':      
            # target = kargs.get('anchor', '309.81') #
            for lcs, dx in lcsmap.items():
                if target in lcs: 
                    lcsmap2[lcs] = dx 
            # N = len(lcsmap2)
        else: 
            lcsmap2 = lcsmap
            print('filter_by_anchor> cohort=%s > noop!' % cohort_name)
            # noop 

        N = len(lcsmap2)
        print('filter_by_anchor> LCS ~ target=%s > N: %d -> %d' % (target, N0, N))

        return lcsmap2
    def filter_lcs_by_uniq(topn=20, min_df=50): # params: lcsmap
        # "try" to preserve LCSs with different code set 
        # see constraints 1 - 3

        ### constraint #1: DF
        candidates = []
        for lcs, dx in lcsmap.items():  # lcsmap is the result of comparing to the entire document set
            nd = len(dx)
            if nd >= min_df: 
                candidates.append(lcs)
        nl = len(candidates)
        print('filter_lcs_by_uniq> Found %d LCSs present in >= %d documents (label=%s)' % \
            (nl, min_df, kargs.get('label', 'n/a'))) # [log] 6494 LCSs present in >= 50 documents

        if nl > topn: 
            print('  + sample a subset of LCSs by diversity (target=%d)' % topn)
            
            ### constraint #2: Diversity
            # candidates2 = apply_diversity_constraint(candidates) # but this has a flaw 
            # nl2 = len(candidates2)

            ### constraint Diversity 2a: No LCS should contain codes that are a subset of another LCS 
            #       When two LCSs are similar (one LCS contains codes that are a subset of those of the other LCS),  
            #       keep the more informative LCS (longer one) as the candidate LCS label. 
            lcsToSubsumed = subsume_subset(candidates)  # lcs to other lcs<i> s.t. lcs > lcs<i>

            # [log] 7138 with sufficient DF down to 2927 diversified LCSs (objective topn: 9)
            candidates2 = lcsToSubsumed.keys() # these LCSs are definitely not a subset of one another
            nl2 = len(candidates2)
            print('  + %d with sufficient DF down to %d diversified LCSs (objective topn: %d)' % (nl, nl2, topn))

            if nl2 < topn:  # need to pad additional (topn-nl2)
                # [policy] if not enough LCSs selected, pick random LCS from original candidate set
                # a. pick those with greatest diff in freq distr 
                # b. pick those with greatest edit distance
                
                undiversified = set(candidates)-set(candidates2)
                candidates3 = random.sample(undiversified, min(topn-nl2, len(undiversified)))
                print('  + padding additoinal %d (=? objective: %d) LCSs that do not satisfy diversity requirement ...' % \
                    (len(candidates3), topn-nl2))
                candidates = candidates3 + candidates2 
            else: # surplus, need to select a subset
                # [policy]
                # always choose the longest? OR random? 
                
                # candidates = random.sample(set(candidates2), topn)
                ### constraint #3: Length
                lcsLengths = [(s, len(s.split(lcs_sep))) for s in candidates2]  # lcsmap is the result of comparing to the entire document set
                lcsLengths = sorted(lcsLengths, key=lambda x: x[1], reverse=True)[:topn]
                print('  + select %d longest LCSs among %d (diversified) candidates with ranks:\n%s\n' % (topn, nl2, lcsLengths))
                candidates = [lcs for lcs, _ in lcsLengths]
            
            assert len(candidates) == topn, "  + size of LCS set (n=%d) <> topn=%d!" % (len(candidates), topn)
        else:  # n(LCS) < topn
            if nl > 0:  # at some LCSs exist with DF >= min_df
                print('  + warning: candidates (n=%d) less than target (topn=%d) => use all LCSs' % (nl, topn))
            else: 
                print('  + Warning: No LCS found with DFreq >= %d' % min_df)
                # lcsmap2 empty; candidates is an empty set
                
        lcsmap2 = {}
        for lcs in candidates: 
            lcsmap2[lcs] = lcsmap[lcs]  # lcs -> {doc}
        
        print('  + concluded with n:%d <=? topn:%d' % (len(lcsmap2), topn))     
        return lcsmap2
    def filter_by_policy(lcsmap, topn=20, min_df=50): # params: lcsmap
        # use only filters above
        N0 = len(lcsmap)  # direct access of lcsmap won't work with nested calls below
        policy = kargs.get('lcs_policy', 'uniqness')
        if policy.startswith('u'):  # uniqnuess
            lcsmap = filter_lcs_by_uniq(topn=topn, min_df=min_df) # select top n LCSs, all of which occurr in at least m documents 
        elif policy.startswith('d'):  # df, document frequency
            lcsmap = filter_lcs_by_docfreq(topn=topn, min_df=min_df)
        elif policy == 'noop': 
            pass
        else: 
            raise NotImplementedError, "unrecognized LCS filter strategy: %s" % policy
        print("filter_by_policy> lcsmap size %d -> %d | policy=%s" % (N0, len(lcsmap), policy))
        return lcsmap
    def diversity_by_set(lcs): # params: lcs_sep 
        # output: a lcs string, sorted, repeated codes removed
        return lcs_sep.join( str(e) for e in sorted(set(lcs.split(lcs_sep))) )
    def apply_diversity_constraint(candidates):
        selected = set()
        for lcs in candidates: 
            s = diversity_by_set(lcs) # set uniquess; sequence of uniq codes in a string
            if not s in selected: 
                # lcsmap2[lcs] = lcsmap[lcs] # lcs -> {doc}
                candidates.append(lcs)  # caveat: subset lcs can still be an entry 
        return candidates
    def is_subset(lcs1, lcs2): # lcs1 < lcs2? 
        lcss1 = set(lcs1.split(lcs_sep))
        lcss2 = set(lcs2.split(lcs_sep))   
        return lcss1 < lcss2  # lcss1 is a subset of lcss2?  
    def subsume_subset0(alist):  # assuming alist is ordered by x >= y >= z in the sense of subset relations
        # e.g. ['x y x z', 'x d', 'x y a b z', 'y x z', 'x y', 'x z', 'z']
        #      'x y a b z' subsumes S: {'y x z', 'x y', 'x z', 'z'} because 'x y a b z' > e in S
        nl = len(alist)
        alist = sorted(alist, cmp=make_comparator(is_subset), reverse=True) # '>' relation; least subset to most subset
        ulist = []  # uniq lcs
        i = 0
        while i < nl: 
            lcs = alist[i]
            ulist.append(lcs)
            if i == nl-1: break 
            for j, lcs2 in enumerate(alist[i+1:]):  # loop until lcs > lcs2 is untrue
                # if lcs<i> is not a subset of anyone (! i < {j}), then it's definitely a candidate
                # if i < j, then j subsumes i to be the candidate
                inext = j+(i+1)
                if is_subset(lcs2, lcs): # lcs subsumes lcs2 
                    # continue
                    pass 
                else: 
                    # ulist.append(lcs)
                    break  # cannot subsume anynore 
            i = inext  # next search point (which is at least i+1)   
        return ulist
    def subsume_subset(alist):
        nl = len(alist)
        alist = sorted(alist, cmp=make_comparator(is_subset), reverse=True) # '>' relation; least subset to most subset
        
        adict = {}
        selected = set() # has the lcs been subsumed by any other lcs? 
        for i, lcs in enumerate(alist):
            if lcs in selected: continue
            adict[lcs] = []
            for j, lcs2 in enumerate(alist[i+1:]): 
                if is_subset(lcs2, lcs): # lcs subsumes lcs2 
                    # if not adict.has_key(lcs): adict[lcs] = []
                    adict[lcs].append(lcs2); selected.add(lcs2)
        return adict    
    def is_subset_to_any(s, aset, n=100): 
        # does the LCS (s) contains codes that are a subset of any of the LCSs in aset? 
        me = set(s.split(lcs_sep))
        aset0 = random.sample(aset, min(len(aset), n))
        for s0 in aset0: 
            you = set(s0.split(lcs_sep))
            if me < you: 
                return True
        return False
    def test_lcs_consistency(): # lcsmapEst vs lcsmap, lcsmapEst should have more counts
        print('... test_lcs_consistency ...')
        print('  + total LCS of all lengths: %d, filtered: %d' % (len(lcsmapEst), len(lcsmap)))
        n, nT = 0, 100
        for s, dx in lcsmapEst.items(): 
            if s in lcsmap: # filtered by min and max lengths, and matched against all docs by analyzeLCS
               dx2 = lcsmap[s]
               assert len(dx2) >= len(dx), \
                   "matching with full document set should get more count but lcsmap[s]=%d < lcsmapEst[s]=%d" % (len(dx2), len(dx))
               print('  + random match: %s -> %s (%d)' % (s, dx, len(dx)))
               print('  + full match:   %s -> %s (%d)' % (s, dx2, len(dx2)))
               n += 1 
            if n >= nT: break
        return  
    def test_min_docs(): # lcsmap (after filter_lcs operation), minNDoc
        # nL0, nL = len(lcsmapEst), len(lcs_candidates)  # must follow select_lcs_by_length
        print("deriveLCS> nLCS: %d (initial) -> %d (length qualified) -> %d (consolidated? %s) -> %d (topn)" % \
            (nL0, nL, nLc, tAggLCS, len(lcsmap))) # [log] 89435 -> 51216 -> 20

        for lcs, dx in lcsmap.items():
            assert len(dx) >= minNDoc, "df is only at %d for LCS: %s" % (len(dx), lcs)
        return 
    def show_params(cid='n/a'): 
        cid = normalize_cluster_id(cid)
        print('  + [cluster=%s] topn_lcs: %d, max_n_pairs: %d, remove_duplicates? %s, pairing_policy=%s' % \
            (cid, topn_lcs, maxNPairs, removeDups, pairing_policy))
        print('  +              min_length: %d, max_length: %d' % (lcsMinLength, lcsMaxLength))
        print('  +              min_ndocs: %d' % minNDoc)
        print('  +              splice policy: %s' % kargs.get('splice_policy', 'noop'))
        print('  +              lcs filter policy: %s' % kargs.get('lcs_filter_policy', 'uniq'))
        print('  +              consolidate similar LCS? %s' % tAggLCS)
        return
    def normalize_cluster_id(cid): 
        label_id = kargs.get('label', 'n/a')
        if label_id != 'n/a': cid = process_label(label_id)
        return cid
    def process_label(l): 
        # remove spaces e.g. CKD Stage 4
        return ''.join(str(e) for e in l.split())
    def do_splice(): # only for file ID, see secondary_id()
        if not kargs.has_key('splice_policy'): return False
        if kargs['splice_policy'].startswith(('noop', 'reg', 'compl', )):  # effective operations 
            return False 
        return True
    def lcs_file_id(): # params: {cohort, lcs_type, lcs_policy, seq_ptype, splice_policy, consolidate_lcs, length}
        # [note] lcs_policy: 'freq' vs 'length', 'diversity'
        #        suppose that we settle on 'freq' (because it's more useful) 
        #        use pairing policy for the lcs_policy parameter: {'random', 'longest_first', }
        adict['cohort'] = kargs.get('cohort', 'generic')  # this is a mandatory arg in makeLCSTSet()
        adict['lcs_type'] = ltype = kargs.get('lcs_type', 'global')  # global: Common LCSs defined in global scope (as opposed to cluster or local scopes)
        adict['lcs_policy'] = kargs.get('lcs_policy', 'uniq') # definition of common LCSs

        # use {seq_ptype, splice_policy, length,} as secondary id
        adict['suffix'] = adict['meta'] = suffix = secondary_id()
        return adict
    def secondary_id(): # attach extra info in suffix
        ctype = kargs.get('seq_ptype', 'regular')
        suffix = kargs.get('suffix', ctype) # vector.D2V.d2v_method
        if do_splice(): suffix = '%s-%s' % (suffix, kargs['splice_policy'])
        if kargs.get('consolidate_lcs', True): suffix = '%s-%s' % (suffix, 'iso') # LCSs up to permutations consider as the same? 
        # suffix = '%s-len%s' % (suffix, kargs.get('min_length', 3))

        label_id = kargs.get('label', None)
        if label_id is not None: suffix = '%s-L%s' % (suffix, label_id)
        # if kargs.get('simplify_code', False):  suffix = '%s-simple' % suffix
        return suffix 
    def load_lcs(): # [note] for now, only used in makeLCSTSet()
        adict = lcs_file_id()
        cohort = adict['cohort']
        ltype, lcs_select_policy = adict['lcs_type'], adict['lcs_policy']
        suffix = adict['meta'] 
        return Pathway.load(cohort, lcs_type=ltype, lcs_policy=lcs_select_policy, suffix=suffix, dir_type='pathway')
    def save_lcs(df):  
        adict = lcs_file_id()
        cohort = adict['cohort']
        ltype, lcs_select_policy = adict['lcs_type'], adict['lcs_policy']
        suffix = adict['meta']         
        # fpath = Pathway.getFullPath(cohort=cohort, lcs_type=ltype, lcs_policy=lcs_select_policy, suffix=suffix) # [params] dir_type='pathway' 
        print('  + saving LCS labels ...')
        Pathway.save(df, cohort=cohort, lcs_type=ltype, lcs_policy=lcs_select_policy, suffix=suffix, dir_type='pathway')
        return
    def eff_docs(): # params: lcsmap
        # find document subset eventually included that lead to the final set of LCSs 
        deff = set()
        for lcs, dx in lcsmap.items(): 
            deff.update(dx)
        return deff
    import motif as mf
    from seqparams import Pathway
    import itertools
    # import vector

    topn_lcs = kargs.get('topn_lcs', 20)   # but predominent will be shorter ones
    minNDoc = kargs.get('min_ndocs', 50)  # candidate LCSs must appear in >= min_ndocs documents
   
    maxNPairs = kargs.get('max_n_pairs', 100000) # Running pair-wise LCS comparisons is expensive, may need to set an upperbound for maximum parings
    nSubsampling = kargs.get('n_subsampling', 100000)

    removeDups = kargs.get('remove_duplicates', True) # remove duplicate codes with consecutive occurrences (and preserve only the first one)
    pairing_policy = kargs.get('pairing_policy', 'random') # policy = {'longest_first', 'random', }

    

    # only see LCS of length >= 5; set to None to prevent from filtering
    lcsMinLength, lcsMaxLength = kargs.get('min_length', 2), kargs.get('max_length', 20) 
    lcsMinCount = kargs.get('min_count', 10)  # min local frequency (used for 'diveristy' policy)
    tAggLCS = kargs.get('consolidate_lcs', True)
    # lcsSelectionPolicy  # options: frequency (f), global frequency (g), longest (l)
    show_params()    

    nDoc = 0
    DCluster = normalize_docs() # D can be a cluster or a list of documents
    assert len(DCluster) == 1, "use deriveLCS2() to consider multiple clusters"

    lcsPersonMap = {}   # LCS -> person ids  # can be very large
    lcsFMapGlobal = {}  # LCS -> count (global across all clusters)
    lcsFMap = {cid:{} for cid in DCluster.keys()}  # LCS frequency map: cid -> lcs -> count
    lcsMap = {} # cid -> canidate LCS
    lcs_sep = Pathway.lcs_sep # ' '  # lcs_sep = peek_sep()

    lcsmapEst = {}  # LCS -> document IDs
    label_id = kargs.get('label', None)  # used for label data which subsumes cid in deriveLCS where only 1 cluster is considered
    for cid, documents in DCluster.items(): # unless specified otherwise, assume that all documents (D) are one big cluster
        assert cid < 1 
        # cid = normalize_cluster_id(cid) # use label if available as cid

        lcsx = set()  # per-cluster LCS set
        nDoc += len(documents)

        # pairwise lcs  # [todo] 
        # pairs = make_pairs(D_subset=documents, cid=cid, policy=pairing_policy) # [params] policy='longest_first'
        # n_total_pairs = len(pairs)
        show_params(cid=cid)

        n_pairs = n_void = 0
        # for pair in pairs:  # likely data source: condition_drug_timed_seq-PTSD-regular.csv
        for _ in range(nSubsampling): 
            # i, j = pair

            # doc1, doc2 = documents[i], documents[j]

            # if len(doc1) == 0 or len(doc2) == 0: continue 
            # assert isinstance(doc1, list) and isinstance(doc2, list), "doc1=%s, doc2=%s" % (doc1, doc2)

            # [expensive]
            # sl = mf.lcs(doc1, doc2)  # sl is a list of codes since doc1: list, doc2: list
            sl = lcs(D_subset)

            if removeDups: sl = [e[0] for e in itertools.groupby(sl)]

            # convert list of codes to a string for indexing
            s = lcs_sep.join(sl)  

            # count local frequencies
            if s: # don't add emtpy strings
                lcsx.add(s) 
                
                # cannot do this if there were > 1 clusters
                if not lcsmapEst.has_key(s): lcsmapEst[s] = set()

                # count local/cluster frequencies
                if not s in lcsFMap[cid]: lcsFMap[cid][s] = 0
                lcsFMap[cid][s] += 1 

                # count global frequencies
                if not s in lcsFMapGlobal: lcsFMapGlobal[s] = 0
                lcsFMapGlobal[s] += 1

                # if not s in lcsPersonMap: lcsPersonMap[s] = set()
                # lcsPersonMap[s].update([i, j])

                n_pairs += 1   # effective pairs excluding empty ones
                r = n_pairs/(n_total_pairs+0.0)
                
                # r_percent = int(math.ceil(r*100))
                # percentages = {interval: 0 for interval in range(0, 100, 10)}
                if n_pairs % 500 == 0: 
                    print('  + [cluster %s] finished computing (%d out of %d ~ %f%%) pairwise LCS ...' % \
                        (cid, n_pairs, n_total_pairs, r*100))
            else: 
                n_void += 1  # at least one empty 
        
        lcsMap[cid] = lcsx
        verify_lcs(cid)  # [test] 
    ### end foreach (cid, doc)

    # filter LCSs 
    # [note] may need to just focus on certain lengths
    #        high frequency LCSs tend to be short
    # n_pairs0 = len(lcsFMapGlobal)  # or use lcsFMap: cluster level map
    # lcsCounts = filter_by_length(lcs_counts=lcsFMapGlobal, sep=lcs_sep) # [params] lcsFMapGlobal
    lcs_candidates = select_lcs_by_length() # lcsMap, lcsMinLength, lcsMaxLength
    nL0, nL = len(lcsmapEst), len(lcs_candidates)

    # lcsmap: lcs -> docId  ... given an LCS, find which documents contain it? 
    lcsmap = analyzeLCS(D, lcs_set=lcs_candidates)  # lcs_sep=Pathway.sep
    test_lcs_consistency()
    
    # consolidate LCSs of exactly the same set of codes (but different ordering)
    # no-op if consolidate_lcs is set to False
    lcsmap = consolidate_permutations(lcsmap) # lcsmap 
    nLc = len(lcsmap)

    # lcsmap: lcs => documents (containing the lcs)

    ### now, filter the LCSs
    # lcsCounts = filter_by_length2() # lcsmapEst
    # lcsmap = filter_lcs_by_frequency(topn=topn_lcs)  # return only topn_lcs entries from lcsmap to make labels
    # lcsmap = filter_lcs_by_docfreq(topn=topn_lcs, min_df=minNDoc)

    #> baseline filter
    lcsmap = filter_by_anchor()  # e.g. PTSD-related LCS must contain 309.81

    #> strategy-based filter e.g. document frequency, uniqueness, lengths
    # [note] it's necessary to pass on lcsmap due to nested calls within the inner function
    lcsmap = filter_by_policy(lcsmap, topn=topn_lcs, min_df=minNDoc)  # option: lcs_policy
    test_min_docs() # params: nL0, nL, nLc, lcsmap

    # compute statistics and save data
    header = ['length', 'lcs', 'n_uniq', 'count', 'df', ] # Pathway.header_global_lcs # ['length', 'lcs', 'count', 'n_uniq', 'df', ] 
    adict = {h:[] for h in header}

    nDEff = len(eff_docs()) # <- lcsmap
    print('  + effective number of documents where candidate LCSs are present: %d (out of %d => r: %f)' % (nDEff, nDoc, nDEff/(nDoc+0.0)))
    for lcs, dx in lcsmap.items(): 
        ss = lcs.split(lcs_sep)
        cnt = len(dx)
        adict['length'].append(len(ss))
        adict['lcs'].append(lcs)
        adict['count'].append(cnt)            # number of documents containing the LCS
        adict['df'].append( round(cnt/(nDoc+0.0), 3))    # document frequency
        adict['n_uniq'].append(len(set(ss)))  # diversity: number of unique tokens within the LCS

    df = DataFrame(adict, columns=header)
    df = df.sort_values(['length', 'count'], ascending=True)
    save_lcs(df)

    return lcsmap # LCS -> document IDs  (this is needed to label the document set)


def t_lcs(): 
    def to_list_repr(strings):
        slx = []
        for string in strings: 
            slx.append([e for e in string]) 
        return slx
    def to_str(lists): 
        strl = []
        for tokens in lists: 
            strl.append(' '.join(tokens))
        return strl

    # ['123.5', '374.7', 'J23'] is a subseq of ['123.5', 'X27.1', '374.7', '334.7', '111', 'J23', '223.4']? True
    q1 = ['123.5', '374.7', 'J23']
    r1 = ['123.5', 'X27.1', '374.7', '334.7', '111', 'J23', '223.4']
    r2 = ['123.5', 'y', 'z', 'X27.1', 'y', '374.7', 'z', 'z', '334.7', '111', 'y', 'J23', 'y', 'z', 'y', 'x', '223.4']
    r3 = ['y', 'z', '374.7', 'x', 'x', '374.7', 'x', '334.7', 'J23']  # missing 123.5

    ### Inputs are strings
    q = ['666222054263314443712', '5432127413542377777', '6664664565464057425']
    q = [q1, r1, r2, r3]
    # q2 = to_list_repr(q) # doesn't work 
    # q = to_str(q)
    s = lcs(q)

    print('  + %s ~>\n  %s\n' % (q, s))

    return

def test():
    t_lcs()
    return

if __name__ == "__main__": 
    test()




