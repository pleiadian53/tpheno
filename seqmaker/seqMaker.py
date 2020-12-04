#!/usr/bin/env python

import bulklearn
import os, re, gc, sys
import numpy as np

import pandas as pd
from pandas import DataFrame, Series
# from bulklearn import *

import collections, itertools, random
import argparse

# suppress warnings 
import warnings
warnings.filterwarnings("ignore")

# [test]
import inspect

# known matplotlib warning 
# python2.7/site-packages/matplotlib/__init__.py:1350: UserWarning:  This call to matplotlib.use() has no effect
# because the backend has already been chosen;
# matplotlib.use() must be called *before* pylab, matplotlib.pyplot,
# or matplotlib.backends is imported for the first time.

def main(**kargs): 
    """
    {0: diagReader, 1: t_make_sequence, 2: t_make_sequence_doc}

    """
    print('info> calling main() ...')
    import argparse
    parser = argparse.ArgumentParser()
    # parser.add_argument('--diag_dir', type=str, default='data-diag',
    #                    help='diag directory containing all diagnostic files.')
    # parser.add_argument('--save_dir', type=str, default='save',
    #                    help='directory to store derived files (e.g. code sequence or cseq files)')
    # parser.add_argument('-c', action='store_const', dest='constant_value',
    #                 const='a const value assigned simply by specifying -c',
    #                 help='Store a constant value')
    # parser.add_argument('-f', action='store_false', default=False,
    #                     dest='boolean_switch',
    #                     help='Set a switch to false')

    parser.add_argument('--diag-reader', action='store_const', dest='opt_num', 
                        const=0,
                        help='invoke diagReader')
    parser.add_argument('--seq', action='store_const', dest='opt_num', 
                         const=1,
                         help='make diag code sequence for each patience')

    parser.add_argument('--doc', action='store_const', dest='opt_num', 
                        const=2,
                        help='from diag_cseq.csv, make sequence document')

    args = parser.parse_args()
    if args.opt_num is None: args.opt_num = 2
    print('sys> opt_num: %d < args:\n%s\n' % (args.opt_num, args))
    
    # app(args)
    test(args, **kargs)  # kargs

    return 

# [old]
def makeDiagSeq(**kargs): 
    """    
    Subsumed by the module seqMaker2 in which data are gathered via SQL queries ... 12.23.2016

    Create a file containing the sequence of diagnostic codes ordered according to their timestamps. 
	Each patient is considered as a "document". Sequences associated with different patients are 
	separated by period. 

    Input 
    -----
    diag_all.csv

	Memo
	----
	1. Record protocol

    2. 

    * groupby 
    sdf = df.groupby(g_fields)[['n_zeros']].sum().add_prefix('total_')

    * iterate groups 
    gm = fs.groupby('model', as_index=False)
    for i, (m, group) in enumerate(gm): 

    3. Statistics of diag files 
       + 2,328,314 unique patients


    """
    def find_midpoint(d1, d2):  # yyyy-mm-dd 
    	d1 = pd.Timestamp(d1)
    	d2 = pd.Timestamp(d2)
    	dm = d1 + (d2-d1)/2 

    	# str(dm.year) + '-' + str(dm.month) + '-' + str(dm.day)
    	return dm
    def find_midpoint2(df, start='admission_date', end='discharge_date'): 
    	d1, d2 = df[start], df[end]
    	d1 = pd.Timestamp(d1)
    	d2 = pd.Timestamp(d2)
    	if d1 == d2: return d1
    	dm = d1 + (d2-d1)/2 
        # str(dm.year) + '-' + str(dm.month) + '-' + str(dm.day)
    	return dm
    def formatDate(dm):
        # dm = pd.Timestamp(d) 
        return str(dm.year) + '-' + str(dm.month) + '-' + str(dm.day)

    # load diag_all.csv
    basedir, ifile = ('data-diag', 'diag_all.csv') # [params]

    assert os.path.exists(basedir)
    ipath = os.path.join(basedir, ifile)

    dtypes = {'mrn': str}
    df = pd.read_csv(ipath, sep='|', header=0, index_col=False, error_bad_lines=True, dtype=dtypes)
    print('info> input diag_all dim: %d by %d' % df.shape)


    # [params]
    group_id = 'mrn'
    d1, d2 = 'admission_date', 'discharge_date'
    default_sortby_attr = 'admission_date'

    use_mid_point = False 
    sortby_attr = 'midpoint_date' if use_mid_point else default_sortby_attr

    seq_attr = 'icd9_code'
    tsep = ';' # time separator

    # [iparams] (inferred params)
    n_uniq = 0 

    # kargs params 
    f_suffix = kargs.get('suffix_', None)
    n_limit = kargs.get('n_limit', np.inf)
    if f_suffix is None and n_limit < np.inf: f_suffix = 'n_%s' % n_limit

    # [debug] passed
    # dim0 = df.shape
    # df[group_id].replace('', np.nan, inplace=True)
    # df = df[pd.notnull(df[group_id])]
    # print('info> diag_all original dim: %s, -n/a on %s => %s' % (str(dim0), group_id, str(df.shape)))

    # [debug] unique mrns
    mrns = list(set(df['mrn'].values))
    n_uniq = len(mrns)
    print('info> n_uniq: %d, example mrns:\n%s\n' % (n_uniq, mrns[:10]))

    # groupby MRN and sort by time 
    pg = df.groupby(group_id)  # set as_index to False => pid remains as a column rather than becoming an index
    assert len(pg) == n_uniq
    # print('info> number of groups %d =?= %d' % (len(pg), len(mrns)))  # [log] number of groups 2328314 =?= 2328314

    # sys.exit(0)
    
    aset = {}
    alist = []
    tlist = []  # time stamps
    # [protocol] one patient per document 
    
    # [debug] passed
    # i = 0
    # for p, dfi in pg: 
    #     if i > 50:
    #         print('> test complete.')
    #         break
    #     try: 
    #         int(p)
    #     except: 
    #         msg = "patient ID is not an integer?\n%s\n" % p
    #         raise RuntimeError, msg
    #     i += 1
    # print('info> format of groups seem correct.')

    print('config> max patients: %s, file suffix: %s | n_uniq: %d' % (n_limit, f_suffix, n_uniq))

    dcnt = n_nulldiag = 0
    for p, dfi in pg: 
    # for i, mrn in enumerate(mrns): 
    	# dfi = df.loc[df[group_id] == mrn]
    	# p = mrn 
        mrn = p
        assert not dfi.empty, 'mrn %s does not have data?' % p

        if dcnt >= n_limit: 
            print('constraint> number of desired unique patients reached (%d =?= %d =?= %d).' % (dcnt, len(alist), len(tlist)))
            # assert len(alist) == len(tlist) and dcnt == len(alist)
            break 
        else: # track progress 
            print('progress> ... processed %d patients (%f%% of the total %d) ...' % (dcnt, (dcnt/(n_uniq+0.0))*100, n_uniq))

        print('group #%d>\n%s\n   (dfi dim: %s)' % (dcnt+1, p, str(dfi.shape)))
        if dcnt < 10: 
            mrns = list(set(dfi['mrn'].values))
            print('debug> mrn %s =?= %s' % (mrns, p))
            print('\n%s\n' % dfi[['mrn', d1, d2]])

        dim0 = dfi.shape
        # assert not aset.has_key(p) 
        # aset.update([p])  

        # remove rows with seq_attr (e.g. ICD9) having empty strings
        dfi[seq_attr].replace('', np.nan, inplace=True)
        # df = df[np.isfinite(df['EPS'])]
        dfi = dfi[pd.notnull(dfi[seq_attr])]
        print('info> dfi dim: %s -n/a => %s' % (str(dim0), str(dfi.shape)))

        if dfi.empty: 
        	print('info> mrn %s were not given diagnostic codes?' % mrn)
            # n_nulldiag += 1 
        	continue
        else: 
        	dcnt += 1

        # find midpoint between two dates
        if use_mid_point: 
            dfi['midpoint_date'] = dfi.apply(find_midpoint2, axis=1)  # passing Series, row-wise
            if dcnt < 10: 
                print('debug> midpoint:\n%s\n' % dfi['midpoint_date'].values[:5])

        # sort according to ascending time stamps 
        dfi.sort_values(sortby_attr, ascending=True, inplace=True)
        if dcnt % 100 == 0: 
            print dfi[['mrn', 'admission_date']].head(5)

		# extract ICD9 codes from ... 

        cseq = dfi[seq_attr].astype('str').str.cat(sep=tsep)
        if dcnt % 500 == 0: 
            print('info> ID: %s => %s' % (p, cseq))
        alist.append((p, cseq))

        if use_mid_point: 
            dfi['midpoint_date'] = dfi['midpoint_date'].apply(formatDate)
            tseq = dfi['midpoint_date'].astype('str').str.cat(sep=tsep)
        else: 
            tseq = dfi[sortby_attr].astype('str').str.cat(sep=tsep)
        tlist.append((p, tseq))

        # assert len(cseq.split(tsep)) == len(tseq.split(tsep))

        if dcnt % 500 == 0: 
            print('info> ID: %s => %s' % (p, tseq))

    # [stats]   
    print('info> Out of %d unique mrns %d have diag codes, %d empty => (ratio: %f)' % (n_uniq, dcnt, dcnt-n_uniq , dcnt/(len(mrns)+0.0) ))

	# convert to dataframe 
    # [params]
    header = ['mrn', 'sequence']

    df = DataFrame(alist, columns=header)
    f_stem = 'cseq'
    ofile = 'diag_%s.csv' % f_stem if not f_suffix else 'diag_%s_%s.csv' % (f_stem, f_suffix)
    opath = os.path.join(basedir, ofile)

    print('info> output df dim: %d by %d' % df.shape)
    print('info> storing diag code sequence to %s' % opath)
    df.to_csv(opath, sep='|', index=False, header=True)

    # timestamps associated with the above code sequences
    header = ['mrn', 'time']
    df = DataFrame(tlist, columns=header)

    f_stem = 'tseq'
    ofile = 'diag_%s.csv' % f_stem if not f_suffix else 'diag_%s_%s.csv' % (f_stem, f_suffix)
    opath = os.path.join(basedir, ofile)

    print('info> storing corresponding timestamps to %s' % opath)
    df.to_csv(opath, sep='|', index=False, header=True)

    return df

# [summary]
def countUniqDiagCodes(**kargs):
    basedir, ifile = ('data-diag', 'diag_cseq_t1.csv') 

    ipath = os.path.join(basedir, ifile)
    assert os.path.exists(ipath)

    # load sequence file
    # dtypes = {'mrn': str}
    df = pd.read_csv(ipath, sep='|', header=0, index_col=False, error_bad_lines=True) # dtype=dtypes
    print('info> input diag_all dim: %d by %d' % df.shape)

    # [params]
    customize, eval_freq = True, True
    header = ['mrn', 'sequence']
    s = 'sequence'
    sep, tsep, psep = (',', ';', '.')  # codes within each visit, same patient across visists, patient level

    min_freq = 2  # set to None, so that count dict will not be filtered by frequency
    cset = set()  # unique code set
    rare_codes = set()
    cset = collections.Counter()

    for i, (rid, row) in enumerate(df.iterrows()):  # foreach patient
        seq = row[s]
        tlist = seq.split(tsep)
        if i < 50: print('info> example per-patient code sequence: %s' % tlist)
        for visit in tlist: 
            codes = visit.split(sep)
            if i < 10: t_code_format(codes)  # [debug]
            cset.update(codes)  

    # if min_freq is not None: 
    #     cset = {k: v for k, v in cset.items() if v >= min_freq}  

    if min_freq is not None:   
        rare_codes = {k for k, v in cset.items() if v < min_freq}
        print('info> n_rare_codes: %d | ex: %s' % (len(rare_codes), rare_codes[:10]))

    # persistence 


    return cset

# [summary]
def countUniqDiagCodes(**kargs):
    basedir, ifile = ('data-diag', 'diag_cseq_t1.csv') 

    ipath = os.path.join(basedir, ifile)
    assert os.path.exists(ipath)

    # load sequence file
    # dtypes = {'mrn': str}
    df = pd.read_csv(ipath, sep='|', header=0, index_col=False, error_bad_lines=True) # dtype=dtypes
    print('info> input diag_all dim: %d by %d' % df.shape)

    # [params]
    customize, eval_freq = True, True
    header = ['mrn', 'sequence']
    s = 'sequence'
    sep, tsep, psep = (',', ';', '.')  # codes within each visit, same patient across visists, patient level

    min_freq = 2  # set to None, so that count dict will not be filtered by frequency
    cset = set()  # unique code set
    rare_codes = set()
    cset = collections.Counter()

    for i, (rid, row) in enumerate(df.iterrows()):  # foreach patient
        seq = row[s]
        tlist = seq.split(tsep)
        if i < 50: print('info> example per-patient code sequence: %s' % tlist)
        for visit in tlist: 
            codes = visit.split(sep)
            if i < 10: t_code_format(codes)  # [debug]
            cset.update(codes)  

    # if min_freq is not None: 
    #     cset = {k: v for k, v in cset.items() if v >= min_freq}  

    if min_freq is not None:   
        rare_codes = {k for k, v in cset.items() if v < min_freq}
        print('info> n_rare_codes: %d | ex: %s' % (len(rare_codes), list(rare_codes)[:10]))

    # persistence 


    return cset

def basicConfig(**kargs): 
    import inspect 
    curframe = inspect.currentframe()
    calframe = inspect.getouterframes(curframe, 2)
    print('Printing basic configuration for %s ...' % calframe[1][3])

    for k, v in kargs.items(): 
        print('  + %s: %s' % (k, v))
    print('-' * 100)
    return 

def makeSeqDoc(**kargs):
    """
    * Document format: 

    1. Per-patient doc: 
       each patient MRN has multiple visits, each of which consists of a set of codes
        + in default mode: 
            one patient, one sentence, codes from different visists are simply concatenated 

            x1; x3, x4 => [x1, x3, x4] => x1 x3 x4 . => one sentence

        + in unroll mode: each sentence is a concatenation of codes that follow the temporal ordering 
                         across visits

                         x1; x3, x4 => [[x1], [x3, x4]] => x1 x3 . x1 x4 .  -> two sentences
    

    2. cohort-level doc:  
       One doc (docstr) corresponds to a concatenation of multiple per-patient docstrings
       for which the patients belong to the same disease cohort

    * Code filtering 

    1. candidate codes: filter rare codes that appear less than x times 

    Note 
    ----
    1. Input format 
    mrn1|493.90;493.90,995.3;493.90;493.90;493.90;493.90;493.90;493.90;250.01,493.90;493.90;V65.40
    mrn2|176.4
    mrn3|970.8,291.81
        where mrn: 1xyz120

    2. Quote from Temporal Properties of Diagnostic Code Time Series

       Each disease was represented as a set of ICD9 codes. For each condition, a set of subtrees of the ICD9 hierarchy was identified 
       to represent that condition. Any documentation of an ICD9 code within that set of subtrees was considered a positive incident 
       for that condition. For example, the subtrees identified for representing meningitis have roots as ICD9 code 320, 321, and 322.

    Memo
    ----
    1. collection.Counter update support single element and list 
    2. Unroll diagnostic code time series 
       Each 'sentence' corresponds to a diag code time series where each event is rep by (sampled) diagnostic 


    Todo
    ---------
    1. Challenge: how to separate truly independent clinical visits? 

    """
    def t_code_format(cx, ftype='icd9'): 
        if ftype == 'icd9': 
            for c in cx: 
                if not icd9utils.isCode(c): 
                    raise ValueError, "%s in %s is not a valid ICD-9" % (c, cx)
        else: 
            # raise NotImplementedError
            print('info> not implemented yet')
            pass 
        return
    def f_supplementary(codes, exclusive=True): 

        # XOR 
        # return [c for c in codes if not icd9utils.isSupplementary(c)] if exclusive else \
        #           [c for c in codes if icd9utils.isSupplementary(c)]
        newlist = ifilterfalse(icd9utils.isSupplementary, codes) if exclusive else filter(icd9utils.isSupplementary, codes) 
        return list(newlist)

    def transform(codes, invalid_as_rare=False): # transform original diag code to its base form (e.g. first 3 digits for ICD-9)
        codes_old, codes_new = codes[:], []
        if code_root_only:
            codes_old = codes[:]
            n_before = len(codes) 
            if invalid_as_rare: 
                n_after = n_before
                for c in codes: 
                    if icd9utils.getRootCode(c, exception=False, verbose=True) is not None: 
                        codes_new.append(c)
                    else: 
                        codes_new.append(token_rare_code)
                        n_after -= 1
                codes = codes_new
            else: 
                codes = [c for c in codes if icd9utils.getRootCode(c, exception=False, invalid_as_is=True, verbose=True)]
            n_after = len(codes)
            if n_before != n_after: 
                print("transform> bad codes in it (%d => %d)?\n%s\n" % (n_before, n_after, codes_old))
        return codes

    def define_rare(cset): # cset maps codes to frequencies 
        if rare_include_ve: 
            return {k for k, v in cset.items() if k.lower().startswith(('v', 'e')) and v < min_freq}
        return {k for k, v in cset.items() if v < min_freq}

    def transform_rare(codes, verbose=False): 
        assert len(rare_codes) > 0, "rare codes were not computed yet."
        codes_old, codes_new = codes[:], []
        if rare_include_invalid: # include/consider invalid codes as 'rare' codes
            for c in codes:
                c_eff = icd9utils.getRootCode(c, exception=False, invalid_as_is=False) # invalid codes => None 
                if c_eff is None or c_eff in rare_codes: 
                    codes_new.append(token_rare_code)
                else: 
                    codes_new.append(c_eff)
                
            codes = codes_new
            if verbose: 
                print('test>\n  + codes_old: %s\n  + codes_new: %s\n' % (codes_old, codes))
        else: # remove invalid code 
            n_skip = 0
            n_original = len(codes)
            for c in codes:
                c_eff = icd9utils.getRootCode(c, exception=False, invalid_as_is=False) # invalid codes => None 
                if c_eff is None: 
                    n_skip += 1
                    continue # skip
                if c_eff in rare_codes: 
                    codes_new.append(token_rare_code)
                else: 
                    codes_new.append(c_eff)
            # print('transform_rare> skipped %d (total: %d) invalid codes' % (n_skip, n_original))
            codes = codes_new

            if verbose: 
                print('test> n_skip: %d (n_orig: %d)\n  + codes_old: %s\n  + codes_new: %s\n' % (n_original, n_skip, codes_old, codes))
        return codes

    import bulklearn.icd9utils as icd9utils
    from itertools import ifilterfalse

    params = {}
    basedir, ifile, n_limit_ = ('data-diag', 'diag_cseq_n_200000.csv', 200000)  # 'diag_cseq_t1.csv')
    ipath = os.path.join(basedir, ifile)
    assert os.path.exists(ipath)
    params['base dir'] = basedir; params['input file'] = ifile

    # load sequence file
    # dtypes = {'mrn': str}
    # header: mrn, sequence
    df = pd.read_csv(ipath, sep='|', header=0, index_col=False, error_bad_lines=True) # dtype=dtypes
    print('info> input diag_all dim: %d by %d' % df.shape)

    # [todo] filter MRNs by cohort definitions 

    # file parameters
    f_suffix = kargs.get('suffix_', None)
    n_limit = kargs.get('n_limit', df.shape[0]) if n_limit_ is None else n_limit_
    if f_suffix is None and n_limit < np.inf: f_suffix = 'n_%s' % n_limit

    customize, eval_freq = True, True
    header = ['mrn', 'sequence']
    s = 'sequence'

    # sequence parameters (when the number of possible temporal paths is too huge, sample only a subset of paths)
    visit_set_max, n_codes_max = 10, 50
    n_paths = 120

    rare_codes = set()
    target_codes = set() # leading_codes + comorbidities
    leading_codes = set() 
    comorbidities = set() 
    
    cset = collections.Counter()
    errset = collections.Counter()

    docstr = docstr2 = ''

    f_stem = 'cseq'
    ofile = 'diag_%s.doc' % f_stem if not f_suffix else 'diag_%s_%s.doc' % (f_stem, f_suffix)
    ofile_readable = 'diag_%s_readable.doc' % f_stem if not f_suffix else 'diag_%s_%s_readable.doc' % (f_stem, f_suffix)

    # ofile, ofile_readable = ('diag_cseq.doc', 'diag_cseq_readable.doc')

    # parameters for defining rare codes
    min_freq = 2  # set to None, so that count dict will not be filtered by frequency
    rare_include_ve = False # if True, V codes and E codes are considered rare 
    rare_include_invalid = True # consider invalid as rare codes in the final doc
    rare_codes = set()
    params['consider invalid codes as rare?'] = 'Yes' if rare_include_invalid else 'No'

    # policy parameters 
    doctypes = {0: 'combined_visits', 1: 'unroll', 2: 'cohort', }
    doctype = 1
    code_root_only = True
    params['document type'] = doctypes[doctype]

    if not doctype in doctypes: customize = False

    # doc parameters 
    sep, tsep, psep = (',', ';', '.')  # codes within each visit, same patient across visists, patient level
    token_doc_begin = 'BEGIN'
    token_doc_end = 'END'
    token_rare_code = 'X'

    # determine code frequencies 
    if eval_freq: 
        for i, (rid, row) in enumerate(df.iterrows()):  # foreach patient
            seq = row[s]
            tlist = seq.split(tsep)
            # if i < 50: print('info> example per-patient code sequence: %s' % tlist)
            for visit in tlist: 
                codes = visit.split(sep)
                 
                # transform to root (first 3 digit for icd9) if code_root_only is true; invalid codes remain as they are
                codes = transform(codes)  

                # if i < 10: t_code_format(codes)  # [debug]
                cset.update(codes)  # cset is a Counter

        # if min_freq is not None: 
        #     cset = {k: v for k, v in cset.items() if v >= min_freq}  

        # [criteria] rare code
        if min_freq is not None:   
            # rare_codes = {k for k, v in cset.items() if v < min_freq}
            rare_codes = define_rare(cset)

            rare_examples = list(rare_codes)[:10]
            print('info> n_rare_codes: %d | ex: %s' % (len(rare_codes), rare_examples))
            params['number of rare codes'] = len(rare_codes); params['rare codes'] = rare_examples

    basicConfig(**params)
    gc.collect()

    codemap = {} # from complete code to its base (i.e. first 3 digits)

    if customize: 
        print('info> doctype: %s' % doctypes[doctype])
        # merge codes across visits 
        if doctype == 0:

            sentences = []
            for i, (rid, row) in enumerate(df.iterrows()):  # foreach patient (with codes corresponding to visits at different dates)
                seq = row[s]
                tlist = seq.split(tsep)
                patient_level_seq = []

                for visit in tlist:  # foreach visit (consisting of set of codes sep by ;)
                    codes = visit.split(sep)

                    # transform codes to base forms? 
                    codes = transform_rare(codes)
                
                    # flat structure [c1, c2, c3, ...]
                    patient_level_seq.extend(codes)
            
                # assuming one patient for one sentence, which combines all the visits at different dates
                sent = ' '.join(patient_level_seq)
                sent += ' ' + psep
                if i < 20: print('info> iter=%d | ex: %s' % (i, sent))
                sentences.append(sent)

            # per-patient doc string 
            docstr = ' '.join(sentences) # a doc is a collection of sentences (ended with a period)

        elif doctype == 1: # unroll
            assert len(docstr) == 0 and len(docstr2) == 0, "Non-empty document strings."

            for i, (rid, row) in enumerate(df.iterrows()):  # foreach patient (with multiple visits)
                seq = row[s]   # take the sequence field
                tlist = seq.split(tsep)  # separate each visit (clinical visits at different times)

                visit_set = []
                n_codes = 0
                for visit in tlist:  # foreach visit (each of which corresponds to a set of codes ~ different dates)
                    codes = visit.split(sep)
                    # n_prior = len(codes)
                    codes = transform_rare(codes, verbose=(i<10))
                    # n_posterior = len(codes)
                    n_codes += len(codes)
                    visit_set.append(codes)  # [[], []]
                if i < 50: print('verbose> visit set (n_list: %d, n_codes_flat: %d):\n  => %s\n' % (len(visit_set), n_codes, visit_set))

                # unroll multiple visits; each 'sentence' corresponds to a diag code time series where each event is rep by (sampled) diagnostic 
                # from each visit
                sentences = [] 

                if len(visit_set) > visit_set_max or n_codes > n_codes_max: # large path volume (e.g. visit_set_max: 10, n_code_max: 50)
                    # generate sample path and always keep target codes (main+comorbidities) along
                    if target_codes: # only consider particular cohort(s) defined by a set of target codes
                        raise NotImplementedError 
                    else: 
                        n_empty_vset = 0
                        for _ in range(n_paths): # only sample n paths
                            scodes = []
                            for vset in visit_set: 
                                if vset: 
                                    scodes.extend(random.sample(vset, 1))
                                else: 
                                    n_empty_vset += 1
                                    continue
                                    # print('warning> empty vset:\n%s\n' % visit_set)  # probably due to removal of rare codes
                                    # break 
                            # if n_empty_vset > 0: break 

                            sent = ' '.join(scodes)
                            sent += ' ' + psep
                        sentences.append(sent)
                else: # small path volume 
                    # [debug] 
                    if i % 10 == 0: 
                        n_combo = len([vseq for vseq in itertools.product(*visit_set)])
                        print('debug> mrn: %s, n_combo: %d' % (row['mrn'], n_combo))

                    # peek 
                    n_path_total = list(itertools.product(*visit_set))
                    print('warning> large path volume even with n_visit_set = %d and n_codes = %d' % (len(visit_set), n_codes))
                    if n_path_total > n_paths: 
                        for vseq in random.sample(itertools.product(*visit_set), n_paths):
                            sent = ' '.join(vseq)
                            sent += ' ' + psep  # end of sentence token
                            sentences.append(sent)  # per-patiet docstr in a set of sentences
                    else:  
                        for vseq in itertools.product(*visit_set): # expensive!!! not feasible for long lists
                            sent = ' '.join(vseq)
                            sent += ' ' + psep   # one possible combination per sentence followed by a period
                            sentences.append(sent)  # per-patiet docstr in a set of sentences

                # doc for training, not quite readable
                docstr += ' '.join(sentences)
                docstr += ' \n' # again use just a single space to separate doc (which represents total records of a single patient)

                # [debug]
                if i < 5: print('info> example doc str:\n%s\n' % docstr)

                # doc2: better visual 
                per_patient_doc = '%s\n' % token_doc_begin
                per_patient_doc += '\n'.join(sentences) # last one doesn't have \n

                # document ending token 
                per_patient_doc += '\n%s\n' % token_doc_end

                # [debug]
                if i < 10: print('info> example doc str2:\n%s\n' % per_patient_doc)

                docstr2 += per_patient_doc + '\n'

            # population level docstr  # memory!!!
            # doctstr = ' '.join(sentences)  # assuming each sentence already as an end token

        elif doctype == 2: # cohort-level doc

            cohorts = selectCohorts(df, diseases=['diabetes'])  # cohorts expressed via sets of mrns
            for cohort in cohorts: 
                for i, (rid, row) in df.loc[df['mrn'].isin(cohort)]: 
                    pass
        
    else: 
        # treating each patient as a sentence
        docstr = df[s].astype('str').str.cat(sep=psep)
        print('info> snapshot:\n%s\n' % docstr[:100])

    assert len(docstr) > 0

    # save
    opath = os.path.join(basedir, ofile)
    fp = open(opath, "w")
    fp.write(docstr)
    fp.close()
    print('info> docstr saved to %s' % opath)

    if docstr2: 
        opath2 = os.path.join(basedir, ofile_readable)
        fp = open(opath2, "w")
        fp.write(docstr2)
        fp.close() 
        print('info> docstr2(readable format) saved to %s' % opath2)

    return docstr

def selectCohorts(df, policy=None, **kargs): 
    # header = ['mrn', 'sequence']

    s = 'field_id'

    # default
    cohort = list(set(df[s].values))
    cohorts = [cohort]  # in general, there are several cohorts 

    # cohort by diseases

    return cohorts

def load(ifile=None, basedir=None, **kargs):
    if basedir is None: basedir = 'data-diag'
    if ifile is None: ifile = 'diag_cseq.csv'

    ipath = os.path.join(basedir, ifile)
    assert os.path.exists(ipath)

    # header = ['mrn', 'sequence']
    df = pd.read_csv(ipath, sep='|', header=0, index_col=False, error_bad_lines=True) # dtype=dtypes
    print('load> input diag_all dim: %d by %d' % df.shape)   

    return df 

def app(**kargs): 
    pass 

def test(args, **kargs): 
    """


    Options
    -------
    Note that parameters are usually specified within template or test functions.

    0. good old diagReader.py from bulk_learn module
    1. make diag code sequence for each patience and store the result in 
       { diag_cseq.csv, diag_tseq.csv }
    2. from diag_cseq.csv, make sequence document

    """
    import bulklearn.diagReader as diagReader
    
    opt = args.opt_num

    options = {0: diagReader, 1: t_make_sequence, 2: t_make_sequence_doc}

    # n_uniq = 200000
    if opt == 1: 
        # kargs['n_limit'] = n_uniq
        pass 
    elif opt == 2: 
        pass

    # import bulklearn.diagReader as diagReader
    # import diagReader    # doesn't work even if i put import bulklearn.diagReader as diagReader in __init__.py
    # diagReader.test2()

    # make diag code sequence for each patience and store the result in diag_tseries.csv
    # n_uniq = 20000
    # t_make_sequence(suffix_='n_%s' % n_uniq, n_limit=n_uniq)
    # t_make_sequence()

    # from diag_cseq.csv, make sequence document
    # t_make_sequence_doc(**kargs)

    options[opt](**kargs)

    return

def t_make_sequence(**kargs):
    n_limit = kargs.get('n_limit', 200000)
    if n_limit is not None: print('t_make_sequence> processing %s unique patients ...' % n_limit)
    df = makeDiagSeq(**kargs)
    # find summary statistics
    return

def t_make_sequence_doc(**kargs):
    docstr = makeSeqDoc(**kargs) 
    return  

if __name__ == "__main__": 
	main()
