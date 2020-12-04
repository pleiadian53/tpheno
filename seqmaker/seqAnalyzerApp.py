# encoding: utf-8

# local modules 
from batchpheno import icd9utils, sampling, qrymed2
from config import seq_maker_config, sys_config
from batchpheno.utils import div
from pattern import medcode as pmed
import seqparams  # all algorithm variables and settings
import analyzer, vector 

### App Module of seqAnalyzer
import seqAnalyzer as sa 


####################################################################################################
#
#  Related 
#  -------
#  pathAnalyzer 
#  pathwayAnalyzer (older version of pathAnalyzer focusing on cluster analysis)
#
#

def t_deidentify(**kargs):  # template function refactored from seqAnalyzer
    def transform_(seqx, pol='seq'):
        if pol == 'seq':
            return seqx # do nothing 
        elif pol == 'set': 
            seqx2 = []
            for i, seq in enumerate(seqx): 
                seqx2.append(sorted(set(seq)))
            return seqx2 
        else: 
            raise NotImplementedError, 'Unknown policy: %s' % pol 

        return seqx         
            
    import collections 
    import algorithms  # local library

    seq_ptype = seqparams.normalize_ptype(kargs.get('seq_ptype', 'diag'))
    read_mode = 'timed'  # 'doc', 'seq'

    # data source from sequencing (entire population)
    n_parts = 10
    basedir = sys_config.read('DataExpRoot')
    basedir_sequencing = os.path.join(basedir, 'sequencing')  #

    # partition of the patient sets in sequencing data is identified by specialized cohort names 
    # e.g. condition_drug_timed_seq-group-10.dat  
    fbase = 'condition_drug_timed_seq'  # regular: condition_drug_seq
    policies = ['seq', 'set', ]

    cache = {}  # set to None if caching is not desired
    tCaching = True if isinstance(cache, dict) else False
    
    for policy in policies: 
        polling = {} 
        n_persons = 0
        seqx, tseqx = [], []
        for i in range(n_parts):
            cohort_name = 'group-%s' % (i+1)  
            ifile = os.path.join(basedir_sequencing, '%s-%s.dat' % (fbase, cohort_name))
            assert os.path.exists(ifile), "invalid ifile: %s" % ifile

            # if cache is None or not cache.has_key(cohort_name): 
            #     # no_throw: if True, then don't throw exception when time|code cannot be parsed successfully but simply ignore that pair
            #     seqx, tseqx = read2(load_=False, save_=False, simplify_code=False, read_mode='timed', verify_=False, 
            #                                     seq_ptype=seq_ptype, ifile=ifile, cohort=cohort_name, no_throw=True) 
            #     if isinstance(cache, dict): 
            #         cache[cohort_name] = seqx, tseqx
            # else: 
            #     seqx, tseqx = cache[cohort_name]
                
            seqx, tseqx = sa.read2(load_=False, save_=False, simplify_code=False, read_mode='timed', verify_=False, 
                                seq_ptype=seq_ptype, ifile=ifile, cohort=cohort_name, no_throw=True) 
        
            # [todo] verify their ids 
            n_persons += len(seqx) 
            
            # [policy]
            n0 = len(seqx)
            seqx = transform_(seqx, pol=policy)
            n1 = len(seqx); assert n0 == n1

            # convert seqx to string for indexing 
            for seq in seqx: 
                seqstr = '_'.join(str(e) for e in seq)     
                if not polling.has_key(seqstr): polling[seqstr] = 0 
                polling[seqstr] += 1
        ### end foreach dataset 

        # [params] control 
        n_comm, n_lcomm = (100, 100)  # most common, least common 

        n_uniq_signatures = len(polling)
        r = n_uniq_signatures/(n_persons+0.0)

        countSig = collections.Counter(polling)
        comm_sig = countSig.most_common(100) 
        div(message='Most common sigatures (n=%d, policy=%s, ctype=%s) ...' % (n_comm, policy, seq_ptype))

        # [note] n_persons: 2302000 (2.3M), n_uniq_signatures: 1476414 (1.4M), ratio: 0.641361
        print('  + n_persons: %d, n_uniq_signatures: %d, ratio: %f' % (n_persons, n_uniq_signatures, r))

        for cs, cnt in comm_sig: 
            print('  + [cnt=%d] %s' % (cnt, cs))

        div(message='Least common sigatures (n=%d, policy=%s, ctype=%s) ...' % (policy, seq_ptype))
        print('  + n_persons: %d, n_uniq_signatures: %d, ratio: %f' % (n_persons, n_uniq_signatures, r))
        l_comm_sig = algorithms.least_common(countSig, 100)
        l_comm_sig = sorted(l_comm_sig, key=lambda x: x[1], reverse=True)

        for cs, cnt in l_comm_sig:  # reverse ordering for clarity? 
            print('  + [cnt=%d] %s' % (cnt, cs))

        # [todo] use panda plotting utility 

        print('status> analysis (policy=%s, ptype=%s) complete ------* '  % (policy, seq_ptype))

    return

def t_deidentify_batch(**kargs): 

    ptypes = ['regular', 'diag', 'med', ] # 'diag', 'med', 
    for ptype in ptypes: 
        t_deidentify(seq_ptype=ptype)

    return

def test(**kargs): 

    # singularity, uniqueness of coding sequences
    t_deidentify_batch(**kargs)

    return

if __name__ == "__main__": 
    test()