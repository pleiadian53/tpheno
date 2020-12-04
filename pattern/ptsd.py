import os, sys, re, random, collections

# install tsne via pip
# from tsne import bh_sne

import pandas as pd 
from pandas import DataFrame

# temporal sequence modules
from config import seq_maker_config, sys_config
from batchpheno.utils import div

try:
    import cPickle as pickle
except:
    import pickle

# from seqmaker import seqTransform as st # [design] this leads to mutual dependency!

# for impl (approximate) phenotyping algorithm based on 
# from seqmaker import seqAnalyzer as sa
# from seqmaker import seqReader as sr

p_diabetes = re.compile(r"(?P<base>250)\.(?P<subclass>[0-9]{1,2})")
p_secondary_diabetes = re.compile(r"(?P<base>249)\.(?P<subclass>[0-9]{1,2})")
p_gestational_diabetes = re.compile(r"(?P<base>648)\.(?P<subclass>[0-9]{1,2})")

### Define Disease-specific Regex Pattern Here
p_condition = re.compile(r'(?P<base>309)\.(?P<subclass>81)')   # 81

GDiagnoses = ['309.81', 'F43.1', ]


### SNOMED #### 
# 192042008           Acute post-trauma stress state SNOMED
# 313182004           Chronic post-traumatic stress disorder   SNOMED
# 317816007           Stockholm syndrome     SNOMED
# 318784009           Posttraumatic stress disorder, delayed onset     SNOMED
# 443919007           Complex posttraumatic stress disorder  SNOMED
# 446175003           Acute posttraumatic stress disorder following military combat    SNOMED
# 446180007           Delayed posttraumatic stress disorder following military combat               SNOMED
# 699241002           Chronic post-traumatic stress disorder following military combat               SNOMED

# [design] primary API (e.g. transform, phenotype) should go on top


##################################
##
#  To create labels, do the followings 
#  
#  1. define 
#     phenotypeDoc
#     label 
#  2. Coding class (for decoder)
#
#

class Coding(object): 
    bitSize = 1

    @staticmethod
    def decode(lx):
        assert len(lx) == Coding.bitSize 
        lx = tuple(lx)
        
        ### diabetes example
        # lv, lc = 'type_0', 0  # default: no diabetes
        # if lx == (0, 0, 0):
        #     lv, lc = 'type_0', 0  # no diabetes
        # elif lx == (1, 0, 0): 
        #     lv, lc = 'type_1', 1  # type I
        # elif lx == (0, 1, 0): 
        #     lv, lc = 'type_2', 2   # type II
        # elif lx == (0, 0, 1): 
        #     lv, lc = 'type_3', 3  # gestational
        # elif lx == (1, 1, 0): 
        #     lv, lc = 'type_1_2', 4  # type I & II
        # elif lx == (1, 0, 1): 
        #     lv, lc = 'type_1_3', 5  # type I & gestational
        # elif lx == (0, 1, 1): 
        #     lv, lc = 'type_2_3', 6  # type II and gestational
        # elif lx == (1, 1, 1):      
        #     lv, lc = 'type_1_2_3', 7

        return ('PTSD', 1)


def transform(seq, **kargs):
    """

    Memo
    ----
    1. use case: 
       transform(seq, policy='noop', seq_ptype='diag')

    """
    # if not kargs.has_key('predicate'): kargs['predicate'] = isCase
 
    # # [args] policy: noop to bypass cut op. 
    # if not kargs.has_key('policy'): kargs['policy'] = 'prior'  # i.e. only looking at the sequence prior to the first diabetes-related diagnosis
    # if not kargs.has_key('seq_ptype'): kargs['seq_ptype'] = 'regular'
    # return st.transform(seq, **kargs)  # operations: cut -> transform_by_ptype (e.g. diag, med)
    raise ValueError, "Use seqTransform for this operation!"

def cut(seq, **kargs): 
    """

    Input
    -----
    * policy: 
        'regular' => noop
        'prior' => coding sequence segment prior to the cut point determined by the predicate
        'posterior'
    """ 
    # if not kargs.has_key('predicate'): kargs['predicate'] = isCase
    # if not kargs.has_key('policy'): kargs['policy'] = 'prior'
    # return st.cut(seq, **karsg)
    raise ValueError, "Use seqTransform for this operation!"

def isCondition(code): 
    """
    Use strict regex match to identify a condition (PTSD in this case).
    """
    scode = str(code).strip()
    m = p_condition.match(scode)
    if m:
        return True
    return False

def isCase(code): 
    """
    Use a set of ICD-9 or ICD-10 to identify a target disease. 
    """
    scode = str(code).strip()
    if scode in GDiagnoses: 
        return True 
    return False

def generate(seqx, **kargs): 
    return genLabels(seqx, **kargs)
def genLabels(seqx, **kargs): # phenotypeDoc + phenotypeindex 
    """
    Input
    -----
    seqx: a list of medical coding sequences (e.g. temporal sequences of diagnoses and treatments)
          

    """
    labelsets = phenotypeDoc(docs, **kargs)
    return phenotypeindex(labelsets, **kargs)

def phenotypeindex(labelsets, **kargs):
    """
    Given phenotyped labels (upon completion of phenotypeDoc), find document indices 
    for different disease subtypes (e.g. type I, type II, type I + type II, etc.)
    """
    # header = ['type_0', 'type_1', 'type_2', 'gestational', 'type_3', 'type_1_2', 'type_1_3', 'type_2_3', 'type_1_2_3']
    header = ['type_0', ]
    res = {h: [] for h in header} # result set 

    res = {}
    for i, lset in enumerate(labelsets): 
        lkey = tuple(lset)
        if not res.has_key(lkey): res[lkey] = [] 
        res[lkey].append(i)  # class tuples  => positions
        # where 
        #   class tuples constains 0,1 encoding of multiple labels  
        #   positions: document positions/indices 

    # rename keys
    for lx, idx in res.items():  
    	if lx == (1, ) or lx == 1:
    		res['type_0'] = res.pop(lx)    # index by position
        else: 
        	raise ValueError, "Invalid label set: %s" % str(lx)

    # one-class classification only 
    res[1] = res['type_0']  # res contains 1) label-to-position mapping and 2) label-to-name mapping

    return res

# [policy]
def phenotype(doc, **kargs):
    min_count = kargs.get('min_count', 1) 

    # [policy]
    n_type1 = n_type2 = n_birth = 0
    labels = [0, 0, 0]  # default: False for all labels (in the order of type I, II, gestational)
    for e in doc: 
        if is_diabetes_type1(e): 
            n_type1 += 1 
            if n_type1 >= min_count: labels[0] = 1
        if is_diabetes_type2(e): 
            n_type2 += 1 
            if n_type2 >= min_count: labels[1] = 1
        if is_diabetes_gestational(e): 
            n_birth += 1
            if n_birth >= min_count: labels[2] = 1
        
        # [todo] add other conditions here after increasing the dim of labels appropriately


    return labels # multilabel

# [precond] input sequences/docs should contain diagnostic codes
def phenotypeDoc(docs, **kargs):
    """
    phenotype input documents (consisting of temporal sequences of codes) according to 
    the mention of related diagnostic codes. 

    Related 
    -------
    1. phenotypeIndex()

    """
    # no known PTSD subtype 
    
    return [(1, )] * len(docs) 

def label(seqx, **kargs): 
    """

    Input: documents

    Params: 
        min_count: minimum number of occurrences of disease codes before declaring positive

    Output: 

    """
    
    # encode documents in phenotypic/class labels 
    labelsets = phenotypeDoc(seqx, **kargs)  # [params] min_count, seq_ptype, cohort
    ret = labelToID(labelsets)  # this is only label-to-ID format

    # convert to a list of labels matching docs positions 
    labels = [None] * len(seqx)
    ulabels = set()
    if kargs.get('numeric_', True):
        for label, docIDs in ret.items():  # 
            if not isinstance(label, int): continue
            for i in docIDs: 
                labels[i] = label
            ulabels.add(label) # [test]
    else: # canonical
        for label, docIDs in ret.items(): 
            if isinstance(label, int): continue
            for i in docIDs: 
                labels[i] = label
            ulabels.add(label) # [test]

    assert len(ulabels) * 2 == len(ret), "Inconsistent canonical and numeric labels (got: %d but total: %d)" % \
        (len(ulabels), len(ret))
    # [condition] no unlabeled docs 
    unlabeled = [i for i, l in enumerate(labels) if l is None]
    assert len(unlabeled) == 0, "There exist unlabeled documents at positions: %s" % unlabeled

    return labels

def labelToID(labelsets):
    """
    Map labels in multilabel format (e.g. [1, 0, 0] as type I diabetes) to: 
        lv: canonical class label 
        lc: numerical class label

    """
    ret = {}
    for i, lset in enumerate(labelsets): 
        lkey = tuple(lset)
        
        lv, lc = Coding.decode(lkey)  # canonical class label, numeric class label 
        
        for l in [lv, lc, ]: 
            if not ret.has_key(l): ret[l] = [] 
 
        # canonical class label 
        ret[lv].append(i)

        # numeric class label
        ret[lc].append(i)
    return ret 

def getNumericLabels(seqx, **kargs):
    kargs['numeric_'] = True
    return label(seqx, **kargs)

def getCanonicalLabels(seqx, **kargs):
    kargs['numeric_'] = False 
    return label(seqx, **kargs)
def getNamedLabels(seqx, **kargs):
    kargs['numeric_'] = False 
    return label(seqx, **kargs) 

def has_type1(seq): 
    for e in seq: 
    	if is_diabetes_type1(e): 
    		return True 
    return False

def has_type2(seq):
    for e in seq: 
    	if is_diabetes_type2(e): 
    		return True 
    return False  

def has_gestational(seq):
    for e in seq: 
    	if is_diabetes_gestational(e): 
    		return True 
    return False 
def has_type3(seq): 
    return has_gestational(seq)    

def cohort_diabetes(code): 
    cmap = {}
    cmap['250'] = {}
    cmap['type_1'] = ['250.01', '250.03', '250.11', '250.13', '250.21', '250.23', '250.31', '250.33', ]   

    # .00 type II or unspecified type, not stated as uncontrolled .02  type II or unspecified type, uncontrolled
    cmap['type_2'] = ['250.00', '250.02', '250.10', '250.12', '250.20', '250.22', '250.30', '250.32', ] 

    return cmap

def is_diabetes_type1(code): 
    scode = str(code).strip()
    # print('type1> input: %s' % scode)
    m = p_diabetes.match(scode)
    # if m: print('m: %s, n: %d, last digit: %s' % (m.group(), len(scode), scode[-1]))
    # base, subclass = 
    if m:
        subc = m.group('subclass')
    	if len(subc) == 2 and subc[-1] in ('1', '3', ): 
    		return True 
    return False

def is_diabetes_type2(code): 
    scode = str(code)
    m = p_diabetes.match(scode)
    if m: 
        subc = m.group('subclass')
    	if len(subc) == 2 and subc[-1] in ('0', '2', ): 
    		return True 
    return False

def is_diabetes_secondary(code): 
    scode = str(code)
    if scode[:3] == '249': 
    	return True 
    return False

def is_diabetes_gestational(code): 
	# 64800 64801 64802 64803 64804 64880 64881 64882 64883 64884
    scode = str(code)
    m = p_gestational_diabetes.match(scode)
    # if scode[:3] == '648': 
    if m: 
        subc = m.group('subclass')
        if subc[0] in ('0', '8', ):
    	    return True 
    return False

def is_diabetes(code): 
    scode = str(code)
    if scode[:3] in ('249', '250', '648', ): 
    	return True 
    return False	

def has_abnormal_blood(code): 
    # 7902 79021 79022 79029 
    scode = str(code)
    if scode[:3] == '790': 
    	if scode in ('790.2', '790.21', '790.22', '790.29'): # abnormal glocose
            return True
    return False 

def has_abnormal_urine(code): 
    # 7915 7916 
    scode = str(code)
    if scode[:3] == '791': 
    	if scode in ('791.5', '791.6', ):  # glycosuria, acetonuria
    	    return True 
    return False

def involve_insulin_pump(code): 
	# V4585 V5391 V6546 
    # V45.85: presence of insulin pump
    # V53.91: fitting and adjustment of insulin pump	
    # V65.46: Encounter for insulin pump training
    scode = str(code)
    if scode in ('V45.85', 'V53.91', 'V65.46', ): 
    	return True
    return False 

def test(): 
    codes = ['250.03', '250.00', 250.2, 250.24, 249.00, 250.01, '648.1', '648.00', '648.82', '649.12']
    print('input> %s' % codes)
    adict = {'type I': is_diabetes_type1, 'type II': is_diabetes_type2, 'secondary': is_diabetes_secondary, 'gestational': is_diabetes_gestational}
    for typ, routine in adict.items(): 
    	print "> %s => %s" % (typ, [c for c in codes if routine(c)])

    # transform 
    seq = ['786.50', '786.50', '67894', '413.9', '250.00', '17.2']
    subseq = transform(seq, policy='prior', inclusive=True, seq_ptype='diag')
    print('> diag seq: %s' % subseq)


if __name__ == "__main__": 
    test()



