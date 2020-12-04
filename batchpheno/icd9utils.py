import os, sys, commands
import re, csv, random
from predicate import isNumber

from pandas import DataFrame 
import pandas as pd
import numpy as np
from utils import div

from config import sys_config
from coding import icd9
# import qrymed2
# import ConfigParser

RefDir = sys_config.read('RefDir')  # reference directory where ICD-9 descriptions can be found 
# ProjDir = sys_config.read('ProjDir')
InputDir = sys_config.read('DataIn')  # note that config file is not case sensitive
# RefDir2 = sys_config.read('batchpheno')
FILE_ICD9_INFO, FILE_PARENT_CHILD = ('ICD9_descriptions', 'ICD9_parent_child_relations')
ICD9_JSON_PATH = os.path.join(InputDir, 'codes.json') # relative to coding
assert os.path.exists(ICD9_JSON_PATH), "Invalid icd-9 json doc path: %s" % ICD9_JSON_PATH
InfectionScope = ('001', '139')


class myDialect(csv.Dialect):
    """

    [debug]
      1. TypeError: "quoting" must be an integer
          -> add uoting = csv.QUOTE_MINIMAL
    """
    lineterminator = '\n'
    delimiter = '\t'
    quotechar = '"'
    quoting = csv.QUOTE_MINIMAL  # see debug [1]

class Code: 
    """

    [note]
    1. for tuberculosis, don't use the scope 010-018 since 018.xx will not be covered
       use max(018.xx) as the upper bound. 
    """
    p = re.compile(r'(?P<char>v|e)?(?P<base>\d+(?![-]))(\.(?P<decimal>\d+))?', re.I)
    p_icd9 = re.compile(r'(?P<char>v|e)?(?P<base>\d+)(\.(?P<decimal>\d+))?', re.I) # re.compile(r'\d+\.\d+')
    s = re.compile(r'(?P<char>v|e)?\d+[-]\d+')

    # \w ~ [a-zA-Z0-9_]
    p_generic = re.compile(r'(?P<category>\w+)(\.(?P<subclass>\w+))?')  # [todo] this also matches S8.

    # header = ('code', 'description')
    codes = []
    
    scopetb = {'infection': ('001', '139'), 'neoplasms': ('140', '239'),
                }
    subscopetb = {}

    # [1]
    subscopetb['infection'] = {'tuberculosis': ('010', '018.96'),  # 018.96
                               'intestinal': ('001', '009.1'), 
                               'immunodeficiency': ('042', '044.9'), }

    extra_infections = ['481', ]                         

    @staticmethod 
    def collect(x): 
        Code.codes.append(x)

    @staticmethod 
    def max(codes): 
        max_ = -1e12
        for code in codes: 
            if not isCode(code): continue
            if numerize(code, convert_char=True) > numerize(max_, convert_char=True): 
                max_ = code
        return max_
        # return max(numerize(code, convert_char=True) for code in codes)

    @staticmethod 
    def min(codes): 
        min_ = 1e12
        for code in codes: 
            if not isCode(code): continue
            if numerize(code, convert_char=True) < numerize(min_, convert_char=True): 
                min_ = code
        return min_
        # return min(numerize(code, convert_char=True) for code in codes)

class ICD9(object):  # this is not the same as coding.icd9.ICD9
    description = 'ICD9_descriptions'
    parent_child_relation = 'ICD9_parent_child_relations'
    dheader = ('code', 'description')
    rheader = ('parent', 'child')

    @staticmethod
    def table(_file=None): 
        if _file is None: _file = ICD9.description
        # print "ICD9::table> reference: %s" % os.path.join(RefDir, _file)
        df = pd.read_csv(os.path.join(RefDir, _file), names=ICD9.dheader, 
                             sep=myDialect.delimiter, header=None, index_col=False) 
        return df

class ICD10(object):   
    """
    See pattern/medcode.py
    """
    pass


class ICD9Map(object): 
    codes = {} 
    codes['diabetes_with_complications'] = """24901 24910 24911 24920 24921 24930 24931 24940 24941 24950 24951 24960 24961 24970 24971 24980 24981 24990 24991 25002
                    25003 25010 25011 25012 25013 25020 25021 25022 25023 25030 25031 25032 25033 25040 25041 25042 25043 25050 25051 25052
                    25053 25060 25061 25062 25063 25070 25071 25072 25073 25080 25081 25082 25083 25090 25091 25092 25093"""
    codes['diabetes_without_complications'] = "24900 25000 25001 7902 79021 79022 79029 7915 7916 V4585 V5391 V6546"

    
    

#<helper>
def _exec(cmd): 
    st, output = commands.getstatusoutput(cmd)
    if st != 0:
        raise RuntimeError, "Could not exec %s" % cmd
    # print "[debug] output: %s" % output
    return output  #<format?>

def read(_file=None, _test=False):
    """

    [path]
       1. TestDir/cerner/part1
    """
    if _file is None: _file = os.path.join(RefDir, FILE_ICD9_INFO)
    reader = csv.reader(open(_file), dialect=myDialect)
    for row in reader:   # each row is a list
        code, value = row[0], row[1]
        if not isCode(code):
            Code.collect(code) 
            continue
        if _test: 
            try: 
                if not isSupplementary(code): 
                    float(numerize(code))
            except: 
                print "read> cannot convert to float? %s" % code 
                raise RuntimeError
        # print "%s, %s" % (code, value)
        yield (code, value)

def lookup(code, _file=None, _xmlFile=None):
    """
    Given a ICD9 code, find its description. 
     : mapping from code to description
    """
    for row in read(_file=_file): 
        icd9c, description = row[0], row[1] 
        if str(icd9c) == code: 
            return description
    return None
    
def lookup2(code, _file=None, default=None): 
    df = ICD9.table(_file)
    row = df[df['code']==code]
    if not row.empty: 
        return df[df['code']==code].iloc[0]['description']

    # call external library 
    tree = icd9.ICD9(ICD9_JSON_PATH)
    node = tree.find(code)
    if node is not None: 
        return node.description

    return default
getName = lookup2

def reverse_lookup(x, _file=None): 
    """
    Given a regex x, find all icd9 codes with description matching x
    """
    df = ICD9.table(_file)  # df = df[df['comp_code'].map(lambda e: isNumber(e))]
    df = df[df['description'].map(lambda e: isMatched(x, e))]
    return [code for code in df['code']] 
getCode = reverse_lookup

def isMatched(s, value, ignore_case=True):
    """
    Does the regex s match with the string value? 
    """
    p = re.compile(s)
    if ignore_case: 
        m = p.search(value, re.I)
    else: 
        m = p.search(value)
    return True if m else False

def hasNDigits(code, n=3, loc='base'):
    """


    Use
    ---

    012.81  base has 3 digits and decimal has 2
    if n = 3, loc='base' => True
    if n = 3, loc='decimal' => False
    """
    m = Code.p_icd9.match(str(code))
    prefix = m.group('char')
    try: 
        base = m.group('base')
    except: 
        pass 
    try: 
        decimal = m.group('decimal')
    except: 
        pass 
    return 
 
def getMedCode(code): 
    """
    Given an icd9 code, find it's corresponding medcode. 

    Memo
    ----
    1. Not all ICD9 codes will get successful returned medcodes. 
       <error> Error. Too few command line arguments to execute qrymed.


    """
    cmd = 'qrymed -isval 48 %s' % code
    mcode = _exec(cmd)
    try: 
        int(mcode)
    except: 
        return None 
    return mcode
medCode = getMedCode

def getRootCode0(code, exception=True, invalid_as_is=True, verbose=False): 
    m = Code.p_icd9.match(str(code))
    if not m and invalid_as_is: return code  # no-op

    try: 
        prefix = m.group('char')
        base = m.group('base')
    except: 
        msg = 'info> invalid ICD-9 code: %s' % code
        if verbose: print(msg)
        if invalid_as_is: return code
        if exception: 
            raise ValueError, msg
        else: 
            return None

    if prefix: 
        return prefix + base
    return base

def getRootCode(code):
    m = Code.p_icd9.match(str(code))
    if not m: 
        return code 
    try: 
        prefix = m.group('char')
        base = m.group('base')
    except: 
        msg = 'info> unrecognized ICD-9 code: %s' % code
        raise ValueError, msg 
    
    if prefix: 
        return prefix + base 
    return base
getRoot = getRootCode # [alias]

def getRootCode2(code): 
    """
    Take into account of both ICD-9 and ICD-10

    Memo
    ----
    1. if the input itself is a '.' then output => ['', '']

    """
    subclass_sep = '.'
    base = str(code).split(subclass_sep)[0]
    base = base.strip()

    return base

    # [todo] 

def getRootSequence(codes, sep=' '):
    """
    Input
    -----
    codes: a string or a list of diag codes

    Convert the input string of sequence of full ICD-9 codes into a string of 
    sequence of root codes. 
    """
    if isinstance(codes, str): 
        return sep.join([getRootCode2(code=c) for c in codes.split(sep)])
    return [getRootCode2(code=c) for c in codes]

def sampleICD9(**kargs):
    pass 

def getInfectiousParasiticCodes(diff=None, n_samples=None, filter_=None, is_in=True, verbose=False): 
    df = ICD9.table()
    scope = Code.scopetb['infection']
    print('debug> looking at scope: %s' % str(scope))
    if is_in: 
        df = df[df['code'].map(lambda e: isWithin(e, scope))] 
    else: 
        df = df[df['code'].map(lambda e: not isWithin(e, scope))]
    codes = df['code'].values  # a numpy array
    if diff is not None: 
        assert hasattr(diff, '__iter__')  
        codesp = codes
        codes = list( set(codes).difference(set([str(e) for e in diff])) )
        if verbose: print('getInfectiousParasiticCodes> removed codes: %s' % (set(codesp)-set(codes)))
    # add extra codes not in scope 
    codes = np.append(codes, Code.extra_infections)
    if n_samples: 
        codes = random.sample(list(codes), n_samples)
    if filter_ is not None: 
        assert hasattr(filter_, '__call__')
        return [code for code in codes if filter_(code)]
    print "getInfectiousParasiticCodes> size: %d" % len(codes)
    return codes

def getAllInfectiousParasiticCodes(is_in=True, verbose=False): 
    return getInfectiousParasiticCodes(is_in=is_in, verbose=verbose, n_samples=None)

def evalRoot(targets, scope=None, verbose=True, **kargs): 
    """
    Evaluate root codes. 

    Arguments
    ---------
    * scope: a set of base codes to match against targets (in order to 
        find out which base codes do not exist in targets yet)

    Output
    ------
    2-tuple (tb, freespots) where 
       tb: a table mapping from base code (integer part) to a set of codes of the same base code 
       freespots: given a contiuous integer scope, freespots are those integers that are not 
                yet being taken by any of the targets
    """
    tb = {}
    for c in targets: 
        m = Code.p_icd9.match(str(c))
        if m: 
            base = m.group('base')
            if not tb.has_key(base): tb[base] = []
            tb[base].append(c)
        else: 
            print('evalRoot> Warning: invalid code %s' % c)

    # [analysis]
    freespots = []
    if verbose: 
        alist = []
        for c in tb.keys():
            icode = int(c)
            alist.append(icode)
        if not scope: 
            scope = range(1, 139+1)
            scope.append(481)
        freespots = list(set(scope)-set(alist))
        print('evalRoot> still missing:\n%s\n' % freespots)

    return tb, freespots

def assignRoot(targets, freespots, verbose=True, **kargs):
    """
    Assign a set of target ICD9 codes to the free spots (i.e. base codes
    not in targets yet). 
    """ 
    tb = {}
    for c in targets: 
        m = Code.p_icd9.match(str(c))
        if m: 
            base = m.group('base')
            if int(base) in freespots: 
                if not tb.has_key(base): tb[base] = []
                tb[base].append(c)
        else: 
            print('assignRoot> Warning: invalid code %s' % c)
    return tb

def makeDescription(targets, **kargs):
    """

    Use 
    ---
    infectionAnalyzer
    labAnalyzer

    """
    import icd9utils, os, dfUtils
    # if targets is None: targets = Params.bulk_train
    # code, icd9utils.lookup(code)) 
    
    sep = kargs.get('sep', ',')
    opath = kargs.get('opath', kargs.get('path', None)) 
    if opath is None: 
        opath = os.getcwd()
    else: 
        assert os.path.exists(opath), "makeDescription> Invalid output path: %s" % opath
    fname = 'icd9-desc.csv' # len(targets)
    extended = kargs.get('include_short', True)
    prefix_diag = kargs.get('prefix', 'c')
    fp = os.path.join(opath, fname)
    try: 
        print('makeDescription> Writing %d targets to %s' % (len(targets), fp))
        fd = open(fp, 'w')
        header = ['code', 'desc'] if not extended else ['code', 'desc', 'short']
        fd.write(sep.join(header)+'\n')
        for c in targets: 
            v = icd9utils.lookup(c)
            assert v is not None

            if not extended: 
                fd.write(str(c)+sep+v+'\n')
            else: 
                # use default value unless otherwise specified 
                v2 = '%s%s' % (prefix_diag, c) if not c in kargs else kargs[c]
                fd.write(str(c)+sep+v+sep+v2+'\n')
    finally: 
        fd.close()

    if kargs.get('sort_', True): 
        dtypes = {'code': str}
        df = dfUtils.load_df(_file=fp, sep=sep, verbose=False, dtypes=dtypes) 
        df = df.sort('code')
        dfUtils.save_df(df, _file=fp, sep=sep, verbose=False)
        
    return 

def isWithin0(e, scope, soft_=False):
    if not isCode(e): return False
    if isinstance(e, str): e = e.lower()
    if isSupplementary(e): 
        # assert isinstance(scope[0], str)
        lc = scope[0][0].lower()  # the first char of the lowerbound
        uc = scope[1][0].lower()
        if not lc in ('e', 'v',) or (lc > uc): return False
         
        tval = False
        x = Code.p.match(e).group('char')
        if not (lc <= x and x <= uc): return False 
        
        decimal = Code.p.match(e).group('decimal')  # E15.40   base:15, decimal:40
        # assert decimal is not None  # can't just have leading E or V 
        e = Code.p.match(e).group('base')+'.'+decimal if decimal else Code.p.match(e).group('base')
        lower, upper = scope[0][1:], scope[1][1:]
    else: 
        lower, upper = scope[0], scope[1]

    e_ = numerize(e) 
    return numerize(lower) <= e_ and e_ <= numerize(upper)

def isValidScope(scope): 
    pass

def isIn(e, codes, base_only=True): 
    """
    Check if an icd9 code e is in the set codes. 
    """
    if base_only: 
        for code in codes: 
            if eq(e, code): # or use eq2(...) for string match
                return True 
    else: 
        for code in codes: 
            if abs_eq(e, code): 
                return True 
    return False

def findMatched(target, candidates, base_only=True):
    return match(target, candidates, base_only=True)
def match(target, candidates, base_only=True):
    """
    Find diagnostic code(s) in candidates that match *target*
    if base_only is set to true, a loose match on the first 3 digits count as a match. 

    Output
    ------
    A list of tuples containing the matched codes within candidates and their positions 
    in the format of (position, matched_code).
    """ 
    if not hasattr(candidates, '__iter__'): # just a string will not pass
        candidates = [candidates]

    iset = [] # indexed set
    if base_only: 
        for i, code in enumerate(candidates): 
            if eq2(target, code):  # or use eq2(...) for string match
                iset.append((i, code))
    else: 
        for i, code in enumerate(candidates): 
            if abs_eq(target, code): # always a string match
                iset.append((i, code))       
    return iset        

def isA(x, y, strict=False): 
    """Check if x is a y (e.g. 112.0 is a 112)
    x is expected to be more specific than y 

    Arguments
    ---------
    strict: if True, then x IS A y iff their longest prefix match 
            is identical
            if False, then x IS A y as long as they share the same 
            root code
    """
    x, y = (str(x), str(y))
    if len(x) < len(y): return False 
    if strict: 
        return x[:len(y)] == y 
        
    return getRootCode(x) == getRootCode(y)

def isWithin(e, scope, soft_=False):
    if not isCode(e): return False
    if isinstance(e, str): e = e.lower()
    eb = Code.p.match(e).group('base')
    lb, ub = Code.p.match(str(scope[0])).group('base'), Code.p.match(str(scope[1])).group('base')

    # codes of different base lengths (number of base digits) cannot be compared
    if len(eb) != len(lb) or len(eb) != len(ub): return False

    if isSupplementary(e): 
        lc = scope[0][0].lower()  # the first char of the lowerbound
        uc = scope[1][0].lower()
        if not lc in ('e', 'v',) or (lc > uc): return False
        x = Code.p.match(e).group('char')
        if not (lc <= x and x <= uc): return False 
    
        decimal = Code.p.match(e).group('decimal')  # E15.40   base:15, decimal:40
        e = eb+'.'+decimal if decimal else eb
        lower, upper = scope[0][1:], scope[1][1:]
    else: 
        lower, upper = scope[0], scope[1]

    # now just compare the numeric parts of the codes 
    return gt_eq(e, lower) and lt_eq(e, upper)

def isCode(x): 
    return str(x).find('-') < 0 and Code.p.match(str(x))

def isSupplementary(code): 
    if str(code).lower().startswith(('e', 'v', )): 
        return True
    return False 

def numerize(code, default=None, convert_char=False): 
    if isinstance(code, str): 
        if isSupplementary(code): 
            if convert_char: 
                if code[0].lower() == 'e':
                    return 5000 + numerize(code[1:])
                elif code[0].lower() == 'v':
                    return 22000 + numerize(code[1:])
            return default 
        code = code.lstrip('0')
        if not code: 
            return 0
        try: 
            return float(code)
        except: 
            print "numerize> %s is a number? %s" % (code, isNumber(code))
    elif isNumber(code): 
        return float(code)
    raise ValueError, "numeric> possibly ill-formatted code: %s" % code  
 
def gt(x, y):
    """

    Use
    ---

    012.5 < 013

    but 013.25 !< 013
    """
    xb, yb = base(x), base(y)
    if len(xb) != len(yb): return False   # codes with different lengths/digits can't be compared 
    return numerize(xb) > numerize(yb)

def lt(x, y):
    xb, yb = base(x), base(y)
    if len(xb) != len(yb): return False   # codes with different lengths/digits can't be compared 
    return numerize(xb) < numerize(yb)  

def eq2(x, y): 
    """

    Memo
    ----
    1. c = '123450' 
       c.split('.') ~>  ['123450']
    """
    xs, ys = str(x), str(y)
    xs, ys = xs.strip(), ys.strip()

    # 123.52 ~ 123.72  # but 123.52 != 123 ? even though their bases are the same but 123 could not a non-diag code
    xl = xs.split('.')
    yl = ys.split('.')

    return xl[0] == yl[0] # and len(xl) == len(yl)
    
def eq(x, y): 
    """

    18.34 == 18.3 == 18 because their bases are the same 
    but 18.3 is a child of 18 
        18.34 is a child of 18.3
    """ 
    return numerize(base(x)) == numerize(base(y)) 

def abs_eq(x, y):  # absolutely equal 
    return str(x).strip().lower() == str(y).strip().lower() 

def gt_eq(x, y):
    return gt(x, y) or eq(x, y)
def lt_eq(x, y): 
    return lt(x, y) or eq(x, y)

def base(e):
    assert isCode(e)
    e = str(e).lower() 
    return Code.p.match(e).group('base')  

def decimal(e): 
    assert isCode(e)
    e = str(e).lower() 
    return Code.p.match(e).group('decimal')

def prefix(e):
    assert isCode(e)
    e = str(e).lower() 
    return Code.p.match(e).group('char')      

def isParent(): 
    pass 
def isChild(): 
    pass 

# [predicate]
def isInfectiousParasitic(code): 
    return isWithin(code, Code.scopetb['infection']) or (code in Code.extra_infections)

def hasInfectiousParasitic(codes, negate_=False): 
    for code in codes: 
        if isInfectiousParasitic(code): 
            return not negate_ 
    return negate_
def hasNoInfectiousParasitic(codes): 
    return hasInfectiousParasitic(codes, negate_=True)

def isNeoplasms(code): 
    return isWithin(code, Code.scopetb['neoplasms'])

def isTuberculosis(code): 
    if not code.startswith('0'): return False
    return isWithin(code, Code.subscopetb['infection']['tuberculosis'])
def getTuberculosisCodes(): 
    return getInfectiousParasiticCodes(filter_=isTuberculosis)

def codeStringToList(codes, base_only=False, no_VE=False, sep=' ', dotpos=3):
    return preproc_code(codes, base_only=base_only, no_ve_code=no_VE)

def preproc_code(code_str, **kargs):
    """
    Given a set of "flat codes," convert them to standard ICD-9 codes.
    e.g. 25015 => 250.15 

    Memo
    ----
    Diabetes mellitus without complication
        24900 25000 25001 7902 79021 79022 79029 7915 7916 V4585 V5391 V6546                

    Diabetes mellitus with complications
        24901 24910 24911 24920 24921 24930 24931 24940 24941 24950 24951 24960 24961 24970 24971 24980 24981 24990 24991 25002
        25003 25010 25011 25012 25013 25020 25021 25022 25023 25030 25031 25032 25033 25040 25041 25042 25043 25050 25051 25052
        25053 25060 25061 25062 25063 25070 25071 25072 25073 25080 25081 25082 25083 25090 25091 25092 25093 

    Diabetes or abnormal glucose tolerance complicating pregnancy; childbirth; or the puerperium
        64800 64801 64802 64803 64804 64880 64881 64882 64883 64884     

    """ 
    sep, dpos = kargs.get('sep', ' '), kargs.get('dotpos', 3)
    base_only = kargs.get('base_only', False) or kargs.get('simplify_code', False)
    filter_ve = kargs.get('no_ve_code', False)

    codes = code_str.split()
    if base_only: 
        for i, c in enumerate(codes): 
            c = c.strip()
            codes[i] = c[0:dpos]
    else: 
        for i, c in enumerate(codes): 
            c = c.strip()
            base, category = c[0:dpos], c[dpos:]
            codes[i] = c[0:dpos] + '.' + c[dpos:] if len(category) > 0 else c[0:dpos] 

    if filter_ve: # no V or E codes 
        codes = [c for c in codes if not c.lower().startswith(('v', 'e'))]

    if kargs.get('unique', False): 
        return list(set(codes))

    return codes

def preproc_code_base(code_str, **kargs): 
    sep, dpos = kargs.get('sep', ' '), kargs.get('dotpos', 3)
    codes = code_str.split()
    for i, c in enumerate(codes): 
        c = c.strip()
        codes[i] = c[0:dpos] 

    return codes    


def test_icd9info(icd9targets=None, **kargs):
    def query(codes, lookup_func): 
        for code in codes: 
            print "%s -> %s" % (code, lookup_func(code))
        return

    cnt = 0 
    for i, row in enumerate(read()): 
        icd9c, value = row[0], row[1]
        # if i > 10: break
        cnt+=1
        # print "%s -> %s" % (icd9c, value)  
    print "%d codes" % cnt 
    # print "> non codes: %s" % Code.codes
    
    # lookup
    codes = ['E900', 'E909.9', '481', '654.44', '1000', '00', '045.22']
    # for func in (lookup, lookup2, ): 
        # print(timeit.timeit("test()", setup="from __main__ import test"))
    query(codes, lookup2)

    return

def t_description(): 
    import configure 
    targets = configure.Params.bt_codes

    # e.g. ['008.45', '011.93', '481', '036.0', '041.85', '133.0', '061', '070.51', '083.2', '097.1', '123.1', '112.5']
    makeDescription(targets=targets, opath='data-feature', sep='|', sort_=True)

    return 

def test_predicate(): 
    e = '02'
    scope = ('00', '01.1')
    print "is %s within %s? %s" % (e, str(scope), isWithin(e, scope))

    e = 'e79'
    scope = ('v001', 'v79.01')
    print "is %s within %s? %s" % (e, str(scope), isWithin(e, scope))

    cases = [('02', ('1.15', '3'))]


    codes = [481, '005', '039.1', 'V42.0', '010.2', '010.01', 'I9_xxxyy']
    code = 'v42.0'
    print("> is %s in %s? %s (base only)" % (code, codes, isIn(code, codes, base_only=True)) )
    print("> is %s in %s? %s" % (code, codes, isIn(code, codes, base_only=False)) )

    print("\n> extracting root codes ...\n")
    for code in codes: 
        print('> code: %s => root: %s' % (code, getRootCode(code, exception=False, invalid_as_is=True)))

    return

def display(adict, title=None, msg_per_entry=None):
    if title: div(message='icd9utils: %s' % title, symbol='*')
    for k, v in adict.items(): 
        if msg_per_entry: 
            print('[%s] %s (%s)' % (k, v, msg_per_entry))
        else: 
            print('[%s] %s' % (k, v))
    return

def t_hierarchy(): 

    import utils, configure

    X = [481.01, '112', '130.09']
    Y = [481.0, '112.0', '130.9']
    for i, x in enumerate(X): 
        if isA(x, Y[i]): 
            print('> %s is a %s' % (x, Y[i]))

    # root analysis 
#     targets = ['047.8','112.2','038.10','038.11','112.5','047.9','038.19','090.9','135','041.9','041.6',
# '090.1','138','041.3','001.1','017.00','011.93','112.4','003.0','094.9','008.45',
# '054.2','070.71','052.7','088.81','041.7','027.0','131.01','041.89','041.85','049.9',
# '046.3','009.2','009.3','009.0','009.1','038.2','117.3','038.0','091.3','117.5',
# '038.8','117.9','054.10','041.19','136.3','041.10','041.11','031.2','031.0','031.9',
# '031.8','112.3','033.9','041.02','041.01','041.00','079.0','079.6','041.09','079.4',
# '054.13','070.51','007.1','070.32','070.30','038.3','038.49','038.43','038.42','038.40',
# '054.79','053.19','110.0','110.3','137.0','075','057.9','112.89','112.84','097.9',
# '097.1','078.5','078.0','070.70','054.3','099.9','127.4','005.9','136.9','053.9',
# '054.11','083.2','054.19','481','130.7','036.0','130.0','008.69','053.79','087.9',
# '008.61','111.9']

    otra = '112.1,112.0,112.9,072.9,096,056.9,041.8,098.86,041.4,041.5,041.2,041.0,011.12,091.0,026.9,001.9,091.9,123.1,003.1,074.0,003.9,074.8,077.99,098.0,008.6,098.2,054.0,054.6,008.8,099.40,099.41,052.9,129,088.82,057.0,039.9,008.43,010.10,131.9,039.1,133.0,079.53,040.82,099.50,099.53,099.55,099.54,039.8,090.2,035,092.9,010.01,010.00,041.1,094.0,131.00,079.51,079.83,041.86,131.09,079.88,079.89,049.8,048,042,038.1,038.9,094.89,136.1,136.8,031.1,079.98,066.3,139.8,033.0,070.54,041.04,041.03,074.3,079.2,079.1,070.22,054.40,054.43,007.4,045.90,007.2,070.59,061,078.19,077.8,070.31,078.10,078.11,004.9,046.1,038.44,038.41,058.10,053.12,053.11,084.0,084.6,110.1,070.41,110.2,110.5,110.4,110.9,110.8,054.8,134.0,054.9,010.90,057.8,078.89,078.88,040.0,055.9,112.81,078.8,097.0,078.2,078.1,111.0,002.0,127.2,099.1,099.0,099.3,054.12,053.21,070.3,053.0,034.0,034.1,130.9,111.8,036.2,132.9,088.8,008.62,132.2,132.1,132.0,088.0'
    otra = otra.split(',')

    # get annotated codes
    gfiles = ['gold_candidates_neg_random_gh.csv', 'gold_candidates_pos_random_gh.csv']
    acodes = set()
    sdict = {}
    for i, f in enumerate(gfiles): 
        fp = os.path.join('data-gold', f)
        df = pd.read_csv(fp, sep='|', header=0, index_col=False, error_bad_lines=True, dtype={'icd9': str})
        codes = df['icd9'].values
        sdict[i] = codes 
        acodes.update(codes)
    n_annotated = len(acodes)
    print('info> n_annotated: %d' % n_annotated)  # 54
    overlap = set(sdict[0]).intersection(sdict[1])
    print('info> overlap? size: %d, %s' % (len(overlap), overlap))

    total_set = configure.Params.code_set
    print('info> size of total targets: %d' % len(total_set))
    targets = list(set(total_set)-set(acodes))
    n_remaining = len(targets)
    n_targets = 100
    n_to_draw = n_targets - n_annotated
    print('info> number of remaining: %d but only need %d' % (n_remaining, n_to_draw))
    # otra = configure.Params.otra

    cur, freespots = evalRoot(targets, scope=None, verbose=True)
    print('> n_roots:%d, current roots:\n%s\n' % (len(cur), cur.keys()))
    utils.div()
    display(cur)
    n = n_to_draw   # setting too high may take time for UpSetR to finish 
    candidates = utils.sample_hashtable(cur, n_sample=n)
    print('> sample existing %d=?=%d candidates:\n%s\n' % (n, len(candidates), list(candidates)))

    print('-' * 100)
    acodes.update(candidates)
    wanted = list(acodes)
    print('info> %d candidates:\n%s\n' % (len(wanted), wanted))
    print('-' * 100)

    newcodes = assignRoot(otra, freespots)
    print('> suggested pick:\n%s\n' % newcodes)

    n = 10 
    candidates = utils.sample_hashtable(newcodes, n_sample=n)
    print('> sample %d=?=%d candidates:\n%s\n' % (n, len(candidates), list(candidates)))
    
    return

def t_hierarchy2(): 
    import utils

    # get annotated codes
    gfiles = ['gold_candidates_neg_random_gh.csv', 'gold_candidates_pos_random_gh.csv']
    acodes = set()
    sdict = {}
    for i, f in enumerate(gfiles): 
        fp = os.path.join('data-gold', f)
        df = pd.read_csv(fp, sep='|', header=0, index_col=False, error_bad_lines=True, dtype={'icd9': str})
        codes = df['icd9'].values
        sdict[i] = codes 
        acodes.update(codes)
    n_annotated = len(acodes)
    print('info> n_annotated: %d' % n_annotated)  # 54

    codes = ['112.3', '047.9', '038.10', '038.11', '112.5', '031.0', '038.19', '031.9', '031.8', '090.9', '041.09', '135', '041.9', '041.6', 
    '090.1', '138', '033.9', '049.9', '031.2', '003.0', '001.1', '017.00', '011.93', '041.00', '079.0', '079.6', '123.1', '079.4', '112.4', 
    '009.0', '112.2', '070.51', '034.0', '007.1', '061', '070.32', '070.30', '054.79', '054.2', '054.3', '054.10', '046.3', '052.7', '038.42', 
    '038.40', '088.81', '053.19', '010.10', '133.0', '110.0', '110.3', '137.0', '040.82', '008.45', '098.0', '075', '057.9', '112.89', '041.7', 
    '112.84', '027.0', '097.1', '078.5', '136.9', '078.0', '009.1', '070.70', '131.01', '070.71', '099.9', '041.89', '127.4', '041.85', '097.9', 
    '005.9', '054.13', '053.9', '054.11', '047.8', '009.3', '083.2', '054.19', '481', '117.3', '091.3', '117.5', '130.7', '038.8', '117.9', '036.0', 
    '094.9', '130.0', '136.3', '008.69', '053.79', '087.9', '041.10', '041.11', '008.61', '111.9']

    assert len(set(acodes) - set(codes)) == 0

    print('info> size: %d' % len(codes))

    cur, freespots = evalRoot(codes, scope=None, verbose=True)
    print('> n_roots:%d, current roots:\n%s\n' % (len(cur), cur.keys()))
    utils.div()
    display(cur)
    n = 100   # setting too high may take time for UpSetR to finish 
    candidates = utils.sample_hashtable(cur, n_sample=n)
    print('> sample existing %d=?=%d candidates:\n%s\n' % (n, len(candidates), list(candidates)))
    

    return

def t_predicate(): 
    alist = ['fset_antibio_feature_description_full_bt_190.csv', 'archive', 'fset_urine_feature_description_full_bt_190.csv', 
    'log', '111.9', '061', '057.9', 'fset_blood_feature_description_full_bt_190.csv', '053.79', '054.11', 'README.txt', '070.32', '090.1', 
    'learner.log', 'fset_microbio_feature_description_full_bt_190.csv', '038.0', '041.02', '054.10', 'exitcode']

    codes = []
    for e in alist: 
        if isCode(e):
            codes.append(e)
    print('info> Found %d diag codes:\n%s\n' % (len(codes), codes))

    testdir = 'test'
    ipath = 'data-learner/%s' % testdir

    assert len(os.listdir(ipath)) > 0
    codes = [d for d in os.listdir(ipath) if os.path.isdir(d) ] # and isCode(d)
    print('info> Found %d diag code dirs:\n%s\n' % (len(codes), codes))

    return

def t_convert(): 
    icd9x = ['058.11', '005', '039.1', 'V42.0', '010.2', '010.01']
    mcodes = [] 
    for code in icd9x: 
        mcode = getMedCode(code)
        print("> icd9 %s => med %s" % (code, mcode))

def test_query(): 
    codes = getInfectiousParasiticCodes()
    print "how many? %d" % len(codes)
    print "max? %s" % Code.max(codes)
    print "min? %s" % Code.min(codes)

    except_ = [481, '005', '039.1', 'V42.0', '010.2', '010.01']  # 3 not valid
    print "min max of except_: %s << %s" % (Code.min(except_), Code.max(except_))
    codes = getInfectiousParasiticCodes(diff=except_, verbose=True)
    print "how many? %d" % len(codes)
    print "max? %s" % Code.max(codes)
    print "min? %s" % Code.min(codes) 

    n = 100
    print('> randomly select %d infectious diseases' % n)
    codes = getInfectiousParasiticCodes(n_samples=n)
    print "how many? %d | they are: %s" % (len(codes), codes)
    print "max? %s" % Code.max(codes)
    print "min? %s" % Code.min(codes) 
    print "type: %s" % type(codes)  
    div(message='> mapping from codes to names ...')
    for i, code in enumerate(codes): 
        print "[%d] %s -> %s" % (i, code, getName(code)) 

    print "-" * 60
    regex = 'meningitis' # 'mening.*'   
    print "Getting %s-related codes ..." % regex
    codes = getCode(regex)
    div(message='Found %d codes with %s as a keyword.' % (len(codes), regex))
    for code in codes: 
        print "  + %s -> %s" % (code, getName(code)) 

    print "-" * 60
    codes = getInfectiousParasiticCodes(filter_=isTuberculosis)
    codes2 = getCode('tubercu')
    print "> size %d =?= %d" % (len(codes), len(codes2))
    # for code in codes: 
    #     print "  + %s -> %s" % (code, getName(code))    

def t_transform(): 
    codes = [481, '005', '039.1', 'V42.0', '010.2', '010.01', 11100, 'MED:31415', 'Test', "lepsis 20mg once every week", "S02.412A"]
    print('input> %s' % codes)
    print('> %s' % getRootSequence(codes))
   
    print('info> input as code string ...')
    code_str = ' '.join([str(e) for e in codes])
    print('> %s' % getRootSequence(code_str))

    return

def t_preproc(**kargs):
    
    ### input 
    # code_str = '24900 25000 25001 7902 79021 79022 79029 7915 7916 V4585 V5391 V6546'
    # code_str += ' ' + """24901 24910 24911 24920 24921 24930 24931 24940 24941 24950 24951 24960 24961 24970 24971 24980 24981 24990 24991 25002
    #     25003 25010 25011 25012 25013 25020 25021 25022 25023 25030 25031 25032 25033 25040 25041 25042 25043 25050 25051 25052
    #     25053 25060 25061 25062 25063 25070 25071 25072 25073 25080 25081 25082 25083 25090 25091 25092 25093"""
    # code_str += ' ' + "64800 64801 64802 64803 64804 64880 64881 64882 64883 64884" 

    code_str = '585 5851 5852 5853 5854 5855 5856 5859 7925 V420 V451 V4511 V4512 V560 V561 V562 V5631 V5632 V568'

    # [params] 
    base_only = False
       
    codes = preproc_code(code_str, base_only=base_only)
    print('> n_codes: %d\n> codes:\n%s\n' % (len(codes), codes))

    codes_minus_ve = preproc_code(code_str, base_only=base_only, no_ve_code=True)
    print('> n_codes: %d\n> codes:\n%s\n' % (len(codes_minus_ve), codes_minus_ve))

    # [status] ok 
    # print('\nNow, do base only\n')
    # codes = preproc_code(code_str, base_only=False)
    # print('> n_codes: %d\n> codes:\n%s\n' % (len(codes), codes))
    # print('> codeset:\n%s\n' % set(codes))  # [log] set(['791', '790', 'V65', 'V45', 'V53', '648', '250', '249'])

    div(message='Now, testing lookup ...')
    n_limit = 100
    for j, c in enumerate(codes[:20]): 
        description = lookup2(c)
        print('+ code: %s => %s' % (c, description))
        if j >= n_limit: break

    return 

def test(**kargs):
    # test_icd9info()

    # test_predicate()
    # test_query()

    ### Find wanted diagnostic codes 
    # t_hierarchy()

    # t_hierarchy2()

    ### description file 
    # t_description()

    ### code preprocessing, formatting 
    t_preproc()

    ### transform codes ### 
    # t_transform()

    ### Misc 
    # t_convert()
    # t_predicate()

    return

if __name__ == "__main__":
    # import timeit
    test()
    # print(timeit.timeit("test()", setup="from __main__ import test"))
