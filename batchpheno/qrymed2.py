import os, sys, re, random
import commands
from collections import deque

from config import seq_maker_config, sys_config

DataRoot = sys_config.read('DataRoot')
DataExpRoot = sys_config.read('DataExpRoot')
TestDir = sys_config.read('TestDir')
RefDir = sys_config.read('RefDir')  # external support data (e.g. ICD9_descriptions)
BinDir = sys_config.read('bin')

ICD9Dir = RefDir
ICD9Tbf = os.path.join(ICD9Dir, 'ICD9_descriptions')

try:
    import cPickle as pickle
except:
    import pickle

# [todo] use a config file 
LabTest = {}

# for microbiology test 
LabTest['microbiology'] = ['2235', ]  

ExeName = 'qrymed'
Qrymed = os.path.join(BinDir, ExeName)
assert os.path.exists(Qrymed), "kernel qrymed does not exist at: %s" % Qrymed

class Params(object): 
    sep = '|'
    cmd = 'qrymed'
    descr_header = description_header = ['code', 'description']   # or 'feature'

def preprocess():
    if not os.path.exists(icd9tbf):
        raise RuntimeError, "[preprocess] Could not find ICD9 map file: %s" % icd9tbf
    # [todo] 

    return    

#<helper>
def _exec(cmd): 
    """
    
    [note]
    1. running qrymed with this command returns a 'list' of medcodes or None if 
       empty set
    """
    st, output = commands.getstatusoutput(cmd)
    if output[:10].lower().find('error') >= 0: st = 2
    if st != 0:
        raise RuntimeError, "Could not exec %s" % cmd
    #print "[debug] output: %s" % output
    return output  #<format?>
def _exec2(cmd, err_default=None, verbose=True):
    # print('debug> input cmd: %s' % cmd)
    st, output = commands.getstatusoutput(cmd)
    # print('debug> st: %s, output: %s' % (st, output))
    if output[:10].lower().find('error') >= 0: st = 2
    if st != 0: 
        # output = '?' if err_default is None else err_default
        output = err_default
        if verbose: print('error> %s' % cmd)
        return (st, output)
    return (st, output) 

def eval(cmd, err_default=None, verbose=True):
    st, output = _exec2(cmd, err_default=err_default, verbose=verbose)
    return output  

def _htemplate(switch, self_=True, expand=False):
    switch = switch    
    if self_: switch = 'i%s' % switch
    cmd = '%s -%s %s' % (Qrymed, switch, code)
    if not expand:
        output = eval(cmd)
        if output is not None:  
            return output.split()
        else: 
            return None
        
    cmd += ' | ' + '%s -e -pnm' % Qrymed
    return eval(cmd)

def _query_code(switch, args, to_int=True, verbose=False): 
    cmd = 'qrymed -%s %s' % (switch, args)
    codes = eval(cmd, verbose=verbose)
    if codes is not None: 
        codes = codes.split()  # can also split '\n' 
        if to_int:
            return map(int, codes)
        else: 
            return codes
    return None

def max_n_digits(): 
    cmd = 'qrymed -desc 1 | tail -n 10'
    alist = eval(cmd)
    codes = alist.split()
    # print [len(str(code)) for code in alist.split()]
    return max(len(str(code)) for code in alist.split())

def getName0(codes):
    cmd = 'qrymed -pnm %s' % codes 
    return _exec(cmd)
def getName(codes, **kargs):
    cmd = 'qrymed -pnm %s' % codes 
    return eval(cmd, **kargs)

def getName2(codes, **kargs): 
    # print('debug> codes: %s' % codes)
    # cmd = 'qrymed -pnm %s' % codes 

    # err_default = kargs.get('err_default', '')
    # verbose = kargs.get('verbose', False)
    # st, output = _exec2(cmd, err_default=err_default, verbose=verbose)

    output = getName(codes, **kargs)
    if output is None or output.lower().find('error') >= 0: 
        return kargs.get('err_default', '')
    return output

def getDescription(codes): 
    return getName(codes)
    
def getDescendent(code, expand=False, self_=False, 
                  write_to_file=False, _file=None, to_int=True, verbose=False):
    switch = 'desc'    
    if self_: switch = 'idesc'
    if not expand: 
        return _query_code(switch, code, to_int=to_int, verbose=verbose)     
    cmd += ' | ' + 'qrymed -e -pnm'
    return eval(cmd, verbose=verbose)

def idesc(code, **kargs): 
    return getDescendent(code=code, self_=True, **kargs)
def desc(code, **kargs): 
    return getDescendent(code=code, self_=False, **kargs)

def getAncestor(code, expand=False, self_=True, 
                  write_to_file=False, _file=None, to_int=True, verbose=False):
    switch = 'anc'    
    if self_: switch = 'ianc'
    if not expand: 
        return _query_code(switch, code, to_int=to_int, verbose=verbose)    
    cmd += ' | ' + 'qrymed -e -pnm'
    return eval(cmd, verbose=verbose)

def ianc(code, **kargs): 
    return getAncestor(code=code, self_=True, **kargs)
def anc(code, **kargs): 
    return getAncestor(code=code, self_=False, **kargs)

def getParents(code, to_int=True): 
    return _query_code('par', code, to_int=to_int) 
    
def getChildren(code, to_int=True):
    return _query_code('child', code, to_int=to_int)  

def _ntemplate(name='microbiology', postfix='desc', _getter=None, use_table=False, 
                 dup=False, save_=True, self_=True, to_int=True): 
    tb = {}
    if use_table and LabTest.has_key(name): 
        codes = LabTest[name]
    else: 
        codes = find(name)  
    if not hasattr(_getter, '__call__'): 
        raise RuntimeError, "_ntemplate> given getter is not callable: %s" % _getter
    for code in codes:
        codeset = _getter(code, expand=False, self_=self_) 
        if not dup: codeset = list(set(codeset))
        if to_int:
            tb[int(code)] = [int(code) for code in codeset] 
        else: 
            tb[code] = codeset  
    if save_: 
        f = os.path.join(DataExpRoot, '%s_%s.pkl' % (name, posfix)) # [todo]
        pickle.dump(desctb, open(f, "wb" ))         
    return tb

def getAncestorByName(name='microbiology', use_table=False, dup=False, save_=True, self_=True, to_int=True):  
    tb = _ntemplate(name=name, posfix='anc', _getter=getAncestor, dup=dup, 
                       save_=save_, self_=self_, to_int=to_int)
    return tb

def getDescendentByName(name='microbiology', use_table=False, dup=False, save_=True, self_=False, 
                          to_int=True, verbose=False): 
    """

    [note] 1. use set for faster search as a filter
    """
    desctb = {}
    if use_table and LabTest.has_key(name): 
        codes = LabTest[name]
    else: 
        codes = find(name, to_int=to_int)
    # then only find selected descendents 
    if verbose: print "desc> name: %s maps %s heads : %s" % (name, len(codes), codes)
    for code in codes:
        # if not desctb.has_key(code): 
        #     desctb[code] = []
        codeset = set(getDescendent(code, expand=False, self_=self_, to_int=to_int))
        if verbose > 2: print "desc> %s has %s descendents ->\n   %s" % (code, len(codeset), list(codeset)[:10])
        desctb[code] = codeset  
    if save_: 
        f = os.path.join(DataExpRoot, '%s_desc.pkl' % name)
        pickle.dump(desctb, open(f, "wb" ))
    return desctb
def codePair(desctb): 
    heads, members = [], set([])
    for k, v in desctb.items(): 
        heads.append(k)
        members.update(v)
    return (heads, list(members))
def codeSet(desctb): 
    codes = set()
    for k, v in desctb.items():
        codes.update([k]+list(v))
    return codes


def groupFromWithin(codes): 
    """
    Try to group medcodes based on parent-child relation from within 
    members of the set. 
    """
    # from treelib import Node, Tree
    from tree import children
    codeset = set(codes)
    desctb, anctb = ({}, {})
    for code in codes: 
        parents, children = (getParents(code, to_int=True), getChildren(code, to_int=True))
        desctb[int(code)] = [child for child in children if child in codeset]
        anctb[int(code)] = [parent for parent in parents if parent in codeset]
    return (desctb, anctb)

def group(codes, max_level=1): 
    pass

def getDescendentByName2(name='microbiology', dup=False, save_=True, self_=False, 
                          to_int=True, verbose=0): 
    """Build a list of mapping from parents to children (so that later on it can be 
    converted to a tree structure) 
    
    Note
    -----
    1. use set for faster search as a filter
    """
    desctb = {}
    try: 
        from collections import OrderedDict
        desctb = OrderedDict() 
    except: 
        print "desc2> ordered dictionary not available"
    
    codes = find(name, to_int=to_int)
    # then only find selected descendents 
    if verbose: print "desc2> name: %s maps %s heads : %s" % (name, len(codes), codes)
    for code in codes:
        codeset = set(getChildren(code, to_int=to_int))
        if verbose > 2: print "desc2> %s has %s children ->\n   %s" % (code, len(codeset), list(codeset)[:10])
        desctb[code] = codeset 
    
    print "desc2> now building further hierarchy for a total of %d codes" % len(codeSet(desctb))
    Nt = len(desctb); print "desc2> before adding children as new heads, size=%d" % len(desctb)
    for head, members in desctb.items(): 
        # level[head] = offset
        mx = deque(members)
        while mx: # foreach child 
            m = mx.popleft()  # pop the child
            dmx = set(getChildren(m, to_int=True))
            if dmx: 
                overlap = dmx.intersection(mx)
                if overlap: print "desc2> %s found siblings as decendents: %s" % (m, overlap)
                # assert not desctb.has_key(m), "%s has already served as a head" % m
            if desctb.has_key(m):  # m with >0 children, this child is also 
                print "desc2> %s->%s already serves as a head with %d children =?= new query %d" % (head, m, len(desctb[m]), len(dmx))
                # overlap = dmx.intersection(desctb[m]); print "desc2> overlapped? how many? %s =?= %d" % (len(overlap), len(desctb[m]))
                # dmx.update(set(desctb[m]))
            else: 
                desctb[m] = dmx   # limit the query result to existing members? overlap.update(set(desctb[m]))
                       
    Ntp = len(desctb); print "desc2> after, size = %d, delta = %d" % (Ntp, Ntp-Nt)
    if save_: 
        f = os.path.join(DataExpRoot, '%s_desc.pkl' % name)
        pickle.dump(desctb, open(f, "wb" ))
    return desctb


def getAncestorHierarchy(name='microbiology', save_=True, load_=False): 
    pass

def getDescendentHierarchy(name='microbiology', save_=True, load_=False):
    """

    [note]
    1. Given a medcode, find its descendents (set X); then 
       pick a member of X and find its descendents (set Y) again, 
       you find that Y is not a subset of X. 
    """
    desctb = None
    if load_: 
        desctb = load(labtest=name, typ='hierarchy') 
        if not desctb is None: return desctb
        desctb = load(labtest=name, typ='desc')
    if not desctb: 
        desctb = getDescendentByName(name=name, self_=False, 
                          dup=False, save_=False, to_int=True, use_table=False)
    # heads, members = codePair(desctb)
    # anctb, level = {}, {}
    # offset = 0
    print "hierarchy> now building further hierarchy for a total of %d codes" % len(codeSet(desctb))
    Nt = len(desctb); print "hierarchy> before adding decendents as new heads, size=%d" % len(desctb)
    for head, members in desctb.items(): 
        # level[head] = offset
        mx = deque(members)
        while mx: 
            m = mx.popleft()
            dmx = set(getDescendent(m, self_=False, to_int=True))
            if dmx: 
                overlap = dmx.intersection(mx)
                # assert not (dmx-overlap), "inconsistent descendnts: %s should already contain\n%s" % (head, overlap)
                if overlap: print "hierarchy> %s found existing memebers as decendents: %s" % (m, overlap)
                # assert not desctb.has_key(m), "%s has already served as a head" % m
            if desctb.get(m, None): 
                print "hierarchy> %s already serves as a head with %d descendents" % (m, len(desctb[m]))
                dmx.update(set(desctb[m]))
            desctb[m] = dmx   # limit the query result to existing members? overlap.update(set(desctb[m]))
    Ntp = len(desctb); print "hierarchy> after, size = %d, delta = %d" % (Ntp, Ntp-Nt)
    if save_: 
        f = os.path.join(DataExpRoot, '%s_hierarchy.pkl' % name)
        pickle.dump(desctb, open(f, "wb" ))
    return desctb  
dhierarchy = getDescendentHierarchy    
 
def nameDescendentHierarchy(name='microbiology', dup=False, save_=True, to_int=True, limit=20, load_=True): 
    if load_: 
        nametb = load(labtest=name, typ='name')
        if not nametb is None: return nametb
    
    desctb = getDescendentHierarchy(name=name, save_=save_, load_=load_)  
    # always re-evaluate names of codes
    nametb = {}       
    for head, memebers in desctb.items(): 
        if limit: 
            nametb[getName(head)] = getName(members)[:limit]
        else: 
            nametb[getName(head)] = getName(members)
    if save_: save(labtest=name, typ='name')
    return nametb

def save(obj, _file=None, labtest='microbiology', typ='desc', verbose=True): 
    if _file is None: 
        if typ.find('desc') >= 0: f = "%(root)s/%(name)s_desc.pkl"
        elif typ.find('hierar') >= 0: f = "%(root)s/%(name)s_hierarchy.pkl" 
        else: f = "%(root)s/%(name)s_names.pkl"
        f = f % {'root': DataExpRoot, 'name': labtest}
    else: 
        dataroot, f = os.path.dirname(_file), os.path.basename(_file)
        if not dataroot: dataroot = DataExpRoot
        f = os.path.join(dataroot, f)
    pickle.dump(obj, open(f, "wb" ))
    if verbose: print "save> input saved to %s" % f
    return 

def load(_file=None, labtest='microbiology', typ='desc'):  
    print "%s> loading existing .pkl for lab test: %s" % (whosdaddy().lower(), labtest) 
    if _file is None: 
        if typ.find('desc') >= 0: f = "%(root)s/%(name)s_desc.pkl"
        elif typ.find('hierar') >= 0: f = "%(root)s/%(name)s_hierarchy.pkl" 
        else: f = "%(root)s/%(name)s_names.pkl"
        f = f % {'root': DataExpRoot, 'name': labtest}
    else: 
        dataroot, f = os.path.dirname(_file), os.path.basename(_file)
        if not dataroot: dataroot = DataExpRoot
        f = os.path.join(dataroot, f)
    desctb = None
    try: 
        desctb = pickle.load(open(f, "rb" ))
    except Exception, e: 
        msg = "load> could not load data from %s\n: %s" % (f, e)
        print msg
    return desctb

def filterByName(name='microbiology', load_=False, to_int=True, desc=True):
    """
    Find all medcodes and all their decendents associated with *name
    """
    has_data = False 
    tb = {} 
    if load_: 
        print "filterByName> searching existing .pkl for lab test: %s" % name 
        f = "%s/%s_%s.pkl" % (DataExpRoot, name, 'desc' if desc else 'anc')
        try: 
            tb = pickle.load(open(f, "rb" ))
            has_data = True
        except Exception, e: 
            msg = "filterByName> could not load data from %s\n: %s" % (f, e)
            print msg
            has_data = False 

    if not has_data:
        if desc:  
            tb = getDescendentByName(name, self_=True, dup=False, to_int=to_int)
        else: 
            tb = getAncestorByName(name, self_=True, dup=False, to_int=to_int)

    codes = set()
    for k, v in tb.items(): 
        codes.update(v)
    return codes

def filterByCode(codes=None, load_=False, to_int=True, desc=True): 
    """

    Arguments
    ---------


    Note
    ----
      2235 Microbiology Procedure
      23945 Antibiotic Preparations
    """
    # has_data = False 
    if codes is None: codes = [2235, 23945]
    tb = {}
    # if load_: 
    #     print "filterByCode> searching existing .pkl for lab test:\n" # (%d: %s)" % (code, getName(code))  
    #     for code in codes: 
    #         print " + %d: %s" % (code, getName(code))

    #     f = "%s/%s_%s.pkl" % (DataExpRoot, codes[0], 'desc' if desc else 'anc')
    #     try: 
    #         tb = pickle.load(open(f, "rb" ))
    #         has_data = True
    #     except Exception, e: 
    #         msg = "filterByCode> could not load data from %s\n: %s" % (f, e)
    #         print msg
    #         has_data = False
    _getter = idesc
    if not desc: _getter = ianc
        
    assert hasattr(_getter, '__call__')
    if to_int: 
        for code in codes: 
            tb[int(code)] = [int(c) for c in _getter(code)]
    else: 
        for code in codes: 
            tb[str(code)] = [str(c) for c in _getter(code, to_int=False)]
    # find the union
    union_ = set()
    for _, c in tb.items(): 
        union_.update(c)
    return union_

def size(code): 
    if isinstance(code, dict): 
        cnt = 0
        for k, v in code.items(): 
            cnt += len(v)
        # print "size> %d" % cnt
        return sum(len(vals) for vals in code.values())
    return len(code)    

def find(name, to_int=True): 
    return _query_code('find', name, to_int=to_int)
def findMedCode(**kargs):
    return find(**kargs)
    
def getMedCodeTable(names):
    """Return dict that maps queries to their related medcodes
    """
    mct = {}
    # convert each infection to its related medcodes
    for name in names:
        mct[name] = set(find(name))
    return mct

def medToICD9(code): 
    cmd = 'qrymed -val 48 %s' %  code
    return eval(cmd)
def getICD9(code):
    return medToICD9(code)
def medToICD(code):
    return medToICD9(code)

def ICD9ToMed(code): 
    cmd = 'qrymed -isval 48 %s' % code
    return eval(cmd)
def getMedCode(code): 
    return ICD9ToMed(code)

### predicates ### 

def isA(c1, c2, self_=True, lookup=None, ancestor_lookup=None, **kargs): # is c1 a kind of c2?  c1 < c2
    """
    Input
    -----
    c1: MED code in int
    c2: MED code in int

    """
    if self_ and c1 == c2: 
        return True

    # method 1: expensive! 
    if lookup is not None: 
        try: 
            return c1 in lookup[c2] # is c1 a decendent of c2  (c1 <- c2?)
        except: 
            return False 
    if ancestor_lookup is not None: # this map may be more practical if we know queried codes are mostly leaves
        try: 
            return c2 in anc_lookup[c1] # is c2 an ancestor of c1  (c1 <- c2?)
        except: 
            return False

    codes = getDescendent(c2, to_int=True, self_=False) # to_int: True, identify case has been handled up front  
    if codes is not None: 
        if int(c1) in codes: 
            return True
    return False

### utilities ###

# [todo]  
def getLabResult(name="microbiology", keywords=None, delimit=','): 
    """
    
    [note] useful commands: 
    
    """
    cmd = "qrymed -find %s | qrymed -e -pnm" % name
    if keywords: 
        for keyword in [w.replace(" ", "") in keywords.split(delimit)]:
            cmd += " " + "grep -i %s" % keyword  # [todo]
    output = _exec(cmd)
    
    # for each of the list, find all its descendents (specific tests)
        
    return output

def descriptionToCSV(file_, prefix=None, verbose=False): 
    """Convert the output of a 'qrymed -pnm <code>' type of query 
       (whose output assumes the format of code and description) 
       into a csv format.

    Use
    ---
    analyzer.py 

    """
    import sutils  # shell utility 
    prefix, base = os.path.dirname(file_), os.path.basename(file_)
    if not prefix: 
        prefix = configure.DirAnaDir
    ipath = os.path.join(prefix, base)
    assert os.path.exists(ipath), 'Invalid input file: %s' % ipath
    fname, ext = os.path.splitext(base)
    if verbose: print('qrymed2::descriptionToCSV> input file ext: %s' % ext)
    opath = os.path.join(prefix, fname+'.csv')
    try: 
        fd = open(opath, 'w')
        fd.write(Params.sep.join(Params.descr_header)+'\n')
        acc = 0
        for line in open(ipath, 'r'): 
            row = line.split()
            fd.write(row[0]+Params.sep+' '.join(row[1:])+'\n')
            acc += 1
    finally: 
        fd.close()
    if verbose: print("qrymed2::descriptionToCSV> processed %d entries" % acc)
    return acc
    
def icd9ToMed0(code, icd9mapfile=None, delimit="\t"):  
    """
    see icd9ToMed()
    [archive] 
    [todo] 1. change the sep of ICD9_descritpions to '|'
           2. some entries may specify alternative names
              e.g. 481 Pneumococcal pneumonia [Streptococcus pneumoniae pneumonia]
              
           &quot;Ventilation&quot; pneumonitis
           Farmers&apos; lung
           Chronic obstructive asthma, without mention of status asthmaticus
           Control of (postoperative) hemorrhage of anus
    """
    def expand_name(name):
        if name.find('[') >= 0: 
            #specp = re.compile(r'(?P<main>\w+(\s\w+)*)\s+\[(?P<alternative>[^[]+)\]')
            mat = specp.match(name)
            if mat:
                return [mat.group('main'), mat.group('alternative')]
            else: 
                raise RuntimeError, "[icd9ToMed] ill-formatted name: %s" % name
        return [name]
    
    if icd9mapfile is None: icd9mapfile=ICD9Tbf
    assert os.path.exists(icd9mapfile), "Invalid ICD9 file: %s" % icd9mapfile
   
    f = open(icd9mapfile, 'r')
    name = ""
    for j, line in enumerate(f):
        fields = line.split(delimit)
        icd9 = fields[0].strip(' ') #fields[0].lstrip('0').strip(' ')
        if code == icd9: 
            name = fields[1] 
            break
    
    print "> name: %s" % name
    cmd = "qrymed -find %s" % name
    output = _exec(cmd)
    return output.split('\n')


def writeTo():
    pass

def testcode():
    """

    [log]

    1. @cerner
    <num>
    59339|CPMC Laboratory Test: EBV-Vca,IgG 3|290|51|nan,6.09,6.46,3.71,3.392,3.53,4.19,7.269,0.6,1.96,5.24,5.09,7.5,0.46,4.57,2.7,5.64,2.941,8.141,3.41,3.37,6.99,5.46,0.06,4.566,5.363,7.358,2.731,7.07,4.575,6.18,6.51,6.17,4.08,5.36,3.93,7.23,5.15,4.48,6.178,1.539,1.964,2.47,3.154,2.551,0.71,2.934,9.437,5.83,1.849,4.92,0.91,4.94,9.84
36220|CPMC Laboratory Test: Toxoplasma Ab, IgG|179|84|nan,45,0.38,44,0.34,1,0,3,5,8,6,69,81
71526|CPMC LABORATORY TEST: EBV,PCR|136|7|nan,70000,11000,6900,22000,46000,360
92173|CPMC LABORATORY TEST: NTX (NM BCE/MM CREATININE)|27|3|nan,43,66,41
65027|CPMC Laboratory Test: Vitamin E (b-gamma Toco)|13|1|nan,0.8
89570|CPMC LABORATORY TEST: CREATININE, URINE -MG/DL|10|3|nan,32,38,67
92056|CPMC LABORATORY TEST: IMMUNOGLOBULIN D, SERUM|9|1|nan,176.1
65032|CPMC Laboratory Test: Vitamin E (alpha Toco)|7|1|nan,5.6


    <char> 
    7307|Patient Name|1108|547
    30353|Ordering Physician|1042|511
    70611|Specimen Collection Time|683|329
    70612|Specimen Accession Time|683|329
    36163|CPMC Laboratory Test: Specimen Description|523|170

    """
    labtest1 = 'microbiology'
    codes1 = filterByName(labtest1, load_=False)
    print "> number of %s-code: %d" % (labtest1, len(codes1)) #  number of microbiology-code: 11770
    labtest2 = 'antibiotic'
    codes2 = filterByName(labtest2, load_=False)
    print "> number of %s-code: %d" % (labtest2, len(codes2)) # number of antibiotic-code: 3248


    print "> do they intersect? by how much?"
    iset = codes1.intersection(codes2)
    print "> size of intersection: %d" % len(iset)
    print "> %f%% of %s" % ((len(iset)+0.0)/len(codes1), labtest1)
    print "> %f%% of %s" % ((len(iset)+0.0)/len(codes2), labtest2)


    testcodes = None 
    # testcodes = microbio_codes = [67845, 1030, 35853, 96912, 37998, 1057, 41255, 39336, 555, 46382, 560, 562, 78267, 448, 39368, 463, 472, 39901, 36449, 71526, 615, 39918, 37999, 38385, 494, 1019]
    testcodes = cerner_num = [59339, 36220, 71526, 92173, 65027, 89570, 92056, 65032]
    print "> checking potential ovelaps ..."
    overlaps = []
    for c in iset: 
        if c in testcodes: 
            overlaps.append(c)
    print "> overlaps: %s" % overlaps 

    print "> test microbio codes from cdr/lab..." 
    microbio_codes = [67845, 1030, 35853, 96912, 37998, 1057, 41255, 39336, 555, 46382, 560, 562, 78267, 448, 39368, 463, 472, 39901, 36449, 71526, 615, 39918, 37999, 38385, 494, 1019]
    # for c in microbio_codes: 
    assert not (set(microbio_codes) - set(codes1)), "> not an empty set? %s" % str(set(microbio_codes) - set(labtest1))
    print "> good, no assertions on %s" % labtest1 

    print "> in %s but not in %s" % (labtest2, labtest1)
    diff = set(codes2)-set(codes1)
    print "> set diff %s - %s: %s" % (labtest2, labtest1, list(diff)[:10])
    print "> size: %d" % len(diff)

    print "> in %s but not in %s" % (labtest1, labtest2)
    diff = set(codes1)-set(codes2)
    print "> set diff %s - %s: %s" % (labtest1, labtest2, list(diff)[:10])
    print "> size: %d" % len(diff)

    # common feature names 
    char_codes = [7307, 30353, 70611, 70612, 36163, ]
    print "> common features %s in above labtests?" % char_codes
    for c in char_codes: 
        if c in codes1 and c in codes2: 
            print "> %s is in both %s and %s" % (c, labtest1, labtest2)

    print "> test filtering by codes ..."
    codeset = [2235, 23945, 6527, ]
    codes = filterByCode(codes=codeset)
    print " + size of desc of %s: %d" % (codeset, len(codes))
    print " + dtype: %s" % type(list(codes)[random.randint(0, len(codes)-1)])

    return

def test1():
    """

    [log]
    test> size of all codes: 5020
    size of antibiotic-code: 3580
      => dups? 

    desc> name: antibiotic maps 116 heads 
    """
    import random
    # output = icd9ToMed('481')
    # print output
   
    labtest = 'microbiology' # 'microbiology' 
    print "test> head-to-descendents rep ... "
    codetb = getDescendentByName(labtest, verbose=0)
    print "test> size of code table: %d" % len(codetb)
    # allcodes = []
    # for k, v in codetb.items(): 
    #     print "base: %s => desc: %s" % (k, str(v))
        # allcodes.append(v)
    print "test> num. codes in head-to-descendents: %d" % len(codeSet(codetb)) 

    print "test> head-to-children rep ... "
    codetb = getDescendentByName2(labtest, verbose=0)
    print "test> size of code table: %d" % len(codetb)
    print "test> num. codes in head-to-children: %d" % len(codeSet(codetb)) 

    print "test> testing filters ..."
    labtest = 'blood.*culture'; medcodes = (40094, )
    codes = filterByName(labtest, load_=False)
    codes2 = filterByCode(medcodes)
    print " + size of %s-code: %d, %d" % (labtest, len(codes), len(codes2))
    print " + dtype: %s" % type(list(codes)[random.randint(0, len(codes)-1)])

    return

def test2(): 
    # kargs = {'to_int': False} 
    # codes = getDescendent(code=1000, self_=True, **kargs)
    # print codes, type(codes)
    labtest = 'antibiotic'
    desctb = getDescendentHierarchy(name=labtest, load_=False)
    print "%s has %d leading entries" % (labtest, len(desctb))
    heads, members = codePair(desctb)
    h = set(heads)
    h.update(members)
    print "%s has a total of %d codes" % (labtest, len(h))

    nametb = nameDescendentHierarchy(name='microbiology')
    print "%d entries" % len(nametb)

def test3(): 
    from tree import children
    codes = [60420,60456,60511,60513,60520,]
    desctb, anctb = groupFromWithin(codes)  # partial adjacency list
    for k, v in desctb.items(): 
        print "base: %s => desc: %s" % (k, str(v)) # 

    for code in codes: 
        codewalk = children(code, desctb, mode='dfs') 
        print "base: %s => walk: %s" % (code, codewalk)  

    # for k, v in anctb.items(): 
    #     print "base: %s => anc: %s" % (k, str(v))
def test3(): 
    print "test> testing filters ..."
    # labtest = 'blood'; medcodes = (40094, )
    labtest = 'fever'; medcodes = []
    codes = filterByName(labtest, load_=False)
    codes2 = filterByCode(medcodes)
    print " + size of %s-code: %d, %d" % (labtest, len(codes), len(codes2))
    print " + dtype: %s" % type(list(codes)[random.randint(0, len(codes)-1)])

def test_filter(): 
    medcodes = [315, 2235, 41901, 1287]
    desc_codes = filterByCode(medcodes, to_int=False)
    print("test> size of codes: %s | type: %s | element type: %s" % (len(desc_codes), type(desc_codes), type(list(desc_codes)[0])))

    return

def t_commands(): 
    code = '316'
    text = getName(code)
    print('demo> code: %s, description: %s' % (code, text))

    return

def t_stats(): 
    model = {'microbiology': [315, 2235, 41901, 1287, ], 
             'blood': [41999, 32099, ],
             'antibiotic': [6527, 23945, 1181, ],
             'urine': [32103, 2648, ], 
             'specimen': [49, ],  
            }  # [const]

    for m, medcodes in model.items(): 
        print "model: %s" % m
        codes = filterByCode(medcodes, to_int=True)
        print('> number of targets: %d =?= %d' % (len(codes), len(set(codes))))
    return 

def t_filter(): 
    models = {'microbio': [315, 2235, 41901, 1287, ], 
             'blood': [41999, 32099, ],
             'antibio': [6527, 23945, 1181, ],
             'urine': [32103, 2648, ], 
             'specimen': [49, ],  
            }  # [const]    
    model = 'antibio'    # blood: 1027, microbio: 570, urine: 160, antibio:3
    codes = filterByCode(models[model], to_int=False)
    print('info> codes: %s' % list(codes)[:5])
    dirs = os.listdir('data/cdr/lab/cerner')
    mdirs = set(dirs).intersection([str(c) for c in codes])
    n_total = len(dirs)
    n_mdir = len(mdirs)
    print('info> n_dirs: %d | model: %s => n_dirs: %d' % (n_total, model, n_mdir))
    return 

def t_query(): 
    medcode = '60005'
    output = getName2(medcode, err_default='')
    print('t_query> code: %s => %s' % (medcode, output))

    medcode = '600057'
    output = getName2(medcode, err_default='')
    print('t_query> code: %s => %s' % (medcode, output))

    return

def t_map(): 
    
    # map related MED codes to the same category (various lab tests measuring the same substance)
    slot, code = 16, '35789'
    # slot = 18
    cmd = 'qrymed -val %s %s' % (slot, code)  # e.g. 35789     slot 16: entity measured

    output = eval(cmd, err_default=None)
    print(output)

    codes = getAncestor(code, to_int=True)
    print('> codes[0]: %s, dtype: %s' % (codes[-1], type(codes[-1])))

    codes = getDescendent(code) # '70745'
    if codes is not None: 
        print('> n_codes: %d' % len(codes))
    else: 
        print('> n_codes: 0')

    # find all codes with slot 7 (parts-of)
    slot = 7 
    # slot = 18
    codes = getDescendent(1) # '70745'
    has_partsofx = []
    for code in codes: 
        cmd = 'qrymed -val %s %s' % (slot, code)  # e.g. 35789     slot 16: entity measured
        output = eval(cmd, err_default=None, verbose=False)
        # [log] example output: 1186\n1197\n1220\n1227\n1232\n1237
        if output is not None: 
            print("> %s has slot 7: %s" % (code, output))
            pcx = output.split('\n')
            has_partsofx.append(pcx)
        if len(has_partsofx) >= 10: 
            break
    print('> has slot 7 (parts-of):\n%s\n' % has_partsofx)

    return

def t_lookup_diagnosis(**kargs): 
    import icd9utils

    alist = ['296.30', 'F33.9', '311', '296.20', ]

    header = ['code', 'med', 'description']
    adict = {h: [] for h in header}
    for code in alist: 
        medcode = ICD9ToMed(code)
        print('  + %s -> med: %s' % (code, medcode))

        if medcode is None: 
            name = icd9utils.getName(code)
        else: 
            name = getName2(medcode)
        print('  + %s -> desc: %s' % ('?' if medcode in (None, '') else medcode, name))

        adict['code'].append(code)
        adict['med'].append(medcode)
        adict['description'].append(name)

    print adict

    medcodes = ['2388', '5589', '7890', 2822, '6819']
    for code in medcodes: 
        c2 = medToICD(code)
        print " + %s => %s (len=%s)" % (code, c2, len(c2))

    return 

def test(): 
    # test3()
    # test1()
    # test_filter()
    # print("> max number of digits: %d" % max_n_digits())

    # demo codes 
    # t_commands()

    # t_stats()

    ### basic query 
    # t_query()
    
    ### filtering
    # t_filter()

    # mapping lab codes 
    # t_map()

    # max number of digits so far? 
    n = max_n_digits()
    print('status> A MED code at most has %d digits' % n)

    # code lookup: ICD-9, 10? => MED => description
    t_lookup_diagnosis()

    return

if __name__ == "__main__": 
    test()