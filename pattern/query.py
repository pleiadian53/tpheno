# encoding: utf-8
import os, sys, re, random, collections

# install tsne via pip
# from tsne import bh_sne

import pandas as pd 
from pandas import DataFrame

try:
    import cPickle as pickle
except:
    import pickle

##########################################################################################################################
#
#  Query API for the descriptions of medical codes (e.g. RxNorm)
#
#
#
def segment(code, rs):
    cr = str(code)
    assert len(cr) == sum(rs)
    s1, s2, s3 = rs # e.g. lengths 4,4,3
    return [cr[:s1], cr[s1:s1+s2], cr[s1+s2:]]

def normalize(code, sep_prefix=':', sep_code='-'): 
    # remove prefix 
    cr = str(code)
    if sep_prefix in cr: 
        prefix, code = cr.split(sep_prefix)  
    else: 
        pass   # noop
    
    # [test]
    assert not sep_code in code
    return code

def isCase(seq, code_set=[], code_prefix='', logic='any', threshold=1):
    """
    Given a sequencce of codes/tokens (i.e. an MCS), determine if it is a case 
    based on its matching with 
    
    1. a set of codes, specified by code_set or 
    2. prefix of code such as the first 3 digits of an ICD-9 

    """
    tVal = False
    positions = []  # matched positions
    if code_prefix: # threshold only applies to this use case
        n_hits = 0
        for i, tok in enumerate(seq): 
            if tok.startswith(code_prefix):
                n_hits += 1 
                # positions.append(i)
            if n_hits >= threshold: 
                tVal = True; break 
    elif len(code_set) > 0:  # code_set is a set of exact diagnostic codes (not just prefixes)
        nc0 = len(code_set)
        nc = len(set(code_set)-set(seq)) 
        if logic == 'any': 
            if nc < nc0: 
                tVal = True 
        else: # all have to be matched 
            if nc == 0: 
                tVal = True
    else: 
        raise ValueError, "Either code_set or code_prefix has to be provided."

    return tVal

def parseNDC(code, n_digits=10, sep='-'):
    """

    Memo
    ----
    (*) National Drug Code (NDC): coding format 
        10 digits

           4-4-2 
           5-3-2
           5-4-1
    """

    # return all possible interpretations 
    scode = str(code)
    assert len(code) == n_digits
    
    configs = [[4, 4, 2], [5, 3, 2], [5, 4, 1]]
    candidates = []
    # 4-4-2, 5-3-2, 5-4-1
    for cfg in configs: 
        candidates.append(sep.join(segment(code, cfg)))

    return candidates


def t_lookup_restful(**kargs): # look up codes via RESTful API 
    """

    Reference 
    ---------
    1. HIPPASpace: https://www.hipaaspace.com/Medical_Web_Services/Test.Drive.RESTful.Web.Services?Type=NDC#rt
    2. openFDA 

    Memo
    ----
    (*) HIPAASpace 

        https://www.HIPAASpace.com/api/{domain}/{operation}?q={query}&rt={result type}&token={token}
        
        Three query parameters are required with each “search” request:

        Use the domain parameter to specify required data domain.
        Use the operation parameter to specify “format_check” operation.
        Use the q (query) parameter to specify your query.
        Use the token (API key) query parameter to identify your application.

        Use the rt (result type) query parameter to specify required result type (json/xml/min.json/min.xml).

        Examples 

        https://www.hipaaspace.com/api/npi/getcode?q=1285636522&rt=xml&token=3932f3b0-cfab-11dc-95ff-0800200c9a663932f3b0-cfab-11dc-95ff-0800200c9a66

    (*) Coding format 
        National Drug Code, NDC  
           4-4-2 
           5-3-2
           5-4-1

    """
    import requests, json
    from urllib2 import Request, urlopen, URLError

    operations = {'getcode', 'getcodes', 'search', 'search_and_keywords', }
    rts = {'json', 'minjson' 'xml', 'minxml', }
    
    my_token = '2DECE6D8DEFE4158AAF4F936A3CEA5557DBD99D6EE3849D589745897EA74841B'
    demo_token = '3932f3b0-cfab-11dc-95ff-0800200c9a663932f3b0-cfab-11dc-95ff-0800200c9a66'

    params = {'domain': 'ndc', 'operation': 'getcode', 'query': '0093-0832-01', 'rt': 'json', 
               'token': my_token}
    

    # e.g. https://www.hipaaspace.com/api/npi/getcode?q=1285636522&rt=xml&token=3932f3b0-cfab-11dc-95ff-0800200c9a663932f3b0-cfab-11dc-95ff-0800200c9a66
    #      https://www.HIPAASpace.com/api/npi/getcode?q=1285636522&rt=xml&token=2DECE6D8DEFE4158AAF4F936A3CEA5557DBD99D6EE3849D589745897EA74841B
    uri = 'https://www.hipaaspace.com/api/{domain}/{operation}?q={query}&rt={rt}&token={token}'
    rquery = uri.format(**params)
    print("info> URI: %s" % rquery) 

    # [todo] this doesn't work, returns an empty content
    # div(message='\nTry requests.get() ...')
    # resp = requests.get(rquery)
    # if resp.status_code != 200:
    #     # This means something went wrong.
    #     raise ApiError('GET /tasks/ {}'.format(resp.status_code))
    # print resp

    div(message='\nTry using urlopen ...')
    request = Request(rquery)
    try:
        response = urlopen(request)
        ret = response.read()
        print ret
    except URLError, e:
        print 'Something went wrong: %s', e
 
    doc = json.loads(ret)
    print('info> dtype: %s' % type(doc))
    print('info> doc:\n%s\n' % doc['NDC'][0]['ProprietaryName'])
    # print('info> %s > %s' (doc['NDC'], doc['NDC'][0]['ProprietaryName']))
    # for todo_item in resp.json():
    #     print('{} {}'.format(todo_item['id'], todo_item['summary']))

    return
def t_test_restful(): 
    import requests

    # Set up the parameters we want to pass to the API.
    # This is the latitude and longitude of New York City.
    parameters = {"lat": 40.71, "lon": -74}

    # Make a get request with the parameters.
    response = requests.get("http://api.open-notify.org/iss-pass.json", params=parameters)

    # Print the content of the response (the data the server returned)
    print(response.content)

    # This gets the same data as the command above
    response = requests.get("http://api.open-notify.org/iss-pass.json?lat=40.71&lon=-74")
    print(response.content)

    return

def test(**kargs): 
    
    ### Utilities
    code = '0093083201'
    candidates = parseNDC(normalize(code))
    print('info> possilbe configurations:\n%s\n' % candidates)
    ### Lookup drug codes (e.g. NDC) via RESTful API
    # t_lookup_restful(**kargs)
    

    return

if __name__ == "__main__": 
    test()