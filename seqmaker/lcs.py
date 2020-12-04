# encoding: utf-8

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