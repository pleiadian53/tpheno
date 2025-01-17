def _search(forward, source, target, start=0, end=None):
    """Naive search for target in source."""
    m = len(source)
    n = len(target)
    if end is None:
        end = m
    else:
        end = min(end, m)
    if n == 0 or (end-start) < n:
        # target is empty, or longer than source, so obviously can't be found.
        return None
    if forward:
        x = range(start, end-n+1)
    else:
        x = range(end-n, start-1, -1)
    for i in x:
        if source[i:i+n] == target:
            return i
    return None

import functools
search = functools.partial(_search, True)
rsearch = functools.partial(_search, False)


_doc = """\
%(name)s(sequence, subsequence [, start [, end]]) -> int or None

Search a sequence[start:end] for a subsequence starting from the %(dir)s,
returning the offset if it is found, otherwise None.

>>> %(name)s([1, 2, "z", 2, "a", 3, 2, "a"], [2, "a"])
%(value)d

If not given, start and end default to the beginning and end of the sequence.
"""

search.__doc__ = _doc % {'name': 'search', 'value': 3, 'dir': 'left'}
rsearch.__doc__ = _doc % {'name': 'rsearch', 'value': 6, 'dir': 'right'}
search.__name__ = 'search'
rsearch.__name__ = 'rsearch'
del _doc, _search

def test(): 
    q7 = ['J24', '123.5', '374.7',]  # query 
    r7 = ['123.5', '374.7', 'J23', ]   # reference

    q8 = ['720.0', ]
    r8 = ['123.5', 'MED:12345', '720.0', '491.0', '720.0', '720.0', '111.1']

    for (q, r) in [(q7, r7),  (q8, r8),]: 
        x = search(r, q)
        print("query %s in %s => %s" % (q, r, x))

    return

if __name__ == "__main__": 
    test()

