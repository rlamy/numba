from poc import filter2d, numpy

from pypy.translator.interactive import Translation

def ident(x):
    return x

t = Translation(ident)
t.annotate(argtypes=[numpy.ndarray])
t.rtype()
t.source_c()
t.compile_c()
