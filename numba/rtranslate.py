"""
Translate (R)Python code to LLVM IR using RPython toolchain.
"""

def jit(*args, **kws):
    def _jit(func):
        llvm = kws.pop('llvm', True)
        t = RTranslator(func, *args, **kws)
        t.translate()
        return t.get_ctypes_func(llvm)
    return _jit


class RTranslator(object):
    def __init__(self, func, ret_type='d', arg_types=['d'], **kw):
        self.func = func
        self.ret_type = ret_type
        self.arg_types = arg_types
