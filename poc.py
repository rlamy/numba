import numpy
import numpy as np
from pypy.translator.tool.cbuild import ExternalCompilationInfo
from pypy.rpython.extregistry import ExtRegistryEntry
from pypy.rpython.extfunc import ExtFuncEntry
from pypy.rpython.rmodel import Repr
from pypy.rpython.robject import PyObjRepr
from pypy.rpython.lltypesystem.lltype import (Float, Ptr, Struct, PyObject,
    Unsigned, malloc, Void, typeOf)
from pypy.rpython.lltypesystem.rffi import CArray, llexternal
from pypy.rpython.lltypesystem.rtuple import TupleRepr
from pypy.annotation.model import SomeObject, SomeTuple, SomeInteger, SomeFloat
from pypy.tool.pairtype import pairtype

# Gross hack!
from pypy.rlib.rarithmetic import build_int
from pypy.rpython.lltypesystem.lltype import build_number
npy_intp = build_int('npy_intp', True, 32, True)
NPY_INTP = build_number('NPY_INTP', npy_intp)
from pypy.translator.c.primitive import define_c_primitive
define_c_primitive(NPY_INTP, 'npy_intp')
from pypy.rpython.rint import getintegerrepr
getintegerrepr(NPY_INTP, 'int_')

class SomeArray2d(SomeObject):
    knowntype = numpy.ndarray
    immutable = False
    def __init__(self, shape=None):
        if shape is None:
            shape = SomeTuple([SomeInteger(nonneg=True), SomeInteger(nonneg=True)])
        self.shape = shape

    def find_method(self, name):
        if name == 'shape':
            return self.shape
        else:
            return super(SomeArray2d, self).find_method(name)

    def rtyper_makerepr(self, rtyper):
        rows, cols = [rtyper.getrepr(n) for n in self.shape.items]
        return Array2dRepr(rows, cols)

class __extend__(pairtype(SomeArray2d, SomeTuple)):
    def getitem((arr, tup)):
        return SomeFloat()

    def setitem((arr, tup), value):
        pass

ARRAY2D = Struct('ndarray', ('rows', NPY_INTP), ('cols', NPY_INTP),
    ('arr', Ptr(CArray(Float))))

class Array2dRepr(Repr):
    lowleveltype = Ptr(ARRAY2D)
    def __init__(self, rows, cols):
        self.rows = rows
        self.cols = cols

    def find_method(self, name):
        if name == 'shape':
            return TupleRepr((self.rows, self.cols))
        else:
            return super(Array2dRepr, self).find_method(name)

    def rtype_getattr(self, hop):
        r_array = hop.inputarg(self, 0)
        r_tuple = hop.r_result
        TUPLE = hop.inputconst(Void, r_tuple.lowleveltype)
        return hop.gendirectcall(ll_get_shape, r_array, TUPLE)

def ll_get_shape(arr, TUPLE):
    tup = malloc(TUPLE.TO)
    tup.item0 = int(arr.rows)
    tup.item1 = int(arr.cols)
    return tup

import os
numpy_include = os.path.join(np.get_include(), 'numpy')
eci = ExternalCompilationInfo(include_dirs=[numpy_include], includes = ['arrayobject.h'])
get_shape = llexternal("PyArray_DIMS", [Ptr(PyObject)], Ptr(CArray(NPY_INTP)), compilation_info=eci)
get_data = llexternal("PyArray_DATA", [Ptr(PyObject)], Ptr(CArray(Float)), compilation_info=eci)
new_array = llexternal("PyArray_SimpleNewFromData",
        [NPY_INTP, Ptr(CArray(NPY_INTP)), NPY_INTP, Ptr(CArray(Float))],
        Ptr(PyObject), compilation_info=eci)

class __extend__(pairtype(Array2dRepr, TupleRepr)):
    def rtype_setitem((arr, tup), hop):
        v_arr, v_tup, v_value = hop.inputargs(arr, tup, Float)
        return hop.gendirectcall(ll_array_setitem, v_arr, v_tup, v_value)

    def rtype_getitem((arr, tup), hop):
        v_arr, v_tup = hop.inputargs(arr, tup)
        return hop.gendirectcall(ll_array_getitem, v_arr, v_tup)

class __extend__(pairtype(Array2dRepr, PyObjRepr)):
    def convert_from_to((arr, pyobj), v, llops):
        DIMS = llops.genvoidconst(CArray(NPY_INTP))
        return llops.gendirectcall(ll_to_array, v, DIMS)

class __extend__(pairtype(PyObjRepr, Array2dRepr)):
    def convert_from_to((pyobj, arr), v, llops):
        return llops.gendirectcall(ll_from_array, v)

def ll_from_array(pyarr):
    dims = get_shape(pyarr)
    c_arr = malloc(ARRAY2D, flavor='raw')
    c_arr.rows = dims[0]
    c_arr.cols = dims[1]
    c_arr.arr = get_data(pyarr)
    return c_arr

def ll_to_array(c_arr, DIMS):
    dims = malloc(DIMS, 2, flavor='raw')
    dims[0] = npy_intp(c_arr.rows)
    dims[1] = npy_intp(c_arr.cols)
    return new_array(npy_intp(2), dims, npy_intp(5), c_arr.arr)

def ll_array_setitem(ndarr, tup, value):
    i, j = tup.item0, tup.item1
    ndarr.arr[i * ndarr.rows + j] = value
def ll_array_getitem(ndarr, tup):
    i, j = tup.item0, tup.item1
    return ndarr.arr[i * ndarr.rows + j]

class Array2d_Entry(ExtRegistryEntry):
    _type_ = numpy.ndarray
    def compute_annotation(self):
        return SomeArray2d()
    compute_result_annotation = compute_annotation

class ZerosLikeEntry(ExtFuncEntry):
    _about_ = numpy.zeros_like
    signature_args = [numpy.ndarray]
    name = 'zeros_like'
    def normalize_args(self, *args):
        # prevent the arg from turning into SomeObject
        return args

    def compute_result_annotation(self, in_arr):
        return SomeArray2d(shape = in_arr.shape)

    def specialize_call(self, hop):
        hop.exception_cannot_occur()
        return hop.gendirectcall(self.lltypeimpl, *hop.inputargs(*hop.args_r))

    @staticmethod
    def lltypeimpl(in_arr):
        size = in_arr.rows * in_arr.cols
        out_arr = malloc(typeOf(in_arr.arr).TO, size, flavor='raw')
        for i in range(size):
            out_arr[i] = 0.
        nd_arr = malloc(ARRAY2D, flavor='raw')
        nd_arr.rows = in_arr.rows
        nd_arr.cols = in_arr.cols
        nd_arr.arr = out_arr
        return nd_arr

class array2d(object):
    shape = (0, 0)
    _array = []
    def __init__(self, m, n):
        self.shape = (m, n)
        self._array = [[None] * n for _ in range(m)]

    def __getitem__(self, (m, n)):
        return self._array[m][n]

    def __setitem__(self, (m, n), value):
        self._array[m][n] = value

def filter2d(image, filt):
    M, N = image.shape
    Mf, Nf = filt.shape
    Mf2 = Mf // 2
    Nf2 = Nf // 2
    result = numpy.zeros_like(image)
    for i in range(Mf2, M - Mf2):
        for j in range(Nf2, N - Nf2):
            num = 0.0
            for ii in range(Mf):
                for jj in range(Nf):
                    num += (filt[Mf-1-ii, Nf-1-jj] * image[i-Mf2+ii, j-Nf2+jj])
            result[i, j] = num
    return result
