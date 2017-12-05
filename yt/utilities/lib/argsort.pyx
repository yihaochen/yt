# distutils: language = c
# cython: cdivision = True
# cython: boundscheck = False
# cython: wraparound = False
# cython: profile = False

cimport numpy as np
import numpy as np
from libc.stdlib cimport malloc, free
from cython cimport view

cdef extern from "stdlib.h":
    ctypedef void const_void "const void"
    void qsort(void *base, int nmemb, int size,
            int(*compar)(const_void *, const_void *)) nogil

cdef struct Sorter:
    np.int64_t index
    np.float64_t value

cdef int _compare(const_void *a, const_void *b):
    cdef np.float64_t v = ((<Sorter*>a)).value-((<Sorter*>b)).value
    if v <= 0: return -1
    if v > 0: return 1

cdef void cyargsort(np.float64_t[:] data, Sorter * order):
    cdef np.int64_t i
    cdef np.int64_t n = data.shape[0]
    for i in range(n):
        order[i].index = i
        order[i].value = data[i]
    qsort(<void *> order, n, sizeof(Sorter), _compare)

cpdef argsort(np.float64_t[:] data, np.int32_t[:] order):
    cdef np.int32_t i
    cdef np.int32_t n = data.shape[0]
    cdef Sorter *order_struct = <Sorter *> malloc(n * sizeof(Sorter))
    cyargsort(data, order_struct)
    for i in range(n):
        order[i] = order_struct[i].index
    free(order_struct)
