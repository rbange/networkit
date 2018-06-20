
'''
	cython: language_level=3
	Header for all imports
'''

#includes
# needed for collections.Iterable
import collections
import math
import os
import tempfile


try:
	import pandas
except:
	print(""" WARNING: module 'pandas' not found, some functionality will be restricted """)


# C++ operators
from cython.operator import dereference, preincrement

# type imports
from libc.stdint cimport uint64_t
from libc.stdint cimport int64_t

# the C++ standard library
from libcpp cimport bool
from libcpp.vector cimport vector
from libcpp.utility cimport pair
from libcpp.map cimport map
from libcpp.set cimport set
from libcpp.stack cimport stack
from libcpp.string cimport string
from libcpp.unordered_set cimport unordered_set
from libcpp.unordered_map cimport unordered_map
from libcpp.algorithm cimport sort as stdsort
