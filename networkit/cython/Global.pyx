# NetworKit typedefs
ctypedef uint64_t count
ctypedef uint64_t index
ctypedef uint64_t edgeid
ctypedef index node
ctypedef index cluster
ctypedef double edgeweight

cdef extern from "cpp/Globals.h" namespace "NetworKit":
	index _none "NetworKit::none"

none = _none

cdef extern from "<algorithm>" namespace "std":
	void swap[T](T &a,  T &b)
	_Graph move( _Graph t ) nogil # specialized declaration as general declaration disables template argument deduction and doesn't work
	_Partition move( _Partition t) nogil
	_Cover move(_Cover t) nogil
	_Matching move(_Matching) nogil
	vector[double] move(vector[double])
	vector[bool] move(vector[bool])
	vector[count] move(vector[count])
	pair[_Graph, vector[node]] move(pair[_Graph, vector[node]]) nogil
	vector[pair[pair[node, node], double]] move(vector[pair[pair[node, node], double]]) nogil
	vector[pair[node, node]] move(vector[pair[node, node]]) nogil

cdef extern from "cpp/viz/Point.h" namespace "NetworKit":
	cdef cppclass Point[T]:
		Point()
		Point(T x, T y)
		T& operator[](const index i) except +
		T& at(const index i) except +

cdef extern from "cpp/base/Algorithm.h":
	cdef cppclass _Algorithm "NetworKit::Algorithm":
		_Algorithm()
		void run() nogil except +
		bool hasFinished() except +
		string toString() except +
		bool isParallel() except +

# Cython helper functions

def stdstring(pystring):
	""" convert a Python string to a bytes object which is automatically coerced to std::string"""
	pybytes = pystring.encode("utf-8")
	return pybytes

def pystring(stdstring):
	""" convert a std::string (= python byte string) to a normal Python string"""
	return stdstring.decode("utf-8")

cdef extern from "cython/cython_helper.h":
	void throw_runtime_error(string message)

cdef class Algorithm:
	""" Abstract base class for algorithms """
	cdef _Algorithm *_this

	def __init__(self, *args, **namedargs):
		if type(self) == Algorithm:
			raise RuntimeError("Error, you may not use Algorithm directly, use a sub-class instead")

	def __cinit__(self, *args, **namedargs):
		self._this = NULL

	def __dealloc__(self):
		if self._this != NULL:
			del self._this
		self._this = NULL

	def run(self):
		"""
		Executes the algorithm.

		Returns
		-------
		Algorithm:
			self
		"""
		if self._this == NULL:
			raise RuntimeError("Error, object not properly initialized")
		with nogil:
			self._this.run()
		return self

	def hasFinished(self):
		"""
		States whether an algorithm has already run.

		Returns
		-------
		Algorithm:
			self
		"""
		if self._this == NULL:
			raise RuntimeError("Error, object not properly initialized")
		return self._this.hasFinished()

	def toString(self):
		""" Get string representation.

		Returns
		-------
		string
			String representation of algorithm and parameters.
		"""
		if self._this == NULL:
			raise RuntimeError("Error, object not properly initialized")
		return self._this.toString().decode("utf-8")


	def isParallel(self):
		"""
		Returns
		-------
		bool
			True if algorithm can run multi-threaded
		"""
		if self._this == NULL:
			raise RuntimeError("Error, object not properly initialized")
		return self._this.isParallel()
