# matching

cdef extern from "cpp/matching/Matching.h":
	cdef cppclass _Matching "NetworKit::Matching":
		_Matching() except +
		_Matching(count) except +
		void match(node, node) except +
		void unmatch(node, node) except +
		bool isMatched(node) except +
		bool areMatched(node, node) except +
		bool isProper(_Graph) except +
		count size(_Graph) except +
		index mate(node) except +
		edgeweight weight(_Graph) except +
		_Partition toPartition(_Graph) except +
		vector[node] getVector() except +

cdef class Matching:
	""" Implements a graph matching.

 		Matching(z=0)

 		Create a new matching data structure for `z` elements.

		Parameters
		----------
		z : index, optional
			Maximum number of nodes.
	"""
	cdef _Matching _this

	def __cinit__(self, index z=0):
		self._this = move(_Matching(z))

	cdef setThis(self,  _Matching& other):
		swap[_Matching](self._this,  other)
		return self

	def match(self, node u, node v):
		self._this.match(u,v)

	def unmatch(self, node u,  node v):
		self._this.unmatch(u, v)

	def isMatched(self, node u):
		return self._this.isMatched(u)

	def areMatched(self, node u, node v):
		return self._this.areMatched(u,v)

	def isProper(self, Graph G):
		return self._this.isProper(G._this)

	def size(self, Graph G):
		return self._this.size(G._this)

	def mate(self, node v):
		return self._this.mate(v)

	def weight(self, Graph G):
		return self._this.weight(G._this)

	def toPartition(self, Graph G):
		return Partition().setThis(self._this.toPartition(G._this))

	def getVector(self):
		""" Get the vector storing the data

		Returns
		-------
		vector
			Vector indexed by node id to node id of mate or none if unmatched
		"""
		return self._this.getVector()

cdef extern from "cpp/matching/Matcher.h":
	cdef cppclass _Matcher "NetworKit::Matcher"(_Algorithm):
		_Matcher(const _Graph _G) except +
		_Matching getMatching() except +

cdef class Matcher(Algorithm):
	""" Abstract base class for matching algorithms """
	cdef Graph G

	def __init__(self, *args, **namedargs):
		if type(self) == Matcher:
			raise RuntimeError("Instantiation of abstract base class")

	def __dealloc__(self):
		self.G = None # just to be sure the graph is deleted

	def getMatching(self):
		"""  Returns the matching.

		Returns
		-------
		Matching
		"""
		if self._this == NULL:
			raise RuntimeError("Error, object not properly initialized")
		return Matching().setThis((<_Matcher*>(self._this)).getMatching())


cdef extern from "cpp/matching/PathGrowingMatcher.h":
	cdef cppclass _PathGrowingMatcher "NetworKit::PathGrowingMatcher"(_Matcher):
		_PathGrowingMatcher(_Graph) except +
		_PathGrowingMatcher(_Graph, vector[double]) except +

cdef class PathGrowingMatcher(Matcher):
	"""
	Path growing matching algorithm as described by  Hougardy and Drake.
	Computes an approximate maximum weight matching with guarantee 1/2.
	"""
	def __cinit__(self, Graph G not None, edgeScores=None):
		self.G = G
		if edgeScores:
			self._this = new _PathGrowingMatcher(G._this, edgeScores)
		else:
			self._this = new _PathGrowingMatcher(G._this)
