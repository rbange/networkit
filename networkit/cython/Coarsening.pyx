# Module: coarsening

cdef extern from "cpp/coarsening/GraphCoarsening.h":
	cdef cppclass _GraphCoarsening "NetworKit::GraphCoarsening"(_Algorithm):
		_GraphCoarsening(_Graph) except +
		_Graph getCoarseGraph() except +
		vector[node] getFineToCoarseNodeMapping() except +
		map[node, vector[node]] getCoarseToFineNodeMapping() except +

cdef class GraphCoarsening(Algorithm):
	cdef Graph _G

	def __init__(self, *args, **namedargs):
		if type(self) == GraphCoarsening:
			raise RuntimeError("Error, you may not use GraphCoarsening directly, use a sub-class instead")

	def __dealloc__(self):
		self._G = None # just to be sure the graph is deleted

	def run(self):
		"""
		Executes the Graph coarsening algorithm.

		Returns
		-------
		GraphCoarsening:
			self
		"""
		if self._this == NULL:
			raise RuntimeError("Error, object not properly initialized")
		with nogil:
			self._this.run()
		return self

	def getCoarseGraph(self):
		return Graph(0).setThis((<_GraphCoarsening*>(self._this)).getCoarseGraph())

	def getFineToCoarseNodeMapping(self):
		return (<_GraphCoarsening*>(self._this)).getFineToCoarseNodeMapping()

	def getCoarseToFineNodeMapping(self):
		return (<_GraphCoarsening*>(self._this)).getCoarseToFineNodeMapping()


cdef extern from "cpp/coarsening/ParallelPartitionCoarsening.h":
	cdef cppclass _ParallelPartitionCoarsening "NetworKit::ParallelPartitionCoarsening"(_GraphCoarsening):
		_ParallelPartitionCoarsening(_Graph, _Partition, bool) except +


cdef class ParallelPartitionCoarsening(GraphCoarsening):
	def __cinit__(self, Graph G not None, Partition zeta not None, useGraphBuilder = True):
		self._this = new _ParallelPartitionCoarsening(G._this, zeta._this, useGraphBuilder)

cdef extern from "cpp/coarsening/MatchingCoarsening.h":
	cdef cppclass _MatchingCoarsening "NetworKit::MatchingCoarsening"(_GraphCoarsening):
		_MatchingCoarsening(_Graph, _Matching, bool) except +


cdef class MatchingCoarsening(GraphCoarsening):
	"""Coarsens graph according to a matching.
 	Parameters
 	----------
 	G : Graph
	M : Matching
 	noSelfLoops : bool, optional
		if true, self-loops are not produced
	"""

	def __cinit__(self, Graph G not None, Matching M not None, bool noSelfLoops=False):
		self._this = new _MatchingCoarsening(G._this, M._this, noSelfLoops)
