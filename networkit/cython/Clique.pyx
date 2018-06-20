'''
	Module: clique
'''

cdef extern from "cpp/clique/MaxClique.h":
	cdef cppclass _MaxClique "NetworKit::MaxClique":
		_MaxClique(_Graph G, count lb) except +
		void run() nogil except +
		count getMaxCliqueSize() except +

cdef class MaxClique:
	"""
	DEPRECATED: Use clique.MaximumCliques instead.

	Exact algorithm for computing the size of the largest clique in a graph.
	Worst-case running time is exponential, but in practice the algorithm is fairly fast.
	Reference: Pattabiraman et al., http://arxiv.org/pdf/1411.7460.pdf

	Parameters:
	-----------
	G : graph in which the cut is to be produced, must be unweighted.
	lb : the lower bound of the size of the maximum clique.
	"""
	cdef _MaxClique* _this
	cdef Graph _G

	def __cinit__(self, Graph G not None, lb=0):
		self._G = G
		self._this = new _MaxClique(G._this, lb)


	def __dealloc__(self):
		del self._this

	def run(self):
		"""
		Actual maximum clique algorithm. Determines largest clique each vertex
	 	is contained in and returns size of largest. Pruning steps keep running time
	 	acceptable in practice.
	 	"""
		cdef count size
		with nogil:
			self._this.run()

	def getMaxCliqueSize(self):
		"""
		Returns the size of the biggest clique
		"""
		return self._this.getMaxCliqueSize()

cdef extern from "cpp/clique/MaximalCliques.h":
	cdef cppclass _MaximalCliques "NetworKit::MaximalCliques"(_Algorithm):
		_MaximalCliques(_Graph G, bool maximumOnly) except +
		_MaximalCliques(_Graph G, NodeVectorCallbackWrapper callback) except +
		vector[vector[node]] getCliques() except +

cdef class MaximalCliques(Algorithm):
	"""
	Algorithm for listing all maximal cliques.

	The implementation is based on the "hybrid" algorithm described in

	Eppstein, D., & Strash, D. (2011).
	Listing All Maximal Cliques in Large Sparse Real-World Graphs.
	In P. M. Pardalos & S. Rebennack (Eds.),
	Experimental Algorithms (pp. 364–375). Springer Berlin Heidelberg.
	Retrieved from http://link.springer.com/chapter/10.1007/978-3-642-20662-7_31

	The running time of this algorithm should be in O(d^2 * n * 3^{d/3})
	where f is the degeneracy of the graph, i.e., the maximum core number.
	The running time in practive depends on the structure of the graph. In
	particular for complex networks it is usually quite fast, even graphs with
	millions of edges can usually be processed in less than a minute.

	Parameters
	----------
	G : Graph
		The graph to list the cliques for
	maximumOnly : bool
		A value of True denotes that only one maximum clique is desired. This enables
		further optimizations of the algorithm to skip smaller cliques more
		efficiently. This parameter is only considered when no callback is given.
	callback : callable
		If a callable Python object is given, it will be called once for each
		maximal clique. Then no cliques will be stored. The callback must accept
		one parameter which is a list of nodes.
	"""
	cdef NodeVectorCallbackWrapper* _callback;
	cdef Graph _G
	cdef object _py_callback

	def __cinit__(self, Graph G not None, bool maximumOnly = False, object callback = None):
		self._G = G

		if callable(callback):
			# Make sure the callback is not de-allocated!
			self._py_callback = callback
			self._callback = new NodeVectorCallbackWrapper(callback)
			try:
				self._this = new _MaximalCliques(self._G._this, dereference(self._callback))
			except BaseException as e:
				del self._callback
				self._callback = NULL
				raise e
		else:
			self._callback = NULL
			self._this = new _MaximalCliques(self._G._this, maximumOnly);

	def __dealloc__(self):
		if not self._callback == NULL:
			del self._callback
			self._callback = NULL

	def getCliques(self):
		"""
		Return all found cliques unless a callback was given.

		This method will throw if a callback was given and thus the cliques were not stored.
		If only the maximum clique was stored, it will return exactly one clique unless the graph
		is empty.

		Returns
		-------
		A list of cliques, each being represented as a list of nodes.
		"""
		return (<_MaximalCliques*>(self._this)).getCliques()
