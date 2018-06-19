# Module: centrality

cdef extern from "cpp/centrality/Centrality.h":
	cdef cppclass _Centrality "NetworKit::Centrality"(_Algorithm):
		_Centrality(_Graph, bool, bool) except +
		vector[double] scores() except +
		vector[pair[node, double]] ranking() except +
		double score(node) except +
		double maximum() except +
		double centralization() except +


cdef class Centrality(Algorithm):
	""" Abstract base class for centrality measures"""

	cdef Graph _G

	def __init__(self, *args, **kwargs):
		if type(self) == Centrality:
			raise RuntimeError("Error, you may not use Centrality directly, use a sub-class instead")

	def __dealloc__(self):
		self._G = None # just to be sure the graph is deleted

	def scores(self):
		"""
		Returns
		-------
		list
			the list of all scores
		"""
		if self._this == NULL:
			raise RuntimeError("Error, object not properly initialized")
		return (<_Centrality*>(self._this)).scores()

	def score(self, v):
		"""
		Returns
		-------
		the score of node v
		"""
		if self._this == NULL:
			raise RuntimeError("Error, object not properly initialized")
		return (<_Centrality*>(self._this)).score(v)

	def ranking(self):
		"""
		Returns
		-------
		dictionary
			a vector of pairs sorted into descending order. Each pair contains a node and the corresponding score
		"""
		if self._this == NULL:
			raise RuntimeError("Error, object not properly initialized")
		return (<_Centrality*>(self._this)).ranking()

	def maximum(self):
		"""
		Returns
		-------
		the maximum theoretical centrality score for the given graph
		"""
		if self._this == NULL:
			raise RuntimeError("Error, object not properly initialized")
		return (<_Centrality*>(self._this)).maximum()

	def centralization(self):
		"""
		Compute the centralization of a network with respect to some centrality measure.

	 	The centralization of any network is a measure of how central its most central
	 	node is in relation to how central all the other nodes are.
	 	Centralization measures then (a) calculate the sum in differences
	 	in centrality between the most central node in a network and all other nodes;
	 	and (b) divide this quantity by the theoretically largest such sum of
	 	differences in any network of the same size.
		"""
		if self._this == NULL:
			raise RuntimeError("Error, object not properly initialized")
		return (<_Centrality*>(self._this)).centralization()

cdef extern from "cpp/centrality/TopCloseness.h":
	cdef cppclass _TopCloseness "NetworKit::TopCloseness":
		_TopCloseness(_Graph G, count, bool, bool) except +
		void run() except +
		node maximum() except +
		edgeweight maxSum() except +
		count iterations() except +
		count operations() except +
		vector[node] topkNodesList(bool) except +
		vector[edgeweight] topkScoresList(bool) except +


cdef class TopCloseness:
	"""
	Finds the top k nodes with highest closeness centrality faster than computing it for all nodes, based on "Computing Top-k Closeness Centrality Faster in Unweighted Graphs", Bergamini et al., ALENEX16.
	The algorithms is based on two independent heuristics, described in the referenced paper. We recommend to use first_heu = true and second_heu = false for complex networks and first_heu = true and second_heu = true for street networks or networks with large diameters.

	TopCloseness(G, k=1, first_heu=True, sec_heu=True)

	Parameters
	----------
	G: An unweighted graph.
	k: Number of nodes with highest closeness that have to be found. For example, if k = 10, the top 10 nodes with highest closeness will be computed.
	first_heu: If true, the neighborhood-based lower bound is computed and nodes are sorted according to it. If false, nodes are simply sorted by degree.
	sec_heu: If true, the BFSbound is re-computed at each iteration. If false, BFScut is used.
	The worst case running time of the algorithm is O(nm), where n is the number of nodes and m is the number of edges.
	However, for most networks the empirical running time is O(m).
	"""
	cdef _TopCloseness* _this
	cdef Graph _G

	def __cinit__(self,  Graph G, k=1, first_heu=True, sec_heu=True):
		self._G = G
		self._this = new _TopCloseness(G._this, k, first_heu, sec_heu)

	def __dealloc__(self):
		del self._this

	def run(self):
		""" Computes top-k closeness. """
		self._this.run()
		return self

	def topkNodesList(self, includeTrail=False):
		""" Returns a list with the k nodes with highest closeness.
			WARNING: closeness centrality of some nodes below the top-k could be equal
	  		to the k-th closeness, we call them trail. Set the parameter includeTrail
	  		to true to also include those nodes but consider that the resulting vector
	  		could be longer than k.

		Parameters
		----------
		includeTrail: Whether or not to include trail nodes.

		Returns
		-------
		vector
			The k nodes with highest closeness.
		"""
		return self._this.topkNodesList(includeTrail)

	def topkScoresList(self, includeTrail=False):
		""" Returns a list with the scores of the k nodes with highest closeness.
			WARNING: closeness centrality of some nodes below the top-k could be equal
  			to the k-th closeness, we call them trail. Set the parameter includeTrail
	  		to true to also include those centrality values but consider that the
	  		resulting vector could be longer than k.

		Parameters
		----------
		includeTrail: Whether or not to include trail centrality value.

		Returns
		-------
		vector
			The k highest closeness scores.
		"""
		return self._this.topkScoresList(includeTrail)


cdef extern from "cpp/centrality/TopHarmonicCloseness.h":
	cdef cppclass _TopHarmonicCloseness "NetworKit::TopHarmonicCloseness":
		_TopHarmonicCloseness(_Graph G, count, bool) except +
		void run() except +
		vector[node] topkNodesList(bool) except +
		vector[edgeweight] topkScoresList(bool) except +


cdef class TopHarmonicCloseness:
	""" Finds the top k nodes with highest harmonic closeness centrality faster
            than computing it for all nodes. The implementation is based on "Computing
            Top-k Centrality Faster in Unweighted Graphs", Bergamini et al., ALENEX16.
            The algorithms are based on two heuristics. We reccommend to use
            useBFSbound = false for complex networks (or networks with small diameter)
            and useBFSbound = true for street networks (or networks with large
            diameters). Notice that the worst case running time of the algorithm is
            O(nm), where n is the number of nodes and m is the number of edges.
            However, for most real-world networks the empirical running time is O(m).


	TopCloseness(G, k=1, useBFSbound=True)

	Parameters
	----------
	G: An unweighted graph.
	k: Number of nodes with highest closeness that have to be found. For example, if k = 10, the top 10 nodes with highest closeness will be computed.
	useBFSbound: If true, the BFSbound is re-computed at each iteration. If false, BFScut is used.
	The worst case running time of the algorithm is O(nm), where n is the number of nodes and m is the number of edges.
	However, for most networks the empirical running time is O(m).
	"""
	cdef _TopHarmonicCloseness* _this
	cdef Graph _G

	def __cinit__(self,  Graph G, k=1, useBFSbound=False):
		self._G = G
		self._this = new _TopHarmonicCloseness(G._this, k, useBFSbound)

	def __dealloc__(self):
		del self._this

	def run(self):
		""" Computes top-k harmonic closeness. """
		self._this.run()
		return self

	def topkNodesList(self, includeTrail=False):
		""" Returns a list with the k nodes with highest harmonic closeness.
			WARNING: closeness centrality of some nodes below the top-k could be equal
			to the k-th closeness, we call them trail. Set the parameter includeTrail
			to true to also include those nodes but consider that the resulting vector
			could be longer than k.

		Parameters
		----------
		includeTrail: Whether or not to include trail nodes.

		Returns
		-------
		vector
			The k nodes with highest harmonic closeness.
		"""
		return self._this.topkNodesList(includeTrail)

	def topkScoresList(self, includeTrail=False):
		""" Returns a list with the scores of the k nodes with highest harmonic closeness.
			WARNING: closeness centrality of some nodes below the top-k could
		  	be equal to the k-th closeness, we call them trail. Set the parameter
		  	includeTrail to true to also include those centrality values but consider
		  	that the resulting vector could be longer than k.

		Parameters
		----------
		includeTrail: Whether or not to include trail centrality value.

		Returns
		-------
		vector
			The k highest closeness harmonic scores.
		"""
		return self._this.topkScoresList(includeTrail)


cdef extern from "cpp/centrality/DynTopHarmonicCloseness.h":
	cdef cppclass _DynTopHarmonicCloseness "NetworKit::DynTopHarmonicCloseness":
		_DynTopHarmonicCloseness(_Graph G, count, bool) except +
		void run() except +
		vector[pair[node, edgeweight]] ranking(bool) except +
		vector[node] topkNodesList(bool) except +
		vector[edgeweight] topkScoresList(bool) except +
		void update(_GraphEvent) except +
		void updateBatch(vector[_GraphEvent]) except +

cdef class DynTopHarmonicCloseness:
	""" Finds the top k nodes with highest harmonic closeness centrality faster
        than computing it for all nodes and updates them after a single or multiple
        edge update. The implementation is based on "Computing Top-k Closeness
	    Centrality in Fully-dynamic Graphs", Bisenius et al., ALENEX18.
        The implementation is based on the static algorithms by Borassi et al.
	    (complex networks) and Bergamini et al. (large-diameter networks).

	TopCloseness(G, k=1, useBFSbound=True)

	Parameters
	----------
	G: An unweighted graph.
	k: Number of nodes with highest closeness that have to be found. For example, if k = 10, the top 10 nodes with highest closeness will be computed.
	useBFSbound: If true, the BFSbound is re-computed at each iteration. If false, BFScut is used.
	The worst case running time of the algorithm is O(nm), where n is the number of nodes and m is the number of edges.
	However, for most networks the empirical running time is O(m).
	"""
	cdef _DynTopHarmonicCloseness* _this
	cdef Graph _G

	def __cinit__(self,  Graph G, k=1, useBFSbound=False):
		self._G = G
		self._this = new _DynTopHarmonicCloseness(G._this, k, useBFSbound)

	def __dealloc__(self):
		del self._this

	def run(self):
		""" Computes top-k harmonic closeness. """
		self._this.run()
		return self

	def ranking(self, includeTrail = False):
		""" Returns the ranking of the k most central nodes in the graph.
			WARNING: closeness centrality of some nodes below the top-k could be equal
		  	to the k-th closeness, we call them trail. Set the parameter includeTrail
		  	to true to also include those nodes but consider that the resulting vector
		  	could be longer than k.

		Parameters
		----------
		includeTrail: Whether or not to include trail nodes.

		Returns
		-------
		vector
				The ranking.
		"""
		return self._this.ranking(includeTrail)

	def topkNodesList(self, includeTrail = False):
		""" Returns a list with the k nodes with highest harmonic closeness.
			WARNING: closeness centrality of some nodes below the top-k could be equal
			to the k-th closeness, we call them trail. Set the parameter includeTrail
			to true to also include those nodes but consider that the resulting vector
			could be longer than k.

		Parameters
		----------
		includeTrail: Whether or not to include trail nodes.

		Returns
		-------
		vector
			The k nodes with highest harmonic closeness.
		"""
		return self._this.topkNodesList(includeTrail)

	def topkScoresList(self, includeTrail = False):
		""" Returns a list with the scores of the k nodes with highest harmonic closeness.
			WARNING: closeness centrality of some nodes below the top-k could
		  	be equal to the k-th closeness, we call them trail. Set the parameter
		  	includeTrail to true to also include those centrality values but consider
		  	that the resulting vector could be longer than k.

		Parameters
		----------
		includeTrail: Whether or not to include trail centrality value.

		Returns
		-------
		vector
			The k highest closeness harmonic scores.
		"""
		return self._this.topkScoresList(includeTrail)


	""" Updates the list of the k nodes with the highest harmonic closeness in G.

	Parameters
	----------
	event: A GrapEvent
	"""
	def update(self, ev):
		self._this.update(_GraphEvent(ev.type, ev.u, ev.v, ev.w))

	""" Updates the list of the k nodes with the highest harmonic closeness in G
		after a batch of edge updates.

	Parameters
	----------
	batch: A GraphEvent vector
	"""
	def updateBatch(self, batch):
		cdef vector[_GraphEvent] _batch
		for ev in batch:
			_batch.push_back(_GraphEvent(ev.type, ev.u, ev.v, ev.w))
		self._this.updateBatch(_batch)



cdef extern from "cpp/centrality/GroupDegree.h":
	cdef cppclass _GroupDegree "NetworKit::GroupDegree":
		_GroupDegree(_Graph G, count, bool) except +
		void run() except +
		vector[node] groupMaxDegree() except +
		count getScore() except +


cdef class GroupDegree:
	"""
	Finds the group with the highest group degree centrality according to the
  definition proposed in 'The centrality of groups and classes' by Everett et
  al. (The Journal of mathematical sociology, 1999). This is a submodular but
  non monotone function so the algorithm can find a solution that is at least
  1/2 of the optimum. Worst-case running time is quadratic, but usually
  faster in real-world networks.
	The 'countGroupNodes' option also count the nodes inside the group in the
	score, this make the group degree monotone and submodular and the algorithm
	is guaranteed to return a (1 - 1/e)-approximation of the optimal solution.

	GroupDegree(G, k = 1, countGroupNodes = False)

	Parameters
	----------
    G: A graph.
    k: Size of the group of nodes
		countGroupNodes: if nodes inside the group should be counted in the
    centrality score.
	"""
	cdef _GroupDegree* _this
	cdef Graph _G

	def __cinit__(self, Graph G, k = 1, countGroupNodes = False):
		self._G = G
		self._this = new _GroupDegree(G._this, k, countGroupNodes)

	def __dealloc__(self):
		del self._this

	def run(self):
		"""
		Computes the group with maximum degree centrality of the graph passed in
	    the constructor.
		"""
		self._this.run()
		return self

	def groupMaxDegree(self):
		"""
		Returns the group with maximum degree centrality.
		Returns
		-------
		vector
			The group of k nodes with highest degree centrality.
		"""
		return self._this.groupMaxDegree()

	def getScore(self):
		"""
		Returns the score of the group with maximum degree centrality (i.e. the
	    number of nodes outside the group that can be reached in one hop from at
	    least one node in the group).

		Returns
		-------
		count
			The number of nodes outside the group that can be reached in one hop
			from at least one node in the group.
		"""
		return self._this.getScore()



cdef extern from "cpp/centrality/GroupCloseness.h":
	cdef cppclass _GroupCloseness "NetworKit::GroupCloseness":
		_GroupCloseness(_Graph G, count, count) except +
		void run() except +
		vector[node] groupMaxCloseness() except +
		double computeFarness(vector[node], count) except +


cdef class GroupCloseness:
	"""
	Finds the group of nodes with highest (group) closeness centrality. The algorithm is the one proposed in Bergamini et al., ALENEX 2018 and finds a solution that is a (1-1/e)-approximation of the optimum.
	The worst-case running time of this approach is quadratic, but usually much faster in practice.

	GroupCloseness(G, k=1, H=0)

	Parameters
	----------
	G: An unweighted graph.
	k: Size of the group.
	H: If equal 0, simply runs the algorithm proposed in Bergamini et al.. If > 0, interrupts all BFSs after H iterations (suggested for very large networks).
	"""
	cdef _GroupCloseness* _this
	cdef Graph _G

	def __cinit__(self,  Graph G, k=1, H=0):
		self._G = G
		self._this = new _GroupCloseness(G._this, k, H)

	def __dealloc__(self):
		del self._this

	def run(self):
		""" Computes group with maximum closeness. """
		self._this.run()
		return self

	""" Returns group with highest closeness.
	Returns
	-------
	vector
		The group of k nodes with highest closeness.
	"""
	def groupMaxCloseness(self):
		return self._this.groupMaxCloseness()


	""" Computes farness (i.e., inverse of the closeness) for a given group (stopping after H iterations if H > 0).
	"""
	def computeFarness(self, S, H=0):
		return self._this.computeFarness(S, H)



cdef extern from "cpp/centrality/DegreeCentrality.h":
	cdef cppclass _DegreeCentrality "NetworKit::DegreeCentrality" (_Centrality):
		_DegreeCentrality(_Graph, bool normalized, bool outdeg, bool ignoreSelfLoops) except +

cdef class DegreeCentrality(Centrality):
	""" Node centrality index which ranks nodes by their degree.
 	Optional normalization by maximum degree. The run() method runs in O(m) time, where m is the number of
	edges in the graph.

 	DegreeCentrality(G, normalized=False)

 	Constructs the DegreeCentrality class for the given Graph `G`. If the scores should be normalized,
 	then set `normalized` to True.

 	Parameters
 	----------
 	G : Graph
 		The graph.
 	normalized : bool, optional
 		Normalize centrality values in the interval [0,1].
	"""

	def __cinit__(self, Graph G, bool normalized=False, bool outDeg = True, bool ignoreSelfLoops=True):
		self._G = G
		self._this = new _DegreeCentrality(G._this, normalized, outDeg, ignoreSelfLoops)



cdef extern from "cpp/centrality/Betweenness.h":
	cdef cppclass _Betweenness "NetworKit::Betweenness" (_Centrality):
		_Betweenness(_Graph, bool, bool) except +
		vector[double] edgeScores() except +

cdef class Betweenness(Centrality):
	"""
		Betweenness(G, normalized=False, computeEdgeCentrality=False)

		Constructs the Betweenness class for the given Graph `G`. If the betweenness scores should be normalized,
  	then set `normalized` to True. The run() method takes O(nm) time, where n is the number
	 	of nodes and m is the number of edges of the graph.

	 	Parameters
	 	----------
	 	G : Graph
	 		The graph.
	 	normalized : bool, optional
	 		Set this parameter to True if scores should be normalized in the interval [0,1].
		computeEdgeCentrality: bool, optional
			Set this to true if edge betweenness scores should be computed as well.
	"""

	def __cinit__(self, Graph G, normalized=False, computeEdgeCentrality=False):
		self._G = G
		self._this = new _Betweenness(G._this, normalized, computeEdgeCentrality)


	def edgeScores(self):
		""" Get a vector containing the betweenness score for each edge in the graph.

		Returns
		-------
		vector
			The betweenness scores calculated by run().
		"""
		return (<_Betweenness*>(self._this)).edgeScores()


cdef extern from "cpp/centrality/Closeness.h":
	cdef cppclass _Closeness "NetworKit::Closeness" (_Centrality):
		_Closeness(_Graph, bool, bool) except +

cdef class Closeness(Centrality):
	"""
		Closeness(G, normalized=True, checkConnectedness=True)

		Constructs the Closeness class for the given Graph `G`. If the Closeness scores should not be normalized,
  		set `normalized` to False. The run() method takes O(nm) time, where n is the number
	 	 of nodes and m is the number of edges of the graph. NOTICE: the graph has to be connected.

	 	Parameters
	 	----------
	 	G : Graph
	 		The graph.
	 	normalized : bool, optional
	 		Set this parameter to False if scores should not be normalized into an interval of [0,1]. Normalization only for unweighted graphs.
	 	checkConnectedness : bool, optional
			turn this off if you know the graph is connected
	"""

	def __cinit__(self, Graph G, normalized=True, checkConnectedness=True):
		self._G = G
		self._this = new _Closeness(G._this, normalized, checkConnectedness)


cdef extern from "cpp/centrality/HarmonicCloseness.h":
	cdef cppclass _HarmonicCloseness "NetworKit::HarmonicCloseness" (_Centrality):
		_HarmonicCloseness(_Graph, bool) except +

cdef class HarmonicCloseness(Centrality):
	"""
	        HarmonicCloseness(G, normalized=True)

		Constructs the HarmonicCloseness class for the given Graph `G`.
                If the harmonic closeness scores should not be normalized, set
                `normalized` to False.
                The run() method takes O(nm) time, where n is the number
	 	of nodes and m is the number of edges of the graph.

	 	Parameters
	 	----------
	 	G : Graph
	 		The graph.
	 	normalized : bool, optional
	 		Set this parameter to False if scores should not be
                        normalized into an interval of [0,1].
                        Normalization only for unweighted graphs.
	"""

	def __cinit__(self, Graph G, normalized=True):
		self._G = G
		self._this = new _HarmonicCloseness(G._this, normalized)


cdef extern from "cpp/centrality/KPathCentrality.h":
	cdef cppclass _KPathCentrality "NetworKit::KPathCentrality" (_Centrality):
		_KPathCentrality(_Graph, double, count) except +

cdef class KPathCentrality(Centrality):
	"""
		KPathCentrality(G, alpha=0.2, k=0)

		Constructs the K-Path Centrality class for the given Graph `G`.

	 	Parameters
	 	----------
	 	G : Graph
	 		The graph.
	 	alpha : double, in interval [-0.5, 0.5]
			tradeoff between runtime and precision
			-0.5: maximum precision, maximum runtime
	 		 0.5: lowest precision, lowest runtime
		k: maximum length of paths
	"""

	def __cinit__(self, Graph G, alpha=0.2, k=0):
		self._G = G
		self._this = new _KPathCentrality(G._this, alpha, k)


cdef extern from "cpp/centrality/KatzCentrality.h":
	cdef cppclass _KatzCentrality "NetworKit::KatzCentrality" (_Centrality):
		_KatzCentrality(_Graph, double, double, double) except +

cdef class KatzCentrality(Centrality):
	"""
		KatzCentrality(G, alpha=5e-4, beta=0.1, tol=1e-8)

		Constructs a KatzCentrality object for the given Graph `G`.
		Each iteration of the algorithm requires O(m) time.
		The number of iterations depends on how long it takes to reach the convergence
		(and therefore on the desired tolerance `tol`).

	 	Parameters
	 	----------
	 	G : Graph
	 		The graph.
	 	alpha : double
			Damping of the matrix vector product result
		beta : double
			Constant value added to the centrality of each vertex
		tol : double
			The tolerance for convergence.
	"""

	def __cinit__(self, Graph G, alpha=0.2, beta=0.1, tol=1e-8):
		self._G = G
		self._this = new _KatzCentrality(G._this, alpha, beta, tol)

cdef extern from "cpp/centrality/ApproxBetweenness.h":
	cdef cppclass _ApproxBetweenness "NetworKit::ApproxBetweenness" (_Centrality):
		_ApproxBetweenness(_Graph, double, double, double) except +
		count numberOfSamples() except +

cdef class ApproxBetweenness(Centrality):
	""" Approximation of betweenness centrality according to algorithm described in
 	Matteo Riondato and Evgenios M. Kornaropoulos: Fast Approximation of Betweenness Centrality through Sampling

 	ApproxBetweenness(G, epsilon=0.01, delta=0.1, universalConstant=1.0)

 	The algorithm approximates the betweenness of all vertices so that the scores are
	within an additive error epsilon with probability at least (1- delta).
	The values are normalized by default. The run() method takes O(m) time per sample, where  m is
	the number of edges of the graph. The number of samples is proportional to universalConstant/epsilon^2.
	Although this algorithm has a theoretical guarantee, the algorithm implemented in Estimate Betweenness usually performs better in practice
	Therefore, we recommend to use EstimateBetweenness if no theoretical guarantee is needed.

	Parameters
	----------
	G : Graph
		the graph
	epsilon : double, optional
		maximum additive error
	delta : double, optional
		probability that the values are within the error guarantee
	universalConstant: double, optional
		the universal constant to be used in computing the sample size.
		It is 1 by default. Some references suggest using 0.5, but there
		is no guarantee in this case.
	"""

	def __cinit__(self, Graph G, epsilon=0.1, delta=0.1, universalConstant=1.0):
		self._G = G
		self._this = new _ApproxBetweenness(G._this, epsilon, delta, universalConstant)

	def numberOfSamples(self):
		return (<_ApproxBetweenness*>(self._this)).numberOfSamples()


cdef extern from "cpp/centrality/EstimateBetweenness.h":
	cdef cppclass _EstimateBetweenness"NetworKit::EstimateBetweenness" (_Centrality):
		_EstimateBetweenness(_Graph, count, bool, bool) except +


cdef class EstimateBetweenness(Centrality):
	""" Estimation of betweenness centrality according to algorithm described in
	Sanders, Geisberger, Schultes: Better Approximation of Betweenness Centrality

	EstimateBetweenness(G, nSamples, normalized=False, parallel=False)

	The algorithm estimates the betweenness of all nodes, using weighting
	of the contributions to avoid biased estimation. The run() method takes O(m)
	time per sample, where  m is the number of edges of the graph. There is no proven
	theoretical guarantee on the quality of the approximation. However, the algorithm
        was shown to perform well in practice.
        If a guarantee is required, use ApproxBetweenness.

	Parameters
	----------
	G : Graph
		input graph
	nSamples : count
		user defined number of samples
	normalized : bool, optional
		normalize centrality values in interval [0,1]
	parallel : bool, optional
		run in parallel with additional memory cost z + 3z * t
	"""

	def __cinit__(self, Graph G, nSamples, normalized=False, parallel=False):
		self._G = G
		self._this = new _EstimateBetweenness(G._this, nSamples, normalized, parallel)


cdef class ApproxBetweenness2(Centrality):
	""" DEPRECATED: Use EstimateBetweenness instead.

	Estimation of betweenness centrality according to algorithm described in
	Sanders, Geisberger, Schultes: Better Approximation of Betweenness Centrality

	ApproxBetweenness2(G, nSamples, normalized=False, parallel=False)

	The algorithm estimates the betweenness of all nodes, using weighting
	of the contributions to avoid biased estimation. The run() method takes O(m)
	time per sample, where  m is the number of edges of the graph. There is no proven
	theoretical guarantee on the quality of the approximation. However, the algorithm
        was shown to perform well in practice.
        If a guarantee is required, use ApproxBetweenness.

	Parameters
	----------
	G : Graph
		input graph
	nSamples : count
		user defined number of samples
	normalized : bool, optional
		normalize centrality values in interval [0,1]
	parallel : bool, optional
		run in parallel with additional memory cost z + 3z * t
	"""

	def __cinit__(self, Graph G, nSamples, normalized=False, parallel=False):
		from warnings import warn
		warn("ApproxBetweenness2 is deprecated; use EstimateBetweenness instead.", DeprecationWarning)
		self._G = G
		self._this = new _EstimateBetweenness(G._this, nSamples, normalized, parallel)


cdef extern from "cpp/centrality/ApproxCloseness.h":
	enum _ClosenessType "NetworKit::ApproxCloseness::CLOSENESS_TYPE":
		INBOUND,
		OUTBOUND,
		SUM

cdef extern from "cpp/centrality/ApproxCloseness.h":
	cdef cppclass _ApproxCloseness "NetworKit::ApproxCloseness" (_Centrality):
		_ClosenessType type
		_ApproxCloseness(_Graph, count, float, bool, _ClosenessType type) except +
		vector[double] getSquareErrorEstimates() except +



cdef class ApproxCloseness(Centrality):
	""" Approximation of closeness centrality according to algorithm described in
  Cohen et al., Computing Classic Closeness Centrality, at Scale.

	ApproxCloseness(G, nSamples, epsilon=0.1, normalized=False, type=OUTBOUND)

	The algorithm approximates the closeness of all nodes in both directed and undirected graphs using a hybrid estimator.
	First, it takes nSamples samples. For these sampled nodes, the closeness is computed exactly. The pivot of each of the
	remaining nodes is the closest sampled node to it. If a node lies very close to its pivot, a sampling approach is used.
	Otherwise, a pivoting approach is used. Notice that the input graph has to be connected.

	Parameters
	----------
	G : Graph
		input graph (undirected)
	nSamples : count
		user defined number of samples
	epsilon : double, optional
		parameter used for the error guarantee; it is also used to control when to use sampling and when to use pivoting
	normalized : bool, optional
		normalize centrality values in interval [0,1]
	type : _ClosenessType, optional
		use in- or outbound centrality or the sum of both (see paper) for computing closeness on directed graph. If G is undirected, this can be ignored.
	"""

	#cdef _ApproxCloseness _this
	INBOUND = 0
	OUTBOUND = 1
	SUM = 2

	def __cinit__(self, Graph G, nSamples, epsilon=0.1, normalized=False, _ClosenessType type=OUTBOUND):
		self._G = G
		self._this = new _ApproxCloseness(G._this, nSamples, epsilon, normalized, type)

	def getSquareErrorEstimates(self):
		""" Return a vector containing the square error estimates for all nodes.

		Returns
		-------
		vector
			A vector of doubles.
		"""
		return (<_ApproxCloseness*>(self._this)).getSquareErrorEstimates()



cdef extern from "cpp/centrality/PageRank.h":
	cdef cppclass _PageRank "NetworKit::PageRank" (_Centrality):
		_PageRank(_Graph, double damp, double tol) except +

cdef class PageRank(Centrality):
	"""	Compute PageRank as node centrality measure.

	PageRank(G, damp=0.85, tol=1e-9)

	Parameters
	----------
	G : Graph
		Graph to be processed.
	damp : double
		Damping factor of the PageRank algorithm.
	tol : double, optional
		Error tolerance for PageRank iteration.
	"""

	def __cinit__(self, Graph G, double damp=0.85, double tol=1e-9):
		self._G = G
		self._this = new _PageRank(G._this, damp, tol)



cdef extern from "cpp/centrality/EigenvectorCentrality.h":
	cdef cppclass _EigenvectorCentrality "NetworKit::EigenvectorCentrality" (_Centrality):
		_EigenvectorCentrality(_Graph, double tol) except +

cdef class EigenvectorCentrality(Centrality):
	"""	Computes the leading eigenvector of the graph's adjacency matrix (normalized in 2-norm).
	Interpreted as eigenvector centrality score.

	EigenvectorCentrality(G, tol=1e-9)

	Constructs the EigenvectorCentrality class for the given Graph `G`. `tol` defines the tolerance for convergence.

	Parameters
	----------
	G : Graph
		The graph.
	tol : double, optional
		The tolerance for convergence.
	"""

	def __cinit__(self, Graph G, double tol=1e-9):
		self._G = G
		self._this = new _EigenvectorCentrality(G._this, tol)


cdef extern from "cpp/centrality/CoreDecomposition.h":
	cdef cppclass _CoreDecomposition "NetworKit::CoreDecomposition" (_Centrality):
		_CoreDecomposition(_Graph, bool, bool, bool) except +
		_Cover getCover() except +
		_Partition getPartition() except +
		index maxCoreNumber() except +
		vector[node] getNodeOrder() except +

cdef class CoreDecomposition(Centrality):
	""" Computes k-core decomposition of a graph.

	CoreDecomposition(G)

	Create CoreDecomposition class for graph `G`. The graph may not contain self-loops.

	Parameters
	----------
	G : Graph
		The graph.
	normalized : boolean
		Divide each core number by the maximum degree.
	enforceBucketQueueAlgorithm : boolean
		enforce switch to sequential algorithm
	storeNodeOrder : boolean
		If set to True, the order of the nodes in ascending order of the cores is stored and can later be returned using getNodeOrder(). Enforces the sequential bucket priority queue algorithm.

	"""

	def __cinit__(self, Graph G, bool normalized=False, bool enforceBucketQueueAlgorithm=False, bool storeNodeOrder = False):
		self._G = G
		self._this = new _CoreDecomposition(G._this, normalized, enforceBucketQueueAlgorithm, storeNodeOrder)

	def maxCoreNumber(self):
		""" Get maximum core number.

		Returns
		-------
		index
			The maximum core number.
		"""
		return (<_CoreDecomposition*>(self._this)).maxCoreNumber()

	def getCover(self):
		""" Get the k-cores as cover.

		Returns
		-------
		vector
			The k-cores as sets of nodes, indexed by k.
		"""
		return Cover().setThis((<_CoreDecomposition*>(self._this)).getCover())

	def getPartition(self):
		""" Get the k-shells as a partition object.

		Returns
		-------
		Partition
			The k-shells
		"""
		return Partition().setThis((<_CoreDecomposition*>(self._this)).getPartition())

	def getNodeOrder(self):
		"""
		Get the node order.

		This is only possible when storeNodeOrder was set.

		Returns
		-------
		list
			The nodes sorted by increasing core number.
		"""
		return (<_CoreDecomposition*>(self._this)).getNodeOrder()

cdef extern from "cpp/centrality/LocalClusteringCoefficient.h":
	cdef cppclass _LocalClusteringCoefficient "NetworKit::LocalClusteringCoefficient" (_Centrality):
		_LocalClusteringCoefficient(_Graph, bool) except +

cdef class LocalClusteringCoefficient(Centrality):
	"""
		LocalClusteringCoefficient(G, turbo=False)

		Constructs the LocalClusteringCoefficient class for the given Graph `G`. If the local clustering coefficient values should be normalized,
		then set `normalized` to True. The graph may not contain self-loops.

		There are two algorithms available. The trivial (parallel) algorithm needs only a small amount of additional memory.
		The turbo mode adds a (sequential, but fast) pre-processing step using ideas from [0]. This reduces the running time
		significantly for most graphs. However, the turbo mode needs O(m) additional memory. In practice this should be a bit
		less than half of the memory that is needed for the graph itself. The turbo mode is particularly effective for graphs
		with nodes of very high degree and a very skewed degree distribution.

		[0] Triangle Listing Algorithms: Back from the Diversion
		Mark Ortmann and Ulrik Brandes
		2014 Proceedings of the Sixteenth Workshop on Algorithm Engineering and Experiments (ALENEX). 2014, 1-8

	 	Parameters
	 	----------
	 	G : Graph
	 		The graph.
		turbo : bool
			If the turbo mode shall be activated.
	"""

	def __cinit__(self, Graph G, bool turbo = False):
		self._G = G
		self._this = new _LocalClusteringCoefficient(G._this, turbo)


cdef extern from "cpp/centrality/Sfigality.h":
	cdef cppclass _Sfigality "NetworKit::Sfigality" (_Centrality):
		_Sfigality(_Graph) except +

cdef class Sfigality(Centrality):
	"""
	Sfigality is a new type of node centrality measures that is high if neighboring nodes have a higher degree, e.g. in social networks, if your friends have more friends than you. Formally:

		$$\sigma(u) = \frac{| \{ v: \{u,v\} \in E, deg(u) < deg(v) \} |}{ deg(u) }$$

 	Parameters
 	----------
 	G : Graph
 		The graph.
	"""

	def __cinit__(self, Graph G):
		self._G = G
		self._this = new _Sfigality(G._this)



cdef extern from "cpp/centrality/DynApproxBetweenness.h":
	cdef cppclass _DynApproxBetweenness "NetworKit::DynApproxBetweenness":
		_DynApproxBetweenness(_Graph, double, double, bool, double) except +
		void run() nogil except +
		void update(_GraphEvent) except +
		void updateBatch(vector[_GraphEvent]) except +
		vector[double] scores() except +
		vector[pair[node, double]] ranking() except +
		double score(node) except +
		count getNumberOfSamples() except +

cdef class DynApproxBetweenness:
	""" The algorithm approximates the betweenness of all vertices so that the scores are
	  within an additive error @a epsilon with probability at least (1- @a delta).
	  The values are normalized by default.

	DynApproxBetweenness(G, epsilon=0.01, delta=0.1, storePredecessors=True, universalConstant=1.0)

	The algorithm approximates the betweenness of all vertices so that the scores are
	within an additive error epsilon with probability at least (1- delta).
	The values are normalized by default.

	Parameters
	----------
	G : Graph
		the graph
	epsilon : double, optional
		maximum additive error
	delta : double, optional
		probability that the values are within the error guarantee
	storePredecessors : bool, optional
		store lists of predecessors?
	universalConstant: double, optional
		the universal constant to be used in computing the sample size.
		It is 1 by default. Some references suggest using 0.5, but there
		is no guarantee in this case.
	"""
	cdef _DynApproxBetweenness* _this
	cdef Graph _G

	def __cinit__(self, Graph G, epsilon=0.01, delta=0.1, storePredecessors = True, universalConstant=1.0):
		self._G = G
		self._this = new _DynApproxBetweenness(G._this, epsilon, delta, storePredecessors, universalConstant)

	# this is necessary so that the C++ object gets properly garbage collected
	def __dealloc__(self):
		del self._this

	def run(self):
		with nogil:
			self._this.run()
		return self

	def update(self, ev):
		""" Updates the betweenness centralities after the edge insertions.

		Parameters
		----------
		ev : GraphEvent.
		"""
		self._this.update(_GraphEvent(ev.type, ev.u, ev.v, ev.w))

	def updateBatch(self, batch):
		""" Updates the betweenness centralities after the batch `batch` of edge insertions.

		Parameters
		----------
		batch : list of GraphEvent.
		"""
		cdef vector[_GraphEvent] _batch
		for ev in batch:
			_batch.push_back(_GraphEvent(ev.type, ev.u, ev.v, ev.w))
		self._this.updateBatch(_batch)

	def scores(self):
		""" Get a vector containing the betweenness score for each node in the graph.

		Returns
		-------
		vector
			The betweenness scores calculated by run().
		"""
		return self._this.scores()

	def score(self, v):
		""" Get the betweenness score of node `v` calculated by run().

		Parameters
		----------
		v : node
			A node.

		Returns
		-------
		double
			The betweenness score of node `v.
		"""
		return self._this.score(v)

	def ranking(self):
		""" Get a vector of pairs sorted into descending order. Each pair contains a node and the corresponding score
		calculated by run().

		Returns
		-------
		vector
			A vector of pairs.
		"""
		return self._this.ranking()

	def getNumberOfSamples(self):
		"""
		Get number of path samples used in last calculation.
		"""
		return self._this.getNumberOfSamples()

cdef extern from "cpp/centrality/DynBetweenness.h":
	cdef cppclass _DynBetweenness "NetworKit::DynBetweenness":
		_DynBetweenness(_Graph) except +
		void run() nogil except +
		void update(_GraphEvent) except +
		void updateBatch(vector[_GraphEvent]) except +
		vector[double] scores() except +
		vector[pair[node, double]] ranking() except +
		double score(node) except +

cdef class DynBetweenness:
	""" The algorithm computes the betweenness centrality of all nodes
			and updates them after an edge insertion.

	DynBetweenness(G)

	Parameters
	----------
	G : Graph
		the graph
	"""
	cdef _DynBetweenness* _this
	cdef Graph _G

	def __cinit__(self, Graph G):
		self._G = G
		self._this = new _DynBetweenness(G._this)

	# this is necessary so that the C++ object gets properly garbage collected
	def __dealloc__(self):
		del self._this

	def run(self):
		with nogil:
			self._this.run()
		return self

	def update(self, ev):
		""" Updates the betweenness centralities after the edge insertions.

		Parameters
		----------
		ev : GraphEvent.
		"""
		self._this.update(_GraphEvent(ev.type, ev.u, ev.v, ev.w))

	def updateBatch(self, batch):
		""" Updates the betweenness centralities after the batch `batch` of edge insertions.

		Parameters
		----------
		batch : list of GraphEvent.
		"""
		cdef vector[_GraphEvent] _batch
		for ev in batch:
			_batch.push_back(_GraphEvent(ev.type, ev.u, ev.v, ev.w))
		self._this.updateBatch(_batch)

	def scores(self):
		""" Get a vector containing the betweenness score for each node in the graph.

		Returns
		-------
		vector
			The betweenness scores calculated by run().
		"""
		return self._this.scores()

	def score(self, v):
		""" Get the betweenness score of node `v` calculated by run().

		Parameters
		----------
		v : node
			A node.

		Returns
		-------
		double
			The betweenness score of node `v.
		"""
		return self._this.score(v)

	def ranking(self):
		""" Get a vector of pairs sorted into descending order. Each pair contains a node and the corresponding score
		calculated by run().

		Returns
		-------
		vector
			A vector of pairs.
		"""
		return self._this.ranking()


cdef extern from "cpp/centrality/DynBetweennessOneNode.h":
	cdef cppclass _DynBetweennessOneNode "NetworKit::DynBetweennessOneNode":
		_DynBetweennessOneNode(_Graph, node) except +
		void run() nogil except +
		void update(_GraphEvent) except +
		void updateBatch(vector[_GraphEvent]) except +
		double getDistance(node, node) except +
		double getSigma(node, node) except +
		double getSigmax(node, node) except +
		double getbcx() except +

cdef class DynBetweennessOneNode:
	""" Dynamic exact algorithm for updating the betweenness of a specific node

	DynBetweennessOneNode(G, x)

	The algorithm aupdates the betweenness of a node after an edge insertions
	(faster than updating it for all nodes), based on the algorithm
	proposed by Bergamini et al. "Improving the betweenness centrality of a node by adding links"

	Parameters
	----------
	G : Graph
		the graph
	x : node
		the node for which you want to update betweenness
	"""
	cdef _DynBetweennessOneNode* _this
	cdef Graph _G

	def __cinit__(self, Graph G, node):
		self._G = G
		self._this = new _DynBetweennessOneNode(G._this, node)

	# this is necessary so that the C++ object gets properly garbage collected
	def __dealloc__(self):
		del self._this

	def run(self):
		with nogil:
			self._this.run()
		return self

	def update(self, ev):
		""" Updates the betweenness centralities after the batch `batch` of edge insertions.

		Parameters
		----------
		ev : edge insertion.
		"""
		self._this.update(_GraphEvent(ev.type, ev.u, ev.v, ev.w))

	def updateBatch(self, batch):
		""" Updates the betweenness centrality of node x after the batch `batch` of edge insertions.

		Parameters
		----------
		batch : list of GraphEvent.
		"""
		cdef vector[_GraphEvent] _batch
		for ev in batch:
			_batch.push_back(_GraphEvent(ev.type, ev.u, ev.v, ev.w))
		self._this.updateBatch(_batch)

	def getDistance(self, u, v):
		""" Returns the distance between node u and node v.
		"""
		return self._this.getDistance(u, v)

	def getSigma(self, u, v):
		""" Returns the number of shortest paths between node u and node v.
		"""
		return self._this.getSigma(u, v)

	def getSigmax(self, u, v):
		""" Returns the number of shortest paths between node u and node v that go through x.
		"""
		return self._this.getSigmax(u, v)

	def getbcx(self):
		""" Returns the betweenness centrality score of node x
		"""
		return self._this.getbcx()

cdef extern from "cpp/centrality/PermanenceCentrality.h":
	cdef cppclass _PermanenceCentrality "NetworKit::PermanenceCentrality":
		_PermanenceCentrality(const _Graph& G, const _Partition& P) except +
		void run() nogil except +
		double getIntraClustering(node u) except +
		double getPermanence(node u) except +

cdef class PermanenceCentrality:
	"""
	Permanence centrality

	This centrality measure measure how well a vertex belongs to its community. The values are calculated on the fly, the partion may be changed in between the requests.
	For details see

	Tanmoy Chakraborty, Sriram Srinivasan, Niloy Ganguly, Animesh Mukherjee, and Sanjukta Bhowmick. 2014.
	On the permanence of vertices in network communities.
	In Proceedings of the 20th ACM SIGKDD international conference on Knowledge discovery and data mining (KDD '14).
	ACM, New York, NY, USA, 1396-1405. DOI: http://dx.doi.org/10.1145/2623330.2623707

	FIXME: does not use the common centrality interface yet.
	"""
	cdef _PermanenceCentrality *_this
	cdef Graph _G
	cdef Partition _P

	def __cinit__(self, Graph G, Partition P):
		self._this = new _PermanenceCentrality(G._this, P._this)
		self._G = G
		self._P = P

	def __dealloc__(self):
		del self._this

	def run(self):
		with nogil:
			self._this.run()
		return self

	def getIntraClustering(self, node u):
		return self._this.getIntraClustering(u)

	def getPermanence(self, node u):
		return self._this.getPermanence(u)

cdef extern from "cpp/centrality/LocalPartitionCoverage.h":
	cdef cppclass _LocalPartitionCoverage "NetworKit::LocalPartitionCoverage" (_Centrality):
		_LocalPartitionCoverage(_Graph, _Partition) except +

cdef class LocalPartitionCoverage(Centrality):
	"""
	The local partition coverage is the amount of neighbors of a node u that are in the same partition as u.
	The running time of the run() method is O(m), where m is the number of edges in the graph.

	LocalPartitionCoverage(G, P)

	Parameters
	----------
	G : Graph
		The graph.
	P : Partition
		The partition to use
	"""
	cdef Partition _P

	def __cinit__(self, Graph G not None, Partition P not None):
		self._G = G
		self._P = P
		self._this = new _LocalPartitionCoverage(G._this, P._this)

cdef extern from "cpp/centrality/LaplacianCentrality.h":
	cdef cppclass _LaplacianCentrality "NetworKit::LaplacianCentrality" (_Centrality):
		_LaplacianCentrality(_Graph, bool) except +

cdef class LaplacianCentrality(Centrality):
	""" Computes the Laplacian centrality of the graph.

	LaplacianCentrality(G, normalized=False)

	The implementation is a simplification of the original algorithm proposed by Qi et al. in
	"Laplacian centrality: A new centrality measure for weighted networks".

	See https://dl.acm.org/citation.cfm?id=2181343.2181780 for details.

	Parameters
	----------
	G : Graph
		The graph.
	normalized : bool, optional
		Whether scores should be normalized by the energy of the full graph.
	"""

	def __cinit__(self, Graph G, normalized = False):
		self._G = G
		self._this = new _LaplacianCentrality(G._this, normalized)

cdef extern from "cpp/centrality/SpanningEdgeCentrality.h":
	cdef cppclass _SpanningEdgeCentrality "NetworKit::SpanningEdgeCentrality":
		_SpanningEdgeCentrality(_Graph G, double tol) except +
		void run() nogil except +
		void runApproximation() except +
		void runParallelApproximation() except +
		vector[double] scores() except +

cdef class SpanningEdgeCentrality:
	""" Computes the Spanning Edge centrality for the edges of the graph.

	SpanningEdgeCentrality(G, tol = 0.1)

	Parameters
	----------
	G : Graph
		The graph.
	tol: double
		Tolerance used for the approximation: with probability at least 1-1/n, the approximated scores are within a factor 1+tol from the exact scores.
	"""
	cdef _SpanningEdgeCentrality* _this
	cdef Graph _G
	def __cinit__(self,  Graph G, double tol = 0.1):
		self._G = G
		self._this = new _SpanningEdgeCentrality(G._this, tol)
	def __dealloc__(self):
		del self._this
	def run(self):
		""" This method computes Spanning Edge Centrality exactly. This solves a linear system for each edge, so the empirical running time is O(m^2),
				where m is the number of edges in the graph."""
		with nogil:
			self._this.run()
		return self
	def runApproximation(self):
		""" Computes approximation of the Spanning Edge Centrality. This solves k linear systems, where k is log(n)/(tol^2). The empirical running time is O(km), where n is the number of nodes
 	 			and m is the number of edges. """
		return self._this.runApproximation()

	def runParallelApproximation(self):
		""" Computes approximation (in parallel) of the Spanning Edge Centrality. This solves k linear systems, where k is log(n)/(tol^2). The empirical running time is O(km), where n is the number of nodes
 	 			and m is the number of edges."""
		return self._this.runParallelApproximation()

	def scores(self):
		""" Get a vector containing the SEC score for each edge in the graph.

		Returns
		-------
		vector
			The SEC scores.
		"""
		return self._this.scores()
