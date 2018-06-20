'''
	Module: distance
'''

cdef extern from "cpp/distance/AdamicAdarDistance.h":
	cdef cppclass _AdamicAdarDistance "NetworKit::AdamicAdarDistance":
		_AdamicAdarDistance(const _Graph& G) except +
		void preprocess() except +
		double distance(node u, node v) except +
		vector[double] getEdgeScores() except +

cdef class AdamicAdarDistance:
	"""
	Calculate the adamic adar similarity.

	Parameters
	----------
	G : Graph
		The input graph.
	"""
	cdef _AdamicAdarDistance* _this
	cdef Graph _G

	def __cinit__(self, Graph G):
		self._G = G
		self._this = new _AdamicAdarDistance(G._this)

	def __dealloc__(self):
		del self._this

	def preprocess(self):
		self._this.preprocess()

	def getAttribute(self):
		"""
		Returns
		-------
		vector[double]
			The edge attribute that contains the adamic adar similarity.

		"""
		#### TODO: convert distance to similarity!?! ####
		return self._this.getEdgeScores()

cdef extern from "cpp/distance/SSSP.h":
	cdef cppclass _SSSP "NetworKit::SSSP"(_Algorithm):
		_SSSP(_Graph G, node source, bool storePaths, bool storeNodesSortedByDistance, node target) except +
		void run() nogil except +
		vector[edgeweight] getDistances(bool moveOut) except +
		edgeweight distance(node t) except +
		vector[node] getPredecessors(node t) except +
		vector[node] getPath(node t, bool forward) except +
		set[vector[node]] getPaths(node t, bool forward) except +
		vector[node] getStack(bool moveOut) except +
		vector[node] getNodesSortedByDistance(bool moveOut) except +
		double _numberOfPaths(node t) except +

cdef class SSSP(Algorithm):
	""" Base class for single source shortest path algorithms. """

	cdef Graph _G

	def __init__(self, *args, **namedargs):
		if type(self) == SSSP:
			raise RuntimeError("Error, you may not use SSSP directly, use a sub-class instead")

	def __dealloc__(self):
		self._G = None # just to be sure the graph is deleted

	def getDistances(self, moveOut=True):
		"""
		Returns a vector of weighted distances from the source node, i.e. the
 	 	length of the shortest path from the source node to any other node.

 	 	Returns
 	 	-------
 	 	vector
 	 		The weighted distances from the source node to any other node in the graph.
		"""
		return (<_SSSP*>(self._this)).getDistances(moveOut)

	def distance(self, t):
		return (<_SSSP*>(self._this)).distance(t)

	def getPredecessors(self, t):
		return (<_SSSP*>(self._this)).getPredecessors(t)

	def getPath(self, t, forward=True):
		""" Returns a shortest path from source to `t` and an empty path if source and `t` are not connected.

		Parameters
		----------
		t : node
			Target node.

		Returns
		-------
		vector
			A shortest path from source to `t or an empty path.
		"""
		return (<_SSSP*>(self._this)).getPath(t, forward)

	def getPaths(self, t, forward=True):
		cdef set[vector[node]] paths = (<_SSSP*>(self._this)).getPaths(t, forward)
		result = []
		for elem in paths:
			result.append(list(elem))
		return result

	def getStack(self, moveOut=True):
		""" DEPRECATED: Use getNodesSortedByDistance instead.

		Returns a vector of nodes ordered in increasing distance from the source.

		For this functionality to be available, storeNodesSortedByDistance has to be set to true in the constructor.
		There are no guarantees regarding the ordering of two nodes with the same distance to the source.

		Parameters
		----------
		moveOut : bool
			If set to true, the container will be moved out of the class instead of copying it; default=true.

		Returns
		-------
		vector
			Nodes ordered in increasing distance from the source.
		"""
		from warnings import warn
		warn("getStack is deprecated; use getNodesSortedByDistance instead.", DeprecationWarning)
		return (<_SSSP*>(self._this)).getStack(moveOut)

	def getNodesSortedByDistance(self, moveOut=True):
		""" Returns a vector of nodes ordered in increasing distance from the source.

		For this functionality to be available, storeNodesSortedByDistance has to be set to true in the constructor.
		There are no guarantees regarding the ordering of two nodes with the same distance to the source.

		Parameters
		----------
		moveOut : bool
			If set to true, the container will be moved out of the class instead of copying it; default=true.

		Returns
		-------
		vector
			Nodes ordered in increasing distance from the source.
		"""
		return (<_SSSP*>(self._this)).getNodesSortedByDistance(moveOut)

	def numberOfPaths(self, t):
		return (<_SSSP*>(self._this))._numberOfPaths(t)

cdef extern from "cpp/distance/DynSSSP.h":
	cdef cppclass _DynSSSP "NetworKit::DynSSSP"(_SSSP):
		_DynSSSP(_Graph G, node source, bool storePaths, bool storeStack, node target) except +
		void update(_GraphEvent ev) except +
		void updateBatch(vector[_GraphEvent] batch) except +
		bool modified() except +
		void setTargetNode(node t) except +

cdef class DynSSSP(SSSP):
	""" Base class for single source shortest path algorithms in dynamic graphs. """
	def __init__(self, *args, **namedargs):
		if type(self) == SSSP:
			raise RuntimeError("Error, you may not use DynSSSP directly, use a sub-class instead")

	def update(self, ev):
		""" Updates shortest paths with the edge insertion.

		Parameters
		----------
		ev : GraphEvent.
		"""
		(<_DynSSSP*>(self._this)).update(_GraphEvent(ev.type, ev.u, ev.v, ev.w))

	def updateBatch(self, batch):
		""" Updates shortest paths with the batch `batch` of edge insertions.

		Parameters
		----------
		batch : list of GraphEvent.
		"""
		cdef vector[_GraphEvent] _batch
		for ev in batch:
			_batch.push_back(_GraphEvent(ev.type, ev.u, ev.v, ev.w))
		(<_DynSSSP*>(self._this)).updateBatch(_batch)

	def modified(self):
		return (<_DynSSSP*>(self._this)).modified()

	def setTargetNode(self, t):
		(<_DynSSSP*>(self._this)).setTargetNode(t)

cdef extern from "cpp/distance/BFS.h":
	cdef cppclass _BFS "NetworKit::BFS"(_SSSP):
		_BFS(_Graph G, node source, bool storePaths, bool storeNodesSortedByDistance, node target) except +

cdef class BFS(SSSP):
	""" Simple breadth-first search on a Graph from a given source

	BFS(G, source, [storePaths], [storeNodesSortedByDistance], target)

	Create BFS for `G` and source node `source`.

	Parameters
	----------
	G : Graph
		The graph.
	source : node
		The source node of the breadth-first search.
	storePaths : bool
		store paths and number of paths?
	target: node
		terminate search when the target has been reached
	"""

	def __cinit__(self, Graph G, source, storePaths=True, storeNodesSortedByDistance=False, target=none):
		self._G = G
		self._this = new _BFS(G._this, source, storePaths, storeNodesSortedByDistance, target)

cdef extern from "cpp/distance/DynBFS.h":
	cdef cppclass _DynBFS "NetworKit::DynBFS"(_DynSSSP):
		_DynBFS(_Graph G, node source) except +

cdef class DynBFS(DynSSSP):
	""" Dynamic version of BFS.

	DynBFS(G, source)

	Create DynBFS for `G` and source node `source`.

	Parameters
	----------
	G : Graph
		The graph.
	source : node
		The source node of the breadth-first search.
	storeStack : bool
		maintain a stack of nodes in order of decreasing distance?
	"""
	def __cinit__(self, Graph G, source):
		self._G = G
		self._this = new _DynBFS(G._this, source)

cdef extern from "cpp/distance/Dijkstra.h":
	cdef cppclass _Dijkstra "NetworKit::Dijkstra"(_SSSP):
		_Dijkstra(_Graph G, node source, bool storePaths, bool storeNodesSortedByDistance, node target) except +

cdef class Dijkstra(SSSP):
	""" Dijkstra's SSSP algorithm.
	Returns list of weighted distances from node source, i.e. the length of the shortest path from source to
	any other node.

    Dijkstra(G, source, [storePaths], [storeNodesSortedByDistance], target)

    Creates Dijkstra for `G` and source node `source`.

    Parameters
	----------
	G : Graph
		The graph.
	source : node
		The source node.
	storePaths : bool
		Paths are reconstructable and the number of paths is stored.
	storeNodesSortedByDistance: bool
		Store a vector of nodes ordered in increasing distance from the source.
	target : node
		target node. Search ends when target node is reached. t is set to None by default.
    """
	def __cinit__(self, Graph G, source, storePaths=True, storeNodesSortedByDistance=False, node target=none):
		self._G = G
		self._this = new _Dijkstra(G._this, source, storePaths, storeNodesSortedByDistance, target)

cdef extern from "cpp/distance/DynDijkstra.h":
	cdef cppclass _DynDijkstra "NetworKit::DynDijkstra"(_DynSSSP):
		_DynDijkstra(_Graph G, node source) except +

cdef class DynDijkstra(DynSSSP):
	""" Dynamic version of Dijkstra.

	DynDijkstra(G, source)

	Create DynDijkstra for `G` and source node `source`.

	Parameters
	----------
	G : Graph
		The graph.
	source : node
		The source node of the breadth-first search.

	"""
	def __cinit__(self, Graph G, source):
		self._G = G
		self._this = new _DynDijkstra(G._this, source)

cdef cppclass PathCallbackWrapper:
	void* callback
	__init__(object callback):
		this.callback = <void*>callback
	void cython_call_operator(vector[node] path):
		cdef bool error = False
		cdef string message
		try:
			(<object>callback)(path)
		except Exception as e:
			error = True
			message = stdstring("An Exception occurred, aborting execution of iterator: {0}".format(e))
		if (error):
			throw_runtime_error(message)

cdef extern from "cpp/distance/AllSimplePaths.h":
	cdef cppclass _AllSimplePaths "NetworKit::AllSimplePaths":
		_AllSimplePaths(_Graph G, node source, node target, count cutoff) except +
		void run() nogil except +
		count numberOfSimplePaths() except +
		vector[vector[node]] getAllSimplePaths() except +
		void forAllSimplePaths[Callback](Callback c) except +

cdef class AllSimplePaths:
	""" Algorithm to compute all existing simple paths from a source node to a target node. The maximum length of the paths can be fixed through 'cutoff'.
		CAUTION: This algorithm could take a lot of time on large networks (many edges), especially if the cutoff value is high or not specified.

	AllSimplePaths(G, source, target, cutoff=none)

	Create AllSimplePaths for `G`, source node `source`, target node 'target' and cutoff 'cutoff'.

	Parameters
	----------
	G : Graph
		The graph.
	source : node
		The source node.
	target : node
		The target node.
	cutoff : count
		(optional) The maximum length of the simple paths.

	"""

	cdef _AllSimplePaths* _this
	cdef Graph _G

	def __cinit__(self,  Graph G, source, target, cutoff=none):
		self._G = G
		self._this = new _AllSimplePaths(G._this, source, target, cutoff)

	def __dealloc__(self):
		del self._this

	def run(self):
		self._this.run()
		return self

	def numberOfSimplePaths(self):
		"""
		Returns the number of simple paths.

		Returns
		-------
		count
			The number of simple paths.
		"""
		return self._this.numberOfSimplePaths()

	def getAllSimplePaths(self):
		"""
		Returns all the simple paths from source to target.

		Returns
		-------
		A vector of vectors.
			A vector containing vectors which represent all simple paths.
		"""
		return self._this.getAllSimplePaths()

	def forAllSimplePaths(self, object callback):
		""" More efficient path iterator. Iterates over all the simple paths.

		Parameters
		----------
		callback : object
			Any callable object that takes the parameter path
		"""
		cdef PathCallbackWrapper* wrapper
		try:
			wrapper = new PathCallbackWrapper(callback)
			self._this.forAllSimplePaths[PathCallbackWrapper](dereference(wrapper))
		finally:
			del wrapper

cdef extern from "cpp/distance/APSP.h":
	cdef cppclass _APSP "NetworKit::APSP"(_Algorithm):
		_APSP(_Graph G) except +
		vector[vector[edgeweight]] getDistances() except +
		edgeweight getDistance(node u, node v) except +

cdef class APSP(Algorithm):
	""" All-Pairs Shortest-Paths algorithm (implemented running Dijkstra's algorithm from each node, or BFS if G is unweighted).

    APSP(G)

    Computes all pairwise shortest-path distances in G.

    Parameters
	----------
	G : Graph
		The graph.
    """
	cdef Graph _G

	def __cinit__(self, Graph G):
		self._G = G
		self._this = new _APSP(G._this)

	def __dealloc__(self):
		self._G = None

	def getDistances(self):
		""" Returns a vector of vectors of distances between each node pair.

 	 	Returns
 	 	-------
 	 	vector of vectors
 	 		The shortest-path distances from each node to any other node in the graph.
		"""
		return (<_APSP*>(self._this)).getDistances()

	def getDistance(self, node u, node v):
		""" Returns the length of the shortest path from source 'u' to target `v`.

		Parameters
		----------
		u : node
			Source node.
		v : node
			Target node.

		Returns
		-------
		int or float
			The distance from 'u' to 'v'.
		"""
		return (<_APSP*>(self._this)).getDistance(u, v)

cdef extern from "cpp/distance/DynAPSP.h":
	cdef cppclass _DynAPSP "NetworKit::DynAPSP"(_APSP):
		_DynAPSP(_Graph G) except +
		void update(_GraphEvent ev) except +
		void updateBatch(vector[_GraphEvent] batch) except +

cdef class DynAPSP(APSP):
	""" All-Pairs Shortest-Paths algorithm for dynamic graphs.

		DynAPSP(G)

		Computes all pairwise shortest-path distances in G.

		Parameters
	----------
	G : Graph
		The graph.
		"""
	def __init__(self, Graph G):
		self._G = G
		self._this = new _DynAPSP(G._this)

	def update(self, ev):
		""" Updates shortest paths with the edge insertion.

		Parameters
		----------
		ev : GraphEvent.
		"""
		(<_DynAPSP*>(self._this)).update(_GraphEvent(ev.type, ev.u, ev.v, ev.w))

	def updateBatch(self, batch):
		""" Updates shortest paths with the batch `batch` of edge insertions.

		Parameters
		----------
		batch : list of GraphEvent.
		"""
		cdef vector[_GraphEvent] _batch
		for ev in batch:
			_batch.push_back(_GraphEvent(ev.type, ev.u, ev.v, ev.w))
		(<_DynAPSP*>(self._this)).updateBatch(_batch)

cdef extern from "cpp/graph/SpanningForest.h":
	cdef cppclass _SpanningForest "NetworKit::SpanningForest":
		_SpanningForest(_Graph) except +
		_Graph generate() except +

cdef class SpanningForest:
	""" Generates a spanning forest for a given graph

		Parameters
		----------
		G : Graph
			The graph.
		nodes : list
			A subset of nodes of `G` which induce the subgraph.
	"""
	cdef _SpanningForest* _this
	cdef Graph _G

	def __cinit__(self, Graph G not None):
		self._G = G
		self._this = new _SpanningForest(G._this)


	def __dealloc__(self):
		del self._this

	def generate(self):
		return Graph().setThis(self._this.generate());

cdef extern from "cpp/distance/Diameter.h" namespace "NetworKit":
	cdef enum DiameterAlgo:
		automatic = 0
		exact = 1
		estimatedRange = 2
		estimatedSamples = 3
		estimatedPedantic = 4

class _DiameterAlgo(object):
	Automatic = automatic
	Exact = exact
	EstimatedRange = estimatedRange
	EstimatedSamples = estimatedSamples
	EstimatedPedantic = estimatedPedantic

cdef extern from "cpp/distance/Diameter.h" namespace "NetworKit::Diameter":
	cdef cppclass _Diameter "NetworKit::Diameter"(_Algorithm):
		_Diameter(_Graph G, DiameterAlgo algo, double error, count nSamples) except +
		pair[count, count] getDiameter() nogil except +

cdef class Diameter(Algorithm):
	cdef Graph _G
	"""
	TODO: docstring
	"""
	def __cinit__(self, Graph G not None, algo = _DiameterAlgo.Automatic, error = -1., nSamples = 0):
		self._G = G
		self._this = new _Diameter(G._this, algo, error, nSamples)

	def getDiameter(self):
		return (<_Diameter*>(self._this)).getDiameter()

cdef extern from "cpp/distance/Eccentricity.h" namespace "NetworKit::Eccentricity":
	pair[node, count] getValue(_Graph G, node v) except +

cdef class Eccentricity:
	"""
	TODO: docstring
	"""

	@staticmethod
	def getValue(Graph G, v):
		return getValue(G._this, v)

cdef extern from "cpp/distance/EffectiveDiameter.h" namespace "NetworKit::EffectiveDiameter":
	cdef cppclass _EffectiveDiameter "NetworKit::EffectiveDiameter"(_Algorithm):
		_EffectiveDiameter(_Graph& G, double ratio) except +
		void run() nogil except +
		double getEffectiveDiameter() except +

cdef class EffectiveDiameter(Algorithm):
	"""
	Calculates the effective diameter of a graph.
	The effective diameter is defined as the number of edges on average to reach a given ratio of all other nodes.

	Parameters
	----------
	G : Graph
		The graph.
	ratio : double
		The percentage of nodes that shall be within stepwidth; default = 0.9
	"""
	cdef Graph _G

	def __cinit__(self, Graph G not None, double ratio=0.9):
		self._G = G
		self._this = new _EffectiveDiameter(G._this, ratio)

	def getEffectiveDiameter(self):
		"""
		Returns
		-------
		double
			the effective diameter
		"""
		return (<_EffectiveDiameter*>(self._this)).getEffectiveDiameter()

cdef extern from "cpp/distance/EffectiveDiameterApproximation.h" namespace "NetworKit::EffectiveDiameterApproximation":
	cdef cppclass _EffectiveDiameterApproximation "NetworKit::EffectiveDiameterApproximation"(_Algorithm):
		_EffectiveDiameterApproximation(_Graph& G, double ratio, count k, count r) except +
		void run() nogil except +
		double getEffectiveDiameter() except +

cdef class EffectiveDiameterApproximation(Algorithm):
	"""
	Calculates the effective diameter of a graph.
	The effective diameter is defined as the number of edges on average to reach a given ratio of all other nodes.

	Implementation after the ANF algorithm presented in the paper "A Fast and Scalable Tool for Data Mining in Massive Graphs"[1]

	[1] by Palmer, Gibbons and Faloutsos which can be found here: http://www.cs.cmu.edu/~christos/PUBLICATIONS/kdd02-anf.pdf

	Parameters
	----------
	G : Graph
		The graph.
	ratio : double
		The percentage of nodes that shall be within stepwidth, default = 0.9
	k : count
		number of parallel approximations, bigger k -> longer runtime, more precise result; default = 64
	r : count
		number of additional bits, important in tiny graphs; default = 7
	"""
	cdef Graph _G

	def __cinit__(self, Graph G not None, double ratio=0.9, count k=64, count r=7):
		self._G = G
		self._this = new _EffectiveDiameterApproximation(G._this, ratio, k, r)

	def getEffectiveDiameter(self):
		"""
		Returns
		-------
		double
			the approximated effective diameter
		"""
		return (<_EffectiveDiameterApproximation*>(self._this)).getEffectiveDiameter()

cdef extern from "cpp/distance/HopPlotApproximation.h" namespace "NetworKit::HopPlotApproximation":
	cdef cppclass _HopPlotApproximation "NetworKit::HopPlotApproximation"(_Algorithm):
		_HopPlotApproximation(_Graph& G, count maxDistance, count k, count r) except +
		void run() nogil except +
		map[count, double] getHopPlot() except +

cdef class HopPlotApproximation(Algorithm):
	"""
	Computes an approxmation of the hop-plot of a given graph.
	The hop-plot is the set of pairs (d, g(g)) for each natural number d
	and where g(d) is the fraction of connected node pairs whose shortest connecting path has length at most d.

	Implementation after the ANF algorithm presented in the paper "A Fast and Scalable Tool for Data Mining in Massive Graphs"[1]

	[1] by Palmer, Gibbons and Faloutsos which can be found here: http://www.cs.cmu.edu/~christos/PUBLICATIONS/kdd02-anf.pdf

	Parameters
	----------
	G : Graph
		The graph.
	maxDistance : double
		maximum distance between considered nodes
		set to 0 or negative to get the hop-plot for the entire graph so that each node can reach each other node
	k : count
		number of parallel approximations, bigger k -> longer runtime, more precise result; default = 64
	r : count
		number of additional bits, important in tiny graphs; default = 7
	"""
	cdef Graph _G

	def __cinit__(self, Graph G not None, count maxDistance=0, count k=64, count r=7):
		self._G = G
		self._this = new _HopPlotApproximation(G._this, maxDistance, k, r)

	def getHopPlot(self):
		"""
		Returns
		-------
		map
			number of connected nodes for each distance
		"""
		cdef map[count, double] hp = (<_HopPlotApproximation*>(self._this)).getHopPlot()
		result = dict()
		for elem in hp:
			result[elem.first] = elem.second
		return result

cdef extern from "cpp/distance/NeighborhoodFunction.h" namespace "NetworKit::NeighborhoodFunction":
	cdef cppclass _NeighborhoodFunction "NetworKit::NeighborhoodFunction"(_Algorithm):
		_NeighborhoodFunction(_Graph& G) except +
		void run() nogil except +
		vector[count] getNeighborhoodFunction() except +

cdef class NeighborhoodFunction(Algorithm):
	"""
	Computes the neighborhood function exactly.
	The neighborhood function N of a graph G for a given distance t is defined
	as the number of node pairs (u,v) that can be reached within distance t.

	Parameters
	----------
	G : Graph
		The graph.
	"""
	cdef Graph _G

	def __cinit__(self, Graph G not None):
		self._G = G
		self._this = new _NeighborhoodFunction(G._this)

	def getNeighborhoodFunction(self):
		"""
		Returns
		-------
		list
			the i-th element denotes the number of node pairs that have a distance at most (i+1)
		"""
		return (<_NeighborhoodFunction*>(self._this)).getNeighborhoodFunction()

cdef extern from "cpp/distance/NeighborhoodFunctionApproximation.h" namespace "NetworKit::NeighborhoodFunctionApproximation":
	cdef cppclass _NeighborhoodFunctionApproximation "NetworKit::NeighborhoodFunctionApproximation"(_Algorithm):
		_NeighborhoodFunctionApproximation(_Graph& G, count k, count r) except +
		void run() nogil except +
		vector[count] getNeighborhoodFunction() except +

cdef class NeighborhoodFunctionApproximation(Algorithm):
	"""
	Computes an approximation of the neighborhood function.
	The neighborhood function N of a graph G for a given distance t is defined
	as the number of node pairs (u,v) that can be reached within distance t.

	Implementation after the ANF algorithm presented in the paper "A Fast and Scalable Tool for Data Mining in Massive Graphs"[1]

	[1] by Palmer, Gibbons and Faloutsos which can be found here: http://www.cs.cmu.edu/~christos/PUBLICATIONS/kdd02-anf.pdf

	Parameters
	----------
	G : Graph
		The graph.
	k : count
		number of approximations, bigger k -> longer runtime, more precise result; default = 64
	r : count
		number of additional bits, important in tiny graphs; default = 7
	"""
	cdef Graph _G

	def __cinit__(self, Graph G not None, count k=64, count r=7):
		self._G = G
		self._this = new _NeighborhoodFunctionApproximation(G._this, k, r)

	def getNeighborhoodFunction(self):
		"""
		Returns
		-------
		list
			the i-th element denotes the number of node pairs that have a distance at most (i+1)
		"""
		return (<_NeighborhoodFunctionApproximation*>(self._this)).getNeighborhoodFunction()

cdef extern from "cpp/distance/NeighborhoodFunctionHeuristic.h" namespace "NetworKit::NeighborhoodFunctionHeuristic::SelectionStrategy":
	enum _SelectionStrategy "NetworKit::NeighborhoodFunctionHeuristic::SelectionStrategy":
		RANDOM
		SPLIT

cdef extern from "cpp/distance/NeighborhoodFunctionHeuristic.h" namespace "NetworKit::NeighborhoodFunctionHeuristic":
	cdef cppclass _NeighborhoodFunctionHeuristic "NetworKit::NeighborhoodFunctionHeuristic"(_Algorithm):
		_NeighborhoodFunctionHeuristic(_Graph& G, const count nSamples, const _SelectionStrategy strategy) except +
		void run() nogil except +
		vector[count] getNeighborhoodFunction() except +

cdef class NeighborhoodFunctionHeuristic(Algorithm):
	"""
	Computes a heuristic of the neighborhood function.
	The algorithm runs nSamples breadth-first searches and scales the results up to the actual amount of nodes.
	Accepted strategies are "split" and "random".

	Parameters
	----------
	G : Graph
		The graph.
	nSamples : count
		the amount of samples, set to zero for heuristic of max(sqrt(m), 0.15*n)
	strategy : enum
		the strategy to select the samples, accepts "random" or "split"
	"""
	cdef Graph _G

	RANDOM = 0
	SPLIT = 1

	def __cinit__(self, Graph G not None, count nSamples=0, strategy=SPLIT):
		self._G = G
		self._this = new _NeighborhoodFunctionHeuristic(G._this, nSamples, strategy)

	def getNeighborhoodFunction(self):
		"""
		Returns
		-------
		list
			the i-th element denotes the number of node pairs that have a distance at most (i+1)
		"""
		return (<_NeighborhoodFunctionHeuristic*>(self._this)).getNeighborhoodFunction()

cdef extern from "cpp/distance/JaccardDistance.h":
	cdef cppclass _JaccardDistance "NetworKit::JaccardDistance":
		_JaccardDistance(const _Graph& G, const vector[count]& triangles) except +
		void preprocess() except +
		vector[double] getEdgeScores() except +

cdef class JaccardDistance:
	"""
	The Jaccard distance measure assigns to each edge the jaccard coefficient
	of the neighborhoods of the two adjacent nodes.

	Parameters
	----------
	G : Graph
		The graph to calculate Jaccard distances for.
	triangles : vector[count]
		Previously calculated edge triangle counts.
	"""

	cdef _JaccardDistance* _this
	cdef Graph _G
	cdef vector[count] triangles

	def __cinit__(self, Graph G, vector[count] triangles):
		self._G = G
		self._triangles = triangles
		self._this = new _JaccardDistance(G._this, self._triangles)

	def __dealloc__(self):
		del self._this

	def getAttribute(self):
		return self._this.getEdgeScores()

cdef extern from "cpp/distance/AlgebraicDistance.h":
	cdef cppclass _AlgebraicDistance "NetworKit::AlgebraicDistance":
		_AlgebraicDistance(_Graph G, count numberSystems, count numberIterations, double omega, index norm, bool withEdgeScores) except +
		void preprocess() except +
		double distance(node, node) except +
		vector[double] getEdgeScores() except +

cdef class AlgebraicDistance:
	"""
	Algebraic distance assigns a distance value to pairs of nodes
    according to their structural closeness in the graph.
    Algebraic distances will become small within dense subgraphs.

	Parameters
	----------
	G : Graph
		The graph to calculate Jaccard distances for.
	numberSystems : count
	 	Number of vectors/systems used for algebraic iteration.
	numberIterations : count
	 	Number of iterations in each system.
	omega : double
	 	attenuation factor in [0,1] influencing convergence speed.
	norm : index
		The norm factor of the extended algebraic distance.
	withEdgeScores : bool
		calculate array of scores for edges {u,v} that equal ad(u,v)
	"""

	cdef _AlgebraicDistance* _this
	cdef Graph _G

	def __cinit__(self, Graph G, count numberSystems=10, count numberIterations=30, double omega=0.5, index norm=0, bool withEdgeScores=False):
		self._G = G
		self._this = new _AlgebraicDistance(G._this, numberSystems, numberIterations, omega, norm, withEdgeScores)

	def __dealloc__(self):
		del self._this

	def preprocess(self):
		self._this.preprocess()
		return self

	def distance(self, node u, node v):
		return self._this.distance(u, v)

	def getEdgeScores(self):
		return self._this.getEdgeScores()

cdef extern from "cpp/distance/CommuteTimeDistance.h":
	cdef cppclass _CommuteTimeDistance "NetworKit::CommuteTimeDistance":
		_CommuteTimeDistance(_Graph G, double tol) except +
		void run() nogil except +
		void runApproximation() except +
		void runParallelApproximation() except +
		double distance(node, node) except +
		double runSinglePair(node, node) except +
		double runSingleSource(node) except +

cdef class CommuteTimeDistance:
	""" Computes the Euclidean Commute Time Distance between each pair of nodes for an undirected unweighted graph.

	CommuteTimeDistance(G)

	Create CommuteTimeDistance for Graph `G`.

	Parameters
	----------
	G : Graph
		The graph.
	tol: double
	"""
	cdef _CommuteTimeDistance* _this
	cdef Graph _G

	def __cinit__(self,  Graph G, double tol = 0.1):
		self._G = G
		self._this = new _CommuteTimeDistance(G._this, tol)

	def __dealloc__(self):
		del self._this

	def run(self):
		""" This method computes ECTD exactly. """
		with nogil:
			self._this.run()
		return self

	def runApproximation(self):
		""" Computes approximation of the ECTD. """
		return self._this.runApproximation()

	def runParallelApproximation(self):
		""" Computes approximation (in parallel) of the ECTD. """
		return self._this.runParallelApproximation()

	def distance(self, u, v):
		"""  Returns the ECTD between node u and node v.

		u : node
		v : node
		"""
		return self._this.distance(u, v)

	def runSinglePair(self, u, v):
		"""  Returns the ECTD between node u and node v, without preprocessing.

		u : node
		v : node
		"""
		return self._this.runSinglePair(u, v)

	def runSingleSource(self, u):
		"""  Returns the sum of the ECTDs from u, without preprocessing.

		u : node
		"""
		return self._this.runSingleSource(u)
