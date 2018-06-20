'''
	Module: graph
'''

cdef extern from "cpp/graph/Graph.h":
	cdef cppclass _Graph "NetworKit::Graph":
		_Graph() except +
		_Graph(count, bool, bool) except +
		_Graph(const _Graph& other) except +
		_Graph(const _Graph& other, bool weighted, bool directed) except +
		void indexEdges(bool) except +
		bool hasEdgeIds() except +
		edgeid edgeId(node, node) except +
		count numberOfNodes() except +
		count numberOfEdges() except +
		pair[count, count] size() except +
		double density() except +
		index upperNodeIdBound() except +
		index upperEdgeIdBound() except +
		count degree(node u) except +
		count degreeIn(node u) except +
		count degreeOut(node u) except +
		double weightedDegree(node u) except +
		bool isIsolated(node u) except +
		_Graph copyNodes() except +
		node addNode() except +
		void removeNode(node u) except +
		bool hasNode(node u) except +
		void restoreNode(node u) except +
		void append(_Graph) except +
		void merge(_Graph) except +
		void addEdge(node u, node v, edgeweight w) except +
		void setWeight(node u, node v, edgeweight w) except +
		void increaseWeight(node u, node v, edgeweight w) except +
		void removeEdge(node u, node v) except +
		void removeSelfLoops() except +
		void swapEdge(node s1, node t1, node s2, node t2) except +
		void compactEdges() except +
		void sortEdges() except +
		bool hasEdge(node u, node v) except +
		edgeweight weight(node u, node v) except +
		vector[node] nodes() except +
		vector[pair[node, node]] edges() except +
		vector[node] neighbors(node u) except +
		void forEdges[Callback](Callback c) except +
		void forNodes[Callback](Callback c) except +
		void forNodePairs[Callback](Callback c) except +
		void forNodesInRandomOrder[Callback](Callback c) except +
		void forEdgesOf[Callback](node u, Callback c) except +
		void forInEdgesOf[Callback](node u, Callback c) except +
		bool isWeighted() except +
		bool isDirected() except +
		string toString() except +
		string getName() except +
		void setName(string name) except +
		edgeweight totalEdgeWeight() except +
		node randomNode() except +
		node randomNeighbor(node) except +
		pair[node, node] randomEdge(bool) except +
		vector[pair[node, node]] randomEdges(count) except +
		Point[float] getCoordinate(node v) except +
		void setCoordinate(node v, Point[float] value) except +
		void initCoordinates() except +
		count numberOfSelfLoops() except +
		_Graph toUndirected() except +
		_Graph toUnweighted() except +
		_Graph transpose() except +
		void BFSfromNode "BFSfrom"[Callback] (node r, Callback c) except +
		void BFSfrom[Callback](vector[node] startNodes, Callback c) except +
		void BFSEdgesFrom[Callback](node r, Callback c) except +
		void DFSfrom[Callback](node r, Callback c) except +
		void DFSEdgesFrom[Callback](node r, Callback c) except +
		bool checkConsistency() except +
		_Graph subgraphFromNodes(unordered_set[node] nodes)  except +

cdef cppclass EdgeCallBackWrapper:
	void* callback
	__init__(object callback):
		this.callback = <void*>callback
	void cython_call_operator(node u, node v, edgeweight w, edgeid eid):
		cdef bool error = False
		cdef string message
		try:
			(<object>callback)(u, v, w, eid)
		except Exception as e:
			error = True
			message = stdstring("An Exception occurred, aborting execution of iterator: {0}".format(e))
		if (error):
			throw_runtime_error(message)

cdef cppclass NodeCallbackWrapper:
	void* callback
	__init__(object callback):
		this.callback = <void*>callback
	void cython_call_operator(node u):
		cdef bool error = False
		cdef string message
		try:
			(<object>callback)(u)
		except Exception as e:
			error = True
			message = stdstring("An Exception occurred, aborting execution of iterator: {0}".format(e))
		if (error):
			throw_runtime_error(message)

cdef cppclass NodeDistCallbackWrapper:
	void* callback
	__init__(object callback):
		this.callback = <void*>callback
	void cython_call_operator(node u, count dist):
		cdef bool error = False
		cdef string message
		try:
			(<object>callback)(u, dist)
		except Exception as e:
			error = True
			message = stdstring("An Exception occurred, aborting execution of iterator: {0}".format(e))
		if (error):
			throw_runtime_error(message)

cdef cppclass NodePairCallbackWrapper:
	void* callback
	__init__(object callback):
		this.callback = <void*>callback
	void cython_call_operator(node u, node v):
		cdef bool error = False
		cdef string message
		try:
			(<object>callback)(u, v)
		except Exception as e:
			error = True
			message = stdstring("An Exception occurred, aborting execution of iterator: {0}".format(e))
		if (error):
			throw_runtime_error(message)

cdef cppclass NodeVectorCallbackWrapper:
	void* callback
	__init__(object callback):
		this.callback = <void*>callback
	# This is called within the run() method which is nogil!
	void cython_call_operator(const vector[node]& nodes) nogil:
		cdef bool error = False
		cdef string message
		# Acquire gil to allow Python code!
		with gil:
			try:
				(<object>callback)(nodes)
			except Exception as e:
				error = True
				message = stdstring("An Exception occurred, aborting execution of iterator: {0}".format(e))
			if (error):
				throw_runtime_error(message)

cdef class Graph:
	""" An undirected graph (with optional weights) and parallel iterator methods.

		Graph(n=0, weighted=False, directed=False)

		Create a graph of `n` nodes. The graph has assignable edge weights if `weighted` is set to True.
	 	If `weighted` is set to False each edge has edge weight 1.0 and any other weight assignment will
	 	be ignored.

	    Parameters
	    ----------
	    n : count, optional
	    	Number of nodes.
	    weighted : bool, optional
	    	If set to True, the graph can have edge weights other than 1.0.
	    directed : bool, optional
	    	If set to True, the graph will be directed.
	"""
	cdef _Graph _this

	def __cinit__(self, n=0, bool weighted=False, bool directed=False):
		if isinstance(n, Graph):
			self._this = move(_Graph((<Graph>n)._this, weighted, directed))
		else:
			self._this = move(_Graph(<count>n, weighted, directed))

	cdef setThis(self, _Graph& other):
		swap[_Graph](self._this, other)
		return self

	def __copy__(self):
		"""
		Generates a copy of the graph
		"""
		return Graph().setThis(_Graph(self._this))

	def __deepcopy__(self, memo):
		"""
		Generates a (deep) copy of the graph
		"""
		return Graph().setThis(_Graph(self._this))

	def __str__(self):
		return "NetworKit.Graph(name={0}, n={1}, m={2})".format(self.getName(), self.numberOfNodes(), self.numberOfEdges())


	def copyNodes(self):
		"""
		Copies all nodes to a new graph

		Returns
		-------
		Graph
			Graph with the same nodes (without edges)
		"""
		return Graph().setThis(self._this.copyNodes())

	def indexEdges(self, bool force = False):
		"""
		Assign integer ids to edges.

		Parameters
		----------
		force : bool
			Force re-indexing of edges.

		"""
		self._this.indexEdges(force)

	def hasEdgeIds(self):
		"""
		Returns true if edges have been indexed

		Returns
		-------
		bool
			if edges have been indexed
		"""
		return self._this.hasEdgeIds()

	def edgeId(self, node u, node v):
		"""
		Returns
		-------
		edgeid
			id of the edge
		"""
		return self._this.edgeId(u, v)

	def numberOfNodes(self):
		"""
		Get the number of nodes in the graph.

	 	Returns
	 	-------
	 	count
	 		The number of nodes.
		"""
		return self._this.numberOfNodes()

	def numberOfEdges(self):
		"""
		Get the number of edges in the graph.

	 	Returns
	 	-------
	 	count
	 		The number of edges.
		"""
		return self._this.numberOfEdges()

	def size(self):
		"""
		Get the size of the graph.

	 	Returns
	 	-------
	 	tuple
	 		a pair (n, m) where n is the number of nodes and m is the number of edges
		"""
		return self._this.size()


	def density(self):
		"""
		Get the density of the graph.

	 	Returns
	 	-------
	 	double
		"""
		return self._this.density()

	def upperNodeIdBound(self):
		"""
		Get an upper bound for the node ids in the graph

		Returns
		-------
		count
			An upper bound for the node ids in the graph
		"""
		return self._this.upperNodeIdBound()

	def upperEdgeIdBound(self):
		"""
		Get an upper bound for the edge ids in the graph

		Returns
		-------
		count
			An upper bound for the edge ids in the graph
		"""
		return self._this.upperEdgeIdBound()

	def degree(self, u):
		"""
		Get the number of neighbors of `v`.

		Parameters
		----------
		v : node
			Node.

		Returns
		-------
		count
			The number of neighbors.
		"""
		return self._this.degree(u)

	def degreeIn(self, u):
		return self._this.degreeIn(u)

	def degreeOut(self, u):
		return self._this.degreeOut(u)

	def weightedDegree(self, v):
		"""
		Returns the weighted degree of v.

		For directed graphs this is the sum of weights of all outgoing edges of v.

		Parameters
		----------
		v : node
			Node.

		Returns
		-------
		double
			The weighted degree of v.
		"""
		return self._this.weightedDegree(v)

	def isIsolated(self, u):
		"""
		If the node `u` is isolated

		Parameters
		----------
		u : node
			Node.

		Returns
		-------
		bool
			If the node is isolated
		"""
		return self._this.isIsolated(u)

	def addNode(self):
		""" Add a new node to the graph and return it.

		Returns
		-------
		node
			The new node.
	 	"""
		return self._this.addNode()

	def removeNode(self, u):
		""" Remove a node `v` and all incident edges from the graph.

	 	Incoming as well as outgoing edges will be removed.

	 	Parameters
	 	----------
	 	u : node
	 		Node.
		"""
		self._this.removeNode(u)

	def restoreNode(self, u):
		""" Restores a previously deleted node `u` with its previous id in the graph.

	 	Parameters
	 	----------
	 	u : node
	 		Node.
		"""
		self._this.restoreNode(u)

	def hasNode(self, u):
		""" Checks if the Graph has the node `u`, i.e. if `u` hasn't been deleted and is in the range of valid ids.

		Parameters
		----------
		u : node
			Node

		Returns
		-------
		bool
			If the Graph has the node `u`
		"""
		return self._this.hasNode(u)

	def append(self, Graph G):
		""" Appends another graph to this graph as a new subgraph. Performs node id remapping.

		Parameters
		----------
		G : Graph
		"""
		self._this.append(G._this)
		return self

	def merge(self, Graph G):
		""" Modifies this graph to be the union of it and another graph.
			Nodes with the same ids are identified with each other.

		Parameters
		----------
		G : Graph
		"""
		self._this.merge(G._this)
		return self

	def addEdge(self, u, v, w=1.0):
		""" Insert an undirected edge between the nodes `u` and `v`. If the graph is weighted you can optionally
		set a weight for this edge. The default weight is 1.0.
		Caution: It is not checked whether this edge already exists, thus it is possible to create multi-edges.

	 	Parameters
	 	----------
	 	u : node
	 		Endpoint of edge.
 		v : node
 			Endpoint of edge.
		w : edgeweight, optional
			Edge weight.
		"""
		self._this.addEdge(u, v, w)
		return self

	def setWeight(self, u, v, w):
		""" Set the weight of an edge. If the edge does not exist, it will be inserted.

		Parameters
		----------
		u : node
			Endpoint of edge.
		v : node
			Endpoint of edge.
		w : edgeweight
			Edge weight.
		"""
		self._this.setWeight(u, v, w)
		return self

	def increaseWeight(self, u, v, w):
		""" Increase the weight of an edge. If the edge does not exist, it will be inserted.

		Parameters
		----------
		u : node
			Endpoint of edge.
		v : node
			Endpoint of edge.
		w : edgeweight
			Edge weight.
		"""
		self._this.increaseWeight(u, v, w)
		return self

	def removeEdge(self, u, v):
		""" Removes the undirected edge {`u`,`v`}.

		Parameters
		----------
		u : node
			Endpoint of edge.
		v : node
			Endpoint of edge.
		"""
		self._this.removeEdge(u, v)
		return self

	def removeSelfLoops(self):
		""" Removes all self-loops from the graph.
		"""
		self._this.removeSelfLoops()

	def swapEdge(self, node s1, node t1, node s2, node t2):
		"""
		Changes the edge (s1, t1) into (s1, t2) and the edge (s2, t2) into (s2, t1).

		If there are edge weights or edge ids, they are preserved. Note that no check is performed if the swap is actually possible, i.e. does not generate duplicate edges.

		Parameters
		----------
		s1 : node
			Source node of the first edge
		t1 : node
			Target node of the first edge
		s2 : node
			Source node of the second edge
		t2 : node
			Target node of the second edge
		"""
		self._this.swapEdge(s1, t1, s2, t2)
		return self

	def compactEdges(self):
		"""
		Compact the edge storage, this should be called after executing many edge deletions.
		"""
		self._this.compactEdges()

	def sortEdges(self):
		"""
		Sorts the adjacency arrays by node id. While the running time is linear this
		temporarily duplicates the memory.
		"""
		self._this.sortEdges()

	def hasEdge(self, u, v):
		""" Checks if undirected edge {`u`,`v`} exists in the graph.

		Parameters
		----------
		u : node
			Endpoint of edge.
		v : node
			Endpoint of edge.

		Returns
		-------
		bool
			True if the edge exists, False otherwise.
		"""
		return self._this.hasEdge(u, v)

	def weight(self, u, v):
		""" Get edge weight of edge {`u` , `v`}. Returns 0 if edge does not exist.

		Parameters
		----------
		u : node
			Endpoint of edge.
		v : node
			Endpoint of edge.

		Returns
		-------
		edgeweight
			Edge weight of edge {`u` , `v`} or 0 if edge does not exist.
		"""
		return self._this.weight(u, v)

	def nodes(self):
		""" Get list of all nodes.

	 	Returns
	 	-------
	 	list
	 		List of all nodes.
		"""
		return self._this.nodes()

	def edges(self):
		""" Get list of edges as node pairs.

	 	Returns
	 	-------
	 	list
	 		List of edges as node pairs.
		"""
		return self._this.edges()

	def neighbors(self, u):
		""" Get list of neighbors of `u`.

	 	Parameters
	 	----------
	 	u : node
	 		Node.

	 	Returns
	 	-------
	 	list
	 		List of neighbors of `u.
		"""
		return self._this.neighbors(u)

	def forNodes(self, object callback):
		""" Experimental node iterator interface

		Parameters
		----------
		callback : object
			Any callable object that takes the parameter node
		"""
		cdef NodeCallbackWrapper* wrapper
		try:
			wrapper = new NodeCallbackWrapper(callback)
			self._this.forNodes[NodeCallbackWrapper](dereference(wrapper))
		finally:
			del wrapper

	def forNodesInRandomOrder(self, object callback):
		""" Experimental node iterator interface

		Parameters
		----------
		callback : object
			Any callable object that takes the parameter node
		"""
		cdef NodeCallbackWrapper* wrapper
		try:
			wrapper = new NodeCallbackWrapper(callback)
			self._this.forNodesInRandomOrder[NodeCallbackWrapper](dereference(wrapper))
		finally:
			del wrapper

	def forNodePairs(self, object callback):
		""" Experimental node pair iterator interface

		Parameters
		----------
		callback : object
			Any callable object that takes the parameters (node, node)
		"""
		cdef NodePairCallbackWrapper* wrapper
		try:
			wrapper = new NodePairCallbackWrapper(callback)
			self._this.forNodePairs[NodePairCallbackWrapper](dereference(wrapper))
		finally:
			del wrapper

	def forEdges(self, object callback):
		""" Experimental edge iterator interface

		Parameters
		----------
		callback : object
			Any callable object that takes the parameter (node, node, edgeweight, edgeid)
		"""
		cdef EdgeCallBackWrapper* wrapper
		try:
			wrapper = new EdgeCallBackWrapper(callback)
			self._this.forEdges[EdgeCallBackWrapper](dereference(wrapper))
		finally:
			del wrapper

	def forEdgesOf(self, node u, object callback):
		""" Experimental incident (outgoing) edge iterator interface

		Parameters
		----------
		u : node
			The node of which incident edges shall be passed to the callback
		callback : object
			Any callable object that takes the parameter (node, node, edgeweight, edgeid)
		"""
		cdef EdgeCallBackWrapper* wrapper
		try:
			wrapper = new EdgeCallBackWrapper(callback)
			self._this.forEdgesOf[EdgeCallBackWrapper](u, dereference(wrapper))
		finally:
			del wrapper

	def forInEdgesOf(self, node u, object callback):
		""" Experimental incident incoming edge iterator interface

		Parameters
		----------
		u : node
			The node of which incident edges shall be passed to the callback
		callback : object
			Any callable object that takes the parameter (node, node, edgeweight, edgeid)
		"""
		cdef EdgeCallBackWrapper* wrapper
		try:
			wrapper = new EdgeCallBackWrapper(callback)
			self._this.forInEdgesOf[EdgeCallBackWrapper](u, dereference(wrapper))
		finally:
			del wrapper

	def toUndirected(self):
		"""
		Return an undirected version of this graph.

	 	Returns
	 	-------
			undirected graph.
		"""
		return Graph().setThis(self._this.toUndirected())


	def toUnweighted(self):
		"""
		Return an unweighted version of this graph.

	 	Returns
	 	-------
			graph.
		"""
		return Graph().setThis(self._this.toUnweighted())

	def transpose(self):
		"""
		Return the transpose of this (directed) graph.

		Returns
		-------
			directed graph.
		"""
		return Graph().setThis(self._this.transpose())

	def isWeighted(self):
		"""
		Returns
		-------
		bool
			True if this graph supports edge weights other than 1.0.
		"""
		return self._this.isWeighted()

	def isDirected(self):
		return self._this.isDirected()

	def toString(self):
		""" Get a string representation of the graph.

		Returns
		-------
		string
			A string representation of the graph.
		"""
		return self._this.toString()

	def getName(self):
		""" Get the name of the graph.

		Returns
		-------
		string
			The name of the graph.
		"""
		return pystring(self._this.getName())

	def setName(self, name):
		""" Set name of graph to `name`.

		Parameters
		----------
		name : string
			The name.
		"""
		self._this.setName(stdstring(name))

	def totalEdgeWeight(self):
		""" Get the sum of all edge weights.

		Returns
		-------
		edgeweight
			The sum of all edge weights.
		"""
		return self._this.totalEdgeWeight()

	def randomNode(self):
		""" Get a random node of the graph.

		Returns
		-------
		node
			A random node.
		"""
		return self._this.randomNode()

	def randomNeighbor(self, u):
		""" Get a random neighbor of `v` and `none` if degree is zero.

		Parameters
		----------
		v : node
			Node.

		Returns
		-------
		node
			A random neighbor of `v.
		"""
		return self._this.randomNeighbor(u)

	def randomEdge(self, bool uniformDistribution = False):
		""" Get a random edge of the graph.

		Parameters
		----------
		uniformDistribution : bool
			If the distribution of the edge shall be uniform

		Returns
		-------
		pair
			Random random edge.

		Notes
		-----
		Fast, but not uniformly random if uniformDistribution is not set,
		slow and uniformly random otherwise.
		"""
		return self._this.randomEdge(uniformDistribution)

	def randomEdges(self, count numEdges):
		""" Returns a list with numEdges random edges. The edges are chosen uniformly at random.

		Parameters
		----------
		numEdges : count
			The number of edges to choose.

		Returns
		-------
		list of pairs
			The selected edges.
		"""
		return self._this.randomEdges(numEdges)

	def getCoordinate(self, v):
		"""
		DEPRECATED: Coordinates should be handled outside the Graph class
		 like general node attributes.

		Get the coordinates of node v.
		Parameters
		----------
		v : node
			Node.

		Returns
		-------
		pair[float, float]
			x and y coordinates of v.
		"""

		return (self._this.getCoordinate(v)[0], self._this.getCoordinate(v)[1])

	def setCoordinate(self, v, value):
		"""
		DEPRECATED: Coordinates should be handled outside the Graph class
		 like general node attributes.

		Set the coordinates of node v.
		Parameters
		----------
		v : node
			Node.
		value : pair[float, float]
			x and y coordinates of v.
		"""
		cdef Point[float] p = Point[float](value[0], value[1])
		self._this.setCoordinate(v, p)

	def initCoordinates(self):
		"""
		DEPRECATED: Coordinates should be handled outside the Graph class
		 like general node attributes.
		"""
		self._this.initCoordinates()

	def numberOfSelfLoops(self):
		""" Get number of self-loops, i.e. edges {v, v}.
		Returns
		-------
		count
			number of self-loops.
		"""
		return self._this.numberOfSelfLoops()

	def BFSfrom(self, start, object callback):
		""" Experimental BFS search interface

		Parameters
		----------
		start: node or list[node]
			One or more start nodes from which the BFS shall be started
		callback : object
			Any callable object that takes the parameter (node, count) (the second parameter is the depth)
		"""
		cdef NodeDistCallbackWrapper *wrapper
		try:
			wrapper = new NodeDistCallbackWrapper(callback)
			try:
				self._this.BFSfromNode[NodeDistCallbackWrapper](<node?>start, dereference(wrapper))
			except TypeError:
				self._this.BFSfrom[NodeDistCallbackWrapper](<vector[node]?>start, dereference(wrapper))
		finally:
			del wrapper

	def BFSEdgesFrom(self, node start, object callback):
		""" Experimental BFS search interface that passes edges that are part of the BFS tree to the callback

		Parameters
		----------
		start: node
			The start node from which the BFS shall be started
		callback : object
			Any callable object that takes the parameter (node, node)
		"""
		cdef EdgeCallBackWrapper *wrapper
		try:
			wrapper = new EdgeCallBackWrapper(callback)
			self._this.BFSEdgesFrom[EdgeCallBackWrapper](start, dereference(wrapper))
		finally:
			del wrapper

	def DFSfrom(self, node start, object callback):
		""" Experimental DFS search interface

		Parameters
		----------
		start: node
			The start node from which the DFS shall be started
		callback : object
			Any callable object that takes the parameter node
		"""
		cdef NodeCallbackWrapper *wrapper
		try:
			wrapper = new NodeCallbackWrapper(callback)
			self._this.DFSfrom[NodeCallbackWrapper](start, dereference(wrapper))
		finally:
			del wrapper

	def DFSEdgesFrom(self, node start, object callback):
		""" Experimental DFS search interface that passes edges that are part of the DFS tree to the callback

		Parameters
		----------
		start: node
			The start node from which the DFS shall be started
		callback : object
			Any callable object that takes the parameter (node, node)
		"""
		cdef NodePairCallbackWrapper *wrapper
		try:
			wrapper = new NodePairCallbackWrapper(callback)
			self._this.DFSEdgesFrom(start, dereference(wrapper))
		finally:
			del wrapper

	def checkConsistency(self):
		"""
		Check for invalid graph states, such as multi-edges.
		"""
		return self._this.checkConsistency()


	def subgraphFromNodes(self, nodes):
		""" Create a subgraph induced by the set `nodes`.

		Parameters
		----------
		nodes : list
			A subset of nodes of `G` which induce the subgraph.

		Returns
		-------
		Graph
			The subgraph induced by `nodes`.

		Notes
		-----
		The returned graph G' is isomorphic (structurally identical) to the subgraph in G,
		but node indices are not preserved.
		"""
		cdef unordered_set[node] nnodes
		for node in nodes:
			nnodes.insert(node);
		return Graph().setThis(self._this.subgraphFromNodes(nnodes))

cdef extern from "cpp/graph/UnionMaximumSpanningForest.h":
	cdef cppclass _UnionMaximumSpanningForest "NetworKit::UnionMaximumSpanningForest"(_Algorithm):
		_UnionMaximumSpanningForest(_Graph) except +
		_UnionMaximumSpanningForest(_Graph, vector[double]) except +
		_Graph getUMSF(bool move) except +
		vector[bool] getAttribute(bool move) except +
		bool inUMSF(edgeid eid) except +
		bool inUMSF(node u, node v) except +

cdef class UnionMaximumSpanningForest(Algorithm):
	"""
	Union maximum-weight spanning forest algorithm, computes the union of all maximum-weight spanning forests using Kruskal's algorithm.

	Parameters
	----------
	G : Graph
		The input graph.
	attribute : list
		If given, this edge attribute is used instead of the edge weights.
	"""
	cdef Graph _G

	def __cinit__(self, Graph G not None, vector[double] attribute = vector[double]()):
		self._G = G

		if attribute.empty():
			self._this = new _UnionMaximumSpanningForest(G._this)
		else:
			self._this = new _UnionMaximumSpanningForest(G._this, attribute)

	def getUMSF(self, bool move = False):
		"""
		Gets the union of all maximum-weight spanning forests as graph.

		Parameters
		----------
		move : boolean
			If the graph shall be moved out of the algorithm instance.

		Returns
		-------
		Graph
			The calculated union of all maximum-weight spanning forests.
		"""
		return Graph().setThis((<_UnionMaximumSpanningForest*>(self._this)).getUMSF(move))

	def getAttribute(self, bool move = False):
		"""
		Get a boolean attribute that indicates for each edge if it is part of any maximum-weight spanning forest.

		This attribute is only calculated and can thus only be request if the supplied graph has edge ids.

		Parameters
		----------
		move : boolean
			If the attribute shall be moved out of the algorithm instance.

		Returns
		-------
		list
			The list with the boolean attribute for each edge.
		"""
		return (<_UnionMaximumSpanningForest*>(self._this)).getAttribute(move)

	def inUMST(self, node u, node v = _none):
		"""
		Checks if the edge (u, v) or the edge with id u is part of any maximum-weight spanning forest.

		Parameters
		----------
		u : node or edgeid
			The first node of the edge to check or the edge id of the edge to check
		v : node
			The second node of the edge to check (only if u is not an edge id)

		Returns
		-------
		boolean
			If the edge is part of any maximum-weight spanning forest.
		"""
		if v == _none:
			return (<_UnionMaximumSpanningForest*>(self._this)).inUMSF(u)
		else:
			return (<_UnionMaximumSpanningForest*>(self._this)).inUMSF(u, v)

cdef extern from "cpp/graph/RandomMaximumSpanningForest.h":
	cdef cppclass _RandomMaximumSpanningForest "NetworKit::RandomMaximumSpanningForest"(_Algorithm):
		_RandomMaximumSpanningForest(_Graph) except +
		_RandomMaximumSpanningForest(_Graph, vector[double]) except +
		void run() except +
		_Graph getMSF(bool move) except +
		vector[bool] getAttribute(bool move) except +
		bool inMSF(edgeid eid) except +
		bool inMSF(node u, node v) except +

cdef class RandomMaximumSpanningForest(Algorithm):
	"""
	Computes a random maximum-weight spanning forest using Kruskal's algorithm by randomizing the order of edges of the same weight.

	Parameters
	----------
	G : Graph
		The input graph.
	attribute : list
		If given, this edge attribute is used instead of the edge weights.
	"""
	cdef vector[double] _attribute
	cdef Graph _G

	def __cinit__(self, Graph G not None, vector[double] attribute = vector[double]()):
		self._G = G
		if attribute.empty():
			self._this = new _RandomMaximumSpanningForest(G._this)
		else:
			self._attribute = move(attribute)
			self._this = new _RandomMaximumSpanningForest(G._this, self._attribute)

	def getMSF(self, bool move = False):
		"""
		Gets the calculated maximum-weight spanning forest as graph.

		Parameters
		----------
		move : boolean
			If the graph shall be moved out of the algorithm instance.

		Returns
		-------
		Graph
			The calculated maximum-weight spanning forest.
		"""
		return Graph().setThis((<_RandomMaximumSpanningForest*>(self._this)).getMSF(move))

	def getAttribute(self, bool move = False):
		"""
		Get a boolean attribute that indicates for each edge if it is part of the calculated maximum-weight spanning forest.

		This attribute is only calculated and can thus only be request if the supplied graph has edge ids.

		Parameters
		----------
		move : boolean
			If the attribute shall be moved out of the algorithm instance.

		Returns
		-------
		list
			The list with the boolean attribute for each edge.
		"""
		return (<_RandomMaximumSpanningForest*>(self._this)).getAttribute(move)

	def inMSF(self, node u, node v = _none):
		"""
		Checks if the edge (u, v) or the edge with id u is part of the calculated maximum-weight spanning forest.

		Parameters
		----------
		u : node or edgeid
			The first node of the edge to check or the edge id of the edge to check
		v : node
			The second node of the edge to check (only if u is not an edge id)

		Returns
		-------
		boolean
			If the edge is part of the calculated maximum-weight spanning forest.
		"""
		if v == _none:
			return (<_RandomMaximumSpanningForest*>(self._this)).inMSF(u)
		else:
			return (<_RandomMaximumSpanningForest*>(self._this)).inMSF(u, v)

cdef extern from "cpp/graph/GraphTools.h" namespace "NetworKit::GraphTools":
	_Graph getCompactedGraph(_Graph G, unordered_map[node,node]) nogil except +
	unordered_map[node,node] getContinuousNodeIds(_Graph G) nogil except +
	unordered_map[node,node] getRandomContinuousNodeIds(_Graph G) nogil except +

cdef class GraphTools:
	@staticmethod
	def getCompactedGraph(Graph graph, nodeIdMap):
		"""
			Computes a graph with the same structure but with continuous node ids.
		"""
		cdef unordered_map[node,node] cNodeIdMap
		for key in nodeIdMap:
			cNodeIdMap[key] = nodeIdMap[key]
		return Graph().setThis(getCompactedGraph(graph._this,cNodeIdMap))

	@staticmethod
	def getContinuousNodeIds(Graph graph):
		"""
			Computes a map of node ids to continuous node ids.
		"""
		cdef unordered_map[node,node] cResult
		with nogil:
			cResult = getContinuousNodeIds(graph._this)
		result = dict()
		for elem in cResult:
			result[elem.first] = elem.second
		return result

	@staticmethod
	def getRandomContinuousNodeIds(Graph graph):
		"""
			Computes a map of node ids to continuous, randomly permutated node ids.
		"""
		cdef unordered_map[node,node] cResult
		with nogil:
			cResult = getRandomContinuousNodeIds(graph._this)
		result = dict()
		for elem in cResult:
			result[elem.first] = elem.second
		return result
