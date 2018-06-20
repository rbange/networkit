'''
	Module: components
'''

cdef extern from "cpp/components/ConnectedComponents.h":
	cdef cppclass _ConnectedComponents "NetworKit::ConnectedComponents":
		_ConnectedComponents(_Graph G) except +
		void run() nogil except +
		count numberOfComponents() except +
		count componentOfNode(node query) except +
		_Partition getPartition() except +
		map[index, count] getComponentSizes() except +

cdef class ConnectedComponents:
	""" Determines the connected components and associated values for an undirected graph.

	ConnectedComponents(G)

	Create ConnectedComponents for Graph `G`.

	Parameters
	----------
	G : Graph
		The graph.
	"""
	cdef _ConnectedComponents* _this
	cdef Graph _G

	def __cinit__(self,  Graph G):
		self._G = G
		self._this = new _ConnectedComponents(G._this)

	def __dealloc__(self):
		del self._this

	def run(self):
		""" This method determines the connected components for the graph given in the constructor. """
		with nogil:
			self._this.run()
		return self

	def getPartition(self):
		""" Get a Partition that represents the components.

		Returns
		-------
		Partition
			A partition representing the found components.
		"""
		return Partition().setThis(self._this.getPartition())

	def numberOfComponents(self):
		""" Get the number of connected components.

		Returns
		-------
		count:
			The number of connected components.
		"""
		return self._this.numberOfComponents()

	def componentOfNode(self, v):
		"""  Get the the component in which node `v` is situated.

		v : node
			The node whose component is asked for.
		"""
		return self._this.componentOfNode(v)

	def getComponentSizes(self):
		return self._this.getComponentSizes()

cdef extern from "cpp/components/ParallelConnectedComponents.h":
	cdef cppclass _ParallelConnectedComponents "NetworKit::ParallelConnectedComponents":
		_ParallelConnectedComponents(_Graph G, bool coarsening) except +
		void run() nogil except +
		count numberOfComponents() except +
		count componentOfNode(node query) except +
		_Partition getPartition() except +

cdef class ParallelConnectedComponents:
	""" Determines the connected components and associated values for
		an undirected graph.
	"""
	cdef _ParallelConnectedComponents* _this
	cdef Graph _G

	def __cinit__(self,  Graph G, coarsening=True	):
		self._G = G
		self._this = new _ParallelConnectedComponents(G._this, coarsening)

	def __dealloc__(self):
		del self._this

	def run(self):
		with nogil:
			self._this.run()
		return self

	def getPartition(self):
		return Partition().setThis(self._this.getPartition())

	def numberOfComponents(self):
		return self._this.numberOfComponents()

	def componentOfNode(self, v):
		return self._this.componentOfNode(v)

cdef extern from "cpp/components/StronglyConnectedComponents.h":
	cdef cppclass _StronglyConnectedComponents "NetworKit::StronglyConnectedComponents":
		_StronglyConnectedComponents(_Graph G, bool iterativeAlgo) except +
		void run() nogil except +
		void runIteratively() nogil except +
		void runRecursively() nogil except +
		count numberOfComponents() except +
		count componentOfNode(node query) except +
		_Partition getPartition() except +

cdef class StronglyConnectedComponents:
	""" Determines the connected components and associated values for
		a directed graph.

		By default, the iterative implementation is used. If edges on the graph have been removed,
		you should switch to the recursive implementation.

		Parameters
		----------
		G : Graph
			The graph.
		iterativeAlgo : boolean
			Specifies which implementation to use, by default True for the iterative implementation.
	"""
	cdef _StronglyConnectedComponents* _this
	cdef Graph _G

	def __cinit__(self, Graph G, iterativeAlgo = True):
		self._G = G
		self._this = new _StronglyConnectedComponents(G._this, iterativeAlgo)

	def __dealloc__(self):
		del self._this

	def run(self):
		with nogil:
			self._this.run()
		return self

	def runIteratively(self):
		with nogil:
			self._this.runIteratively()
		return self

	def runRecursively(self):
		with nogil:
			self._this.runRecursively()
		return self

	def getPartition(self):
		return Partition().setThis(self._this.getPartition())

	def numberOfComponents(self):
		return self._this.numberOfComponents()

	def componentOfNode(self, v):
		return self._this.componentOfNode(v)

cdef extern from "cpp/components/WeaklyConnectedComponents.h":
	cdef cppclass _WeaklyConnectedComponents "NetworKit::WeaklyConnectedComponents":
		_WeaklyConnectedComponents(_Graph G) except +
		void run() nogil except +
		count numberOfComponents() except +
		count componentOfNode(node query) except +
		map[index, count] getComponentSizes() except +
		vector[vector[node]] getComponents() except +

cdef class WeaklyConnectedComponents:
	""" Determines the weakly connected components of a directed graph.

		Parameters
		----------
		G : Graph
			The graph.
	"""
	cdef _WeaklyConnectedComponents* _this
	cdef Graph _G

	def __cinit__(self, Graph G):
		self._G = G
		self._this = new _WeaklyConnectedComponents(G._this)

	def __dealloc__(self):
		del self._this

	def run(self):
		with nogil:
			self._this.run()
		return self

	def numberOfComponents(self):
		""" Returns the number of components.

			Returns
			count
				The number of components.
		"""
		return self._this.numberOfComponents()

	def componentOfNode(self, v):
		""" Returns the the component in which node @a u is.

			Parameters
			----------
			v : node
				The node.
		"""
		return self._this.componentOfNode(v)

	def getComponentSizes(self):
		""" Returns the map from component to size.

			Returns
			map[index, count]
			 	A map that maps each component to its size.
		"""
		return self._this.getComponentSizes()

	def getComponents(self):
		""" Returns all the components, each stored as (unordered) set of nodes.

			Returns
			vector[vector[node]]
				A vector of vectors. Each inner vector contains all the nodes inside the component.
		"""
		return self._this.getComponents()

cdef extern from "cpp/components/BiconnectedComponents.h":
	cdef cppclass _BiconnectedComponents "NetworKit::BiconnectedComponents":
		_BiconnectedComponents(_Graph G) except +
		void run() nogil except +
		count numberOfComponents() except +
		map[count, count] getComponentSizes() except +
		vector[vector[node]] getComponents() except +

cdef class BiconnectedComponents:
	""" Determines the biconnected components of an undirected graph as defined in
		Tarjan, Robert. Depth-First Search and Linear Graph Algorithms. SIAM J.
		Comput. Vol 1, No. 2, June 1972.


		Parameters
		----------
		G : Graph
			The graph.
	"""
	cdef _BiconnectedComponents* _this
	cdef Graph _G

	def __cinit__(self, Graph G):
		self._G = G
		self._this = new _BiconnectedComponents(G._this)

	def __dealloc__(self):
		del self._this

	def run(self):
		""" Computes the biconnected components of the graph given in the
			constructor.
		"""
		with nogil:
			self._this.run()
		return self

	def numberOfComponents(self):
		""" Returns the number of components.

			Returns
			count
				The number of components.
		"""
		return self._this.numberOfComponents()

	def getComponentSizes(self):
		""" Returns the map from component to size.

			Returns
			map[count, count]
			A map that maps each component to its size.
		"""
		return self._this.getComponentSizes()

	def getComponents(self):
		""" Returns all the components, each stored as (unordered) set of nodes.

			Returns
			vector[vector[node]]
				A vector of vectors. Each inner vector contains all the nodes inside the component.
		"""
		return self._this.getComponents()

cdef extern from "cpp/components/DynConnectedComponents.h":
	cdef cppclass _DynConnectedComponents "NetworKit::DynConnectedComponents":
		_DynConnectedComponents(_Graph G) except +
		void run() nogil except +
		void update(_GraphEvent) except +
		void updateBatch(vector[_GraphEvent]) except +
		count numberOfComponents() except +
		count componentOfNode(node query) except +
		map[index, count] getComponentSizes() except +
		vector[vector[node]] getComponents() except +

cdef class DynConnectedComponents:
	""" Determines and updates the connected components of an undirected graph.

		Parameters
		----------
		G : Graph
			The graph.
	"""
	cdef _DynConnectedComponents* _this
	cdef Graph _G

	def __cinit__(self, Graph G):
		self._G = G
		self._this = new _DynConnectedComponents(G._this)

	def __dealloc__(self):
		del self._this

	def run(self):
		with nogil:
			self._this.run()
		return self

	def numberOfComponents(self):
		""" Returns the number of components.

			Returns
			count
				The number of components.
		"""
		return self._this.numberOfComponents()

	def componentOfNode(self, v):
		""" Returns the the component in which node @a u is.

			Parameters
			----------
			v : node
				The node.
		"""
		return self._this.componentOfNode(v)

	def getComponentSizes(self):
		""" Returns the map from component to size.

			Returns
			map[index, count]
			 	A map that maps each component to its size.
		"""
		return self._this.getComponentSizes()

	def getComponents(self):
		""" Returns all the components, each stored as (unordered) set of nodes.

			Returns
			vector[vector[node]]
				A vector of vectors. Each inner vector contains all the nodes inside the component.
		"""
		return self._this.getComponents()

	def update(self, event):
		""" Updates the connected components after an edge insertion or
			deletion.

			Parameters
			----------
			event : GraphEvent
				The event that happened (edge deletion or insertion).
		"""
		self._this.update(_GraphEvent(event.type, event.u, event.v, event.w))

	def updateBatch(self, batch):
		""" Updates the connected components after a batch of edge insertions or
			deletions.

			Parameters
			----------
			batch : vector[GraphEvent]
				A vector that contains a batch of edge insertions or deletions.
		"""
		cdef vector[_GraphEvent] _batch
		for event in batch:
			_batch.push_back(_GraphEvent(event.type, event.u, event.v, event.w))
		self._this.updateBatch(_batch)

cdef extern from "cpp/components/DynWeaklyConnectedComponents.h":
	cdef cppclass _DynWeaklyConnectedComponents "NetworKit::DynWeaklyConnectedComponents":
		_DynWeaklyConnectedComponents(_Graph G) except +
		void run() nogil except +
		void update(_GraphEvent) except +
		void updateBatch(vector[_GraphEvent]) except +
		count numberOfComponents() except +
		count componentOfNode(node query) except +
		map[index, count] getComponentSizes() except +
		vector[vector[node]] getComponents() except +

cdef class DynWeaklyConnectedComponents:
	""" Determines and updates the weakly connected components of a directed graph.

		Parameters
		----------
		G : Graph
			The graph.
	"""
	cdef _DynWeaklyConnectedComponents* _this
	cdef Graph _G

	def __cinit__(self, Graph G):
		self._G = G
		self._this = new _DynWeaklyConnectedComponents(G._this)

	def __dealloc__(self):
		del self._this

	def run(self):
		with nogil:
			self._this.run()
		return self

	def numberOfComponents(self):
		""" Returns the number of components.

			Returns
			count
				The number of components.
		"""
		return self._this.numberOfComponents()

	def componentOfNode(self, v):
		""" Returns the the component in which node @a u is.

			Parameters
			----------
			v : node
				The node.
		"""
		return self._this.componentOfNode(v)

	def getComponentSizes(self):
		""" Returns the map from component to size.

			Returns
			map[index, count]
			 	A map that maps each component to its size.
		"""
		return self._this.getComponentSizes()

	def getComponents(self):
		""" Returns all the components, each stored as (unordered) set of nodes.

			Returns
			vector[vector[node]]
				A vector of vectors. Each inner vector contains all the nodes
				inside the component.

		"""
		return self._this.getComponents()

	def update(self, event):
		""" Updates the connected components after an edge insertion or
			deletion.

			Parameters
			----------
			event : GraphEvent
				The event that happened (edge deletion or insertion).
		"""
		self._this.update(_GraphEvent(event.type, event.u, event.v, event.w))

	def updateBatch(self, batch):
		""" Updates the connected components after a batch of edge insertions or
			deletions.

			Parameters
			----------
			batch : vector[GraphEvent]
				A vector that contains a batch of edge insertions or deletions.
		"""
		cdef vector[_GraphEvent] _batch
		for event in batch:
			_batch.push_back(_GraphEvent(event.type, event.u, event.v, event.w))
		self._this.updateBatch(_batch)
