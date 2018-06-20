'''
	Module: Misc
	Collection of several isolated classes
'''
cdef extern from "cpp/correlation/Assortativity.h":
	cdef cppclass _Assortativity "NetworKit::Assortativity"(_Algorithm):
		_Assortativity(_Graph, vector[double]) except +
		_Assortativity(_Graph, _Partition) except +
		void run() nogil except +
		double getCoefficient() except +

cdef class Assortativity(Algorithm):
	""" """
	cdef Graph G
	cdef vector[double] attribute
	cdef Partition partition

	def __cinit__(self, Graph G, data):
		if isinstance(data, Partition):
			self._this = new _Assortativity(G._this, (<Partition>data)._this)
			self.partition = <Partition>data
		else:
			self.attribute = <vector[double]?>data
			self._this = new _Assortativity(G._this, self.attribute)
		self.G = G

	def getCoefficient(self):
		return (<_Assortativity*>(self._this)).getCoefficient()

cdef class JaccardSimilarityAttributizer:
	"""
	The Jaccard similarity measure assigns to each edge (1 - the jaccard coefficient
	of the neighborhoods of the two adjacent nodes).

	Parameters
	----------
	G : Graph
		The graph to calculate Jaccard similarities for.
	triangles : vector[count]
		Previously calculated edge triangle counts.
	"""

	cdef _JaccardDistance* _this
	cdef Graph _G
	cdef vector[count] _triangles

	def __cinit__(self, Graph G, vector[count] triangles):
		self._G = G
		self._triangles = triangles
		self._this = new _JaccardDistance(G._this, self._triangles)

	def __dealloc__(self):
		del self._this

	def getAttribute(self):
		#convert distance to similarity
		self._this.preprocess()
		return [1 - x for x in self._this.getEdgeScores()]

# stats

def gini(values):
	"""
	Computes the Gini coefficient for the distribution given as a list of values.
	"""
	sorted_list = sorted(values)
	height, area = 0, 0
	for value in sorted_list:
		height += value
		area += height - value / 2.
	fair_area = height * len(values) / 2
	return (fair_area - area) / fair_area

# simulation
cdef extern from "cpp/simulation/EpidemicSimulationSEIR.h":
	cdef cppclass _EpidemicSimulationSEIR "NetworKit::EpidemicSimulationSEIR" (_Algorithm):
		_EpidemicSimulationSEIR(_Graph, count, double, count, count, node) except +
		vector[vector[count]] getData() except +

cdef class EpidemicSimulationSEIR(Algorithm):
	"""
 	Parameters
 	----------
 	G : Graph
 		The graph.
 	tMax : count
 		max. number of timesteps
	transP : double
		transmission probability
	eTime : count
		exposed time
	iTime : count
		infectious time
	zero : node
		starting node
	"""
	cdef Graph G
	def __cinit__(self, Graph G, count tMax, double transP=0.5, count eTime=2, count iTime=7, node zero=none):
		self.G = G
		self._this = new _EpidemicSimulationSEIR(G._this, tMax, transP, eTime, iTime, zero)
	def getData(self):
		return pandas.DataFrame((<_EpidemicSimulationSEIR*>(self._this)).getData(), columns=["zero", "time", "state", "count"])
