'''
	Module: Edgescore
'''

cdef extern from "cpp/edgescores/EdgeScore.h":
	cdef cppclass _EdgeScore "NetworKit::EdgeScore"[T](_Algorithm):
		_EdgeScore(const _Graph& G) except +
		vector[T] scores() except +
		T score(edgeid eid) except +
		T score(node u, node v) except +

cdef class EdgeScore(Algorithm):
	"""
	TODO DOCSTIRNG
	"""
	cdef Graph _G

	cdef bool isDoubleValue(self):
		raise RuntimeError("Implement in subclass")

	def __init__(self, *args, **namedargs):
		if type(self) == EdgeScore:
			raise RuntimeError("Error, you may not use EdgeScore directly, use a sub-class instead")

	def __dealloc__(self):
		self._G = None # just to be sure the graph is deleted

	def score(self, u, v = None):
		if v is None:
			if self.isDoubleValue():
				return (<_EdgeScore[double]*>(self._this)).score(u)
			else:
				return (<_EdgeScore[count]*>(self._this)).score(u)
		else:
			if self.isDoubleValue():
				return (<_EdgeScore[double]*>(self._this)).score(u, v)
			else:
				return (<_EdgeScore[count]*>(self._this)).score(u, v)

	def scores(self):
		if self.isDoubleValue():
			return (<_EdgeScore[double]*>(self._this)).scores()
		else:
			return (<_EdgeScore[count]*>(self._this)).scores()

cdef extern from "cpp/edgescores/ChibaNishizekiTriangleEdgeScore.h":
	cdef cppclass _ChibaNishizekiTriangleEdgeScore "NetworKit::ChibaNishizekiTriangleEdgeScore"(_EdgeScore[count]):
		_ChibaNishizekiTriangleEdgeScore(const _Graph& G) except +

cdef class ChibaNishizekiTriangleEdgeScore(EdgeScore):
	"""
	Calculates for each edge the number of triangles it is embedded in.

	Parameters
	----------
	G : Graph
		The graph to count triangles on.
	"""

	def __cinit__(self, Graph G):
		"""
		G : Graph
			The graph to count triangles on.
		"""
		self._G = G
		self._this = new _ChibaNishizekiTriangleEdgeScore(G._this)

	cdef bool isDoubleValue(self):
		return False

cdef extern from "cpp/edgescores/ChibaNishizekiQuadrangleEdgeScore.h":
	cdef cppclass _ChibaNishizekiQuadrangleEdgeScore "NetworKit::ChibaNishizekiQuadrangleEdgeScore"(_EdgeScore[count]):
		_ChibaNishizekiQuadrangleEdgeScore(const _Graph& G) except +

cdef class ChibaNishizekiQuadrangleEdgeScore(EdgeScore):
	"""
	Calculates for each edge the number of quadrangles (circles of length 4) it is embedded in.

	Parameters
	----------
	G : Graph
		The graph to count quadrangles on.
	"""

	def __cinit__(self, Graph G):
		"""
		Parameters
		----------
		G : Graph
			The graph to count quadrangles on.
		"""
		self._G = G
		self._this = new _ChibaNishizekiQuadrangleEdgeScore(G._this)

	cdef bool isDoubleValue(self):
		return False

cdef extern from "cpp/edgescores/TriangleEdgeScore.h":
	cdef cppclass _TriangleEdgeScore "NetworKit::TriangleEdgeScore"(_EdgeScore[double]):
		_TriangleEdgeScore(const _Graph& G) except +

cdef class TriangleEdgeScore(EdgeScore):
	"""
	Triangle counting.

	Parameters
	----------
	G : Graph
		The graph to count triangles on.
	"""

	def __cinit__(self, Graph G):
		"""
		Parameters
		----------
		G : Graph
			The graph to count triangles on.
		"""
		self._G = G
		self._this = new _TriangleEdgeScore(G._this)

	cdef bool isDoubleValue(self):
		return False

cdef extern from "cpp/edgescores/EdgeScoreLinearizer.h":
	cdef cppclass _EdgeScoreLinearizer "NetworKit::EdgeScoreLinearizer"(_EdgeScore[double]):
		_EdgeScoreLinearizer(const _Graph& G, const vector[double]& attribute, bool inverse) except +

cdef class EdgeScoreLinearizer(EdgeScore):
	"""
	Linearizes a score such that values are evenly distributed between 0 and 1.

	Parameters
	----------
	G : Graph
		The input graph.
	a : vector[double]
		Edge score that shall be linearized.
	"""
	cdef vector[double] _score

	def __cinit__(self, Graph G, vector[double] score, inverse = False):
		self._G = G
		self._score = score
		self._this = new _EdgeScoreLinearizer(G._this, self._score, inverse)

	cdef bool isDoubleValue(self):
		return True

cdef extern from "cpp/edgescores/EdgeScoreNormalizer.h":
	cdef cppclass _EdgeScoreNormalizer "NetworKit::EdgeScoreNormalizer"[T](_EdgeScore[double]):
		_EdgeScoreNormalizer(const _Graph&, vector[T]&, bool inverse, double lower, double upper) except +

cdef class EdgeScoreNormalizer(EdgeScore):
	"""
	Normalize an edge score such that it is in a certain range.

	Parameters
	----------
	G : Graph
		The graph the edge score is defined on.
	score : vector[double]
		The edge score to normalize.
	inverse
		Set to True in order to inverse the resulting score.
	lower
		Lower bound of the target range.
	upper
		Upper bound of the target range.
	"""
	cdef vector[double] _inScoreDouble
	cdef vector[count] _inScoreCount

	def __cinit__(self, Graph G not None, score, bool inverse = False, double lower = 0.0, double upper = 1.0):
		self._G = G
		try:
			self._inScoreDouble = <vector[double]?>score
			self._this = new _EdgeScoreNormalizer[double](G._this, self._inScoreDouble, inverse, lower, upper)
		except TypeError:
			try:
				self._inScoreCount = <vector[count]?>score
				self._this = new _EdgeScoreNormalizer[count](G._this, self._inScoreCount, inverse, lower, upper)
			except TypeError:
				raise TypeError("score must be either a vector of integer or float")

	cdef bool isDoubleValue(self):
		return True

cdef extern from "cpp/edgescores/EdgeScoreBlender.h":
	cdef cppclass _EdgeScoreBlender "NetworKit::EdgeScoreBlender"(_EdgeScore[double]):
		_EdgeScoreBlender(const _Graph&, const vector[double]&, const vector[double]&, const vector[bool]&) except +

cdef class EdgeScoreBlender(EdgeScore):
	"""
	Blends two attribute vectors, the value is chosen depending on the supplied boolean vector

	Parameters
	----------
	G : Graph
		The graph for which the attribute shall be blended
	attribute0 : vector[double]
		The first attribute (chosen for selection[eid] == false)
	attribute1 : vector[double]
		The second attribute (chosen for selection[eid] == true)
	selection : vector[bool]
		The selection vector
	"""
	cdef vector[double] _attribute0
	cdef vector[double] _attribute1
	cdef vector[bool] _selection

	def __cinit__(self, Graph G not None, vector[double] attribute0, vector[double] attribute1, vector[bool] selection):
		self._G = G
		self._attribute0 = move(attribute0)
		self._attribute1 = move(attribute1)
		self._selection = move(selection)

		self._this = new _EdgeScoreBlender(G._this, self._attribute0, self._attribute1, self._selection)

	cdef bool isDoubleValue(self):
		return True

cdef extern from "cpp/edgescores/GeometricMeanScore.h":
	cdef cppclass _GeometricMeanScore "NetworKit::GeometricMeanScore"(_EdgeScore[double]):
		_GeometricMeanScore(const _Graph& G, const vector[double]& a) except +

cdef class GeometricMeanScore(EdgeScore):
	"""
	Normalizes the given edge attribute by the geometric average of the sum of the attributes of the incident edges of the incident nodes.

	Parameters
	----------
	G : Graph
		The input graph.
	a : vector[double]
		Edge attribute that shall be normalized.
	"""
	cdef vector[double] _attribute

	def __cinit__(self, Graph G, vector[double] attribute):
		self._G = G
		self._attribute = attribute
		self._this = new _GeometricMeanScore(G._this, self._attribute)

	cdef bool isDoubleValue(self):
		return True

cdef extern from "cpp/edgescores/EdgeScoreAsWeight.h":
	cdef cppclass _EdgeScoreAsWeight "NetworKit::EdgeScoreAsWeight":
		_EdgeScoreAsWeight(const _Graph& G, const vector[double]& score, bool squared, edgeweight offset, edgeweight factor) except +
		_Graph calculate() except +

cdef class EdgeScoreAsWeight:
	"""
	Assigns an edge score as edge weight of a graph.

	Parameters
	----------
	G : Graph
		The graph to assign edge weights to.
	score : vector[double]
		The input edge score.
	squared : bool
		Edge weights will be squared if set to True.
	offset : edgeweight
		This offset will be added to each edge weight.
	factor : edgeweight
		Each edge weight will be multiplied by this factor.
	"""

	cdef _EdgeScoreAsWeight* _this
	cdef Graph _G
	cdef vector[double] _score

	def __cinit__(self, Graph G, vector[double] score, bool squared, edgeweight offset, edgeweight factor):
		self._G = G
		self._score = score
		self._this = new _EdgeScoreAsWeight(G._this, self._score, squared, offset, factor)

	def __dealloc__(self):
		if self._this is not NULL:
			del self._this
			self._this = NULL

	def getWeightedGraph(self):
		"""
		Returns
		-------
		Graph
			The weighted result graph.
		"""
		return Graph(0).setThis(self._this.calculate())

cdef extern from "cpp/edgescores/PrefixJaccardScore.h":
	cdef cppclass _PrefixJaccardScore "NetworKit::PrefixJaccardScore<double>"(_EdgeScore[double]):
		_PrefixJaccardScore(const _Graph& G, const vector[double]& a) except +

cdef class PrefixJaccardScore(EdgeScore):
	cdef vector[double] _attribute

	def __cinit__(self, Graph G, vector[double] attribute):
		self._G = G
		self._attribute = attribute
		self._this = new _PrefixJaccardScore(G._this, self._attribute)

	cdef bool isDoubleValue(self):
		return True
