'''
	Module: graphio
'''

cdef extern from "cpp/io/GraphReader.h":
	cdef cppclass _GraphReader "NetworKit::GraphReader":
		_GraphReader() nogil except +
		_Graph read(string path) nogil except +

cdef class GraphReader:
	""" Abstract base class for graph readers"""

	cdef _GraphReader* _this

	def __init__(self, *args, **kwargs):
		if type(self) == GraphReader:
			raise RuntimeError("Error, you may not use GraphReader directly, use a sub-class instead")

	def __cinit__(self, *args, **kwargs):
		self._this = NULL

	def __dealloc__(self):
		if self._this != NULL:
			del self._this
		self._this = NULL

	def read(self, path):
		cdef string cpath = stdstring(path)
		cdef _Graph result

		with nogil:
			result = move(self._this.read(cpath)) # extra move in order to avoid copying the internal variable that is used by Cython
		return Graph(0).setThis(result)

cdef extern from "cpp/io/METISGraphReader.h":
	cdef cppclass _METISGraphReader "NetworKit::METISGraphReader" (_GraphReader):
		_METISGraphReader() nogil except +

cdef class METISGraphReader(GraphReader):
	""" Reads the METIS adjacency file format [1]. If the Fast reader fails,
		use readGraph(path, graphio.formats.metis) as an alternative.
		[1]: http://people.sc.fsu.edu/~jburkardt/data/metis_graph/metis_graph.html
	"""
	def __cinit__(self):
		self._this = new _METISGraphReader()

cdef extern from "cpp/io/GraphToolBinaryReader.h":
	cdef cppclass _GraphToolBinaryReader "NetworKit::GraphToolBinaryReader" (_GraphReader):
		_GraphToolBinaryReader() except +

cdef class GraphToolBinaryReader(GraphReader):
	""" Reads the binary file format defined by graph-tool[1].
		[1]: http://graph-tool.skewed.de/static/doc/gt_format.html
	"""
	def __cinit__(self):
		self._this = new _GraphToolBinaryReader()

cdef extern from "cpp/io/EdgeListReader.h":
	cdef cppclass _EdgeListReader "NetworKit::EdgeListReader"(_GraphReader):
		_EdgeListReader() except +
		_EdgeListReader(char separator, node firstNode, string commentPrefix, bool continuous, bool directed)
		map[string,node] getNodeMap() except +

cdef class EdgeListReader(GraphReader):
	""" Reads a file in an edge list format.
		TODO: docstring
	"""
	def __cinit__(self, separator, firstNode, commentPrefix="#", continuous=True, directed=False):
		self._this = new _EdgeListReader(stdstring(separator)[0], firstNode, stdstring(commentPrefix), continuous, directed)

	def getNodeMap(self):
		cdef map[string,node] cResult = (<_EdgeListReader*>(self._this)).getNodeMap()
		result = dict()
		for elem in cResult:
			#result.append((elem.first,elem.second))
			result[(elem.first).decode("utf-8")] = elem.second
		return result

cdef extern from "cpp/io/KONECTGraphReader.h":
	cdef cppclass _KONECTGraphReader "NetworKit::KONECTGraphReader"(_GraphReader):
		_KONECTGraphReader() except +
		_KONECTGraphReader(char separator, bool ignoreLoops)

cdef class KONECTGraphReader(GraphReader):
	""" Reader for the KONECT graph format, which is described in detail on the KONECT website[1].

		[1]: http://konect.uni-koblenz.de/downloads/konect-handbook.pdf
	"""
	def __cinit__(self, separator, ignoreLoops = False):
		self._this = new _KONECTGraphReader(stdstring(separator)[0], ignoreLoops)

cdef extern from "cpp/io/GMLGraphReader.h":
	cdef cppclass _GMLGraphReader "NetworKit::GMLGraphReader"(_GraphReader):
		_GMLGraphReader() except +

cdef class GMLGraphReader(GraphReader):
	""" Reader for the GML graph format, which is documented here [1].

		[1]: http://www.fim.uni-passau.de/fileadmin/files/lehrstuhl/brandenburg/projekte/gml/gml-technical-report.pdf
 	"""
	def __cinit__(self):
		self._this = new _GMLGraphReader()

cdef extern from "cpp/io/METISGraphWriter.h":
	cdef cppclass _METISGraphWriter "NetworKit::METISGraphWriter":
		_METISGraphWriter() except +
		void write(_Graph G, string path) nogil except +

cdef class METISGraphWriter:
	""" Writes graphs in the METIS format"""
	cdef _METISGraphWriter _this

	def write(self, Graph G not None, path):
		 # string needs to be converted to bytes, which are coerced to std::string
		cdef string cpath = stdstring(path)
		with nogil:
			self._this.write(G._this, cpath)

cdef extern from "cpp/io/GraphToolBinaryWriter.h":
	cdef cppclass _GraphToolBinaryWriter "NetworKit::GraphToolBinaryWriter":
		_GraphToolBinaryWriter() except +
		void write(_Graph G, string path) nogil except +

cdef class GraphToolBinaryWriter:
	""" Reads the binary file format defined by graph-tool[1].
		[1]: http://graph-tool.skewed.de/static/doc/gt_format.html
	"""
	cdef _GraphToolBinaryWriter _this

	def write(self, Graph G not None, path):
		 # string needs to be converted to bytes, which are coerced to std::string
		cdef string cpath = stdstring(path)
		with nogil:
			self._this.write(G._this, cpath)

cdef extern from "cpp/io/DotGraphWriter.h":
	cdef cppclass _DotGraphWriter "NetworKit::DotGraphWriter":
		_DotGraphWriter() except +
		void write(_Graph G, string path) nogil except +

cdef class DotGraphWriter:
	""" Writes graphs in the .dot/GraphViz format"""
	cdef _DotGraphWriter _this

	def write(self, Graph G not None, path):
		 # string needs to be converted to bytes, which are coerced to std::string
		cdef string cpath = stdstring(path)
		with nogil:
			self._this.write(G._this, cpath)

cdef extern from "cpp/io/GMLGraphWriter.h":
	cdef cppclass _GMLGraphWriter "NetworKit::GMLGraphWriter":
		_GMLGraphWriter() except +
		void write(_Graph G, string path) nogil except +

cdef class GMLGraphWriter:
	""" Writes a graph and its coordinates as a GML file.[1]
		[1] http://svn.bigcat.unimaas.nl/pvplugins/GML/trunk/docs/gml-technical-report.pdf """
	cdef _GMLGraphWriter _this

	def write(self, Graph G not None, path):
		 # string needs to be converted to bytes, which are coerced to std::string
		cdef string cpath = stdstring(path)
		with nogil:
			self._this.write(G._this, cpath)

cdef extern from "cpp/io/EdgeListWriter.h":
	cdef cppclass _EdgeListWriter "NetworKit::EdgeListWriter":
		_EdgeListWriter() except +
		_EdgeListWriter(char separator, node firstNode, bool bothDirections) except +
		void write(_Graph G, string path) nogil except +

cdef class EdgeListWriter:
	""" Writes graphs in various edge list formats.

	Parameters
	----------
	separator : string
		The separator character.
	firstNode : node
		The id of the first node, this value will be added to all node ids
	bothDirections : bool, optional
		If undirected edges shall be written in both directions, i.e., as symmetric directed graph (default: false)
	"""

	cdef _EdgeListWriter _this

	def __cinit__(self, separator, firstNode, bool bothDirections = False):
		cdef char sep = stdstring(separator)[0]
		self._this = _EdgeListWriter(sep, firstNode, bothDirections)

	def write(self, Graph G not None, path):
		cdef string cpath = stdstring(path)
		with nogil:
			self._this.write(G._this, cpath)

cdef extern from "cpp/io/LineFileReader.h":
	cdef cppclass _LineFileReader "NetworKit::LineFileReader":
		_LineFileReader() except +
		vector[string] read(string path)

cdef class LineFileReader:
	""" Reads a file and puts each line in a list of strings """
	cdef _LineFileReader _this

	def read(self, path):
		return self._this.read(stdstring(path))

cdef extern from "cpp/io/SNAPGraphWriter.h":
	cdef cppclass _SNAPGraphWriter "NetworKit::SNAPGraphWriter":
		_SNAPGraphWriter() except +
		void write(_Graph G, string path) nogil except +

cdef class SNAPGraphWriter:
	""" Writes graphs in a format suitable for the Georgia Tech SNAP software [1]
		[1]: http://snap-graph.sourceforge.net/
	"""
	cdef _SNAPGraphWriter _this

	def write(self, Graph G, path):
		cdef string cpath = stdstring(path)
		with nogil:
			self._this.write(G._this, cpath)

cdef extern from "cpp/io/SNAPGraphReader.h":
	cdef cppclass _SNAPGraphReader "NetworKit::SNAPGraphReader"(_GraphReader):
		_SNAPGraphReader() except +
		unordered_map[node,node] getNodeIdMap() except +

cdef class SNAPGraphReader(GraphReader):
	""" Reads a graph from the SNAP graph data collection [1] (currently experimental)
		[1]: http://snap.stanford.edu/data/index.html
	"""
	def __cinit__(self):
		self._this = new _SNAPGraphReader()

	def getNodeIdMap(self):
		cdef unordered_map[node,node] cResult = (<_SNAPGraphReader*>(self._this)).getNodeIdMap()
		result = []
		for elem in cResult:
			result.append((elem.first,elem.second))
		return result

cdef extern from "cpp/io/PartitionReader.h":
	cdef cppclass _PartitionReader "NetworKit::PartitionReader":
		_PartitionReader() except +
		_Partition read(string path) except +

cdef class PartitionReader:
	""" Reads a partition from a file.
		File format: line i contains subset id of element i.
	 """
	cdef _PartitionReader _this

	def read(self, path):
		return Partition().setThis(self._this.read(stdstring(path)))

cdef extern from "cpp/io/PartitionWriter.h":
	cdef cppclass _PartitionWriter "NetworKit::PartitionWriter":
		_PartitionWriter() except +
		void write(_Partition, string path) nogil except +

cdef class PartitionWriter:
	""" Writes a partition to a file.
		File format: line i contains subset id of element i.
	 """
	cdef _PartitionWriter _this

	def write(self, Partition zeta, path):
		cdef string cpath = stdstring(path)
		with nogil:
			self._this.write(zeta._this, cpath)

cdef extern from "cpp/io/EdgeListPartitionReader.h":
	cdef cppclass _EdgeListPartitionReader "NetworKit::EdgeListPartitionReader":
		_EdgeListPartitionReader() except +
		_EdgeListPartitionReader(node firstNode, char sepChar) except +
		_Partition read(string path) except +

cdef class EdgeListPartitionReader:
	""" Reads a partition from an edge list type of file
	 """
	cdef _EdgeListPartitionReader _this

	def __cinit__(self, node firstNode=1, sepChar = '\t'):
		self._this = _EdgeListPartitionReader(firstNode, stdstring(sepChar)[0])

	def read(self, path):
		return Partition().setThis(self._this.read(stdstring(path)))

cdef extern from "cpp/io/SNAPEdgeListPartitionReader.h":
	cdef cppclass _SNAPEdgeListPartitionReader "NetworKit::SNAPEdgeListPartitionReader":
		_SNAPEdgeListPartitionReader() except +
		_Cover read(string path, unordered_map[node,node] nodeMap,_Graph G) except +
#		_Partition readWithInfo(string path, count nNodes) except +

cdef class SNAPEdgeListPartitionReader:
	""" Reads a partition from a SNAP 'community with ground truth' file
	 """
	cdef _SNAPEdgeListPartitionReader _this

	def read(self,path, nodeMap, Graph G):
		cdef unordered_map[node,node] cNodeMap
		for (key,val) in nodeMap:
			cNodeMap[key] = val
		return Cover().setThis(self._this.read(stdstring(path), cNodeMap, G._this))

cdef extern from "cpp/io/CoverReader.h":
	cdef cppclass _CoverReader "NetworKit::CoverReader":
		_CoverReader() except +
		_Cover read(string path,_Graph G) except +

cdef class CoverReader:
	""" Reads a cover from a file
		File format: each line contains the space-separated node ids of a community
	 """
	cdef _CoverReader _this

	def read(self, path, Graph G):
		return Cover().setThis(self._this.read(stdstring(path), G._this))

cdef extern from "cpp/io/CoverWriter.h":
	cdef cppclass _CoverWriter "NetworKit::CoverWriter":
		_CoverWriter() except +
		void write(_Cover, string path) nogil except +

cdef class CoverWriter:
	""" Writes a partition to a file.
		File format: each line contains the space-separated node ids of a community
	 """
	cdef _CoverWriter _this

	def write(self, Cover zeta, path):
		cdef string cpath = stdstring(path)
		with nogil:
			self._this.write(zeta._this, cpath)

cdef extern from "cpp/io/EdgeListCoverReader.h":
	cdef cppclass _EdgeListCoverReader "NetworKit::EdgeListCoverReader":
		_EdgeListCoverReader() except +
		_EdgeListCoverReader(node firstNode) except +
		_Cover read(string path, _Graph G) except +

cdef class EdgeListCoverReader:
	""" Reads a cover from an edge list type of file
		File format: each line starts with a node id and continues with a list of the communities the node belongs to
	 """
	cdef _EdgeListCoverReader _this

	def __cinit__(self, firstNode=1):
		self._this = _EdgeListCoverReader(firstNode)

	def read(self, path, Graph G):
		return Cover().setThis(self._this.read(stdstring(path), G._this))
