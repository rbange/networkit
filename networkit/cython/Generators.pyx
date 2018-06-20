'''
	Module: generators
'''

cdef extern from "cpp/generators/BarabasiAlbertGenerator.h":
	cdef cppclass _BarabasiAlbertGenerator "NetworKit::BarabasiAlbertGenerator":
		_BarabasiAlbertGenerator() except +
		_BarabasiAlbertGenerator(count k, count nMax, count n0, bool batagelj) except +
		_BarabasiAlbertGenerator(count k, count nMax, const _Graph & initGraph, bool batagelj) except +
		_Graph generate() except +

cdef class BarabasiAlbertGenerator:
	"""
	This generator implements the preferential attachment model as introduced by Barabasi and Albert[1].
	The original algorithm is very slow and thus, the much faster method from Batagelj and Brandes[2] is
	implemented and the current default.
	The original method can be chosen by setting \p batagelj to false.
	[1] Barabasi, Albert: Emergence of Scaling in Random Networks http://arxiv.org/pdf/cond-mat/9910332.pdf
	[2] ALG 5 of Batagelj, Brandes: Efficient Generation of Large Random Networks https://kops.uni-konstanz.de/bitstream/handle/123456789/5799/random.pdf?sequence=1

	Parameters
	----------
	k : count
		number of edges that come with a new node
	nMax : count
		maximum number of nodes produced
	n0 : count
		number of starting nodes
	batagelj : bool
		Specifies whether to use batagelj's method or the original one.
	"""
	cdef _BarabasiAlbertGenerator _this

	def __cinit__(self, count k, count nMax, n0=0, bool batagelj=True):
		if isinstance(n0, Graph):
			self._this = _BarabasiAlbertGenerator(k, nMax, (<Graph>n0)._this, batagelj)
		else:
			self._this = _BarabasiAlbertGenerator(k, nMax, <count>n0, batagelj)

	def generate(self):
		return Graph().setThis(self._this.generate())

	@classmethod
	def fit(cls, Graph G, scale=1):
		(n, m) = G.size()
		k = math.floor(m / n)
		return cls(nMax=scale * n, k=k, n0=k)

cdef extern from "cpp/generators/PubWebGenerator.h":
	cdef cppclass _PubWebGenerator "NetworKit::PubWebGenerator":
		_PubWebGenerator(count numNodes, count numberOfDenseAreas, float neighborhoodRadius, count maxNumberOfNeighbors) except +
		_Graph generate() except +

cdef class PubWebGenerator:
	""" Generates a static graph that resembles an assumed geometric distribution of nodes in
	a P2P network.

	The basic structure is to distribute points randomly in the unit torus
	and to connect vertices close to each other (at most @a neighRad distance and none of
	them already has @a maxNeigh neighbors). The distribution is chosen to get some areas with
	high density and others with low density. There are @a numDenseAreas dense areas, which can
	overlap. Each area is circular, has a certain position and radius and number of points.
	These values are strored in @a denseAreaXYR and @a numPerArea, respectively.

	Used and described in more detail in J. Gehweiler, H. Meyerhenke: A Distributed
	Diffusive Heuristic for Clustering a Virtual P2P Supercomputer. In Proc. 7th High-Performance
	Grid Computing Workshop (HPGC'10), in conjunction with 24th IEEE Internatl. Parallel and
	Distributed Processing Symposium (IPDPS'10), IEEE, 2010.

	PubWebGenerator(numNodes, numberOfDenseAreas, neighborhoodRadius, maxNumberOfNeighbors)

	Parameters
	----------
	numNodes : count
		Up to a few thousand (possibly more if visualization is not desired and quadratic
		time complexity has been resolved)
	numberOfDenseAreas : count
		Depending on number of nodes, e.g. [8, 50]
	neighborhoodRadius : float
		The higher, the better the connectivity [0.1, 0.35]
	maxNumberOfNeighbors : count
		Maximum degree, a higher value corresponds to better connectivity [4, 40]
	"""
	cdef _PubWebGenerator* _this

	def __cinit__(self, numNodes, numberOfDenseAreas, neighborhoodRadius, maxNumberOfNeighbors):
		self._this = new _PubWebGenerator(numNodes, numberOfDenseAreas, neighborhoodRadius, maxNumberOfNeighbors)

	def __dealloc__(self):
		del self._this

	def generate(self):
		return Graph(0).setThis(self._this.generate())

cdef extern from "cpp/generators/ErdosRenyiGenerator.h":
	cdef cppclass _ErdosRenyiGenerator "NetworKit::ErdosRenyiGenerator":
		_ErdosRenyiGenerator(count nNodes, double prob, bool directed) except +
		_Graph generate() except +

cdef class ErdosRenyiGenerator:
	""" Creates random graphs in the G(n,p) model.
	The generation follows Vladimir Batagelj and Ulrik Brandes: "Efficient
	generation of large random networks", Phys Rev E 71, 036113 (2005).

	ErdosRenyiGenerator(count, double)

	Creates G(nNodes, prob) graphs.

	Parameters
	----------
	nNodes : count
		Number of nodes n in the graph.
	prob : double
		Probability of existence for each edge p.
	directed : bool
		Generates a directed
	"""

	cdef _ErdosRenyiGenerator* _this

	def __cinit__(self, nNodes, prob, directed=False):
		self._this = new _ErdosRenyiGenerator(nNodes, prob, directed)

	def __dealloc__(self):
		del self._this

	def generate(self):
		return Graph(0).setThis(self._this.generate())

	@classmethod
	def fit(cls, Graph G, scale=1):
		""" Fit model to input graph"""
		(n, m) = G.size()
		if G.isDirected():
			raise Exception("TODO: figure out scaling scheme for directed graphs")
		else:
			p = (2 * m) / (scale * n * (n-1))
		return cls(scale * n, p)

cdef extern from "cpp/generators/DorogovtsevMendesGenerator.h":
	cdef cppclass _DorogovtsevMendesGenerator "NetworKit::DorogovtsevMendesGenerator":
		_DorogovtsevMendesGenerator(count nNodes) except +
		_Graph generate() except +

cdef class DorogovtsevMendesGenerator:
	""" Generates a graph according to the Dorogovtsev-Mendes model.

 	DorogovtsevMendesGenerator(nNodes)

 	Constructs the generator class.

	Parameters
	----------
	nNodes : count
		Number of nodes in the target graph.
	"""

	cdef _DorogovtsevMendesGenerator* _this

	def __cinit__(self, nNodes):
		self._this = new _DorogovtsevMendesGenerator(nNodes)

	def __dealloc__(self):
		del self._this

	def generate(self):
		""" Generates a random graph according to the Dorogovtsev-Mendes model.

		Returns
		-------
		Graph
			The generated graph.
		"""
		return Graph(0).setThis(self._this.generate())

	@classmethod
	def fit(cls, Graph G, scale=1):
		return cls(scale * G.numberOfNodes())

cdef extern from "cpp/generators/RegularRingLatticeGenerator.h":
	cdef cppclass _RegularRingLatticeGenerator "NetworKit::RegularRingLatticeGenerator":
		_RegularRingLatticeGenerator(count nNodes, count nNeighbors) except +
		_Graph generate() except +

cdef class RegularRingLatticeGenerator:
	"""
	Constructs a regular ring lattice.

	RegularRingLatticeGenerator(count nNodes, count nNeighbors)

	Constructs the generator.

	Parameters
	----------
	nNodes : number of nodes in the target graph.
	nNeighbors : number of neighbors on each side of a node
	"""

	cdef _RegularRingLatticeGenerator* _this

	def __cinit__(self, nNodes, nNeighbors):
		self._this = new _RegularRingLatticeGenerator(nNodes, nNeighbors)

	def __dealloc__(self):
		del self._this

	def generate(self):
		""" Generates a rgular ring lattice.

		Returns
		-------
		Graph
			The generated graph.
		"""
		return Graph(0).setThis(self._this.generate())

cdef extern from "cpp/generators/WattsStrogatzGenerator.h":
	cdef cppclass _WattsStrogatzGenerator "NetworKit::WattsStrogatzGenerator":
		_WattsStrogatzGenerator(count nNodes, count nNeighbors, double p) except +
		_Graph generate() except +

cdef class WattsStrogatzGenerator:
	""" Generates a graph according to the Watts-Strogatz model.

	First, a regular ring lattice is generated. Then edges are rewired
		with a given probability.

	WattsStrogatzGenerator(count nNodes, count nNeighbors, double p)

	Constructs the generator.

	Parameters
	----------
	nNodes : Number of nodes in the target graph.
	nNeighbors : number of neighbors on each side of a node
	p : rewiring probability
	"""

	cdef _WattsStrogatzGenerator* _this

	def __dealloc__(self):
		del self._this

	def __cinit__(self, nNodes, nNeighbors, p):
		self._this = new _WattsStrogatzGenerator(nNodes, nNeighbors, p)

	def generate(self):
		""" Generates a random graph according to the Watts-Strogatz model.

		Returns
		-------
		Graph
			The generated graph.
		"""
		return Graph(0).setThis(self._this.generate())

cdef extern from "cpp/generators/ClusteredRandomGraphGenerator.h":
	cdef cppclass _ClusteredRandomGraphGenerator "NetworKit::ClusteredRandomGraphGenerator":
		_ClusteredRandomGraphGenerator(count, count, double, double) except +
		_Graph generate() except +
		_Partition getCommunities() except +

cdef class ClusteredRandomGraphGenerator:
	""" The ClusteredRandomGraphGenerator class is used to create a clustered random graph.

	The number of nodes and the number of edges are adjustable as well as the probabilities
	for intra-cluster and inter-cluster edges.

	ClusteredRandomGraphGenerator(count, count, pin, pout)

	Creates a clustered random graph.

	Parameters
	----------
	n : count
		number of nodes
	k : count
		number of clusters
	pin : double
		intra-cluster edge probability
	pout : double
		inter-cluster edge probability
	"""

	cdef _ClusteredRandomGraphGenerator* _this

	def __cinit__(self, n, k, pin, pout):
		self._this = new _ClusteredRandomGraphGenerator(n, k, pin, pout)

	def __dealloc__(self):
		del self._this

	def generate(self):
		""" Generates a clustered random graph with the properties given in the constructor.

		Returns
		-------
		Graph
			The generated graph.
		"""
		return Graph(0).setThis(self._this.generate())

	def getCommunities(self):
		""" Returns the generated ground truth clustering.

		Returns
		-------
		Partition
			The generated ground truth clustering.
		"""
		return Partition().setThis(self._this.getCommunities())

cdef extern from "cpp/generators/ChungLuGenerator.h":
	cdef cppclass _ChungLuGenerator "NetworKit::ChungLuGenerator":
		_ChungLuGenerator(vector[count] degreeSequence) except +
		_Graph generate() except +

cdef class ChungLuGenerator:
	"""
		Given an arbitrary degree sequence, the Chung-Lu generative model
		will produce a random graph with the same expected degree sequence.

		see Chung, Lu: The average distances in random graphs with given expected degrees
		and Chung, Lu: Connected Components in Random Graphs with Given Expected Degree Sequences.
		Aiello, Chung, Lu: A Random Graph Model for Massive Graphs describes a different generative model
		which is basically asymptotically equivalent but produces multi-graphs.
	"""

	cdef _ChungLuGenerator* _this

	def __cinit__(self, vector[count] degreeSequence):
		self._this = new _ChungLuGenerator(degreeSequence)

	def __dealloc__(self):
		del self._this

	def generate(self):
		""" Generates graph with expected degree sequence seq.

		Returns
		-------
		Graph
			The generated graph.
		"""
		return Graph(0).setThis(self._this.generate())

	@classmethod
	def fit(cls, Graph G, scale=1):
		""" Fit model to input graph"""
		(n, m) = G.size()
		degSeq = DegreeCentrality(G).run().scores()
		return cls(degSeq * scale)

cdef extern from "cpp/generators/HavelHakimiGenerator.h":
	cdef cppclass _HavelHakimiGenerator "NetworKit::HavelHakimiGenerator":
		_HavelHakimiGenerator(vector[count] degreeSequence, bool ignoreIfRealizable) except +
		_Graph generate() except +
		bool isRealizable() except +
		bool getRealizable() except +

cdef class HavelHakimiGenerator:
	""" Havel-Hakimi algorithm for generating a graph according to a given degree sequence.

		The sequence, if it is realizable, is reconstructed exactly. The resulting graph usually
		has a high clustering coefficient. Construction runs in linear time O(m).

		If the sequence is not realizable, depending on the parameter ignoreIfRealizable, either
		an exception is thrown during generation or the graph is generated with a modified degree
		sequence, i.e. not all nodes might have as many neighbors as requested.

		HavelHakimiGenerator(sequence, ignoreIfRealizable=True)

		Parameters
		----------
		sequence : vector
			Degree sequence to realize. Must be non-increasing.
		ignoreIfRealizable : bool, optional
			If true, generate the graph even if the degree sequence is not realizable. Some nodes may get lower degrees than requested in the sequence.
	"""

	cdef _HavelHakimiGenerator* _this


	def __cinit__(self, vector[count] degreeSequence, ignoreIfRealizable=True):
		self._this = new _HavelHakimiGenerator(degreeSequence, ignoreIfRealizable)

	def __dealloc__(self):
		del self._this

	def isRealizable(self):
		return self._this.isRealizable()

	def getRealizable(self):
		return self._this.getRealizable();

	def generate(self):
		""" Generates degree sequence seq (if it is realizable).

		Returns
		-------
		Graph
			Graph with degree sequence seq or modified sequence if ignoreIfRealizable is true and the sequence is not realizable.
		"""
		return Graph(0).setThis(self._this.generate())

	@classmethod
	def fit(cls, Graph G, scale=1):
		degSeq = DegreeCentrality(G).run().scores()
		return cls(degSeq * scale, ignoreIfRealizable=True)

cdef extern from "cpp/generators/EdgeSwitchingMarkovChainGenerator.h":
	cdef cppclass _EdgeSwitchingMarkovChainGenerator "NetworKit::EdgeSwitchingMarkovChainGenerator":
		_EdgeSwitchingMarkovChainGenerator(vector[count] degreeSequence, bool ignoreIfRealizable) except +
		_Graph generate() except +
		bool isRealizable() except +
		bool getRealizable() except +

cdef class EdgeSwitchingMarkovChainGenerator:
	"""
	Graph generator for generating a random simple graph with exactly the given degree sequence based on the Edge-Switching Markov-Chain method.

	This implementation is based on the paper
	"Random generation of large connected simple graphs with prescribed degree distribution" by Fabien Viger and Matthieu Latapy,
	available at http://www-rp.lip6.fr/~latapy/FV/generation.html, however without preserving connectivity (this could later be added as
	optional feature).

	The Havel-Hakami generator is used for the initial graph generation, then the Markov-Chain Monte-Carlo algorithm as described and
	implemented by Fabien Viger and Matthieu Latapy but without the steps for ensuring connectivity is executed. This should lead to a
	graph that is drawn uniformly at random from all graphs with the given degree sequence.

	Note that at most 10 times the number of edges edge swaps are performed (same number as in the abovementioned implementation) and
	in order to limit the running time, at most 200 times as many attempts to perform an edge swap are made (as certain degree distributions
	do not allow edge swaps at all).

	Parameters
	----------
	degreeSequence : vector[count]
		The degree sequence that shall be generated
	ignoreIfRealizable : bool, optional
		If true, generate the graph even if the degree sequence is not realizable. Some nodes may get lower degrees than requested in the sequence.
	"""
	cdef _EdgeSwitchingMarkovChainGenerator *_this

	def __cinit__(self, vector[count] degreeSequence, bool ignoreIfRealizable = False):
		self._this = new _EdgeSwitchingMarkovChainGenerator(degreeSequence, ignoreIfRealizable)

	def __dealloc__(self):
		del self._this

	def isRealizable(self):
		return self._this.isRealizable()

	def getRealizable(self):
		return self._this.getRealizable()

	def generate(self):
		"""
		Generate a graph according to the configuration model.

		Issues a INFO log message if the wanted number of edge swaps cannot be performed because of the limit of attempts (see in the description of the class for details).

		Returns
		-------
		Graph
			The generated graph.
		"""
		return Graph().setThis(self._this.generate())

	@classmethod
	def fit(cls, Graph G, scale=1):
		degSeq = DegreeCentrality(G).run().scores()
		return cls(degSeq * scale, ignoreIfRealizable=True)

cdef extern from "cpp/generators/HyperbolicGenerator.h":
	cdef cppclass _HyperbolicGenerator "NetworKit::HyperbolicGenerator":
		# TODO: revert to count when cython issue fixed
		_HyperbolicGenerator(unsigned int nodes,  double k, double gamma, double T) except +
		void setLeafCapacity(unsigned int capacity) except +
		void setTheoreticalSplit(bool split) except +
		void setBalance(double balance) except +
		vector[double] getElapsedMilliseconds() except +
		_Graph generate() except +
		_Graph generate(vector[double] angles, vector[double] radii, double R, double T) except +

cdef class HyperbolicGenerator:
	""" The Hyperbolic Generator distributes points in hyperbolic space and adds edges between points with a probability depending on their distance. The resulting graphs have a power-law degree distribution, small diameter and high clustering coefficient.
For a temperature of 0, the model resembles a unit-disk model in hyperbolic space.

 		HyperbolicGenerator(n, k=6, gamma=3, T=0)

 		Parameters
		----------
		n : integer
			number of nodes
		k : double
			average degree
		gamma : double
			exponent of power-law degree distribution
		T : double
			temperature of statistical model

	"""

	cdef _HyperbolicGenerator* _this

	def __cinit__(self,  n, k=6, gamma=3, T=0):
		if gamma <= 2:
				raise ValueError("Exponent of power-law degree distribution must be > 2")
		self._this = new _HyperbolicGenerator(n, k, gamma, T)

	def __dealloc__(self):
		del self._this

	def setLeafCapacity(self, capacity):
		self._this.setLeafCapacity(capacity)

	def setBalance(self, balance):
		self._this.setBalance(balance)

	def setTheoreticalSplit(self, theoreticalSplit):
		self._this.setTheoreticalSplit(theoreticalSplit)

	def getElapsedMilliseconds(self):
		return self._this.getElapsedMilliseconds()

	def generate(self):
		""" Generates hyperbolic random graph

		Returns
		-------
		Graph

		"""
		return Graph(0).setThis(self._this.generate())

	def generate(self, angles, radii, R, T=0):
		# TODO: documentation
		return Graph(0).setThis(self._this.generate(angles, radii, R, T))

	@classmethod
	def fit(cls, Graph G, scale=1):
		""" Fit model to input graph"""
		degSeq = DegreeCentrality(G).run().scores()
		gamma = max(-1 * PowerlawDegreeSequence(degSeq).getGamma(), 2.1)
		(n, m) = G.size()
		k = 2 * (m / n)
		return cls(n * scale, k, gamma)

cdef extern from "cpp/generators/RmatGenerator.h":
	cdef cppclass _RmatGenerator "NetworKit::RmatGenerator":
		_RmatGenerator(count scale, count edgeFactor, double a, double b, double c, double d, bool weighted, count reduceNodes) except +
		_Graph generate() except +

cdef class RmatGenerator:
	"""
	Generates static R-MAT graphs. R-MAT (recursive matrix) graphs are
	random graphs with n=2^scale nodes and m=nedgeFactor edges.
	More details at http://www.graph500.org or in the original paper:
	Deepayan Chakrabarti, Yiping Zhan, Christos Faloutsos:
	R-MAT: A Recursive Model for Graph Mining. SDM 2004: 442-446.

	RmatGenerator(scale, edgeFactor, a, b, c, d)

	Parameters
	----------
	scale : count
		Number of nodes = 2^scale
	edgeFactor : count
		Number of edges = number of nodes * edgeFactor
	a : double
		Probability for quadrant upper left
	b : double
		Probability for quadrant upper right
	c : double
		Probability for quadrant lower left
	d : double
		Probability for quadrant lower right
	weighted : bool
		result graph weighted?
	"""

	cdef _RmatGenerator* _this
	paths = {"kronfitPath" : None}

	def __cinit__(self, count scale, count edgeFactor, double a, double b, double c, double d, bool weighted=False, count reduceNodes=0):
		self._this = new _RmatGenerator(scale, edgeFactor, a, b, c, d, weighted, reduceNodes)

	def __dealloc__(self):
		del self._this

	def generate(self):
		""" Graph to be generated according to parameters specified in constructor.

		Returns
		-------
		Graph
			The generated graph.
		"""
		return Graph(0).setThis(self._this.generate())

	@classmethod
	def setPaths(cls, kronfitPath):
		cls.paths["kronfitPath"] = kronfitPath

	@classmethod
	def fit(cls, G, scale=1, initiator=None, kronfit=True, iterations=50):
		import math
		import re
		import subprocess
		import os
		import random
		from networkit import graphio
		if initiator:
			(a,b,c,d) = initiator
		else:
			if kronfit:
				with tempfile.TemporaryDirectory() as tmpdir:
					if cls.paths["kronfitPath"] is None:
						raise RuntimeError("call setPaths class method first to configure")
					# write graph
					tmpGraphPath = os.path.join(tmpdir, "{0}.edgelist".format(G.getName()))
					tmpOutputPath = os.path.join(tmpdir, "{0}.kronfit".format(G.getName()))
					graphio.writeGraph(G, tmpGraphPath, graphio.Format.EdgeList, separator="\t", firstNode=1, bothDirections=True)
					# call kronfit
					args = [cls.paths["kronfitPath"], "-i:{0}".format(tmpGraphPath), "-gi:{0}".format(str(iterations)), "-o:{}".format(tmpOutputPath)]
					subprocess.call(args)
					# read estimated parameters
					with open(tmpOutputPath) as resultFile:
						for line in resultFile:
							if "initiator" in line:
								matches = re.findall("\d+\.\d+", line)
								weights = [float(s) for s in matches]
			else:
				# random weights because kronfit is slow
				weights = (random.random(), random.random(), random.random(), random.random())
			# normalize
			nweights = [w / sum(weights) for w in weights]
			(a,b,c,d) = nweights
		print("using initiator matrix [{0},{1};{2},{3}]".format(a,b,c,d))
		# other parameters
		(n,m) = G.size()
		scaleParameter = math.ceil(math.log(n * scale, 2))
		edgeFactor = math.floor(m / n)
		reduceNodes = (2**scaleParameter) - (scale * n)
		print("random nodes to delete to achieve target node count: ", reduceNodes)
		return RmatGenerator(scaleParameter, edgeFactor, a, b, c, d, False, reduceNodes)

cdef extern from "cpp/generators/PowerlawDegreeSequence.h":
	cdef cppclass _PowerlawDegreeSequence "NetworKit::PowerlawDegreeSequence":
		_PowerlawDegreeSequence(count minDeg, count maxDeg, double gamma) except +
		_PowerlawDegreeSequence(_Graph) except +
		_PowerlawDegreeSequence(vector[double]) except +
		void setMinimumFromAverageDegree(double avgDeg) nogil except +
		void setGammaFromAverageDegree(double avgDeg, double minGamma, double maxGamma) nogil except +
		double getExpectedAverageDegree() except +
		count getMinimumDegree() const
		count getMaximumDegree() const
		double getGamma() const
		double setGamma(double) const
		void run() nogil except +
		vector[count] getDegreeSequence(count numNodes) except +
		count getDegree() except +

cdef class PowerlawDegreeSequence:
	"""
	Generates a powerlaw degree sequence with the given minimum and maximum degree, the powerlaw exponent gamma

	If a list of degrees or a graph is given instead of a minimum degree, the class uses the minimum and maximum
	value of the sequence and fits the exponent such that the expected average degree is the actual average degree.

	Parameters
	----------
	minDeg : count, list or Graph
		The minium degree, or a list of degrees to fit or graphs
	maxDeg : count
		The maximum degree
	gamma : double
		The powerlaw exponent, default: -2
	"""
	cdef _PowerlawDegreeSequence *_this

	def __cinit__(self, minDeg, count maxDeg = 0, double gamma = -2):
		if isinstance(minDeg, Graph):
			self._this = new _PowerlawDegreeSequence((<Graph>minDeg)._this)
		elif isinstance(minDeg, collections.Iterable):
			self._this = new _PowerlawDegreeSequence(<vector[double]?>minDeg)
		else:
			self._this = new _PowerlawDegreeSequence((<count?>minDeg), maxDeg, gamma)

	def __dealloc__(self):
		del self._this

	def setMinimumFromAverageDegree(self, double avgDeg):
		"""
		Tries to set the minimum degree such that the specified average degree is expected.

		Parameters
		----------
		avgDeg : double
			The average degree that shall be approximated
		"""
		with nogil:
			self._this.setMinimumFromAverageDegree(avgDeg)
		return self

	def setGammaFromAverageDegree(self, double avgDeg, double minGamma = -1, double maxGamma = -6):
		"""
		Tries to set the powerlaw exponent gamma such that the specified average degree is expected.

		Parameters
		----------
		avgDeg : double
			The average degree that shall be approximated
		minGamma : double
			The minimum gamma to use, default: -1
		maxGamma : double
			The maximum gamma to use, default: -6
		"""
		with nogil:
			self._this.setGammaFromAverageDegree(avgDeg, minGamma, maxGamma)
		return self

	def getExpectedAverageDegree(self):
		"""
		Returns the expected average degree. Note: run needs to be called first.

		Returns
		-------
		double
			The expected average degree.
		"""
		return self._this.getExpectedAverageDegree()

	def getMinimumDegree(self):
		"""
		Returns the minimum degree.

		Returns
		-------
		count
			The minimum degree
		"""
		return self._this.getMinimumDegree()

	def setGamma(self, double gamma):
		"""
		Set the exponent gamma

		Parameters
		----------
		gamma : double
			The exponent to set
		"""
		self._this.setGamma(gamma)
		return self

	def getGamma(self):
		"""
		Get the exponent gamma.

		Returns
		-------
		double
			The exponent gamma
		"""
		return self._this.getGamma()

	def getMaximumDegree(self):
		"""
		Get the maximum degree

		Returns
		-------
		count
			The maximum degree
		"""
		return self._this.getMaximumDegree()

	def run(self):
		"""
		Executes the generation of the probability distribution.
		"""
		with nogil:
			self._this.run()
		return self

	def getDegreeSequence(self, count numNodes):
		"""
		Returns a degree sequence with even degree sum.

		Parameters
		----------
		numNodes : count
			The number of nodes/degrees that shall be returned

		Returns
		-------
		vector[count]
			The generated degree sequence
		"""
		return self._this.getDegreeSequence(numNodes)

	def getDegree(self):
		"""
		Returns a degree drawn at random with a power law distribution

		Returns
		-------
		count
			The generated random degree
		"""
		return self._this.getDegree()

cdef extern from "cpp/generators/LFRGenerator.h":
	cdef cppclass _LFRGenerator "NetworKit::LFRGenerator"(_Algorithm):
		_LFRGenerator(count n) except +
		void setDegreeSequence(vector[count] degreeSequence) nogil except +
		void generatePowerlawDegreeSequence(count avgDegree, count maxDegree, double nodeDegreeExp) nogil except +
		void setCommunitySizeSequence(vector[count] communitySizeSequence) nogil except +
		void setPartition(_Partition zeta) nogil except +
		void generatePowerlawCommunitySizeSequence(count minCommunitySize, count maxCommunitySize, double communitySizeExp) nogil except +
		void setMu(double mu) nogil except +
		void setMu(const vector[double] & mu) nogil except +
		void setMuWithBinomialDistribution(double mu) nogil except +
		_Graph getGraph() except +
		_Partition getPartition() except +
		_Graph generate() except +

cdef class LFRGenerator(Algorithm):
	"""
	The LFR clustered graph generator as introduced by Andrea Lancichinetti, Santo Fortunato, and Filippo Radicchi.

	The community assignment follows the algorithm described in
	"Benchmark graphs for testing community detection algorithms". The edge generation is however taken from their follow-up publication
	"Benchmarks for testing community detection algorithms on directed and weighted graphs with overlapping communities". Parts of the
	implementation follow the choices made in their implementation which is available at https://sites.google.com/site/andrealancichinetti/software
	but other parts differ, for example some more checks for the realizability of the community and degree size distributions are done
	instead of heavily modifying the distributions.

	The edge-switching markov-chain algorithm implementation in NetworKit is used which is different from the implementation in the original LFR benchmark.

	You need to set a degree sequence, a community size sequence and a mu using the additionally provided set- or generate-methods.

	Parameters
	----------
	n : count
		The number of nodes
	"""
	params = {}
	paths = {}

	def __cinit__(self, count n):
		self._this = new _LFRGenerator(n)

	def setDegreeSequence(self, vector[count] degreeSequence):
		"""
		Set the given degree sequence.

		Parameters
		----------
		degreeSequence : collections.Iterable
			The degree sequence that shall be used by the generator
		"""
		with nogil:
			(<_LFRGenerator*>(self._this)).setDegreeSequence(degreeSequence)
		return self

	def generatePowerlawDegreeSequence(self, count avgDegree, count maxDegree, double nodeDegreeExp):
		"""
		Generate and set a power law degree sequence using the given average and maximum degree with the given exponent.


		Parameters
		----------
		avgDegree : count
			The average degree of the created graph
		maxDegree : count
			The maximum degree of the created graph
		nodeDegreeExp : double
			The (negative) exponent of the power law degree distribution of the node degrees
		"""
		with nogil:
			(<_LFRGenerator*>(self._this)).generatePowerlawDegreeSequence(avgDegree, maxDegree, nodeDegreeExp)
		return self

	def setCommunitySizeSequence(self, vector[count] communitySizeSequence):
		"""
		Set the given community size sequence.

		Parameters
		----------
		communitySizeSequence : collections.Iterable
			The community sizes that shall be used.
		"""
		with nogil:
			(<_LFRGenerator*>(self._this)).setCommunitySizeSequence(communitySizeSequence)
		return self

	def setPartition(self, Partition zeta not None):
		"""
		Set the partition, this replaces the community size sequence and the random assignment of the nodes to communities.

		Parameters
		----------
		zeta : Partition
			The partition to use
		"""
		with nogil:
			(<_LFRGenerator*>(self._this)).setPartition(zeta._this)
		return self

	def generatePowerlawCommunitySizeSequence(self, count minCommunitySize, count maxCommunitySize, double communitySizeExp):
		"""
		Generate a powerlaw community size sequence with the given minimum and maximum size and the given exponent.

		Parameters
		----------
		minCommunitySize : count
			The minimum community size
		maxCommunitySize : count
			The maximum community size
		communitySizeExp : double
			The (negative) community size exponent of the power law degree distribution of the community sizes
		"""
		with nogil:
			(<_LFRGenerator*>(self._this)).generatePowerlawCommunitySizeSequence(minCommunitySize, maxCommunitySize, communitySizeExp)
		return self

	def setMu(self, mu):
		"""
		Set the mixing parameter, this is the fraction of neighbors of each node that do not belong to the node's own community.

		This can either be one value for all nodes or an iterable of values for each node.

		Parameters
		----------
		mu : double or collections.Iterable
			The mixing coefficient(s), i.e. the factor of the degree that shall be inter-cluster degree
		"""
		if isinstance(mu, collections.Iterable):
			(<_LFRGenerator*>(self._this)).setMu(<vector[double]>mu)
		else:
			(<_LFRGenerator*>(self._this)).setMu(<double>mu)
		return self

	def setMuWithBinomialDistribution(self, double mu):
		"""
		Set the internal degree of each node using a binomial distribution such that the expected mixing parameter is the given @a mu.

		The mixing parameter is for each node the fraction of neighbors that do not belong to the node's own community.

		Parameters
		----------
		mu : double
			The expected mu that shall be used.
		"""
		with nogil:
			(<_LFRGenerator*>(self._this)).setMuWithBinomialDistribution(mu)
		return self

	def getGraph(self):
		"""
		Return the generated Graph.

		Returns
		-------
		Graph
			The generated graph.
		"""
		return Graph().setThis((<_LFRGenerator*>(self._this)).getGraph())

	def generate(self, useReferenceImplementation=False):
		"""
		Generates and returns the graph. Wrapper for the StaticGraphGenerator interface.

		Returns
		-------
		Graph
			The generated graph.
		"""
		if useReferenceImplementation:
			from networkit import graphio
			os.system("{0}/benchmark {1}".format(self.paths["refImplDir"], self.params["refImplParams"]))
			return graphio.readGraph("network.dat", graphio.Format.EdgeListTabOne)
		return Graph().setThis((<_LFRGenerator*>(self._this)).generate())

	def getPartition(self):
		"""
		Return the generated Partiton.

		Returns
		-------
		Partition
			The generated partition.
		"""
		return Partition().setThis((<_LFRGenerator*>(self._this)).getPartition())

	@classmethod
	def setPathToReferenceImplementationDir(cls, path):
		cls.paths["refImplDir"] = path


	@classmethod
	def fit(cls, Graph G, scale=1, vanilla=False, communityDetectionAlgorithm=PLM, plfit=False):
		""" Fit model to input graph"""
		(n, m) = G.size()
		# detect communities
		communities = communityDetectionAlgorithm(G).run().getPartition()
		# get degree sequence
		degSeq = DegreeCentrality(G).run().scores()
		# set number of nodes
		gen = cls(n * scale)
		if vanilla:
			# fit power law to degree distribution and generate degree sequence accordingly
			#print("fit power law to degree distribution and generate degree sequence accordingly")
			avgDegree = int(sum(degSeq) / len(degSeq))
			maxDegree = max(degSeq)
			if plfit:
				degSeqGen = PowerlawDegreeSequence(G)
				nodeDegreeExp = -1 * degSeqGen.getGamma()
				degSeqGen.run()
				gen.setDegreeSequence(degSeqGen.getDegreeSequence(n * scale))
			else:
				nodeDegreeExp = 2
				gen.generatePowerlawDegreeSequence(avgDegree, maxDegree, -1 * nodeDegreeExp)
			print(avgDegree, maxDegree, nodeDegreeExp)
			# fit power law to community size sequence and generate accordingly
			#print("fit power law to community size sequence and generate accordingly")
			communitySize = communities.subsetSizes()
			communityAvgSize = int(sum(communitySize) / len(communitySize))
			communityMaxSize = max(communitySize)
			communityMinSize = min(communitySize)

			localCoverage = LocalPartitionCoverage(G, communities).run().scores()
			mu = 1.0 - sum(localCoverage) / len(localCoverage)
			# check if largest possible internal degree can fit in the largest possible community
			if math.ceil((1.0 - mu) * maxDegree) >= communityMaxSize:
				# Make the maximum community size 5% larger to make it more likely
				# the largest generated degree will actually fit.
				communityMaxSize = math.ceil(((1.0 - mu) * maxDegree + 1) * 1.05)
				print("Increasing maximum community size to fit the largest degree")

			if plfit:
				communityExp = -1 * PowerlawDegreeSequence(communityMinSize, communityMaxSize, -1).setGammaFromAverageDegree(communityAvgSize).getGamma()
			else:
				communityExp = 1
			pl = PowerlawDegreeSequence(communityMinSize, communityMaxSize, -1 * communityExp)

			try: # it can be that the exponent is -1 because the average would be too low otherwise, increase minimum to ensure average fits.
				pl.setMinimumFromAverageDegree(communityAvgSize)
				communityMinSize = pl.getMinimumDegree()
			except RuntimeError: # if average is too low with chosen exponent, this might not work...
				pl.run()
				print("Could not set desired average community size {}, average will be {} instead".format(communityAvgSize, pl.getExpectedAverageDegree()))


			gen.generatePowerlawCommunitySizeSequence(minCommunitySize=communityMinSize, maxCommunitySize=communityMaxSize, communitySizeExp=-1 * communityExp)
			# mixing parameter
			#print("mixing parameter")
			gen.setMu(mu)
			# Add some small constants to the parameters for the reference implementation to
			# ensure it won't say the average degree is too low.
			refImplParams = "-N {0} -k {1} -maxk {2} -mu {3} -minc {4} -maxc {5} -t1 {6} -t2 {7}".format(n * scale, avgDegree + 1e-4, maxDegree, mu, max(communityMinSize, 3), communityMaxSize, nodeDegreeExp + 0.001, communityExp)
			cls.params["refImplParams"] = refImplParams
			print(refImplParams)
		else:
			if scale > 1:
				# scale communities
				cData = communities.getVector()
				cDataCopy = cData[:]
				b = communities.upperBound()
				for s in range(1, scale):
					cDataExtend = [i + (b * s) for i in cDataCopy]
					cData = cData + cDataExtend
				assert (len(cData) == n * scale)
				gen.setPartition(Partition(0, cData))
			else:
				gen.setPartition(communities)
			# degree sequence
			gen.setDegreeSequence(degSeq * scale)
			# mixing parameter
			localCoverage = LocalPartitionCoverage(G, communities).run().scores()
			gen.setMu([1.0 - x for x in localCoverage] * scale)
		return gen

cdef extern from "cpp/generators/DynamicPathGenerator.h":
	cdef cppclass _DynamicPathGenerator "NetworKit::DynamicPathGenerator":
		_DynamicPathGenerator() except +
		vector[_GraphEvent] generate(count nSteps) except +

cdef class DynamicPathGenerator:
	""" Example dynamic graph generator: Generates a dynamically growing path. """
	cdef _DynamicPathGenerator* _this

	def __cinit__(self):
		self._this = new _DynamicPathGenerator()

	def __dealloc__(self):
		del self._this

	def generate(self, nSteps):
		return [GraphEvent(ev.type, ev.u, ev.v, ev.w) for ev in self._this.generate(nSteps)]

cdef extern from "cpp/generators/DynamicDorogovtsevMendesGenerator.h":
	cdef cppclass _DynamicDorogovtsevMendesGenerator "NetworKit::DynamicDorogovtsevMendesGenerator":
		_DynamicDorogovtsevMendesGenerator() except +
		vector[_GraphEvent] generate(count nSteps) except +

cdef class DynamicDorogovtsevMendesGenerator:
	""" Generates a graph according to the Dorogovtsev-Mendes model.

 	DynamicDorogovtsevMendesGenerator()

 	Constructs the generator class.
	"""
	cdef _DynamicDorogovtsevMendesGenerator* _this

	def __cinit__(self):
		self._this = new _DynamicDorogovtsevMendesGenerator()

	def __dealloc__(self):
		del self._this

	def generate(self, nSteps):
		""" Generate event stream.

		Parameters
		----------
		nSteps : count
			Number of time steps in the event stream.
		"""
		return [GraphEvent(ev.type, ev.u, ev.v, ev.w) for ev in self._this.generate(nSteps)]

cdef extern from "cpp/generators/DynamicPubWebGenerator.h":
	cdef cppclass _DynamicPubWebGenerator "NetworKit::DynamicPubWebGenerator":
		_DynamicPubWebGenerator(count numNodes, count numberOfDenseAreas,
			float neighborhoodRadius, count maxNumberOfNeighbors) except +
		vector[_GraphEvent] generate(count nSteps) except +
		_Graph getGraph() except +

cdef class DynamicPubWebGenerator:
	cdef _DynamicPubWebGenerator* _this

	def __cinit__(self, numNodes, numberOfDenseAreas, neighborhoodRadius, maxNumberOfNeighbors):
		self._this = new _DynamicPubWebGenerator(numNodes, numberOfDenseAreas, neighborhoodRadius, maxNumberOfNeighbors)

	def __dealloc__(self):
		del self._this

	def generate(self, nSteps):
		""" Generate event stream.

		Parameters
		----------
		nSteps : count
			Number of time steps in the event stream.
		"""
		return [GraphEvent(ev.type, ev.u, ev.v, ev.w) for ev in self._this.generate(nSteps)]

	def getGraph(self):
		return Graph().setThis(self._this.getGraph())

cdef extern from "cpp/generators/DynamicHyperbolicGenerator.h":
	cdef cppclass _DynamicHyperbolicGenerator "NetworKit::DynamicHyperbolicGenerator":
		_DynamicHyperbolicGenerator(count numNodes, double avgDegree, double gamma, double T, double moveEachStep, double moveDistance) except +
		vector[_GraphEvent] generate(count nSteps) except +
		_Graph getGraph() except +
		vector[Point[float]] getCoordinates() except +

cdef class DynamicHyperbolicGenerator:
	cdef _DynamicHyperbolicGenerator* _this

	def __cinit__(self, numNodes, avgDegree = 6, gamma = 3, T = 0, moveEachStep = 1, moveDistance = 0.1):
		""" Dynamic graph generator according to the hyperbolic unit disk model.

		Parameters
		----------
		numNodes : count
			number of nodes
		avgDegree : double
			average degree of the resulting graph
		gamma : double
			power-law exponent of the resulting graph
		T : double
			temperature, selecting a graph family on the continuum between hyperbolic unit disk graphs and Erdos-Renyi graphs
		moveFraction : double
			fraction of nodes to be moved in each time step. The nodes are chosen randomly each step
		moveDistance: double
			base value for the node movements
		"""
		if gamma <= 2:
				raise ValueError("Exponent of power-law degree distribution must be > 2")
		self._this = new _DynamicHyperbolicGenerator(numNodes, avgDegree = 6, gamma = 3, T = 0, moveEachStep = 1, moveDistance = 0.1)

	def __dealloc__(self):
		del self._this

	def generate(self, nSteps):
		""" Generate event stream.

		Parameters
		----------
		nSteps : count
			Number of time steps in the event stream.
		"""
		return [GraphEvent(ev.type, ev.u, ev.v, ev.w) for ev in self._this.generate(nSteps)]

	def getGraph(self):
		return Graph().setThis(self._this.getGraph())

	def getCoordinates(self):
		""" Get coordinates in the Poincare disk"""
		return [(p[0], p[1]) for p in self._this.getCoordinates()]

cdef extern from "cpp/generators/DynamicForestFireGenerator.h":
	cdef cppclass _DynamicForestFireGenerator "NetworKit::DynamicForestFireGenerator":
		_DynamicForestFireGenerator(double p, bool directed, double r) except +
		vector[_GraphEvent] generate(count nSteps) except +
		_Graph getGraph() except +

cdef class DynamicForestFireGenerator:
	""" Generates a graph according to the forest fire model.
	 The forest fire generative model produces dynamic graphs with the following properties:
     heavy tailed degree distribution
     communities
     densification power law
     shrinking diameter

    see Leskovec, Kleinberg, Faloutsos: Graphs over Tim: Densification Laws,
    Shringking Diameters and Possible Explanations

 	DynamicForestFireGenerator(double p, bool directed, double r = 1.0)

 	Constructs the generator class.

 	Parameters
 	----------
 	p : forward burning probability.
 	directed : decides whether the resulting graph should be directed
 	r : optional, backward burning probability
	"""
	cdef _DynamicForestFireGenerator* _this

	def __cinit__(self, p, directed, r = 1.0):
		self._this = new _DynamicForestFireGenerator(p, directed, r)

	def __dealloc__(self):
		del self._this

	def generate(self, nSteps):
		""" Generate event stream.

		Parameters
		----------
		nSteps : count
			Number of time steps in the event stream.
		"""
		return [GraphEvent(ev.type, ev.u, ev.v, ev.w) for ev in self._this.generate(nSteps)]
