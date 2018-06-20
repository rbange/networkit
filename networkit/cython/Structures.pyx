'''
	Module: structures
'''

cdef extern from "cpp/structures/Partition.h":
	cdef cppclass _Partition "NetworKit::Partition":
		_Partition() except +
		_Partition(index) except +
		_Partition(_Partition) except +
		_Partition(vector[index]) except +
		index subsetOf(index e) except +
		index extend() except +
		void remove(index e) except +
		void addToSubset(index s, index e) except +
		void moveToSubset(index s, index e) except +
		void toSingleton(index e) except +
		void allToSingletons() except +
		void mergeSubsets(index s, index t) except +
		void setUpperBound(index upper) except +
		index upperBound() except +
		index lowerBound() except +
		void compact(bool useTurbo) except +
		bool contains(index e) except +
		bool inSameSubset(index e1, index e2) except +
		vector[count] subsetSizes() except +
		map[index, count] subsetSizeMap() except +
		set[index] getMembers(const index s) except +
		count numberOfElements() except +
		count numberOfSubsets() except +
		vector[index] getVector() except +
		void setName(string name) except +
		string getName() except +
		set[index] getSubsetIds() except +
		index operator[](index) except +

cdef class Partition:
	""" Implements a partition of a set, i.e. a subdivision of the
 		set into disjoint subsets.

 		Partition(z=0)

 		Create a new partition data structure for `z` elements.

		Parameters
		----------
		size : index, optional
			Maximum index of an element. Default is 0.
	"""
	cdef _Partition _this

	def __cinit__(self, index size=0, vector[index] data=[]):
		if data.size() != 0:
			self._this = move(_Partition(data))
		else:
			self._this = move(_Partition(size))

	def __len__(self):
		"""
		Returns
		-------
		count
			Number of elements in the partition.
		"""
		return self._this.numberOfElements()

	def __getitem__(self, index e):
		""" Get the set (id) in which the element `e` is contained.

	 	Parameters
	 	----------
	 	e : index
	 		Index of element.

	 	Returns
	 	-------
	 	index
	 		The index of the set in which `e` is contained.
		"""
		return self._this.subsetOf(e)

	def __setitem__(self, index e, index s):
		""" Set the set (id) in which the element `e` is contained.

		Parameters
		----------
		e : index
			Index of the element
		s : index
			Index of the subset
		"""
		self._this.addToSubset(s, e)

	def __copy__(self):
		"""
		Generates a copy of the partition
		"""
		return Partition().setThis(_Partition(self._this))

	def __deepcopy__(self):
		"""
		Generates a copy of the partition
		"""
		return Partition().setThis(_Partition(self._this))

	cdef setThis(self,  _Partition& other):
		swap[_Partition](self._this,  other)
		return self

	def subsetOf(self, e):
		""" Get the set (id) in which the element `e` is contained.

	 	Parameters
	 	----------
	 	e : index
	 		Index of element.

	 	Returns
	 	-------
	 	index
	 		The index of the set in which `e` is contained.
		"""
		return self._this.subsetOf(e)

	def extend(self):
		""" Extend the data structure and create a slot	for one more element.

		Initializes the entry to `none` and returns the index of the entry.

		Returns
		-------
		index
			The index of the new element.
		"""
		return self._this.extend()

	def addToSubset(self, s, e):
		""" Add a (previously unassigned) element `e` to the set `s`.

		Parameters
		----------
		s : index
			The index of the subset.
		e : index
			The element to add.
		"""
		self._this.addToSubset(s, e)

	def moveToSubset(self, index s, index e):
		"""  Move the (previously assigned) element `e` to the set `s.

		Parameters
		----------
		s : index
			The index of the subset.
		e : index
			The element to move.
		"""
		self._this.moveToSubset(s, e)

	def toSingleton(self, index e):
		""" Creates a singleton set containing the element `e`.

		Parameters
		----------
		e : index
			The index of the element.
		"""
		self._this.toSingleton(e)

	def allToSingletons(self):
		""" Assigns every element to a singleton set. Set id is equal to element id. """
		self._this.allToSingletons()

	def mergeSubsets(self, index s, index t):
		""" Assigns the elements from both sets to a new set and returns the id of it.

		Parameters
		----------
		s : index
			Set to merge.
		t : index
			Set to merge.

		Returns
		-------
		index
			Id of newly created set.
		"""
		self._this.mergeSubsets(s, t)

	def setUpperBound(self, index upper):
		""" Sets an upper bound for the subset ids that **can** be assigned.

		Parameters
		----------
		upper : index
			Highest assigned subset id + 1
		"""
		self._this.setUpperBound(upper)

	def upperBound(self):
		""" Return an upper bound for the subset ids that have been assigned.
	 	(This is the maximum id + 1.)

	 	Returns
	 	-------
	 	index
	 		The upper bound.
		"""
		return self._this.upperBound()

	def lowerBound(self):
		""" Get a lower bound for the subset ids that have been assigned.

		Returns
		-------
		index
			The lower bound.
		"""
		return self._this.lowerBound()

	def compact(self, useTurbo = False):
		""" Change subset IDs to be consecutive, starting at 0.

		Parameters
		----------
		useTurbo : bool
			Default: false. If set to true, a vector instead of a map to assign new ids
	 		which results in a shorter running time but possibly a large space overhead.

		"""
		self._this.compact(useTurbo)

	def contains(self, index e):
		""" Check if partition assigns a valid subset to the element `e`.

		Parameters
		----------
		e : index
			The element.

		Returns
		-------
		bool
			True if the assigned subset is valid, False otherwise.
		"""
		return self._this.contains(e)

	def inSameSubset(self, index e1, index e2):
		""" Check if two elements `e1` and `e2` belong to the same subset.

		Parameters
		----------
		e1 : index
			An Element.
		e2 : index
			An Element.

		Returns
		-------
		bool
			True if `e1` and `e2` belong to same subset, False otherwise.
		"""
		return self._this.inSameSubset(e1, e2)

	def subsetSizes(self):
		""" Get a list of subset sizes. Indices do not necessarily correspond to subset ids.

	 	Returns
	 	-------
	 	vector
	 		A vector of subset sizes.
		"""
		return self._this.subsetSizes()

	def subsetSizeMap(self):
		""" Get a map from subset id to size of the subset.

		Returns
		-------
		dict
			A map from subset id to size of the subset.
		"""
		return self._this.subsetSizeMap()

	def getMembers(self, s):
		""" Get the members of the subset `s`.

		Parameters
		----------
		s : index
			The subset.

		Returns
		-------
		set
			A set containing the members of `s.
		"""
		return self._this.getMembers(s)

	def numberOfElements(self):
		"""
		Returns
		-------
		count
			Number of elements in the partition.
		"""
		return self._this.numberOfElements()

	def numberOfSubsets(self):
		""" Get the current number of sets in this partition.

		Returns
		-------
		count
			The current number of sets.
		"""
		return self._this.numberOfSubsets()

	def getVector(self):
		""" Get the actual vector representing the partition data structure.

		Returns
		-------
		vector
			Vector containing information about partitions.
		"""
		return self._this.getVector()

	def setName(self, string name):
		"""  Set a human-readable identifier `name` for the instance.

		Parameters
		----------
		name : string
			The name.
		"""
		self._this.setName(name)

	def getName(self):
		""" Get the human-readable identifier.

		Returns
		-------
		string
			The name of this partition.
		"""
		return self._this.getName()

	def getSubsetIds(self):
		""" Get the ids of nonempty subsets.

		Returns
		-------
		set
			A set of ids of nonempty subsets.
		"""
		return self._this.getSubsetIds()


cdef extern from "cpp/structures/Cover.h":
	cdef cppclass _Cover "NetworKit::Cover":
		_Cover() except +
		_Cover(_Partition p) except +
		_Cover(count n) except +
		set[index] subsetsOf(index e) except +
#		index extend() except +
		void remove(index e) except +
		void addToSubset(index s, index e) except +
		void removeFromSubset(index s, index e) except +
		void moveToSubset(index s, index e) except +
		void toSingleton(index e) except +
		void allToSingletons() except +
		void mergeSubsets(index s, index t) except +
		void setUpperBound(index upper) except +
		index upperBound() except +
		index lowerBound() except +
#		void compact() except +
		bool contains(index e) except +
		bool inSameSubset(index e1, index e2) except +
		vector[count] subsetSizes() except +
		map[index, count] subsetSizeMap() except +
		set[index] getMembers(const index s) except +
		count numberOfElements() except +
		count numberOfSubsets() except +
#		vector[index] getVector() except +
#		void setName(string name) except +
#		string getName() except +
		set[index] getSubsetIds() except +


cdef class Cover:
	""" Implements a cover of a set, i.e. an assignment of its elements to possibly overlapping subsets. """
	cdef _Cover _this

	def __cinit__(self, n=0):
		if isinstance(n, Partition):
			self._this = move(_Cover((<Partition>n)._this))
		else:
			self._this = move(_Cover(<count?>n))

	cdef setThis(self, _Cover& other):
		swap[_Cover](self._this, other)
		return self

	def subsetsOf(self, e):
		""" Get the ids of subsets in which the element `e` is contained.

		Parameters
		----------
		e : index
			An element

		Returns
		-------
		set
			A set of subset ids in which `e` 	is contained.
		"""
		return self._this.subsetsOf(e)

#	def extend(self):
#		self._this.extend()

	def addToSubset(self, s, e):
		""" Add the (previously unassigned) element `e` to the set `s`.

		Parameters
		----------
		s : index
			A subset
		e : index
			An element
		"""
		self._this.addToSubset(s, e)

	def removeFromSubset(self, s, e):
		""" Remove the element `e` from the set `s`.

		Parameters
		----------
		s : index
			A subset
		e : index
			An element
		"""
		self._this.removeFromSubset(s, e)

	def moveToSubset(self, index s, index e):
		""" Move the element `e` to subset `s`, i.e. remove it from all other subsets and place it in the subset.

		Parameters
		----------
		s : index
			A subset
		e : index
			An element
		"""
		self._this.moveToSubset(s, e)

	def toSingleton(self, index e):
		""" Creates a singleton set containing the element `e` and returns the index of the new set.

		Parameters
		----------
		e : index
			An element

		Returns
		-------
		index
			The index of the new set.
		"""
		self._this.toSingleton(e)

	def allToSingletons(self):
		""" Assigns every element to a singleton set. Set id is equal to element id. """
		self._this.allToSingletons()

	def mergeSubsets(self, index s, index t):
		""" Assigns the elements from both sets to a new set.

		Parameters
		----------
		s : index
			A subset
		t : index
			A subset
		"""
		self._this.mergeSubsets(s, t)

	def setUpperBound(self, index upper):
		self._this.setUpperBound(upper)

	def upperBound(self):
		""" Get an upper bound for the subset ids that have been assigned.
	   	(This is the maximum id + 1.)

	   	Returns
	   	-------
	   	index
	   		An upper bound.
		"""
		return self._this.upperBound()

	def lowerBound(self):
		""" Get a lower bound for the subset ids that have been assigned.

		Returns
		-------
		index
			A lower bound.
		"""
		return self._this.lowerBound()

#	def compact(self):
#		self._this.compact()

	def contains(self, index e):
		"""  Check if cover assigns a valid subset to the element `e`.

		Parameters
		----------
		e : index
			An element.

		Returns
		-------
		bool
			True, if `e` is assigned to a valid subset, False otherwise.

		"""
		return self._this.contains(e)

	def inSameSubset(self, index e1, index e2):
		"""  Check if two elements `e1` and `e2` belong to the same subset.

	 	Parameters
	 	----------
	 	e1 : index
			An element.
		e2 : index
			An element.

		Returns
		-------
		bool
			True, if `e1` and `e2` belong to the same subset, False otherwise.
		"""
		return self._this.inSameSubset(e1, e2)

	def subsetSizes(self):
		""" Get a list of subset sizes.

		Returns
		-------
		list
			A list of subset sizes.

		Notes
		-----
		Indices do not necessarily correspond to subset ids.
		"""
		return self._this.subsetSizes()

	def subsetSizeMap(self):
		""" Get a map from subset id to size of the subset.

	 	Returns
	 	-------
	 	dict
	 		A map from subset id to size of the subset.
		"""
		return self._this.subsetSizeMap()

	def getMembers(self, s):
		""" Get the members of a specific subset `s`.

		Returns
		-------
		set
			The set of members of subset `s`.
		"""
		return self._this.getMembers(s)

	def numberOfElements(self):
		""" Get the current number of elements in this cover.

		Returns
		-------
		count
			The current number of elements.
		"""
		return self._this.numberOfElements()

	def numberOfSubsets(self):
		"""  Get the current number of sets in this cover.

		Returns
		-------
		count
			The number of sets in this cover.
		"""
		return self._this.numberOfSubsets()

#	def getVector(self):
#		return self._this.getVector()

#	def setName(self, string name):
#		self._this.setName(name)

#	def getName(self):
#		return self._this.getName()

	def getSubsetIds(self):
		""" Get the ids of nonempty subsets.

		Returns
		-------
		set
			A set of ids of nonempty subsets.
		"""
		return self._this.getSubsetIds()# Module: structures
