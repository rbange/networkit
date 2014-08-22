/*
 * Partition.h
 *
 *  Created on: 03.10.2013
 *      Author: cls
 */

#ifndef PARTITION_H_
#define PARTITION_H_

#include <cinttypes>
#include <set>
#include <vector>
#include <map>
#include <cassert>
#include <limits>

#include "../graph/Graph.h"



namespace NetworKit {



/**
 * @ingroup structures
 * Implements a partition of a set, i.e. a subdivision of the
 * set into disjoint subsets.
 */
class Partition {

public:

	Partition();

	/**
	 * Create a new partition data structure for @a z elements.
	 *
	 * @param[in]	z	maximum index
	 */
	Partition(index z);

		/**
	 * Create a new partition data structure for @a z elements.
	 * Initialize each entry to the default value.
	 * WARNING: this circumvents the standard interface and may leave the object
	 * in an inconsistent state. Use only in exceptional cases.
	 *
	 * @param[in]	z	maximum index
	 * @param[in]	defaultValue
	 */
	Partition(index z, index defaultValue);

	virtual ~Partition() = default;

	/**
	 *  Index operator.
	 *
	 *  @param[in]	e	an element
	 */
	inline index& operator [](const index& e) {
		return this->data[e];
	}
	/**
	 * Index operator for const instances of this class.
	 *
	 * @param[in]	e	an element
	 */
	inline const index& operator [](const index& e) const {
		return this->data[e];
	}

	/**
	 * Get the set (id) in which the element @a e is contained.
	 *
	 * @param e Index of element.
	 * @return The index of the set in which @a e is contained.
	 */
	inline index subsetOf(index e) const {
		assert (e < this->numberOfElements());
		return this->data[e];
	}


	/**
	 * Extend the data structure and create a slot
	 * for one more element. Initializes the entry to none
	 * and returns the index of the entry.
	 */
	inline index extend() {
		data.push_back(none);
		z++;
		assert (z == data.size()); //(data.size() - 1)
		return z-1;
	}


	/**
	 * Removes the entry for the given element
	 * by setting it to none.
	 */
	inline void remove(index e) {
		assert (e < z);
		data[e] = none;
	}

	/**
	 * Add a (previously unassigned) element @a e to the set @a s.
	 *
	 * @param s The index of the subset.
	 * @param e The element to add.
	 */
	inline void addToSubset(index s, index e) {
		assert (data[e] == none);	// guarantee that element was unassigned
		assert (s <= omega);		// do not create new subset ids
		data[e] = s;
	}


	/**
	 * Move the (previously assigned) element @a e to the set @a s.
	 *
	 * @param s The index of the subset.
	 * @param e The element to move.
	 */
	inline void moveToSubset(index s, index e) {
		assert (this->contains(e));
		assert (s <= omega); 		// do not create new subset ids
		data[e] = s;
	}

	/**
	 * Creates a singleton set containing the element @a e.
	 *
	 * @param e The index of the element.
	 */
	inline void toSingleton(index e) {
		data[e] = newSubsetId();
	}

	/**
	 * Assigns every element to a singleton set.
	 * Set id is equal to element id.
	 */
	void allToSingletons();

	/**
	 * Assigns every element to the same subset.
	 * Set id is equal to zero.
	 */
	void allToOnePartition();

	/**
	 * Assigns the elements from both sets to a new set and returns the id of it.
	 *
	 * @param s Set to merge.
	 * @param t Set to merge.
	 * @return Id of newly created set.
	 */
	index mergeSubsets(index s, index t);


	/**
	 * Check if partition is a 1-partition,
	 * i.e. every element is assigned to the same set.
	 */
	//bool isOnePartition(Graph& G);


	/**
	 * Check if partition is a singleton partition,
	 * i.e. every element is assigned to a different set.
	 */
	//bool isSingletonPartition(Graph& G) const;

	/**
	 * Sets an upper bound for the subset ids that CAN be assigned.
	 *
	 * @param[in]	upper	highest assigned subset ID + 1
	 */
	inline void setUpperBound(index upper) {
		this->omega = upper-1;
	}

	/**
	 * Return an upper bound for the subset ids that have been assigned.
	 * (This is the maximum id + 1.)
	 *
	 * @return The upper bound.
	 */
	inline index upperBound() const {
		return omega+1;
	}

	/**
	 * Get a lower bound for the subset ids that have been assigned.
	 *
	 * @return The lower bound.
	 */
	inline index lowerBound() const {
		return 0;
	}


	/**
	 * Change subset IDs to be consecutive, starting at 0.
	 */
	void compact();


	/**
	 * Check if partition assigns a valid subset to the element @a e.
	 *
	 * @param e The element.
	 * @return @c true if the assigned subset is valid, @c false otherwise.
	 */
	inline bool contains(index e) const {
		return (e < z) && (data[e] != none);	// e is in the element index range and the entry is not empty
	}


	/**
	 * Check if two elements @a e1 and @a e2 belong to the same subset.
	 *
	 * @param e1 Element.
	 * @param e2 Element.
	 * @return @c true if @a e1 and @a e2 belong to same subset, @c false otherwise.
	 */
	inline bool inSameSubset(index e1, index e2) const {
		assert (data[e1] != none);
		assert (data[e2] != none);
		return (data[e1] == data[e2]);
	}


	/**
	 * Get a list of subset sizes. Indices do not necessarily correspond to subset ids.
	 *
	 * @return A vector of subset sizes.
	 */
	std::vector<count> subsetSizes() const;


	/**
	 * Get a map from subset id to size of the subset.
	 *
	 * @return A map from subset id to size of the subset.
	 */
	std::map<index, count> subsetSizeMap() const;


	/**
	 * Get the members of the subset @a s.
	 *
	 * @param s The subset.
	 * @return A set containing the members of @a s.
	 */
	std::set<index> getMembers(const index s) const;


	/**
	 * @return number of elements in the partition.
	 */
	inline count numberOfElements() const {
		return z;	// z is the maximum element id
	}


	/**
	 * Get the current number of sets in this partition.
	 *
	 * @return The current number of sets.
	 */
	count numberOfSubsets() const;

	/**
	 * Get the actual vector representing the partition data structure.
	 * @return vector containing information about partitions.
	 */
	std::vector<index> getVector() const;


	/**
	 * @return the subsets of the partition as a set of sets.
	 */
	std::set<std::set<index> > getSubsets();


	/**
	 * Get the ids of nonempty subsets.
	 *
	 * @return A set of ids of nonempty subsets.
	 */
	std::set<index> getSubsetIds();

	/**
	 * Set a human-readable identifier @a name for the instance.
	 *
	 * @param name The name.
	 */
	inline void setName(std::string name) {
		this->name = name;
	}


	/**
	 * Get the human-readable identifier.
	 *
	 * @return The name of this partition.
	 */
	inline std::string getName() const {
		return this->name;
	}

	/**
	 * Iterate over all entries (node, cluster id) and execute callback function @a func (lambda closure).
	 *
	 * @param func Takes parameters <code>(node, index)</code>
	 */
	template<typename Callback> void forEntries(Callback func) const;

	/**
	 * Iterate over all entries (node, cluster id) in parallel and execute callback function @a handle (lambda closure).
	 *
	 * @param handle Takes parameters <code>(node, index)</code>
	 */
	template<typename Callback> void parallelForEntries(Callback handle) const;


private:
	index z;	//!< maximum element index that can be mapped
	index omega;	//!< maximum subset index ever assigned
	std::vector<index> data;  	//!< data container, indexed by element index, containing subset index
	std::string name;

	/**
	 * Allocates and returns a new subset id.
	 */
	inline index newSubsetId() {
		index s = ++omega;
		return s;
	}
};

template<typename Callback>
inline void Partition::forEntries(Callback handle) const {
	for (index e = 0; e < this->z; e++) {
		handle(e, data[e]);
	}
}

template<typename Callback>
inline void Partition::parallelForEntries(Callback handle) const {
	#pragma omp parallel for
	for (index e = 0; e < this->z; e++) {
		handle(e, this->data[e]);
	}
}

} /* namespace NetworKit */

#endif /* PARTITION_H_ */