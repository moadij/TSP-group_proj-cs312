#!/usr/bin/python3

from PyQt5.QtCore import QLineF, QPointF



import copy
import time
import numpy as np
from TSPClasses import *
import heapq
import itertools



class TSPSolver:
	def __init__( self, gui_view ):
		self._scenario = None

	def setupWithScenario( self, scenario ):
		self._scenario = scenario


	''' <summary>
		This is the entry point for the default solver
		which just finds a valid random tour.  Note this could be used to find your
		initial BSSF.
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of solution, 
		time spent to find solution, number of permutations tried during search, the 
		solution found, and three null values for fields not used for this 
		algorithm</returns> 
	'''

	def defaultRandomTour( self, time_allowance=60.0 ):
		results = {}
		cities = self._scenario.getCities()
		ncities = len(cities)
		foundTour = False
		count = 0
		bssf = None
		start_time = time.time()
		while not foundTour and time.time()-start_time < time_allowance:
			# create a random permutation
			perm = np.random.permutation( ncities )
			route = []
			# Now build the route using the random permutation
			for i in range( ncities ):
				route.append( cities[ perm[i] ] )
			bssf = TSPSolution(route)
			count += 1
			if bssf.cost < np.inf:
				# Found a valid route
				foundTour = True
		end_time = time.time()
		results['cost'] = bssf.cost if foundTour else math.inf
		results['time'] = end_time - start_time
		results['count'] = count
		results['soln'] = bssf
		results['max'] = None
		results['total'] = None
		results['pruned'] = None
		return results


	''' <summary>
		This is the entry point for the greedy solver, which you must implement for 
		the group project (but it is probably a good idea to just do it for the branch-and
		bound project as a way to get your feet wet).  Note this could be used to find your
		initial BSSF.
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of best solution, 
		time spent to find best solution, total number of solutions found, the best
		solution found, and three null values for fields not used for this 
		algorithm</returns> 
	'''

	'''
	This function performs a greedy algorithm for finding an initial route that works, but
	likely isn't optimal. It serves as the starting point for the branchAndBound function
	below, and provides a default return solution should the time limit be reached before a
	full solution is found. It operates by selecting the smallest possible next node from the
	current node, starting with 0.
	
	This function uses createMatrix, which runs in O(n^2) time and space complexity, as well
	as creates a copy of the matrix for manipulation. There are also several loops over each
	city, resulting in O(n^2) loops as cities are visited and paths changed. Overall, the
	time and space complexity of this function are O(n^2) and O(n^2), respectively.
	'''

	def greedy( self,time_allowance=60.0 ):
		startTime = time.time()												#O(1), O(1)
		results = {}														#O(1), O(1)
		cities = self._scenario.getCities()									#O(1), O(n)
		nCities = len(cities)												#O(1), O(1)
		count = 0															#O(1), O(1)

		currentSpot = 0														#O(1), O(1)
		masterMatrix = self.createMatrix(cities)							#O(n^2), O(n^2)
		path = []															#O(1), O(1)
		cityMatrix = np.array(copy.deepcopy(masterMatrix))					#O(n^2), O(n^2)
		cityMatrix[:, currentSpot] = math.inf								#O(n), O(1)
		for city in cities:													#O(n), O(1)
			path.append(currentSpot)										#O(1), O(n)
			minColumn = np.min(cityMatrix[currentSpot])						#O(n), O(1)
			minColumnIndex = 0												#O(1), O(1)
			if minColumn == math.inf:										#O(1), O(1)
				for i in range(nCities):									#O(n), O(1)
					if i not in path:										#O(1), O(1)
						minColumnIndex = i									#O(1), O(1)
			else:															#O(1), O(1)
				minColumnIndex = np.argmin(cityMatrix[currentSpot])			#O(n), O(1)
			cityMatrix[currentSpot][minColumnIndex] = math.inf				#O(1), O(1)
			cityMatrix[currentSpot, :] = math.inf							#O(n), O(1)
			cityMatrix[:, minColumnIndex] = math.inf						#O(n), O(1)
			currentSpot = minColumnIndex									#O(1), O(1)

		route = []															#O(1), O(1)
		for i in range(nCities):											#O(n), O(1)
			route.append(cities[path[i]])									#O(1), O(n)

		bssf = TSPSolution(route)											#O(1), O(1)

		endTime = time.time()												#O(1), O(1)
		results['cost'] = bssf.cost											#O(1), O(1)
		results['time'] = endTime - startTime								#O(1), O(1)
		results['count'] = count											#O(1), O(1)
		results['soln'] = bssf												#O(1), O(1)
		results['max'] = None												#O(1), O(1)
		results['total'] = None												#O(1), O(1)
		results['pruned'] = None											#O(1), O(1)

		return results														#O(1), O(1)

	''' <summary>
			This is the entry point for the branch-and-bound algorithm that you will implement
			</summary>
			<returns>results dictionary for GUI that contains three ints: cost of best solution, 
			time spent to find best solution, total number solutions found during search (does
			not include the initial BSSF), the best solution found, and three more ints: 
			max queue size, total number of states created, and number of pruned states.</returns> 
	'''

	'''
    This function performs the branchAndBound algorithm responsible for finding the
    optimal solution for the traveling salesperson problem, or as close to optimal as
    possible within the given time allowance. It starts by using the greedy algorithm
    above to find an initial solution, which provides an upper limit for the cost, as
    no solution should be accepted if it isn't better than the solution found by the
    greedy algorithm. This allows for pruning more nodes, as the upper limit allows us
    to determine if a particular path is likely to be fruitful.

    The worst case running time of this algorithm is quite large, O((n-1)!). Without
    pruning, the full exploration tree would include a starting node, followed by
    reduced cost matrices for every other node for which there is an edge. From
    there, we would create another node for each city reachable from each state in
    the previous search, and so on, resulting in a search pattern that would give
    (n-1)! nodes. In actuality, having an upper limit from the greedy algorithm
    allows for extensive "pruning" of these nodes, as any pathway that has reached
    a cost greater than the upper limit can never result in a more optimal solution.
    As such, the running time is actually much lower.

    Another way we can increase the running time of the algorithm is by using a
    priority queue for each state to be explored, expanding those with the lowest
    cost and longest path first, in order to find a solution faster and potentially
    further reduce our upper limit.
    '''

	def branchAndBound(self, time_allowance=60.0):
		startTime = time.time()  											# O(1), O(1)
		results = {} 										 				# O(1), O(1)
		greedy = self.greedy()  											# O(n^2), O(n^2)
		lowestCost = greedy['cost']  										# O(1), O(1)
		num_updates = 0  													# O(1), O(1)
		maxHeapLen = 1  													# O(1), O(1)
		pruned = 0  														# O(1), O(1)
		totalCreated = 0  													# O(1), O(1)
		bssf = greedy['soln']  												# O(1), O(1)

		heap = []  															# O(1), O(1)
		heapq.heapify(heap)  												# O(1), O(1)

		cities = self._scenario.getCities()  								# O(1), O(n)
		nCities = len(cities)  												# O(1), O(1)

		masterMatrix = self.createMatrix(cities)  							# O(n^2), O(n^2)
		cityMatrix = np.array(copy.deepcopy(masterMatrix))  				# O(n^2), O(n^2)
		startBound = self.reduceMatrix(cityMatrix, cities)  				# O(n^2), O(n)

		startPath = []  													# O(1), O(1)
		startPath.append(0)  												# O(1), O(1)
		remCities = ([x._index for x in cities], [y._index for y in cities])  # O(n^2), O(n^2)

		'''
        When saving data to a node, there are three things that increase
        the space requirement: cityMatrix (O(n^2)), remCities (O(2n)),
        and path O(n). As more destinations are reach, path will grow
        as remCities decreases, up to a full n array. As such, the space
        requirement is better represented as the worst case of remCities,
        or O(2n).
        '''
		startNode = Node(0, cityMatrix, startBound, remCities, startPath)  	# O(1), O(2n + n^2)
		heapq.heappush(heap, (startNode.bound, startNode))  			   	# O(1), O(1)

		while len(heap) > 0 and time.time() - startTime < time_allowance:  	# O(n!), O(1)
			# load necessary data
			currentNode = heapq.heappop(heap)[1]  							# O(log n), O(1)
			if currentNode.bound < lowestCost:  							# O(1), O(1)
				currentCity = currentNode.currentCity  						# O(1), O(1)
				currentMatrix = currentNode.RCM  							# O(1), O(1)
				currentBound = currentNode.bound  							# O(1), O(1)
				currentPath = currentNode.path  							# O(1), O(1)
				remCities = currentNode.remCities  							# O(1), O(1)
				if len(currentPath) == nCities:  							# O(1), O(1)
					route = []  											# O(1), O(1)
					for i in range(nCities):  								# O(n), O(1)
						route.append(cities[currentPath[i]])  				# O(1), O(n)

					bssf = TSPSolution(route)  								# O(1), O(1)
					lowestCost = bssf.cost  								# O(1), O(1)
					num_updates += 1  										# O(1), O(1)

				# expand destinations
				for sink in remCities[1]:  									# O(n), O(1)
					if sink != currentCity:  								# O(1), O(1)
						totalCreated += 1  									# O(1), O(1)
						if currentMatrix[currentCity][sink] == math.inf:  	# O(1), O(1)
							pruned += 1  									# O(1), O(1)
						else:  												# O(1), O(1)
							newPath = copy.deepcopy(currentPath)  			# O(n), O(n)
							newPath.append(sink)  							# O(1), O(n)
							newMatrix = np.array(copy.deepcopy(currentMatrix))  # O(n^2), O(n^2)
							newRemCities = (copy.deepcopy(remCities[0]), copy.deepcopy(remCities[1]))  # O(n), O(n)
							if currentCity in newRemCities[0]:  			# O(1), O(1)
								newRemCities[0].remove(currentCity)  		# O(1), O(1)
							newRemCities[1].remove(sink)  					# O(1), O(1)
							pathReductionCost = self.reduceMatrixPath(newMatrix, cities, currentCity, sink,
																	  newRemCities)  # O(n^2), O(n)
							newNode = Node(sink, newMatrix, (currentBound + pathReductionCost), newRemCities,
										   newPath)  						# O(1), O(2n + n^2)
							if newNode.bound < lowestCost:  				# O(1), O(1)
								heapq.heappush(heap, (newNode.bound, newNode))  # O(1), O(1)
								if len(heap) > maxHeapLen:  				# O(1), O(1)
									maxHeapLen = len(heap)  				# O(1), O(1)
							else:  											# O(1), O(1)
								pruned += 1  								# O(1), O(1)
			else:  															# O(1), O(1)
				pruned += 1  												# O(1), O(1)

		pruned += len(heap)  												# O(1), O(1)

		endTime = time.time()  												# O(1), O(1)
		results['cost'] = bssf.cost 									 	# O(1), O(1)
		results['time'] = endTime - startTime  								# O(1), O(1)
		results['count'] = num_updates  									# O(1), O(1)
		results['soln'] = bssf  											# O(1), O(1)
		results['max'] = maxHeapLen  										# O(1), O(1)
		results['total'] = totalCreated  									# O(1), O(1)
		results['pruned'] = pruned  										# O(1), O(1)

		return results  													# O(1), O(1)

	''' <summary>
		This is the entry point for the algorithm you'll write for your group project.
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of best solution, 
		time spent to find best solution, total number of solutions found during search, the 
		best solution found.  You may use the other three field however you like.
		algorithm</returns> 
	'''

	def fancy( self,time_allowance=60.0 ):
		#Create two sets: one with all cities S, edges E, and path P
			#Pick a node at random (or one that hasn’t been chosen yet)
			# P += S.pop(node)
			# E += (node, node)
			# get_path(S, P, E, 0)
		# TODO: get_initial_set function that connects the first two nodes (sorted by index) that have paths
		pass

	# while path_not_found:
	# 	for c in cities:
	# 		if c._index not in path:
	#			find closest node
	#			calc incremental costs

	'''
	Initial set function returns an initial path array with a default starting node of 0 and the first
	node in Cities that creates a loop with 0. This serves as the starting point for our algorithm, and 
	can be randomized later.
	'''
	def get_initial_set(self, cityMatrix, remainingCities):
		path = [0]
		for i in range(1, len(cityMatrix[0])):
			if self._scenario._edge_exists[i][0] and self._scenario._edge_exists[0][i]:
				path.append(i)
				return path

	'''
	Helper function to calculate the incremental cost of replace an edge between fromNode and toNode.
	'''
	def calculate_cost(self, costMatrix, fromNode: int, toNode: int, newNode: int):
		cost = 0
		cost += costMatrix[fromNode][newNode]
		cost += costMatrix[newNode][toNode]
		cost -= costMatrix[fromNode][toNode]
		return cost

	'''
	Helper function to find the closest member of the path to a non-member node
	'''
	def find_closest_node(self, cities, newNode, path):
		for i in range(len(path)):
			list.append(math.sqrt(((cities[i]._x - cities[newNode]._x) **2) + ((cities[i]._y - cities[newNode]._y) **2)))
		return path[list.index(min(list))]

	'''
	Helper function createMatrix creates a cost matrix based on all cities contained in the
	scenario. It first creates a 2d matrix filled with infinity in every cell. Then, using
	a nested for loop, each cell is updated based on the edges contained in the scenario.

	As the matrix is made in this function, the space complexity is O(n^2). After the
	matrix is created, values are simply looked up and replaced in the matrix, meaning
	the rest of the function runs with O(1) space complexity. Because two nested loops are
	required to initialize and populate the matrix, the running time is O(2*n^2). In
	conclusion, the total time and space complexity of this function are both O(n^2).
	'''
	def createMatrix(self, cities):
		myMatrix = [[math.inf for x in cities] for x in cities]				#O(n^2), O(n^2)
		for i in cities:													#O(n), O(1)
			for j in cities:												#O(n), O(1)
				if self._scenario._edge_exists[i._index][j._index]:			#O(1), O(1)
					first = cities[i._index]								#O(1), O(1)
					second = cities[j._index]								#O(1), O(1)
					pathCost = first.costTo(second)							#O(1), O(1)
					myMatrix[i._index][j._index] = pathCost					#O(1), O(1)
		return myMatrix														#O(1), O(1)

	'''
	Helper function reduceMatrix reduces a full cost matrix for all rows and columns. This
	particular function is only used for finding the initial reduced cost matrix, RCMs
	related to a node placement are determined using reduceMatrixPath, below.

	As the values are being changed on a matrix that already exists for most parts of this
	function, the space complexity related to the matrix is O(1). When creating an array of
	minimum values (minRow and minColumn), there is one min per city which makes a space
	complexity of O(n) each. The total time and space complexity of this function are
	O(n^2) and O(n), respectively.
	'''
	def reduceMatrix(self, matrix, cities):
		minRow = np.min(matrix, axis=1)														#O(1), O(n)
		bound = np.sum(minRow)																#O(1), O(1)
		for i in cities:																	#O(n), O(1)
			for j in cities:																#O(n), O(1)
				matrix[i._index][j._index] = matrix[i._index][j._index] - minRow[i._index]	#O(1), O(1)
		minColumn = np.min(matrix, axis=0)													#O(1), O(n)
		bound += np.sum(minColumn)															#O(1), O(1)
		for i in cities:																	#O(n), O(1)
			for j in cities:																#O(n), O(1)
				matrix[i._index][j._index] = matrix[i._index][j._index]-minColumn[j._index]	#O(1), O(1)
		return bound																		#O(1), O(1)

	'''
	Helper function reduceMatrixPath reduces a cost matrix based on including a particular
	node, as passed in through pathRow and pathColumn. remCities is passed in as well, for
	reducing the correct rows and columns passed on the remaining city possibilities.

	As the values are being changed on a matrix that already exists for most parts of this
	function, the space complexity related to the matrix is O(1). When creating an array of
	minimum values (minRow and minColumn), there is one min per city which makes a space
	complexity of O(n) each. The total time and space complexity of this function are
	O(n^2) and O(n), respectively.
	'''
	def reduceMatrixPath(self, matrix, cities, pathRow, pathColumn, remCities):
		# calculate the path cost, set row/column to inf
		bound = matrix[pathRow][pathColumn]													#O(1), O(1)
		matrix[:, pathColumn] = math.inf													#O(1), O(1)
		matrix[pathRow, :] = math.inf														#O(1), O(1)
		for i in range(len(cities)):														#O(n), O(1)
			if i not in remCities[1]:														#O(1), O(1)
				matrix[pathColumn][i] = math.inf											#O(1), O(1)

		# calculate the update cost, reduce matrix
		minRow = np.min(matrix, axis=1)														#O(1), O(n)
		for source in remCities[0]:															#O(n), O(1)
			bound += minRow[source]															#O(1), O(1)
			for dest in cities:																#O(n), O(1)
				matrix[source][dest._index] = matrix[source][dest._index] - minRow[source]	#O(1), O(1)
		minColumn = np.min(matrix, axis=0)													#O(1), O(n)
		for sink in remCities[1]:															#O(n), O(1)
			bound += minColumn[sink]														#O(1), O(1)
			for origin in cities:															#O(1), O(1)
				matrix[origin._index][sink] = matrix[origin._index][sink] - minColumn[sink]	#O(1), O(1)

		return bound																		#O(1), O(1)

class Node:
	def __init__(self,
				 currentCity = None,
				 RCM = None,
				 bound = None,
				 remCities = None,
				 path = None,
				 edge = None):
		self.currentCity = currentCity
		self.RCM = RCM
		self.bound = bound
		self.remCities = remCities
		self.path = path
		self.edge = edge

	def __lt__(self, other):
		assert(type(other) == Node)
		return len(self.path) > len(other.path)

class Edge:
	def __init__(self,
				 to_node = None,
				 from_node = None,
				 cost_to_node = None,
				 cost_from_node = None):
		self.cost_to_node = cost_to_node
		self.cost_from_node = cost_from_node
		#what else fo we need here?
