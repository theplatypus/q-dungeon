
from .edge import Edge
from .room import Room

import random
import hashlib

import numpy as np
import pandas as pd

class Dungeon : 
	
	def __init__(self, rows = 4, cols = 4, edges = True) :
		self.rows = rows
		self.cols = cols
		self.rooms = {}
		self.edges = {}
		self.adjacency_list = {}
		self.keypoints = {"In" : None, "Out" : None, "*" : None}

		for (i, j, index) in self.room_iterator() :
			self.add_room(Room(i, j, label = index))
		
		for (A, B, edge_label) in self.edge_iterator() :
			self.add_edge(Edge(A, B, label = edge_label, is_passable = not edges))

	def __repr__(self) : 
		return self.__str__()

	def __str__(self) : 
		rooms = self.as_vector(lambda room : "%s " % (room.keypoint[:1]) if room.keypoint else "  ")
		edges_hor = [ "|" if not self.edges[edge_index].is_passable else " " for (i, j, edge_index) in self.edge_iterator(vertical=False)]
		edges_vert = [ "--" if not self.edges[edge_index].is_passable else "  " for (i, j, edge_index) in self.edge_iterator(horizontal=False)]

		n = self.cols + (self.cols - 1)
		m = self.rows + (self.rows - 1)
		mat = [ "--".join([":"] * (self.cols +1)) ]
		for j in range(m) :
			row = [":"] if j % 2 != 0 else ["|"]
			for i in range(n) :
				if j % 2 != 0 :
					if i % 2 == 0 :
						row.append(edges_vert.pop(0))
					else :
						row.append(":")
				elif i % 2 != 0 : 
					row.append(edges_hor.pop(0))
				else :
					row.append(rooms.pop(0))
			row.append(":" if j % 2 != 0 else "|")
			mat.append(row)
		mat.append("--".join([":"] * (self.cols+1)))
		return '\n'.join([''.join([cell for cell in row]) for row in mat])

	def compute_index(self, i, j) : 
		return str(((j - 1)* self.cols) + i)

	def room_iterator(self, flat = True) : 

		for j in range(1, self.rows + 1) :
			if flat :
				for i in range(1, self.cols + 1) :
					index = self.compute_index(i, j)
					yield (i, j, index)
			else :
				yield [(i, j, self.compute_index(i, j)) for i in range(1, self.cols + 1)]

	def edge_iterator(self, horizontal = True, vertical = True) :

		for (i, j, index_A) in self.room_iterator():

			if i < self.cols and horizontal :
				index_B = self.compute_index(i+1, j)
				yield (index_A, index_B, "%s-%s" % (index_A, index_B))

			if j < self.rows and vertical :
				index_B = self.compute_index(i, j+1)
				yield (index_A, index_B, "%s-%s" % (index_A, index_B))

	def add_room(self, room) :
		"""
		add_room

		We make sure to update the instance state where it matters
		"""
		self.rooms[room.label] = room
		self.adjacency_list[room.label] = []
		return True
	
	def edit_room(self, id_room, keypoint) :
		assert id_room in self.rooms
		assert keypoint in ['In', 'Out', '*']

		self.rooms[id_room].keypoint = keypoint

		# a keypoint must be unique in dungeon
		if self.keypoints[keypoint] :
			self.rooms[self.keypoints[keypoint]].keypoint = None
		
		self.keypoints[keypoint] = id_room
		return True

	def add_edge(self, edge) :

		assert edge.A in self.rooms
		assert edge.B in self.rooms

		self.edges[edge.label] = edge
		self.update_adjacency_list(edge)
		return True

	def edit_edge(self, id_edge, is_passable) : 

		assert id_edge in self.edges
		# edit object 
		self.edges[id_edge].is_passable = is_passable
		# update adjacency list 
		self.update_adjacency_list(self.edges[id_edge])
		return True

	def update_adjacency_list(self, edge) :
		if edge.is_passable :
			if edge.B not in self.adjacency_list[edge.A] :
				self.adjacency_list[edge.A].append(edge.B)
			if edge.A not in self.adjacency_list[edge.B] and not edge.is_directed :
				self.adjacency_list[edge.B].append(edge.A)
		else :
			if edge.B in self.adjacency_list[edge.A] :
				self.adjacency_list[edge.A].remove(edge.B)
			if edge.A in self.adjacency_list[edge.B] and not edge.is_directed :
				self.adjacency_list[edge.B].remove(edge.A)
		return True

	# Encoding related functions

	def get_adjacency_list(self): 
		return self.adjacency_list
	
	def as_vector(self, fn = lambda room : room.label) :
		return [ fn(self.rooms[index]) for (i,j, index) in self.room_iterator()]

	def as_matrix(self, fn = lambda room : room.label) :
		return [
			[ fn(self.rooms[index]) for (i,j, index) in [ col for col in cols ] ] 
			for cols in [ row for row in self.room_iterator(flat=False)]
		]
	
	def as_adj_matrix(self) : 
		N = self.rows * self.cols 
		A = [ [0 for col in range(N)] for row in range(N)]
		lbl2index = lambda label : int(label) - 1

		# Diagonal Values
		keypoints = [ lbl2index(i) for i in self.keypoints.values()]
		for i in keypoints :
			A[i][i] = 1
		
		# Adjacency
		for src, dests in self.get_adjacency_list().items() :
			src_i = lbl2index(src)
			for dest in dests :
				dest_i = lbl2index(dest)
				A[src_i][dest_i] = 1
		return A

	def as_tuple(self, strict = False):

		if strict :
			keypoints = sorted( [(key, sorted(self.keypoints[key])) for key in sorted(self.keypoints)])
		else :
			keypoints = sorted(list(self.keypoints.values()))

		adjacency = [ (key, sorted(self.get_adjacency_list()[key])) for key in sorted(self.get_adjacency_list()) ]
		
		return (keypoints, adjacency)

	def get_state_vector(self) :
		"""
		Minimal-size state vector.
		"""
		keypoints = [ 1 if self.rooms[index].keypoint else 0 for (i, j, index) in self.room_iterator() ]
		edges = [ 1 if self.edges[i].is_passable else 0 for (A, B, i) in self.edge_iterator() ]
		return keypoints + edges

	def state_fingerprint(self, use_hash = True) :
		(keypoints, adjacency) = self.as_tuple()

		state_str = "%s %s" % (keypoints, adjacency)
		if use_hash :
			#state_str = hashlib.sha256(state_str.encode('utf-8'), digest_size = 4).hexdigest()
			state_str = hashlib.blake2s(state_str.encode('utf-8'), digest_size = 8).hexdigest()
		return state_str


	# Rewards related functions

	def get_number_walls(self) :
		return len([ self.edges[i] for (A, B, i) in self.edge_iterator() if not self.edges[i].is_passable])

	def shortest_path(self, start, goal):

		if start == goal:
			return [start]
		
		graph = self.get_adjacency_list()
		explored = []
		queue = [[start]]
		
		while queue :
			path = queue.pop(0)
			node = path[-1]

			if node not in explored:
				neighbours = graph[node]
				
				for neighbour in neighbours:
					new_path = list(path)
					new_path.append(neighbour)
					queue.append(new_path)
					
					if neighbour == goal:
						return new_path
				explored.append(node)

		return None

	def room_reach(self, room_label, visited = None):
		"""
		room_reach

		Number of rooms accessible from a given room
		"""
		assert room_label
		visited = [] if visited is None else visited
		#print("Computing reach from %s ; visited = %s" % (room_label, visited))
		visited.append(room_label)
		reach = [room_label]
		for neighbour in self.adjacency_list[room_label] :
			#print("Looking at %s..." % (neighbour))
			if neighbour not in visited :
				#print("Visiting %s..." % (neighbour))
				reach_neighbor = self.room_reach(neighbour, visited = visited)
				#print("%s returned %s" % (neighbour, reach_neighbor))
				reach += reach_neighbor
		return reach

	def keypoints_reach(self) : 
		return sum([ len(self.room_reach(keypoint)) for keypoint in self.keypoints.values()])
	
	def compute_treasure_path(self) :
		if self.keypoints["In"] and self.keypoints["*"] :
			return self.shortest_path(self.keypoints["In"], self.keypoints["*"])
		else :
			return None
	
	def compute_exit_path(self) :
		if self.keypoints["In"] and self.keypoints["Out"] :
			return self.shortest_path(self.keypoints["In"], self.keypoints["Out"])
		else : 
			return None

	def compute_difficulty(self) : 

		if not self.keypoints["In"] or not self.keypoints["Out"] or not self.keypoints["*"] :
			 return None
		
		treasure_path = self.compute_treasure_path()
		out_path = self.compute_exit_path()

		if not treasure_path or not out_path :
			return None
		
		return len(treasure_path) + len(out_path)

	def connex_compounds(self) :
		return list(set(
				[ str(sorted(reach)) for reach in 
					[ self.room_reach(room) for i, j, room in self.room_iterator() ] 
				]))

	def is_valid(self) : 
		if not self.keypoints["In"] :
			return False
		if not self.keypoints["Out"] :
			return False
		if not self.keypoints["*"] :
			return False
		if not self.shortest_path(self.keypoints['In'], self.keypoints['*']) :
			return False
		if not self.shortest_path(self.keypoints['In'], self.keypoints['Out']) :
			return False
		return True

	# RANDOM GENERATION

	def place_random_keypoints(self, seed = None):
		if seed :
			random.seed(seed)
		keypoints = random.sample(list(self.rooms), 3)
		for keypoint in keypoints :
			self.place_keypoint(keypoint)
		return True
	
	def open_random_walls(self, walls_to_open, seed = None) :
		if seed :
			random.seed(seed)
		walls = random.sample(list(self.edges), walls_to_open)
		for wall in walls :
			self.open_wall(wall)
		return True
		
	def randomize(self, walls_to_open = 10, seed = None):
		if seed :
			random.seed(seed)
		self.place_random_keypoints()
		self.open_random_walls(walls_to_open)
		return True

	# AGENT ACTIONS

	def open_wall(self, edge_label) :
		self.edit_edge(edge_label, is_passable = True)

	def close_wall(self, edge_label) :
		self.edit_edge(edge_label, is_passable = False)

	def toggle_wall_down(self, room_label) :
		room = self.rooms[room_label]
		if room.j == self.rows :
			# no room down -> Illegal
			return False
		i_down = room.i
		j_down = room.j + 1

		edge_label = "%s-%s" % (self.compute_index(room.i, room.j), self.compute_index(i_down, j_down))
		self.edit_edge(edge_label, is_passable = not self.edges[edge_label].is_passable)
		return True
	
	def toggle_wall_right(self, room_label) :
		room = self.rooms[room_label]
		if room.i == self.cols :
			# no wall right -> Illegal
			return False
		i_right = room.i + 1
		j_right = room.j

		edge_label = "%s-%s" % (self.compute_index(room.i, room.j), self.compute_index(i_right, j_right))
		self.edit_edge(edge_label, is_passable = not self.edges[edge_label].is_passable)
		return True 
	
	def place_keypoint(self, room_label, keypoint = None) :

		if not keypoint :
			if not self.keypoints['In'] :
				keypoint = 'In'
			elif not self.keypoints['Out'] :
				keypoint = 'Out'
			elif not self.keypoints['*'] :
				keypoint = '*'
			else :
				# no keypoints left to place -> Illegal
				# we could edit the room and update, but as this method is meant 
				# to be used as agent action, it would not be of help in learning
				return False
		
		if self.rooms[room_label].keypoint :
			# room is already a keypoint -> Illegal
			return False
		else :
			self.edit_room(room_label, keypoint)
			return True

	# TESTS
	
	def test() :
		dungeon = Dungeon(4, 4)
		print("\nPrint as Matrix")
		print(dungeon.as_matrix())

		print("\nPrint as Vector")
		print(dungeon.as_vector())

		print("\nPrint as pretty Matrix (Pandas)")
		print(pd.DataFrame(dungeon.as_matrix()))

		# Iterators 
		
		#for (i, j, index) in dungeon.room_iterator() :
		#	print((i, j, index))
		print("\nRoom flat iteration")
		print([ (i, j, index) for (i, j, index) in dungeon.room_iterator() ])
		print("\nRoom iteration by rows")
		print([row for row in dungeon.room_iterator(flat= False) ])
		
		print("\nEdge Iterator")
		for (A, B, i) in dungeon.edge_iterator() :
			print(dungeon.edges[i])
		
		print("\nInitial djacency List")
		print(dungeon.adjacency_list)

		print("\nDungeon Edition")
		# method 1 : overrides existing edge
		dungeon.add_edge(Edge("1", "2", "1-2",True))

		# method 2 : use higher level function
		dungeon.open_wall("2-6")
		# dungeon.open_wall("1-5")
		dungeon.toggle_wall_down("1")
		dungeon.open_wall("3-4")
		dungeon.open_wall("12-16")
		dungeon.open_wall("6-7")
		dungeon.open_wall("6-10")
		dungeon.open_wall("5-6")
		dungeon.open_wall("10-14")
		#dungeon.open_wall("14-15")
		dungeon.toggle_wall_right("14")
		dungeon.open_wall("15-16")
		dungeon.open_wall("8-12")

		dungeon.place_keypoint('1', "In")
		dungeon.place_keypoint('8', "Out")
		dungeon.place_keypoint('7', "*")

		print(dungeon.adjacency_list)

		print("\nKeypoints")
		print(dungeon.keypoints)
		
		print("\nShortest Paths")
		print("1 -> 7 : %s" % (dungeon.shortest_path('1', '7')))
		print("1 -> 8 : %s" % (dungeon.shortest_path('1', '8')))
		print("1 -> 5 : %s" % (dungeon.shortest_path('1', '5')))
		print("1 -> 4 : %s" % (dungeon.shortest_path('1', '4')))

		print("\nString Representation\n")
		print(dungeon)

		print("\nTuple Representation")
		print(dungeon.as_tuple())

		print("\nMatrix Representation")
		print(np.array(dungeon.as_adj_matrix()))

		print("\nFingerprint")
		print(dungeon.state_fingerprint()) # -> 3b4b6b18085313f8
		
		print("\nState Vector")
		print(dungeon.get_state_vector())

		print("\nDifficulty estimation")
		print(dungeon.compute_difficulty())

	def random_test() :
		print("\n Random Test")
		big_dungeon = Dungeon(20, 14)
		big_dungeon.randomize(320)
		print(big_dungeon)
		print("Fingerprint : %s - Valid : %s - Difficulty : %s" % (big_dungeon.state_fingerprint(), big_dungeon.is_valid(), big_dungeon.compute_difficulty()))

