
from dungeon.dungeon import Dungeon
import random

class DungeonSimulator :

	def __init__(self, rows = 4, cols = 4, level = 2, current_room_encoding = "one-hot") :
		"""
		Dungeon Environment  
		---------------------
		Starts from a random dungeon, and proceeds to repair it

			- in the best way possible (difficulty)
			- as quick as possible

		"""
		self.rows = rows
		self.cols = cols
		self.nb_rooms = (self.rows * self.cols)
		
		self.dungeon = Dungeon(self.rows, self.cols)
		
		self.level = level
		if level == 1 :
			self.dungeon.place_keypoint('1')
			self.dungeon.place_keypoint('7')
			self.dungeon.place_keypoint('9')

		elif level == 2 : 
			self.dungeon.place_random_keypoints()

		elif level == 3 :
			self.dungeon.randomize(5)

		self.current_room_encoding = current_room_encoding
		self.difficulty = 0
		self.closeness = self.dungeon.keypoints_reach()
		self.compounds = len(self.dungeon.connex_compounds())
		self.max_compounds = max( [len(compound) for compound in [ eval(c) for c in self.dungeon.connex_compounds()]])
		self.current_room = 0
		self.phase = 0
		self.tick = 0
		self.finished = False
		self.update()
		self.state_size = len(self.state)
		self.action_size = 3


	def update(self) :
		
		if self.finished :
			# no more updating, the game is finished
			return False
		else :
			# 2 possibles actions / turn
			self.phase = self.tick % 2
			
			if self.phase == 0 :
				self.current_room = self.current_room + 1

			self.dungeon_state = self.dungeon.get_state_vector()
			if self.current_room_encoding == 'one-hot' :
				oh_current = [ 0 ] * self.nb_rooms
				oh_current[self.current_room - 1] = 1
				self.state = self.dungeon_state + oh_current
			else :
				self.state = self.dungeon_state + [ self.current_room / self.nb_rooms ]
			self.tick += 1
		return True
	
	def take_action(self, action):

		assert action in ["TOGGLE_WALL_RIGHT", "TOGGLE_WALL_DOWN", "NOP"]
		current_room_label = str(self.current_room)
		reward = -0.5 # hurry up ! -0.01 * self.tick
		valid = True
		# print("start at reward : %s" % (reward))

		if action == "TOGGLE_WALL_RIGHT" :
			valid = self.dungeon.toggle_wall_right(current_room_label)

		elif action == "TOGGLE_WALL_DOWN" :
			valid = self.dungeon.toggle_wall_down(current_room_label)
		
		# Phase valuation
		if not valid :
			# illegal move, do not waste your time
			reward += -0.5 # -1
			# print("Illegal move ; reward = %s" % (reward))
		else :

			# evaluate agent action
			# gives a positive or negative reward, according to
			# the added difficulty this move provides
			difficulty = len(self.dungeon.compute_treasure_path() or []) + len(self.dungeon.compute_exit_path() or [])
			compounds = len(self.dungeon.connex_compounds())
			max_compounds = max([len(compound) for compound in [ eval(c) for c in self.dungeon.connex_compounds()]])
			closeness = self.dungeon.keypoints_reach() 
			
			reward += (difficulty - self.difficulty) #+ self.dungeon.get_number_walls()
			reward += (closeness - self.closeness) #+ self.dungeon.get_number_walls()
			reward += -(compounds - self.compounds) #+ self.dungeon.get_number_walls()
			reward += max_compounds - self.max_compounds
			# print("closeness = %s" % (closeness))
			# print("difficulty = %s" % (difficulty))
			# print("compounds = %s" % (compounds))
			# print("max_compounds = %s" % (max_compounds))
			# print("reward = %s (%s d_closeness + %s d_difficulty + %s d_compounds + %s d_compound_size)" % 
			# 	(reward, (closeness - self.closeness), (difficulty - self.difficulty), -(compounds - self.compounds), max_compounds - self.max_compounds))
			
			self.difficulty = difficulty
			self.closeness = closeness
			self.compounds = compounds
			self.max_compounds = max_compounds
			#print(difficulty, reward)
		
		# Ending conditions
		if self.dungeon.is_valid() :

			# greatly values more difficult schemes
			self.finished = True
			#reward += 10 #self.dungeon.compute_difficulty() #+ self.dungeon.get_number_walls()
			reward += (100 * (1/self.tick)) * self.dungeon.compute_difficulty()
			# print("Final reward = %s" % (reward))
		#elif self.phase == 1 and self.current_room == (self.rows * self.cols) :
		elif self.current_room == self.nb_rooms :
			# end of the board ! 
			# malus for taking extra time ?
			if self.tick > (2 * self.nb_rooms) :
				print("Abort")
				reward = -100
				self.finished = True
				# print("RESET")
				#self.reset()
			#print(".")

			# but anyway reset iteration
			self.current_room = 0
			self.phase = 0
			self.update()

		self.update()
		
		return self.state, reward, self.finished, None

	def reset(self):

		self.__init__(self.rows, self.cols, self.level, self.current_room_encoding)
		return self.state

	def test(rows = 4, cols = 4):
		"""
		Simulates a random agent
		"""
		game = DungeonSimulator(rows, cols)
		actions = [ "TOGGLE_WALL_RIGHT", "TOGGLE_WALL_DOWN", "NOP" ]
		while not game.finished : 
			action = random.sample(actions, 1).pop()
			state, reward = game.take_action(action)
			print("Room %s ; %s : reward = %s" % (game.current_room, action, reward))
		print(game.dungeon)
		print(game.state)
		print("%s" % ("Bad random work" if not game.dungeon.is_valid() else "Good work"))


