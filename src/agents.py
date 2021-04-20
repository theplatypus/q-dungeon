
import random
import numpy as np
from collections import deque

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# from keras.optimizers import Adam
# from keras.callbacks import LearningRateScheduler

from keras.models import load_model, save_model

 # on my computer ; huge perf gap 
tf.compat.v1.disable_eager_execution() 

from environment import DungeonSimulator

VERBOSE = False

actions = ["TOGGLE_WALL_RIGHT", "TOGGLE_WALL_DOWN", "NOP"]

def trace(msg):
	if VERBOSE :
		print("[agent] %s" % (msg))

class RandomDungeoner:

	def __init__(self):
		pass

	def act(self, state):
		return self.random_action()

	def random_action(self):
		return random.sample(actions, 1).pop()

class Dungeoner:

	def __init__(self, state_size, action_size, epochs_train = 500, learning_rate = 0.01, epsilon_start = 1.0, epsilon_min = 0.2, gamma = 0.9, memory_size = 512):
		
		self.state_size = state_size
		self.action_size = action_size

		self.memory = deque(maxlen = memory_size)
		self.gamma = gamma # discount rate

		self.epsilon = epsilon_start  # exploration rate
		self.epsilon_min = epsilon_min # minimal ratio of random moves
		self.epsilon_epoch = np.linspace(self.epsilon, self.epsilon_min, num = epochs_train)

		self.learning_rate = learning_rate
		self.learning_rate_min = 0.001
		self.lr = np.geomspace(self.learning_rate, self.learning_rate_min, num = epochs_train)
		
		self.model = self.build_model()

	def build_model(self):

		model = keras.Sequential([
			layers.Dense(256, input_dim=self.state_size, activation='relu'),
			layers.Dense(32, activation='relu'),
			layers.Dense(self.action_size, activation='linear'),
		])
		
		model.compile(
			loss='mse',
			optimizer=keras.optimizers.RMSprop(lr=self.learning_rate)
			#optimizer=Adam(lr=self.learning_rate)
		)
		
		return model

	def memorize(self, state, action, reward, next_state, done):
		trace("[Memorize]")
		self.memory.append((state, action, reward, next_state, done))

	def act(self, state):
		trace("[Act]")
		if np.random.rand() <= self.epsilon:
			action = random.randrange(self.action_size)
			trace("Random action : %s" % (actions[action]))
		else :
			act_values = self.model.predict(state)
			action = np.argmax(act_values[0])
			trace("Greedy action : %s [%s]" % (action, act_values))
		return action # returns action

	def replay(self, batch_size):
		trace("[Replay] %s" % (batch_size))

		minibatch = random.sample(self.memory, batch_size)
		X_train = []
		y_train = []

		for state, action, reward, next_state, done in minibatch:
			target = reward
			if not done:
				target = (reward + self.gamma *
						  np.amax(self.model.predict(next_state)[0]))
			target_f = self.model.predict(state)
			target_f[0][action] = target
			X_train.append(state.reshape(self.state_size,))
			y_train.append(target_f.reshape(self.action_size,))

		self.model.fit(np.array(X_train), np.array(y_train), batch_size = batch_size, epochs = 5, verbose=0)

	def update_epsilon(self, epoch) :
		if self.epsilon > self.epsilon_min:
			self.epsilon = self.epsilon_epoch[epoch]
			trace("Update epsilon to %s" % (self.epsilon))

	def load(self, name):
		self.model = load_model(name)

	def save(self, name):
		save_model(self.model, name)
