# -*- coding: utf-8 -*-
import random
import numpy as np

from environment import DungeonSimulator
from agents import Dungeoner

actions = ["TOGGLE_WALL_RIGHT", "TOGGLE_WALL_DOWN", "NOP"]

VERBOSE = False

import argparse

parser = argparse.ArgumentParser(description='Test Dungeon Agents')

parser.add_argument('--rows', action="store", 
	dest="rows", default = 4, type=int)

parser.add_argument('--cols', action="store", 
	dest="cols", default = 4, type=int)

parser.add_argument('--epochs', action="store", 
	dest="epochs", default = 500, type=int)

parser.add_argument('--lr', action="store", 
	dest="lr", default = 0.01, type=float)

parser.add_argument('--memory', action="store", 
	dest="memory", default = 512, type=int)

parser.add_argument('--batch', action="store", 
	dest="batch", default = 32, type=int)

parser.add_argument('--gamma', action="store", 
	dest="gamma", default = 0.85, type=float)

parser.add_argument('--epsilon', action="store", 
	dest="epsilon", default = 1.0, type=float)

parser.add_argument('--mineps', action="store", 
	dest="mineps", default = 0.1, type=float)

parser.add_argument('--level', action="store", 
	dest="level", default = 2, type=int) 

parser.add_argument('--filename', action="store", 
	dest="filename", default = "", type=str) 

args = parser.parse_args()


ROWS = args.rows or 4
COLS = args.cols or 4
EPOCHS = args.epochs or 1000
LEARNING_RATE = args.lr or 0.001,
MEMORY_SIZE = args.memory or 2048
BATCH_SIZE = args.batch or 256
GAMMA = args.gamma or 0.99
EPSILON = args.epsilon or 1.0
EPSILON_MIN = args.mineps or 0.1
LEVEL = args.level or 2
FILENAME = args.filename

def trace(msg):
	if VERBOSE :
		print("[agent] %s" % (msg))


def train_agent(rows = 4, cols = 4, 
	epochs = 1000, 
	learning_rate = 0.01,
	memory_size = 512, batch_size = 32, 
	gamma = 0.9, 
	epsilon_start = 1.0, epsilon_min = 0.1, 
	level = 2,
	savename = None):

	env = DungeonSimulator(rows, cols, level = level)
	state_size = env.state_size
	action_size = env.action_size 

	agent = Dungeoner(state_size, action_size, epochs_train = epochs, epsilon_start = epsilon_start, epsilon_min = epsilon_min, gamma = gamma, memory_size = memory_size)

	done = False
	rewards_log = []
	steps_log = []
	success_log = []
	eps_log = []
	mazes_library = {}

	for e in range(epochs):

		print("#########################################")
		print("GAME #%s" % (e))
		state = env.reset()
		trace(env.dungeon)

		state = np.reshape(state, [1, state_size])
		total_reward = 0 

		while not env.finished :

			trace("#%s.%s"  % (env.current_room, env.phase))

			action = agent.act(state)
			next_state, reward, done, _ = env.take_action(actions[action])# env.step(action)
			total_reward += reward
			next_state = np.reshape(next_state, [1, state_size])
			agent.memorize(state, action, reward, next_state, done)
			state = next_state

			trace(env.dungeon)
			trace("%s -> %s" % (actions[action], reward))
			# print()
			if done:
				rewards_log.append(total_reward)
				success_log.append(1 if env.dungeon.is_valid() else 0)
				steps_log.append(env.tick)
				eps_log.append(agent.epsilon)
				maze_fingerprint = env.dungeon.state_fingerprint()
				if env.dungeon.is_valid() and maze_fingerprint not in mazes_library :
					mazes_library[maze_fingerprint] = env.dungeon
				print("episode: {}/{}, score: {} - {} steps, e: {:.2}"
					  .format(e, epochs, total_reward, env.tick, agent.epsilon))
				trace(env.dungeon)
				break

		agent.update_epsilon(e)

		if len(agent.memory) > batch_size:
			agent.replay(batch_size)

		if savename and e % 100 == 0 :
			agent.save(savename)

	return agent, mazes_library, (rewards_log, success_log, steps_log, eps_log)


experiment_name = "DIM%s%sLVL%s_LR%s_E%s_REPLAY%s-%s_G%s_E%s-%s" % (
	ROWS, COLS,
	LEVEL,
	LEARNING_RATE,
	EPOCHS,
	MEMORY_SIZE, BATCH_SIZE,
	GAMMA,
	EPSILON, EPSILON_MIN
)

#### AGENT TRAINING

trained_agent, mazes, logs = train_agent(
	ROWS, COLS, 
	epochs = EPOCHS, 
	memory_size = MEMORY_SIZE, batch_size = BATCH_SIZE,
	gamma = GAMMA, 
	epsilon_start = EPSILON, epsilon_min = EPSILON_MIN, 
	level = LEVEL,
	savename = experiment_name
)


if FILENAME != "":

	#### DUNGEONS/AGENT SAVING

	import pickle

	pickle.dump(mazes, open("../save/mazes/mazes_%s.pck" % (FILENAME.replace(".", ",")), "wb"))

	trained_agent.save("../save/agents/%s.h5" % (FILENAME))

	#### LEARNING CURVE

	import matplotlib
	import matplotlib.pyplot as plt
	import pandas as pd

	rewards_log, success_log, steps_log, eps_log = logs

	learning = pd.DataFrame(logs).transpose().rename_axis("epoch")
	learning.columns = ['reward', 'success', 'actions', 'epsilon']
	learning['period'] = learning.index // 10

	learning_downsample = learning.groupby('period').mean()

	df = learning_downsample
	df['reward'] = (df['reward'] - min(df['reward'])) / (max(df['reward']) - min(df['reward']))
	df['actions'] = (df['actions'] - min(df['actions'])) / (max(df['actions']) - min(df['actions']))
	df['epsilon'] = (df['epsilon'] - min(df['epsilon'])) / (max(df['epsilon']) - min(df['epsilon']))

	df.plot()
	#plt.show()
	plt.savefig("../save/plots/learning_curve_%s.png" % (FILENAME))

# #### Test on new examples

# Create an environment 
env = DungeonSimulator(ROWS, COLS)

# Create Agent

agent = Dungeoner(env.state_size, env.action_size)

successes = []
moves = []
for i in range(EPOCHS) : 
	state = env.reset()
	# Test on new examples
	i = 0
	while not env.finished :
		print("#%s.%s"  % (env.current_room, env.phase))
		i += 1
		print(env.dungeon)
		state = np.reshape(env.state, [1, env.state_size])
		action = agent.act(state)
		next_state, reward, done, _ = env.take_action(actions[action])
		print("%s -> %s" % (actions[action], reward))
	print("finished in %s actions" % (i))
	print(env.dungeon)
	if env.dungeon.is_valid() : 
		successes.append(env.dungeon)
		moves.append(i)

print("Successes : %s/%s (%0.1f%%)" % (len(successes), EPOCHS, 100*len(successes)/EPOCHS))
print("Average moves : %0.2f" % (sum(moves)/len(successes)))

