from agents import RandomDungeoner, Dungeoner
from environment import DungeonSimulator

import numpy as np

import argparse

parser = argparse.ArgumentParser(description='Test Dungeon Agents')

parser.add_argument('--rows', action="store", 
	dest="rows", default = 4, type=int)

parser.add_argument('--cols', action="store", 
	dest="cols", default = 4, type=int)

parser.add_argument('--randomness', action="store", 
	dest="randomness", default = 0.1, type=float)

parser.add_argument('--tests', action="store", 
	dest="tests", default = 10, type=int)

parser.add_argument('--filename', action="store", 
	dest="filename", default = "", type=str) 

args = parser.parse_args()

ROWS = args.rows or 4
COLS = args.cols or 4
RANDOMNESS = args.randomness or 0.1
TESTS = args.tests or 10
FILENAME = args.filename

actions = ["TOGGLE_WALL_RIGHT", "TOGGLE_WALL_DOWN", "NOP"]

# Create an environment 
env = DungeonSimulator(ROWS, COLS)

# Create Agent

agent = Dungeoner(env.state_size, env.action_size)

agent.epsilon = RANDOMNESS
agent.action_size = 3

if FILENAME != "" :
	agent.model.load_weights('../save/agents/%s.h5' % (FILENAME))

elif ROWS == 4 and COLS == 4 :
	#agent.model.load_weights('../save/agents/selected/4_4/agent_4*4_random_512_099_00125.h5')
	agent.model.load_weights('../save/agents/selected/4_4/1000_256_1e-3_RMS.h5')
	
	# tf format - less convenient imho
	# agent.load("../save/agents/truc")
elif ROWS == 3 and COLS == 3 :
	agent.model.load_weights('../save/agents/selected/3_3/save/agents/selected/3_3/[3*3-random][lr=0,01][gamma=0,8][batch=128].h5')

else :
	print("Error, no agent trained for that, it's going to be very random")
	agent.epsilon = 1


# Test the agent

successes = []
moves = []
for i in range(TESTS) : 
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

print("Successes : %s/%s (%0.1f%%)" % (len(successes), TESTS, 100*len(successes)/TESTS))
print("Average moves : %0.2f" % (sum(moves)/len(successes)))

