
# Dungeon generator by Q-learning 

## Repository Structure

### /src

Source code of the Python software providing a solution to the problem.

A detailed Readme.md can be found in this folder.

### /saves

Some useful files generated during the execution of programs : 

	- `agents` contains some saved agents, to avoid further training
	- `mazes` contains "libraries" of mazes generated during learning
	- `plots` contains learning curves

## Running the code

### Get the repository 

```bash
git clone https://github.com/theplatypus/q-dungeon.git

cd q-dungeon
```

The project is accessible through two endpoints (scripts) :

 - `train.py` : trains an agent to solve dungeons
 - `train.py` : test an agent trained on its ability to solve new dungeons

#### Pythonic way

```bash
pip install -r requirements.txt 
cd src 
python test.py

# should reach 70-80% success
python test.py --tests=30

# should reach 50-60% success, with more average number of moves
python test.py --tests=30 --randomness=1.0

# Play a new kind of dungeon; do not expect a lot from this number of epochs
python train.py --rows=3 --cols=6 --filename=test36 --epochs=50

# Test it
python test.py --rows=3 --cols=6 --filename=test36 --tests=30

```

#### Containerized way

```bash
# Build image
docker build -t qdungeon .

# Run tests
docker run -it qdungeon python test.py 

# should reach 70-80% success
docker run -it qdungeon python test.py --tests=30

# should reach 50-60% success, with more average number of moves
docker run -it qdungeon python test.py --tests=30 --randomness=1.0

# Play a new kind of dungeon; you can expect a bit more of those parameters
# one liner to avoid using volume
docker run -it qdungeon python train.py --rows=3 --cols=6 --filename=test36 --memory=2048 --batch=256 --epochs=1500 && test.py --rows=3 --cols=6 --filename=test36 --tests=30


# Interactive bash in environment
docker run -it qdungeon

# Interactive IPython shell
docker run -it qdungeon ipython
```