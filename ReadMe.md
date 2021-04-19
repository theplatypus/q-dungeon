
# Dungeon generator by Q-learning 

## Repository Structure

### /src

Source code of the Python software providing a solution to the problem.

A detailed Readme.md can be found in this foller.

### /saves

Some useful files generated during the execution of programs : 

	- `agents` contains some saved agents, to avoid further training
	- `mazes` contains "libraries" of mazes generated during learning
	- `plots` contains learning curves

## Running the code

### Get the repository 

```bash
git clone __

cd __
```

#### Pythonic way

```bash
pip install -r requirements.txt 
```

```bash
python test.py

# or if you have Docker
docker build -t qdungeon .
```

#### Containerized way

```bash

docker build -t qdungeon .

docker run -d qdungeon 

docker exec -it %img test.py
```