FROM tensorflow/tensorflow:latest 

WORKDIR /home/qdungeon

COPY . . 

RUN pip install -r requirements.txt

WORKDIR /home/qdungeon/src

RUN chmod +x test.py train.py ..

CMD ["bash"]

