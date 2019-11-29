import zmq
import time
import random
import json

def recv_and_parse_kv(socket):
    message = socket.recv()
    group = message.split("\t")
    if group[0] == "alive":
        return group[0], "0"
    else:
        return group[0], group[1]


context = zmq.Context()
socket = context.socket(zmq.REP)
socket.bind("tcp://*:5555")

while True:
	message = json.loads(socket.recv())
	print(message[1])
        socket.send("list OK!")
