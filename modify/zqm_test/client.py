import zmq
import sys
import json

context = zmq.Context()
socket = context.socket(zmq.REQ)
socket.connect("tcp://localhost:5555")
data = [[1,1,1],[2,2,2],[3,3,3],[4,4,4]]
socket.send(json.dumps(data))
response = socket.recv()
print(response)
