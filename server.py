'''
Simple example of the server.py
'''
import socket
import time
import pickle
import util

HEADERSIZE = 10
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind((socket.gethostname(), 1243))
s.listen(5)

while True:
    # now our endpoint knows about the OTHER endpoint.
    clientsocket, address = s.accept()
    print(f"Connection from {address} has been established.")
    img = util.encodeImage("sample/train.jpg")
    msg = util.objTobytes(img)
    clientsocket.send(msg)
    

    