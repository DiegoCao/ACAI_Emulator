'''
Simple example of the server.py
'''
import socket
import time
import torch
import threading
import pickle
import util
import time

def serverSendweight(clientsocket, filepath):
    weights = util.encodeBytes(filepath)
    msg = util.imgTobytes(weights)
    clientsocket.sendall(msg)
    
    

# def serverRecvimgs(clientsocket):
#     image_list = []
#     while True:
#         msg = 

    


def handle_client(clientsocket, address):   
    print(f"Connection from {address} has been established.")
    connected = True
    while connected:
        # img = util.encodeBytes("/Users/hangruicao/acai/emulator/YOLOv3_Train_Inference/yolo_detector.pt")
        # msg = util.imgTobytes(img)
        # print("the length of message", len(msg))
        # clientsocket.send(msg)
        torchtensor = torch.rand(2,3)
        pobj = pickle.dumps(torchtensor)
        msg = util.imgTobytes(pobj)
        clientsocket.sendall(msg)
        print("message sent")

        while True:
            # receive the ACK 
            new_msg = False
            msg = s.recv(16)
            if new_msg:
                print("new msg len:",msg[:HEADERSIZE])
                msglen = int(msg[:HEADERSIZE])
                new_msg = False

        


if __name__ == "__main__":
    print("[STARTING] Server wakes up...")
    HEADERSIZE = 10
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind((socket.gethostname(), 1234))
    s.listen(5)
    clientsocket = None
    while True:
    # now our endpoint knows about the OTHER endpoint.
        clientsocket, address = s.accept()

        handle_client(clientsocket, address)
        # clientsocket.close()
