'''
Simple example of the client.py
'''
import socket
import pickle
import torch
    
HEADERSIZE = 10
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect((socket.gethostname(), 1234))
image_counter = 0


def clientSendImg(s, imgdir):
    for image_path in imagedir:
        img = util.encodeBytes(image_path)
        s.sendall(image_data)

        time.sleep(500)

ack_msg = "ack"
pt_save_path = "tmp.pt"
print("[STARTING] Client 1 wakes up")
print("Client 1 starts handling requests")
def client_logic():
    image_counter = 0
    while True:
        print("start receiving message")
        full_msg = b''
        new_msg = True
        while True:
            msg = s.recv(1024)
            if new_msg:
                print("new msg len:",msg[:HEADERSIZE])
                msglen = int(msg[:HEADERSIZE])
                new_msg = False
            # print(f"full message length: {msglen}")
            full_msg += msg
            # print("the length of fullmessage is ", len(full_msg)) # need to get wait
            if len(full_msg)-HEADERSIZE == msglen:
                obj = pickle.loads(full_msg[HEADERSIZE:])
                print(obj)
                # recv_wt = open(file_write_path, "wb")
                recv_wt.write(full_msg[HEADERSIZE:])
                recv_wt.close()
                new_msg = True
                full_msg = b""

client_logic()