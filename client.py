'''
Simple example of the client.py
'''

import socket
import pickle


HEADERSIZE = 10
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect((socket.gethostname(), 1243))
image_counter = 0

while True:
    full_msg = b''
    new_msg = True
    while True:
        msg = s.recv(16)
        if new_msg:
            print("new msg len:",msg[:HEADERSIZE])
            msglen = int(msg[:HEADERSIZE])
            new_msg = False

        print(f"full message length: {msglen}")
        full_msg += msg
        if len(full_msg)-HEADERSIZE == msglen:
            file_write_path = "client_" + str(image_counter) + ".jpg"
            recv_img = open(file_write_path, "wb")
            recv_img.write(full_msg[HEADERSIZE:])
            recv_img.close()
            new_msg = True
            full_msg = b""