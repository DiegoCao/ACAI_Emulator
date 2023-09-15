import socket
import random
import string
import time
import os
import sys

def generate_random_text_file(file_name, size_in_bytes):
    with open(file_name, "w") as file:
        file.write(''.join(random.choice(string.ascii_letters) for _ in range(size_in_bytes)))

def client(server_ip, server_port, num_requests, log_address):
    
    with open(log_address, 'r+') as f:
        f.truncate(0)

    for i in range(num_requests):
        file_name = "/data/random_file{}.txt".format(i)
        # Generate random file size between 10MB and 20MB
        file_size = random.randint(10*1024*1024, 20*1024*1024) 
        generate_random_text_file(file_name, file_size)
        print("Generated file of size ", file_size/1024/1024, "MB")

        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_socket.connect((server_ip, server_port))

        # Send the file size to the server
        file_size_str = str(file_size)
        client_socket.sendall(file_size_str.encode('utf-8'))

        # Wait for acknowledgment from the server
        acknowledgment = client_socket.recv(1024)
        print("Received acknowledgment from server:", acknowledgment.decode('utf-8'))

        with open(file_name, "rb") as file:
            file_data = file.read()
        
        start_time = time.time()
        client_socket.sendall(file_data)
        end_time = time.time()

        client_socket.close()

        latency = end_time - start_time
        print("Request {}: Sent {} bytes in {:.4f} seconds".format(
            i + 1, len(file_data), latency))
        with open(log_address, "a") as f:
            f.write("Request {}: Sent {} bytes in {:.4f} seconds\n".format(
            i + 1, len(file_data), latency))

if __name__ == "__main__":
    ip = sys.argv[1]
    port = int(sys.argv[2])
    num_requests = int(sys.argv[3])
    log_address = sys.argv[4]
    client(ip, port, num_requests, log_address)