import socket
import os
import time
import sys

def server(log_address):

    with open(log_address, 'r+') as f:
        f.truncate(0)

    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind(("0.0.0.0", 5001))
    server_socket.listen()

    print("Server listening on {}:{}".format("0.0.0.0", 5001))

    while True:
        conn, addr = server_socket.accept()
        print("Accepted connection from {}:{}".format(addr[0], addr[1]))

        # Receive the file size from the client
        file_size_str = conn.recv(1024).decode('utf-8')
        # Send acknowledgment back to the client
        conn.sendall("File size received".encode('utf-8'))

        file_size = int(file_size_str)
        start_time = time.time()

        with open("/data/received.txt", "wb") as file:
            # Receive the correct amount of data based on the file size
            remaining_data = file_size
            while remaining_data > 0:
                data = conn.recv(min(remaining_data, 1024))
                if not data:
                    break
                file.write(data)
                remaining_data -= len(data)

        # Send acknowledgment back to the client
        conn.sendall("File received successfully".encode('utf-8'))

        end_time = time.time()
        conn.close()

        # Calculate statistics
        file_size = os.path.getsize("received.txt")
        latency = end_time - start_time
        drop_rate = 0  # to be implemented
        bandwidth = file_size / latency / 1024  # KB/s

        print("Received {} bytes(Latency: {:.4f} seconds, Bandwidth: {:.4f} KB/s)".format(
            file_size, latency, bandwidth))
        with open(log_address, "a") as f:
            f.write("Received {} bytes(Latency: {:.4f} seconds, Bandwidth: {:.4f} KB/s)\n".format(
            file_size, latency, bandwidth))

if __name__ == "__main__":
    log_address = sys.argv[1]
    server(log_address)
