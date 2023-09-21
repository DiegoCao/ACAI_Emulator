import socket
import sys 
# HOST = sys.argv[1]
PORT = int(sys.argv[1])

def start_server(port):
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.settimeout(100000)
    server_socket.bind((socket.gethostname(), port))
    server_socket.listen(1)
    print(f"Server listening on :{port}")
    while True:
        client_socket, client_addr = server_socket.accept()
        client_socket.settimeout(100000)
        print(f"Accepted connection from {client_addr}")
        handle_client(client_socket)

def handle_client(client_socket):
    while True:
        data = client_socket.recv(1024)
        if not data:
            break
        
    client_socket.close()

if __name__ == "__main__":
    # HOST = '0.0.0.0'  # Listen on all available network interfaces
    # PORT = 12345  # Choose a suitable port number

    start_server(PORT)
