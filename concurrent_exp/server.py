import socket
import threading

def handle_client(client_socket, addr):
    print(f"Accepted connection from {addr}")

    # Handle data from the client
    data = client_socket.recv(1024)
    print(f"Received data from {addr}: {data.decode('utf-8')}")

    # Send a response back to the client
    client_socket.send(b"Hello from the server!")

    # Close the client socket
    # client_socket.close()
    # print(f"Connection with {addr} closed")

def start_server():
    # Set the host
    host = '127.0.0.1'

    # List of ports to listen on
    ports = [12345, 12346, 12347]

    # Create a separate thread for each port
    for port in ports:
        server_thread = threading.Thread(target=listen_on_port, args=(host, port))
        server_thread.start()

def listen_on_port(host, port):
    # Create a socket object
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    # Bind the socket to the address and port
    server_socket.bind((host, port))
    server_socket.listen(5)
    print(f"Server listening on port {port}")

    # Listen for incoming connections and handle them in separate threads
    while True:
        client_socket, addr = server_socket.accept()
        client_handler = threading.Thread(target=handle_client, args=(client_socket, addr))
        client_handler.start()

if __name__ == "__main__":
    start_server()