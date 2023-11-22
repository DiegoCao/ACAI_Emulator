import socket

def start_client(port):
    # Set the host and port
    host = '127.0.0.1'

    # Create a socket object
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    # Connect to the server
    client_socket.connect((host, port))

    # Send data to the server
    message = "Hello from the client!"
    client_socket.send(message.encode('utf-8'))

    # Receive and print the server's response
    response = client_socket.recv(1024)
    print(f"Received response from server: {response.decode('utf-8')}")

    # Close the client socket
    # client_socket.close()

if __name__ == "__main__":
    # Connect to each port in a separate client instance
    for port in [12345, 12346, 12347]:
        start_client(port)
    # start_client(12345)
