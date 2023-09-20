import socket
import time
import sys
SERVER_HOST = sys.argv[1]
SERVER_PORT = int(sys.argv[2])
DATA_SIZE_MB = int(sys.argv[3])
TIMEOUTVALUE = 1000000
def test_bandwidth(server_host, server_port, data_size_mb):
    data_size_bytes = data_size_mb * 1024 * 1024
    data = b'x' * data_size_bytes

    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.settimeout(TIMEOUTVALUE)
    client_socket.connect((server_host, server_port))
    start_time = time.time()
    client_socket.sendall(data)
    end_time = time.time()

    elapsed_time = end_time - start_time
    bandwidth_mbps = (data_size_bytes / elapsed_time) / (1024 * 1024)

    print(f"Data sent: {data_size_mb} MB")
    print(f"Elapsed time: {elapsed_time:.2f} seconds")
    print(f"Bandwidth: {bandwidth_mbps:.2f} Mbps")

if __name__ == "__main__":
    # SERVER_HOST = 'server_ip_address'  # Replace with the server's IP address
    # SERVER_PORT = 12345  # Use the same port as in server.py
    # DATA_SIZE_MB = 100  # Adjust this to the desired data size in MB

    test_bandwidth(SERVER_HOST, SERVER_PORT, DATA_SIZE_MB)
