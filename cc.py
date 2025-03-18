import socket
import threading

# List to keep track of connected clients
clients = []
# Lock to synchronize access to the clients list
clients_lock = threading.Lock()

# Function to handle communication with each client
def handle_client(client_socket):
    print("Client connected.")
    while True:
        try:
            # Receive message from the client
            message = client_socket.recv(1024).decode('utf-8')
            if not message:
                break  # If the client disconnects
            print(f"Received from client: {message}")
            client_socket.send("Command received".encode('utf-8'))
        except:
            break
    
    # Remove client from the list if they disconnect (thread-safe)
    with clients_lock:
        if client_socket in clients:  # Check if client is still in the list
            clients.remove(client_socket)
    print("Client disconnected.")
    client_socket.close()

# Function to start the server and handle multiple clients
def start_server(host, port):
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((host, port))
    server_socket.listen(5)
    print(f"Server listening on {host}:{port}...")
    
    while True:
        client_socket, client_address = server_socket.accept()
        print(f"Connection from {client_address}")
        with clients_lock:
            clients.append(client_socket)
        client_handler = threading.Thread(target=handle_client, args=(client_socket,))
        client_handler.start()

# Function to send a message to all connected clients
def send_message_to_clients(message):
    # Create a copy of the clients list to prevent issues when clients disconnect during sending
    with clients_lock:
        connected_clients = clients.copy()
    
    for client in connected_clients:
        try:
            client.send(message.encode('utf-8'))
            response = client.recv(1024).decode('utf-8')
            print(f"Response from client: {response}")
        except Exception:
            # Thread-safe removal of the client
            with clients_lock:
                if client in clients:  # Check if client is still in the list
                    clients.remove(client)

# Start the server in a separate thread
def start_server_thread(host, port):
    server_thread = threading.Thread(target=start_server, args=(host, port))
    server_thread.daemon = True
    server_thread.start()

# Start the server
if __name__ == "__main__":
    host = 'zonetwelve-local'  # Accept connections on all network interfaces
    port = 9999
    start_server_thread(host, port)

    # Server interactive mode
    while True:
        message = input("Enter message to send to all clients (or 'exit' to quit): ")
        if message.lower() == 'exit':
            break
        else:
            send_message_to_clients(message)