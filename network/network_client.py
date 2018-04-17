import socket

host = "localhost"


my_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# AF_INET - internet family
# SOCK_STREAM - indicates that it's TCP.
address = (host, 5555)
mysocket.connect(address)

try:
    message = b"Hi, this is a test\n"
    my_socket.sendall(message)
except socket.errno:
    print("Socket Error")
finally:
    my_socket.close()
