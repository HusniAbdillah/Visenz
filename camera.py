import cv2
import socket
import pickle
import struct

server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind(('0.0.0.0', 9999))
server_socket.listen(1)

print("Menunggu koneksi...")
conn, addr = server_socket.accept()
print("Terhubung ke:", addr)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    data = pickle.dumps(frame)
    message = struct.pack("Q", len(data)) + data
    conn.sendall(message)