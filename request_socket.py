import socket
import base64
import cv2
import numpy as np

def test_connection():
    x, y = 1000, 0
    img_height = int(2048 / 8)
    img_width = int(600 / 8)
    # Define server address and port
    server_address = '172.16.1.115'
    server_port = 5555

    # Create a socket object
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    # Connect to the server
    client_socket.connect((server_address, server_port))

    img_name = './data/REAL/QC-1706653456087.jpg'
    img = cv2.imread(img_name)
    cropped_image = img[y:y+2048, x:x+500]
    img = cv2.resize(cropped_image, (img_width, img_height))

    cv2.destroyAllWindows()

    _, img_encode = cv2.imencode('.jpg', img)
    img_bytes = img_encode.tobytes()

    client_socket.send(img_bytes)

    result_label = client_socket.recv(10).decode()

    # Display the result label
    print(f"Image Result: {result_label}")

    client_socket.close()