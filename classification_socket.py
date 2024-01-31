import socketserver
import numpy as np
import cv2
import tensorflow as tf
from keras import utils

class Predict:
    classes = ["ACCEPT", "REJECT", "REWORK"]

    def __init__(self):
        self.model = tf.keras.models.load_model('./models')

    def predict(self, img):
        img_array = utils.img_to_array(img)
        img_array = tf.expand_dims(img_array, axis=0)

        prediction = self.model.predict(img_array)

        predicted_class = np.argmax(prediction[0])
        return self.classes[predicted_class]


predict = Predict()


class ImageProcessingHandler(socketserver.BaseRequestHandler):

    def handle(self):
        # Receive image data from the client
        image_data = b''
        while True:
            data = self.request.recv(100000)
            if not data:
                break
            image_data += data
            data_encode = np.frombuffer(image_data, dtype=np.uint8)

            img_decode = cv2.imdecode(data_encode, cv2.IMREAD_COLOR)

            prediction = predict.predict(img_decode)
            print(prediction)
            self.request.sendall(prediction.encode())



if __name__ == "__main__":
    host, port = '172.16.1.115', 5555
    server = socketserver.TCPServer((host, port), ImageProcessingHandler)
    server.serve_forever(0.5)
