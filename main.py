import tensorflow as tf
import cv2
from tensorflow.keras import layers
from tensorflow.keras.models import load_model
from ultralytics import YOLO
import numpy as np


model = YOLO("Model/face detection/best.pt")


class L1Dist(layers.Layer):

    def __init__(self, **kwargs):
        super().__init__()

    def call(self, input_embedding, validation_embedding):
        return tf.math.abs(input_embedding - validation_embedding)


def load_siamese_model(model_path):
    return load_model(model_path, custom_objects={'L1Dist': L1Dist})


loaded_model = load_siamese_model('Model/face recognition/Model 1/my_model.h5')


image1 = cv2.imread(
    'Media/20220923_131608.jpg')


results = model(image1, device=0)[0]
for detection in results.boxes.data.tolist():
    x1, y1, x2, y2, score, class_id = detection
    car_polygon = [(int(x1), int(y1)), (int(x1), int(
        y2)), (int(x2), int(y2)), (int(x2), int(y1))]

    image1 = image1[int(y1):int(y2), int(x1):int(x2)]

    cv2.imwrite(f"Media/ff.jpg", image1)
    break

image1 = cv2.resize(image1, (70, 70))

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()

    results = model(frame, device=0)[0]

    for detection in results.boxes.data.tolist():

        x1, y1, x2, y2, score, class_id = detection

        extracted_image = frame[int(y1):int(y2), int(x1):int(x2)]

        extracted_image = cv2.resize(extracted_image, (70, 70))
        extracted_image = np.expand_dims(extracted_image, axis=0)
        image1_ = np.expand_dims(image1, axis=0)

        pre = loaded_model.predict([image1_, extracted_image])[0][0]
        print(pre)
        if pre > 0.5:
            pre = "Name"
        else:
            pre = "Unknown"

        cv2.putText(frame, str(pre), (int(x1), int(y1)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    cv2.imshow('Webcam', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
