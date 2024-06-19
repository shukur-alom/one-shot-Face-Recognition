import cv2
import os
from ultralytics import YOLO
model = YOLO("Model/face detection/best.pt")


for i in os.listdir("DataSet/lfw"):
    print(i)
    for j in os.listdir(f"DataSet/lfw/{i}"):
        if not os.path.exists(f"DataSet/lfw2/{i}"):
            os.makedirs(f"DataSet/lfw2/{i}")

        img_path = f"DataSet/lfw/{i}/{j}"
        frame = cv2.imread(img_path)

        results = model(frame, device=0)[0]

        for detection in results.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = detection
            car_polygon = [(int(x1), int(y1)), (int(x1), int(
                y2)), (int(x2), int(y2)), (int(x2), int(y1))]

            extracted_image = frame[int(y1):int(y2), int(x1):int(x2)]

            cv2.imwrite(f"DataSet/lfw2/{i}/{j}", extracted_image)
            break