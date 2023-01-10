import json
import os.path
from pathlib import Path
import cv2
import numpy as np
from grpc_test.torchserve_grpc_client import infer_image, get_inference_stub

WIDTH, HEIGHT = 640, 480
ACTION_API = "127.0.0.1:7070"
MODEL_NAME = "STCNet_8f_CesleaFDD6"  # use 8 frames as input

test_file = "./videos/test.avi"

cap = cv2.VideoCapture(test_file)
fps = 10  # FPS for output video
output_file = Path("output") / "out.avi"
os.makedirs("output", exist_ok=True)
fourcc = cv2.VideoWriter_fourcc(*'DIVX')

out_stream = cv2.VideoWriter(str(output_file), fourcc, fps, (WIDTH, HEIGHT))

while cap.isOpened():
    ret, image = cap.read()
    if not ret: break
    image = cv2.resize(image, (WIDTH, HEIGHT))

    # dimension of original frame
    raw_height, raw_width = image.shape[:-1]

    rgb_input = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Action recognition API Call
    prediction = infer_image(
        # gRPC 통신을 위한 부분. ACTION_API: 추론 API 서버의 URI
        get_inference_stub(target=ACTION_API),
        # 추론에 사용할 모델명 (어떤 모델을 사용할 것인가?)
        MODEL_NAME,
        # RGB 비디오 프레임 (numpy array)
        rgb_input)

    if prediction != "0":
        prediction = json.loads(prediction)
        prediction = list(prediction.items())

        step_size = 20
        start = 70

        banner = np.zeros_like(image)
        cv2.rectangle(banner, (0, 0), (400, 50), (255, 128, 255), -1)
        image = cv2.addWeighted(image, 0.75, banner, 0.25, 0)

        y_pred, proba = prediction[0]  # most significant class

        cv2.putText(image, "{} : {:.2f}%".format(y_pred, proba * 100), (5, 35), cv2.FONT_HERSHEY_DUPLEX, 1.0,
                    (255, 255, 255), 1, cv2.LINE_AA)
        for i, (y_pred, proba) in enumerate(prediction):
            cv2.putText(image, "{} : {:.2f}%".format(y_pred, proba * 100), (5, start + step_size * i),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        print(prediction)

    out_stream.write(image)
    cv2.imshow("camera", image)
    cv2.waitKey(33)

out_stream.release()
cap.release()
cv2.destroyAllWindows()
