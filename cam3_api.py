import cv2
from ultralytics import YOLO
import base64
import numpy as np
from fastapi import FastAPI, Form, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

app = FastAPI()

# CORS 미들웨어 추가
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 모든 출처를 허용하는 예시입니다. 보안을 위해 필요한 출처만 허용하도록 수정하세요.
    allow_credentials=True,
    allow_methods=["*"],  # 모든 HTTP 메소드를 허용합니다.
    allow_headers=["*"],  # 모든 HTTP 헤더를 허용합니다.
)

font = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 1.5
color = (255, 0, 0)
thickness = 1
model = YOLO('yolov8s.pt')

# 웹캠 설정
capture = cv2.VideoCapture(0)

# 모델이 인식하는 클래스 이름 목록
classNames = [
    "person", "bicycle", "car", "motorbike", "aeroplane", "bus",
    "train", "truck", "boat", "traffic light", "fire hydrant",
    "stop sign", "parking meter", "bench", "bird", "cat", "dog",
    "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie",
    "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
    "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
    "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
    "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
    "teddy bear", "hair drier", "toothbrush"
]

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/count/")
async def count_people():
    ret, frame = capture.read()
    if not ret:
        return JSONResponse(content={"error": "Failed to capture image"}, status_code=500)
    
    # 이미지 처리
    frame = cv2.flip(frame, 1)
    #results = model(frame, size=640)
    results = model(frame, stream=True, verbose = False)
    
    class_ids = []
    confidences = []
    bboxes = []
    
    for result in results:
        boxes = result.boxes
        for box in boxes:
            confidence = box.conf
            if confidence > 0.8:
                if str(classNames[int(box.cls)]) == 'person':
                    xyxy = box.xyxy.tolist()[0]
                    bboxes.append(xyxy)
                    confidences.append(float(confidence))
                    class_ids.append(box.cls.tolist())
                
    result_boxes = cv2.dnn.NMSBoxes(bboxes, confidences, 0.25, 0.45, 0.5)
    font = cv2.FONT_HERSHEY_PLAIN
    depth_list = list()
    person_id_list = list()
    count_p = len(result_boxes)

    cv2.putText(img=frame, text=f'Counts People in ROI: {count_p}', org= (30,40), fontFace=font, 
                        fontScale=1.5, color=(255, 0, 0), thickness=2)
    
    for i in range(len(bboxes)):
            
        label = str(classNames[int(class_ids[i][0])]) # 'person'

        # if label == 'person':
        if i in result_boxes:
            bbox = list(map(int, bboxes[i]))
            x1, y1, x2, y2 = bbox
            org = [x1, x2]
            color = (255, 0, 255)
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
            cv2.putText(frame, label, org, font, fontScale, color, thickness)
            cv2.putText(img=frame, text=f'Counts People in ROI: {count_p}', org= (30,40), fontFace=font, 
                        fontScale=1.5, color=(255, 0, 0), thickness=2)

    # 탐지된 사람 수
    people_count = str(count_p)

    # Base64 인코딩된 이미지로 변환
    _, buffer = cv2.imencode('.jpg', frame)
    jpeg_base64 = base64.b64encode(buffer).decode('utf-8')

    return {"count": people_count, "image": jpeg_base64}

    # cv2.imshow("VideoFrame", frame)
    
    # key = cv2.waitKey(1)
    # # Press esc or 'q' to close the image window
    # if key & 0xFF == ord('q') or key == 27:
    #     cv2.destroyAllWindows()
    #     break


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)