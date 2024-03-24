import requests
import cv2
import numpy as np
import base64
import time

# 서버의 /count/ 엔드포인트 URL
url = 'http://127.0.0.1:8000/count/'

while True:
    # 서버로부터 데이터를 받아옵니다.
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        print(f"Detected people count: {data['count']}")

        # Base64 인코딩된 이미지 데이터를 디코딩합니다.
        img_data = base64.b64decode(data['image'])
        nparr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # OpenCV를 사용하여 이미지를 화면에 표시합니다.
        cv2.imshow("Detected People", img)
        
        # 'q'를 누르면 종료합니다.
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        print("Failed to get data from the server.")
        
    # 서버에 요청을 보내는 주기를 설정합니다. 여기서는 1초로 설정했습니다.
    #time.sleep(1)

cv2.destroyAllWindows()
