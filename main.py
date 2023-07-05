from fastapi import FastAPI, File, UploadFile
import torch
import requests

app = FastAPI()


@app.post("/upload_photo/")
async def upload_photo(photo_url: str):

    image_url = photo_url                                  #"http://example.com/image.jpg"
    response = requests.get(image_url)
    response.raise_for_status()
    image_data = response.content

    model = torch.hub.load("ultralytics/yolov5", "yolov5s")  # or yolov5n - yolov5x6, custom

    img = photo_url  # or file, Path, PIL, OpenCV, numpy, list

    results = model(img)
   
    # 여기에서 사진을 다운로드하거나 처리하는 작업을 수행할 수 있습니다.
    # 예를 들어, requests 라이브러리를 사용하여 사진을 다운로드하거나, 이미지 처리를 위해 Pillow 라이브러리를 사용할 수 있습니다.
    # 이 예제에서는 간단히 사진 URL을 출력합니다.
    return results