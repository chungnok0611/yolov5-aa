from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel
import urllib.request as req
import subprocess as sub
import urllib.error
import torch
from pandas import DataFrame

app = FastAPI()

class url(BaseModel):
    photo_url: str

model = torch.hub.load("ultralytics/yolov5", "yolov5s")  # or yolov5n - yolov5x6, custom

@app.post("/upload_photo/")
async def upload_photo(photo_url: url):
    results = model(photo_url.photo_url)
    a: DataFrame = results.pandas().xyxy[0]
    bottle = a['name'][0]
    return bottle

'''

@app.get("/")
def root():

    model = torch.hub.load("ultralytics/yolov5", "yolov5s")  # or yolov5n - yolov5x6, custom

    img = "https://ultralytics.com/images/zidane.jpg"  # or file, Path, PIL, OpenCV, numpy, list

    results = model(img)

    return results'''
 