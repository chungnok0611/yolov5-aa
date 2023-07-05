import torch
from PIL import Image
from pandas import DataFrame


model = torch.hub.load("ultralytics/yolov5", "yolov5s")  # or yolov5n - yolov5x6, custom
results = model("https://www.officemax.co.nz/images/ProductImages/500/2465981.jpg")
#print(type(results))
a = results.pandas().xyxy[0]
a: DataFrame
bottle = a['name'][0]
print(bottle)