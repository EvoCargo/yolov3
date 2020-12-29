from pathlib import Path

import cv2
import numpy as np
import torch
from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import check_img_size, non_max_suppression, scale_coords
from yolo_scripted import YoloFacade


source = 'test_image.jpg'
weights = 'yolov3-spp.torchscript.pt'
imgsz = 640
device = torch.device('cuda:0')


def detect(source: str, weights: str, imgsz: int, device: torch.device):
    model = torch.jit.load(weights, map_location=device)

    max_stride = model.stride.max()
    imgsz = check_img_size(imgsz, s=max_stride)  # check img_size

    # Run inference
    im0 = cv2.imread(source)
    img = letterbox(im0, new_shape=imgsz)[0]
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)

    img = torch.from_numpy(img)
    img = img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    pred = model(img)

    # Process detections
    det = pred[0]
    if len(det) == 0:
        raise ValueError('No predictions')
    # Rescale boxes from img_size to im0 size
    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

    return det
