import time
from pathlib import Path

import cv2
import numpy as np
import torch

from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import (
    check_img_size,
    non_max_suppression,
    scale_coords,
    xyxy2xywh,
)
from utils.plots import plot_one_box
from yolo_scripted import YoloFacade


source = 'test_image.jpg'
weights = 'yolov3-spp.pt'
traced = 'yolov3-spp.torchscript.pt'
imgsz = 640
device = torch.device('cuda:0')

def detect(source: str, weights: str, traced: str, imgsz: int, device: torch.device):
    half = False  # True or False for regular model, False for torchscript

    save_txt = True

    # Directories
    save_dir = Path('./inference')
    (save_dir / 'labels' if save_txt else save_dir).mkdir(
        parents=True, exist_ok=True
    )  # make dir

    # Load model
    if traced:
        model = YoloFacade.from_checkpoint(weights, traced, device)
    else:
        model = attempt_load(weights, map_location=device)  # load FP32 model
        if half:
            model.half()  # to FP16
    max_stride = model.stride.max()
    imgsz = check_img_size(imgsz, s=max_stride)  # check img_size

    # Run inference
    im0 = cv2.imread(source)
    img = letterbox(im0, new_shape=imgsz)[0]
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)

    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # Inference
    augment = False
    conf_thres = 0.25
    iou_thres = 0.45
    classes = None
    agnostic_nms = True

    with torch.no_grad():
        if traced:
            pred = model(img)
        else:
            pred = model(img, augment=augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms)

    # Process detections
    det = pred[0]
    if len(det) == 0:
        raise ValueError('No predictions')
    # Rescale boxes from img_size to im0 size
    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

    print('### Done')
