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


source = 'test_image.jpg'
weights = 'yolov3-spp.pt'
imgsz = 640

view_img = False
save_txt = True
save_img = True

webcam = False
vid_path, vid_writer = None, None
vid_cap = None

# Directories
save_dir = Path('./inference')
(save_dir / 'labels' if save_txt else save_dir).mkdir(
    parents=True, exist_ok=True
)  # make dir
save_path = str(save_dir / source)
txt_path = (save_dir / 'labels' / source).with_suffix('.txt')

# Initialize
device = torch.device('cuda:0')
half = device.type != 'cpu'  # half precision only supported on CUDA

# Load model
model = attempt_load(weights, map_location=device)  # load FP32 model
imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
if half:
    model.half()  # to FP16

# Get names and colors
names = model.module.names if hasattr(model, 'module') else model.names
colors = [[np.random.randint(0, 255) for _ in range(3)] for _ in names]

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
augment = False  # ??
conf_thres = 0.25
iou_thres = 0.45
classes = None
agnostic_nms = True

with torch.no_grad():
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

# s = ''
# s += '%gx%g ' % img.shape[2:]  # print string
# gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh

# # Print results
# for c in det[:, -1].unique():
#     n = (det[:, -1] == c).sum()  # detections per class
#     s += '%g %ss, ' % (n, names[int(c)])  # add to string

# # Write results
# for *xyxy, conf, cls in reversed(det):
#     if save_txt:  # Write to file
#         xywh = (
#             (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
#         )  # normalized xywh
#         save_conf = True
#         line = (
#             (cls, *xywh, conf) if save_conf else (cls, *xywh)
#         )  # label format
#         with open(txt_path, 'a') as f:
#             f.write(('%g ' * len(line)).rstrip() % line + '\n')

#     if save_img or view_img:  # Add bbox to image
#         label = '%s %.2f' % (names[int(cls)], conf)
#         plot_one_box(
#             xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3
#         )

# # Save results (image with detections)
# if save_img:
#     cv2.imwrite(save_path, im0)

# if save_txt or save_img:
#     s = (
#         f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}"
#         if save_txt
#         else ''
#     )
#     print(f"Results saved to {save_dir}{s}")
