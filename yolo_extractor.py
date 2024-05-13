import torch
from ultralytics import YOLO
from ultralytics.data.augment import LetterBox
import cv2
import torch.nn as nn
import numpy as np

def xywh2xyxy(x):
    assert x.shape[-1] == 4, f"input shape last dimension expected 4 but input shape is {x.shape}"
    y = torch.empty_like(x) if isinstance(x, torch.Tensor) else np.empty_like(x)
    dh = x[..., 3] / 2
    dw = x[..., 2] / 2
    y[..., 0] = x[..., 0] - dw
    y[..., 1] = x[..., 1] - dh
    y[..., 2] = x[..., 0] + dw
    y[..., 3] = x[..., 1] + dh
    return y
def extract_features(model, img, layer_index):
    intermediate_features = []
    def hook_fn(module, input, output):
        nonlocal intermediate_features
        intermediate_features.append(output)
    hook = model.model.model[layer_index].register_forward_hook(hook_fn)
    with torch.no_grad():
        model(img, verbose=False)
    hook.remove()
    feats = intermediate_features[0]
    global_avg_pooling = nn.AdaptiveAvgPool2d(1)
    feats = global_avg_pooling(feats)
    return feats.squeeze()
def preprocess_image(img_path):
    transform = LetterBox(640, True, stride=32)
    img = cv2.imread(img_path)
    img = transform(image=img)
    return img
def image_embedding(yolo, img, max_det, layer_index):
    features = extract_features(yolo, img, layer_index=layer_index)
    results = yolo(img, verbose=False, max_det=max_det)[0]
    classes_name = yolo.names
    pred_cls = results.boxes.cls
    pred_cls = pred_cls.detach().cpu().numpy().astype(int)
    pred_cls_names = [classes_name[idx] for idx in pred_cls]
    pred_scores = results.boxes.conf
    boxes_pos = results.boxes.xywh
    # det_data = results.boxes.data
    # return features, det_data
    #make your own return!!
    return features, pred_cls_names, pred_scores, boxes_pos


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = YOLO("train3/weights/best.pt")
    model = model.to(device)
    image = preprocess_image("test.jpg")
    print(image.shape)
    features, pred_cls, pred_scores, boxes_pos = image_embedding(model, image, max_det=50, layer_index=21)
    boxes_pos = xywh2xyxy(boxes_pos)
    # layer_index = 21--> 640
    # layer_index = 20--> 1280
