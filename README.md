# YOLOv8-Image-Embedding

## Introduce
This repo is to extract features from images and create embeddings by YOLOv8. YOLO stands for "You Only Look Once" and is known for its efficiency and accuracy in detecting objects in real-time.
## Requirement
1. python3.8
2. ultralytics
3. cv2
4. numpy
5. torch
## Run
```markdown
python  yolo_extractor.py
```
##Architecture
![image](https://github.com/ycchen218/YOLOv8-Feature-Extractor/blob/main/git-image/yolov8.png)
![image](https://github.com/ycchen218/YOLOv8-Feature-Extractor/blob/main/git-image/feature_extractor.png)
## Detail
1. I use the **global average pooling** to encode the image features from 3 dims to 2 dims, you can unuse it as you want.
2. You can choose which YOLOv8 block you want to output by the attribute **layer_index**. (recommend 15~21)
3. The image resize mechanism is the same as **ultralytics**.
4. The detection block position is xywh, you can transform it by the function **xywh2xyxy(x)**.
5. the **max_det** constrains the max detection amount.
