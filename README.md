# 2023-Graduation-Project-PeopleTrafficAnalize
2023 Graduation Project: A store traffic analysis system based on pytorch and opencv

This is an implementation of a store traffic statistics system. Implemented cross-camera multi-target tracking, basing on:
- Yolov5 - person detection
- DeepSORT - multiple object tracking
- FastReID - Person re-identification

#### In this project using follow models:
1. yolov5x pre-trained model
2. DeepSORT model, trained on dataset MARKET1501
3. FastReid model, The reid model is trained by the fast-reid framework. The resnet34 distilled from resnet101 is very large because the model saves parameters such as the FC layer and optimizer. If these are removed and only resnet34 is kept, the model will be more than 30 MB.

#### Enviroments
This project is trained and tested in Python 3.9.16, in frameword `pytorch 1.13.0 py3.9_cuda11.7_cudnn8_0`
- OS: Windows 10 Professional
- GPU: NVIDIA GEFORCE GTX1050Ti, 4G

#### How to run
1. download yolo model `yolo5x.pt` to `./weights/`
2. download reid model `model_final.pth` to `/kd-r34-r101_ibn`
3. start with `python store_traffic_monitor.py`


---

![avatar](./documents/report-hierarchy-CN.png)
![avatar](./documents/Presentation/1.png)
![avatar](./documents/Presentation/2.png)
![avatar](./documents/Presentation/3.png)
![avatar](./documents/Presentation/4.png)
![avatar](./documents/Presentation/5.png)
![avatar](./documents/Presentation/6.png)
![avatar](./documents/Presentation/7.png)
![avatar](./documents/Presentation/8.png)
![avatar](./documents/Presentation/9.png)
![avatar](./documents/Presentation/10.png)
![avatar](./documents/Presentation/11.png)
![avatar](./documents/Presentation/12.png)
![avatar](./documents/Presentation/13.png)