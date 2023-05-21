#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time : 2021/1/18
# @Author : liyijia

import os
import re
import time
import cv2
import torch
import warnings
import argparse
import numpy as np
import onnxruntime as ort
from utils.datasets import LoadStreams, LoadImages
from utils.draw import draw_boxes_and_text
from utils.general import check_img_size
# from utils.torch_utils import time_synchronized
from yolo_people_detect import YoloPersonDetect
from deep_sort import build_tracker
from utils.parser import get_config
from utils.log import get_logger
from utils.torch_utils import select_device, load_classifier, time_synchronized
# count
from collections import Counter
from collections import deque
import math
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
import sys

# ------------------- 路径 ---------
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # project root directory

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH

ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
# -------------------


def increment_path(path, exist_ok=True, sep=''):
    # Increment path, i.e. runs/exp --> runs/exp{sep}0, runs/exp{sep}1 etc.
    path = os.path.abspath(path)  # os-agnostic
    if (os.path.exists(path) and exist_ok) or (not os.path.exists(path)):
        return str(path)
    else:
        dirs = [d for d in os.listdir(os.path.dirname(path)) if
                re.match(rf"{os.path.basename(path)}{sep}\d+", d)]
        i = [int(re.search(r"\d+", d).group()) for d in dirs]  # indices
        n = max(i) + 1 if i else 2  # increment number
        return f"{os.path.dirname(path)}{os.path.sep}{os.path.basename(path)}{sep}{n}"  # update path



def tlbr_midpoint(box):
    minX, minY, maxX, maxY = box
    midpoint = (int((minX + maxX) / 2), int((minY + maxY) / 2))  # minus y coordinates to get proper xy format
    return midpoint

def makedir(dir_path):
    # 创建目录
    dir_path = os.path.dirname(dir_path)  # 获取路径名，删掉文件名
    bool = os.path.exists(dir_path)  # 存在返回True，不存在返回False
    if bool:
        pass
    else:
        os.makedirs(dir_path)

def draw_line(point_1, point_2, ori_img, color=(0, 255, 255)):
    line = [( int(point_1[0] * ori_img.shape[1]), int(point_1[1] * ori_img.shape[0])),
            ( int(point_2[0] * ori_img.shape[1]), int(point_2[1] * ori_img.shape[0]))
            ]

    cv2.line(ori_img, line[0], line[1], color, 3)
    return line

# def print_statistics_to_frame(down_count, ori_img, total_counter, total_track, up_count):
def draw_statistics_to_frame(down_count, ori_img, total_counter, total_track, up_count):
    label = "TOTAL: {} people cross the yellow line. ({} IN, {} OUT.)".format(str(total_counter), str(up_count), str(down_count))
    t_size = get_size_with_pil(label, 15)  # 原：25
    x1 = 20
    y1 = 850
    color = compute_color_for_labels(2)
    ori_img = put_text_to_cv2_img_with_pil(ori_img, label, (x1 + 5, y1 - t_size[1] - 2), (255, 0, 0))
    return ori_img

def draw_idx_frame(ori_img, idx_frame):
    label = "Frames: {}".format(idx_frame)
    t_size = get_size_with_pil(label, 15)  # 原：25
    x1 = 20
    y1 = 50
    color = compute_color_for_labels(3)
    ori_img = put_text_to_cv2_img_with_pil(ori_img, label, (x1 + 5, y1 - t_size[1] - 2), (255, 0, 0))
    return ori_img

# def draw_newest_info(angle, last_track_id, ori_img):
#     current_time = int(time.time())
#     localtime = time.localtime(current_time)
#     dt = time.strftime('%Y-%m-%d %H:%M:%S', localtime)
#     # ---------------------------------------
#     label = "TIME: {} | Person №{} crossed yellow line. [{}]".format(dt, str(last_track_id), str("IN") if angle >= 0 else str('OUT'))
#     t_size = get_size_with_pil(label, 25)
#     x1 = 20
#     y1 = 900
#     color = compute_color_for_labels(2)
#     ori_img = put_text_to_cv2_img_with_pil(ori_img, label, (x1 + 5, y1 - t_size[1] - 2), (255, 0, 0))
#     return ori_img

def draw_newest_info_binary_lines(is_in, last_track_id, ori_img):
    current_time = int(time.time())
    localtime = time.localtime(current_time)
    dt = time.strftime('%Y-%m-%d %H:%M:%S', localtime)
    # ---------------------------------------
    label = "TIME: {} | Person №{} crossed yellow line. [{}]".format(dt, str(last_track_id),
                                                                     str("IN") if is_in else str('OUT'))
    t_size = get_size_with_pil(label, 25)
    x1 = 20
    y1 = 900
    color = compute_color_for_labels(2)
    ori_img = put_text_to_cv2_img_with_pil(ori_img, label, (x1 + 5, y1 - t_size[1] - 2), (255, 0, 0))
    return ori_img

def intersect(A, B, C, D):
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)


def ccw(A, B, C):
    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])


def vector_angle(midpoint, previous_midpoint):
    x = midpoint[0] - previous_midpoint[0]
    y = midpoint[1] - previous_midpoint[1]
    return math.degrees(math.atan2(y, x))


def get_size_with_pil(label,size=25):
    font = ImageFont.truetype("./configs/simkai.ttf", size, encoding="utf-8")  # simhei.ttf
    return font.getsize(label)


#为了支持中文，用pil
def put_text_to_cv2_img_with_pil(cv2_img,label,pt,color):
    pil_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)  # cv2和PIL中颜色的hex码的储存顺序不同，需转RGB模式
    pilimg = Image.fromarray(pil_img)  # Image.fromarray()将数组类型转成图片格式，与np.array()相反
    draw = ImageDraw.Draw(pilimg)  # PIL图片上打印汉字
    font = ImageFont.truetype("./configs/simkai.ttf", 25, encoding="utf-8") #simhei.ttf
    draw.text(pt, label, color,font=font)
    return cv2.cvtColor(np.array(pilimg), cv2.COLOR_RGB2BGR)  # 将图片转成cv2.imshow()可以显示的数组格式


colors = np.array([
    [1,0,1],
    [0,0,1],
    [0,1,1],
    [0,1,0],
    [1,1,0],
    [1,0,0]
    ]);

def get_color(c, x, max):
    ratio = (x / max) * 5;
    i = math.floor(ratio);
    j = math.ceil(ratio);
    ratio -= i;
    r = (1 - ratio) * colors[i][c] + ratio * colors[j][c];
    return r;

def compute_color_for_labels(class_id, class_total=80):
    offset = (class_id + 0) * 123457 % class_total;
    red = get_color(2, offset, class_total);
    green = get_color(1, offset, class_total);
    blue = get_color(0, offset, class_total);
    return (int(red*256),int(green*256),int(blue*256))


class YoloReid():
    def __init__(self, cfg, args, path):
        imgsz = check_img_size(args.img_size, s=32)  # self.model.stride.max())  # check img_size，必须是s的整数倍
        use_cuda = args.use_cuda and torch.cuda.is_available()
        if not use_cuda:
            warnings.warn("[INFO] Running in cpu mode, which maybe very slow!", UserWarning)
        self.logger = get_logger("root")
        self.args = args
        self.video_path = path
        self.person_detect = YoloPersonDetect(self.args)  # TODO: Person_detect类用来检测行人
        self.dataset = LoadImages(self.video_path, img_size=imgsz) # TODO: LoadImages类用来读取视频流
        self.deepsort = build_tracker(cfg, args.sort, use_cuda=use_cuda) # TODO: DeepSort或者DeepReID用来做目标跟踪

    # ***********************************************************************************************************
    # 用到了LoadImages类对象，Person_detect类对象，DeepSort/DeepReid类对象
    def deep_sort(self):
        """
        输入一个视频帧，检测出向上和向下的人流数目
        """
        idx_frame = 0 # 第几帧
        # results = []
        paths = {}
        track_cls = 0
        last_track_id = -1
        total_track = 0
        angle = -1
        total_counter = 0
        up_count = 0
        down_count = 0
        class_counter = Counter()   # store counts of each detected class， TODO:检测类别的计数，如果只检测人应该可以删掉
        already_counted = deque(maxlen=50)   # temporary memory for storing counted IDs

        for video_path, img, ori_img, vid_cap in self.dataset: # LoadImages对象的, 从视频流里面抓取下一帧
            idx_frame += 1 # 视频帧
            start_time = time_synchronized() # 计时

            # yolo detection
            # TODO: detect()函数返回检测到的bbox[cx,cy,w,h]，置信度，类别id
            bbox_xywh, cls_conf, cls_ids, xy = self.person_detect.detect(video_path, img, ori_img, vid_cap)
            # do tracking
            tracks_output = self.deepsort.update(bbox_xywh, cls_conf, ori_img)

            # 1.视频中间画行黄线
            yellow_line = self.draw_yellow_line_in_middle(ori_img)
            # 2. 统计人数
            # Todo: tracks_output - 检测到的所有 [ bbox + track_id ]
            # 在这里做了触线处理！
            for track in tracks_output:
                bbox = track[:4]
                track_id = track[-1]
                midpoint = tlbr_midpoint(bbox)
                # TODO: origin_midpoint是什么？
                origin_midpoint = (midpoint[0],
                                   ori_img.shape[0] - midpoint[1])  # get midpoint respective to bottom-left
                if track_id not in paths: # TODO: path又是什么鬼
                    paths[track_id] = deque(maxlen=2) # path保存了每个track的最多两个帧的midpoint
                    total_track = track_id
                paths[track_id].append(midpoint)
                previous_midpoint = paths[track_id][0] # 此track前一帧的midpoint
                origin_previous_midpoint = (previous_midpoint[0], ori_img.shape[0] - previous_midpoint[1])
                # TODO: 处理触线的情况
                # 判断 线段(midpoint, previous_midpoint)与yellow_line是否相交
                if intersect(midpoint, previous_midpoint, yellow_line[0], yellow_line[1]) \
                        and track_id not in already_counted:
                    class_counter[track_cls] += 1
                    total_counter += 1
                    last_track_id = track_id; # 记录触线者的ID
                    # draw red yellow_line，触碰线的情况下画红线
                    cv2.line(ori_img, yellow_line[0], yellow_line[1], (0, 0, 255), 2)
                    already_counted.append(track_id)  # Set already counted for ID to true.
                    # TODO: 这个角度计算原理是？
                    angle = vector_angle(origin_midpoint, origin_previous_midpoint) # 计算角度，判断向上还是向下走
                    if angle > 0:
                        up_count += 1
                    if angle < 0:
                        down_count += 1
                    # ------------------ TODO: 获取行人的图像 ----------------

                    # ------------------------------------------------------

                if len(paths) > 50:
                    del paths[list(paths)[0]]

            # 3. 绘制人员
            if len(tracks_output) > 0:
                bbox_tlwh = []
                bbox_xyxy = tracks_output[:, :4]
                track_id = tracks_output[:, -1]
                ori_img = draw_boxes_and_text(ori_img, bbox_xyxy, track_id)
                for bb_xyxy in bbox_xyxy:
                    # 改变左边格式，详见https://devpress.csdn.net/gitcode/6405a8e6986c660f3cf91258.html
                    bbox_tlwh.append(self.deepsort._xyxy_to_tlwh(bb_xyxy))
                # results.append((idx_frame - 1, bbox_tlwh, track_id))
            # 4. 绘制统计信息
            ori_img = self.print_statistics_to_frame(down_count, ori_img, total_counter, total_track, up_count)
            if last_track_id >= 0:
                ori_img = self.print_newest_info(angle, last_track_id, ori_img)
            # 5. 展示处理后的图像
            end_time = time_synchronized()
            if self.args.display:
                cv2.imshow("test", ori_img)
                if cv2.waitKey(1) & 0xFF == 27: # 按下Esc退出
                    break
            # 6. 打印logger信息
            self.logger.info("Index of frame: {} / "
                             "One Image spend time: {:.03f}s, "
                             "fps: {:.03f}, "
                             "detections : {}, "
                             "tracks : {} "
                             "class_counter : {} " \
                             .format( idx_frame, end_time - start_time, 1 / (end_time - start_time),
                                     bbox_xywh.shape[0], len(tracks_output)
                             ,class_counter
                                      ))
    # ***********************************************************************************************************



    # ============================================ 把打印的部分放在一起 ============================================
    def draw_yellow_line_in_middle(self, ori_img):
        line = [(0, int(0.48 * ori_img.shape[0])),
                (int(ori_img.shape[1]), int(0.48 * ori_img.shape[0]))]
        # line()函数， 要画的线所在的图像， 直线起点，直线终点（坐标分别为宽、高,opencv中图像的坐标原点在左上角），直线的颜色， 线条粗细
        cv2.line(ori_img, line[0], line[1], (0, 255, 255), 1)
        return line

    def print_statistics_to_frame(self, down_count, ori_img, total_counter, total_track, up_count):
        label = "客流总数: {}".format(str(total_track))
        t_size = get_size_with_pil(label, 25)
        x1 = 20
        y1 = 400
        color = compute_color_for_labels(2)
        ori_img = put_text_to_cv2_img_with_pil(ori_img, label, (x1 + 5, y1 - t_size[1] - 2), (0, 0, 0))

        label = "穿过黄线人数: {} ({} 向上, {} 向下)".format(str(total_counter), str(up_count), str(down_count))
        t_size = get_size_with_pil(label, 15)  # 原：25
        x1 = 20
        y1 = 450
        color = compute_color_for_labels(2)
        ori_img = put_text_to_cv2_img_with_pil(ori_img, label, (x1 + 5, y1 - t_size[1] - 2), (0, 0, 0))
        return ori_img

    def print_newest_info(self, angle, last_track_id, ori_img):
        # --------- 打印当前的时间 -----------------
        current_time = int(time.time())
        localtime = time.localtime(current_time)
        dt = time.strftime('%Y-%m-%d %H:%M:%S', localtime)
        # ---------------------------------------
        label = "【TIME: {}】 行人{}号{}穿过黄线".format(dt, str(last_track_id), str("向上") if angle >= 0 else str('向下'))
        t_size = get_size_with_pil(label, 25)
        x1 = 20
        y1 = 500
        color = compute_color_for_labels(2)
        # cv2.rectangle(ori_img, (x1 - 1, y1), (x1 + t_size[0] + 10, y1 - t_size[1]), color, 2)
        ori_img = put_text_to_cv2_img_with_pil(ori_img, label, (x1 + 5, y1 - t_size[1] - 2), (255, 0, 0))
        return ori_img
    # =================================================================================================================

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", default='./test_video/MOT16-03.mp4', type=str)
    parser.add_argument("--camera", action="store", dest="cam", type=int, default="-1")
    parser.add_argument('--device', default='cuda:0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    # yolov5
    parser.add_argument('--weights', nargs='+', type=str, default='./weights/yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('--img-size', type=int, default=960, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.4, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--classes', default=[0], type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    # deep_sort
    parser.add_argument("--sort", default=True, help='True: sort model, False: reid model')
    parser.add_argument("--config_deepsort", type=str, default="./configs/deep_sort.yaml")
    parser.add_argument("--display", default=True, help='show resule')
    parser.add_argument("--frame_interval", type=int, default=1)
    parser.add_argument("--cpu", dest="use_cuda", action="store_false", default=True)

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args() # video-path='./test_video/MOT16-03.mp4', weights='./weights/yolov5s.pt', sort=True
    cfg = get_config()
    cfg.merge_from_file(args.config_deepsort) # default='./configs/deep_sort.yaml'

    yolo_reid = YoloReid(cfg, args, path=args.video_path)
    with torch.no_grad():
        yolo_reid.deep_sort()
