import os
from collections import deque
import time
import cv2
import numpy as np
import torch
import warnings
import argparse
from person_count import tlbr_midpoint, intersect, vector_angle, get_size_with_pil, compute_color_for_labels, \
    put_text_to_cv2_img_with_pil, draw_yellow_line, makedir, print_statistics_to_frame, print_newest_info, \
    draw_idx_frame
from utils.datasets import LoadStreams, LoadImages
from utils.draw import draw_boxes_and_text, draw_reid_person, draw_boxes
from utils.general import check_img_size
from person_detect_yolov5 import YoloPersonDetect
from deep_sort import build_tracker, DeepReid
from utils.parser import get_config
from utils.log import get_logger
from utils.torch_utils import select_device, load_classifier, time_synchronized
from sklearn.metrics.pairwise import cosine_similarity
from fast_reid.demo.person_bank import Reid_feature

from pycallgraph2 import PyCallGraph
from pycallgraph2.output import GraphvizOutput





class VideoStreamTracker():
    def __init__(self, yolo_model,
                 deepsort_model,
                 dataset,
                 query_feat,
                 output_people_img_path,
                 is_display, p1, p2):
        self.idx_frame = 0
        self.total_track = 0
        self.total_counter = 0
        self.up_count = 0
        self.down_count = 0
        self.already_counted = deque(maxlen=100)
        self.yolo_model = yolo_model
        self.deepsort = deepsort_model
        self.dataset = dataset
        # 1.画黄线
        self.p1_ratio = p1
        self.p2_ratio = p2
        # 2.处理tracks
        self.output_people_img_path = output_people_img_path
        # 3.ReID
        self.query_feat = query_feat
        # 4.绘制统计信息 & 绘制检测框 & 绘制帧数
        # 5.展示图像，输出结果视频
        self.is_display = is_display
        # 6.销毁窗口 & 打印log

    def process_frame(self):
        print("HERE!")
        paths = {}  # 每一个track的行动轨迹
        last_track_id = -1
        angle = -1

        already_counted = deque(maxlen=100)  # temporary memory for storing counted IDs

        for video_path, img, ori_img, vid_cap in self.dataset:  # 获取视频帧
            self.idx_frame += 1
            start_time = time_synchronized()

            # yolo detection
            bbox_xywh, cls_conf, cls_ids, xy = self.yolo_model.detect(video_path, img, ori_img, vid_cap)
            outputs, features = self.deepsort.update(bbox_xywh, cls_conf, ori_img)  # TODO: 路径问题，一定要放在test_video下才可以

            # 1.画黄线
            yellow_line_in = draw_yellow_line(self.p1_ratio, self.p2_ratio, ori_img)

            # 2. 处理tracks
            for track in outputs:
                bbox = track[:4]
                track_id = track[-1]
                midpoint_1 = tlbr_midpoint(bbox) # TODO: 简化撞线计算
                origin_midpoint = (midpoint_1[0],
                                   ori_img.shape[0] - midpoint_1[1])  # get midpoint_1 respective to bottom-left
                if track_id not in paths:
                    paths[track_id] = deque(maxlen=2)  # path保存了每个track的两个帧的midpoint（运动轨迹）
                    total_track = track_id
                paths[track_id].append(midpoint_1)
                midpoint_0 = paths[track_id][0]  # 此track前一帧的midpoint
                origin_previous_midpoint = (midpoint_0[0], ori_img.shape[0] - midpoint_0[1])

                if intersect(midpoint_1, midpoint_0, yellow_line_in[0], yellow_line_in[1]) \
                        and track_id not in already_counted:
                    self.total_counter += 1
                    last_track_id = track_id;  # 记录触线者的ID
                    cv2.line(ori_img, yellow_line_in[0], yellow_line_in[1], (0, 0, 255), 1)  # 触碰线的情况下画红线
                    already_counted.append(track_id)  # Set already counted for ID to true.
                    angle = vector_angle(origin_midpoint, origin_previous_midpoint)  # 计算角度，判断向上还是向下走
                    if angle > 0:  # 进店
                        self.up_count += 1
                        self.customer_first_enter(bbox, ori_img, track_id, yellow_line_in)
                    if angle < 0:
                        self.down_count += 1
            # 3.重识别结果 - Enter不需要管这个
            self.draw_reid_result_to_frame()
            # 4.绘制统计信息
            ori_img = self.draw_info_to_frame(angle, last_track_id, ori_img, outputs, self.total_track)
            # 5.展示图像，todo:输出结果视频
            if self.is_display:
                cv2.imshow("test", ori_img)
                if cv2.waitKey(1) & 0xFF == 27:
                    break
            end_time = time_synchronized()
            print("Index of frame: {} / "
                              "One Image spend time: {:.03f}s, "
                              "fps: {:.03f}, "
                              "tracks : {}, "
                              "detections : {}, "
                              "features of detections: {}"
                              .format(self.idx_frame, end_time - start_time, 1 / (end_time - start_time)
                                      , bbox_xywh.shape[0]
                                      , len(outputs)
                                      , len(bbox_xywh)
                                      , features.shape
                                      )
                              )
        cv2.destroyAllWindows()  ## 销毁所有opencv显示窗口

    def customer_first_enter(self, bbox, ori_img, track_id, yellow_line_in):
        # todo: 把撞线人的特征抠出来

        # 进店的时候，把人物的图像抠出来
        cv2.line(ori_img, yellow_line_in[0], yellow_line_in[1], (0, 0, 0), 1)  # 消除线条
        ROI_person = ori_img[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
        path = str(self.output_people_img_path + 'track_id-{}.jpg'.format(track_id))
        makedir(path)
        cv2.imwrite(path, ROI_person)
        # 打印当前的时间 & 顾客入店信息
        current_time = int(time.time())
        localtime = time.localtime(current_time)
        dt = time.strftime('%Y-%m-%d %H:%M:%S', localtime)
        print("[Customer came👏] current customer💂‍♂️: {}, "
              "Enter time⏰ : {}".format(
            track_id
            , dt
        ))

    def draw_info_to_frame(self, angle, last_track_id, ori_img, outputs, total_track):
        ori_img = print_statistics_to_frame(self.down_count, ori_img, self.total_counter, total_track, self.up_count)
        ori_img = draw_idx_frame(ori_img, self.idx_frame)
        if last_track_id >= 0:
            ori_img = print_newest_info(angle, last_track_id, ori_img)  # 打印撞线人的信息
        if len(outputs) > 0:  # 展示跟踪结果
            bbox_tlwh = []
            bbox_xyxy = outputs[:, :4]
            identities = outputs[:, -1]
            draw_boxes_and_text(ori_img, bbox_xyxy, identities)  # 给每个detection画框
            for bb_xyxy in bbox_xyxy:
                bbox_tlwh.append(self.deepsort._xyxy_to_tlwh(bb_xyxy))
        return ori_img

    def draw_reid_result_to_frame(self):
        pass