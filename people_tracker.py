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
from pathlib import Path
from pycallgraph2 import PyCallGraph
from pycallgraph2.output import GraphvizOutput


class VideoStreamTracker():
    def __init__(self, yolo_model,
                 reid_model,
                 deepsort_model,
                 dataset,
                 query_feat,
                 query_names,
                 output_people_img_path,
                 is_display, p1, p2, tracker_type_number=-1):
        '''
        todo: 默认参数：cus_features - None, cus_names - [],  is_display - True
        parameters:
            cus_features : reid使用的查询库的特征
            output_people_img_path : 将提取出的人物图像放在什么路径下
            is_display : 是否展示视频
            p1, p2 : 黄线的两个端点占画面的比例（0-1之间），用列表的形式传递
            tracker_type_number : 0代表入口摄像头， 1代表出口摄像头， 其他数字代表店内摄像头
        '''
        self.idx_frame = 0
        self.total_track = 0
        self.total_counter = 0
        self.up_count = 0
        self.down_count = 0
        self.already_counted = deque(maxlen=100)
        self.yolo_model = yolo_model
        self.deepsort = deepsort_model
        self.dataset = dataset
        self.tracker_type_number = tracker_type_number
        # 1.画黄线
        self.p1_ratio = p1
        self.p2_ratio = p2
        # 2.处理tracks
        self._save_dir = output_people_img_path
        # 3.ReID
        self.reid_model = reid_model
        self.query_feat = query_feat
        self.query_names = query_names
        # 4.绘制统计信息 & 绘制检测框 & 绘制帧数
        # 5.展示图像，输出结果视频
        self.is_display = is_display
        self.is_save_vid = False # todo: 增加到参数中
        # 6.销毁窗口 & 打印log

    def tracking(self, query_feat=None, query_names=[]):
        # 如果不是入口摄像头，那么在处理之前要更新一下query_feat, cus_names
        if self.tracker_type_number != 0:
            self.update_reid_query(query_feat, query_names)

        paths = {}  # 每一个track的行动轨迹
        last_track_id = -1
        angle = -1
        already_counted = deque(maxlen=100)  # temporary memory for storing counted IDs

        vid_path = None
        vid_writer = None

        for video_path, img, ori_img, vid_cap in self.dataset:  # 获取视频帧
            self.idx_frame += 1
            start_time = time_synchronized()

            # yolo detection
            bbox_xywh, cls_conf, cls_ids, xy = self.yolo_model.detect(video_path, img, ori_img, vid_cap)
            # outputs, features = self.deepsort.update(bbox_xywh, cls_conf, ori_img)
            if len(bbox_xywh) > 0 and len(cls_conf) > 0: # 加上这句，如果没检测到人就直接跳过这一帧
                outputs, features = self.deepsort.update(bbox_xywh, cls_conf, ori_img)

            # 1.画黄线
            yellow_line = draw_yellow_line(self.p1_ratio, self.p2_ratio, ori_img)
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

                if intersect(midpoint_1, midpoint_0, yellow_line[0], yellow_line[1]) \
                        and track_id not in already_counted:
                    self.total_counter += 1
                    last_track_id = track_id;  # 记录触线者的ID
                    cv2.line(ori_img, yellow_line[0], yellow_line[1], (0, 0, 255), 1)  # 触碰线的情况下画红线
                    already_counted.append(track_id)  # Set already counted for ID to true.
                    angle = vector_angle(origin_midpoint, origin_previous_midpoint)  # 计算角度，判断向上还是向下走
                    if angle > 0:  # 进店
                        self.up_count += 1
                        if self.tracker_type_number == 0:
                            self.customer_enter(bbox, ori_img, track_id, yellow_line)
                        else: # 功能已经实现，但是要调参
                            self.person_search(bbox, ori_img, track_id)
                    if angle < 0:
                        self.down_count += 1

            # 3.重识别结果 - Enter摄像头不需要管这个
            if self.tracker_type_number != 0:
                img, match_names = self.draw_reid_result_to_frame(features, ori_img, xy)
            # 4.绘制统计信息
            ori_img = self.draw_info_to_frame(angle, last_track_id, ori_img, outputs, self.total_track)
            # 5.展示图像，输出结果视频
            if self.is_display:
                cv2.imshow("test", ori_img)
                if cv2.waitKey(1) & 0xFF == 27:
                    break
            end_time = time_synchronized()
            if self.is_save_vid: # 输出视频
                if vid_path != self._save_dir:
                    vid_path = self._save_dir
                    if isinstance(vid_writer, cv2.VideoWriter):
                        vid_writer.release()
                    if vid_cap:  # video
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    else:  # stream
                        fps, w, h = 30, ori_img.shape[1], ori_img.shape[0]
                    save_path = str(
                        Path(self._save_dir).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                    vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                vid_writer.write(ori_img)
            print("Index of frame: {} "
                              "Spend time: {:.03f}s, "
                              "Fps: {:.03f}, "
                              "Tracks : {}, "
                              "Detections : {}, "
                              "Features of detections: {}"
                              .format(self.idx_frame
                                      # , self.total_frame_count
                                      , end_time - start_time
                                      , 1 / (end_time - start_time)
                                      , bbox_xywh.shape[0]
                                      , len(outputs)
                                      , len(bbox_xywh)
                                      , features.shape
                                      )
                              )
        cv2.destroyAllWindows()  ## 销毁所有opencv显示窗口

        if self.tracker_type_number == 0: # 如果这是入口摄像头，需要提取特征后续使用
            return self.feature_extract()
        # vid_writer.release()

    def customer_enter(self, bbox, ori_img, track_id, yellow_line_in):
        # todo: 把撞线人的特征输出来 & 记录入店的时间

        # 进店的时候，把人物的图像抠出来
        cv2.line(ori_img, yellow_line_in[0], yellow_line_in[1], (0, 0, 0), 1)  # 消除线条
        ROI_person = ori_img[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
        path = str(self._save_dir + '/cus{}.jpg'.format(track_id))
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

    def person_search(self, bbox, ori_img, track_id): # todo: bug, 重识别失败了，why？
        ROI_person = ori_img[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
        query_feature_vector = self.reid_model(ROI_person)
        # ---- compare -----------
        cos_sim = cosine_similarity(self.query_feat, query_feature_vector)
        max_idx = np.argmax(cos_sim, axis=1)  # 每行最大值的索引
        maximum = np.max(cos_sim, axis=1)
        print("[ReID DEBUG] maximum = ", maximum)
        # --- fix bug: 如果搜索不到 ---
        if max(maximum) > 0.4: # 0.5大了
            max_idx[maximum < 0.4] = -1
            idx = np.argmax(max_idx)  # todo: bug, 没办法检测全是-1的情况
            person_name = self.query_names[idx]  # 搜寻得到的
        else:
            person_name = "new-{}".format(track_id)

        print("[DEBUG-reid] person_name: ", person_name)
        path = str(self._save_dir + '/{}.jpg'.format(person_name))
        makedir(path)
        cv2.imwrite(path, ROI_person)

        # 打印当前的时间 & 顾客入店信息
        current_time = int(time.time())
        localtime = time.localtime(current_time)
        dt = time.strftime('%Y-%m-%d %H:%M:%S', localtime)
        print("[ReID Result🙌] person: {}, "
              "time⏰ : {}".format(
            person_name
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

    def draw_reid_result_to_frame(self, features, ori_img, xy):
        person_cossim = cosine_similarity(features, self.query_feat)  # 计算features和query_features的余弦相似度
        max_idx = np.argmax(person_cossim, axis=1)
        maximum = np.max(person_cossim, axis=1)
        max_idx[maximum < 0.5] = -1
        score = maximum
        reid_results = max_idx
        img, match_names = draw_reid_person(ori_img, xy, reid_results, self.query_names)  # draw_person name
        print("[ReID] match people names: {} .".format(match_names))
        return img, match_names

    def update_reid_query(self, features, names):
        self.query_feat = features
        self.query_names = names

    def feature_extract(self):
        # reid_feature = Reid_feature() # reid model
        names = []
        embs = np.ones((1, 512), dtype=np.int)
        for image_name in os.listdir(self._save_dir):
            img = cv2.imread(os.path.join(self._save_dir, image_name))
            feat = self.reid_model(img)  # 提取特征，返回正则化的numpy数组
            pytorch_output = feat.numpy()  # 转化成numpy数组
            embs = np.concatenate((pytorch_output, embs), axis=0) # 和已有的特征向量数组embs在第0维上进行拼接，得到更新后的embs数组
            names.append(image_name[0:-4])  # 去除.jpg作为顾客的名字
        names = names[::-1] # 倒序翻转 [1, 2, 3, 4, 5] -> [5, 4, 3, 2, 1]
        names.append("None")

        feat_path = os.path.join(str(self._save_dir), '..', 'query_features')
        names_path = os.path.join(str(self._save_dir), '..', 'names')
        # 实际保存的是embs数组中除了最后一行以外的所有行
        np.save(feat_path, embs[:-1, :]) # 将numpy数组 embs 的第一维度的所有元素（除了最后一个元素）保存为二进制文件的操作，文件名为 feat_path
        np.save(names_path, names)  # save query

        # 代码读取一个.npy文件，该文件中包含了一个特征向量query，将另外一组特征向量embs与query计算余弦相似度
        path = '{}.npy'.format(str(feat_path))
        query = np.load(path)
        cos_sim = cosine_similarity(embs, query)
        max_idx = np.argmax(cos_sim, axis=1)
        maximum = np.max(cos_sim, axis=1)
        max_idx[maximum < 0.6] = -1
        # -------------- 用作reid对比 ？ --------------
        print("Succeed extracting features for ReID.")

        return query, names