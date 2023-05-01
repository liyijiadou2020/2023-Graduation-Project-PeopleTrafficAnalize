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
    draw_idx_frame, increment_path, ROOT
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

from people_tracker import VideoStreamTracker
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser()

    # parser.add_argument("--video_path", default='./test_video/test3', type=str) # ok

    parser.add_argument("--video_path", default='./test_video/cam1.mp4', type=str)
    parser.add_argument("--video_out_path", default='./test_video/cam2.mp4', type=str)
    parser.add_argument("--video3_path", default='./test_video/cam3.mp4', type=str) # todo：加载成功了，但好像是yolo的问题

    parser.add_argument("--camera", action="store", dest="cam", type=int, default="-1")
    parser.add_argument('--device', default='cuda:0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument("--display", default=True, help='True: show window, False: not')
    parser.add_argument("--frame_interval", type=int, default=1)
    parser.add_argument("--cpu", dest="use_cuda", action="store_false", default=True)
    # yolov5
    parser.add_argument('--weights', nargs='+', type=str, default='./weights/yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('--img-size', type=int, default=1080, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.4, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--classes', default=[0], type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    # deep_sort
    parser.add_argument("--sort", default=False, help='True: sort model or False: reid model')
    parser.add_argument("--config_deepsort", type=str, default="./configs/deep_sort.yaml")

    return parser.parse_args()

class TrafficMonitor():
    def __init__(self, cfg, args, path_in, path_out, path3):
        self._logger = get_logger("root")
        self.args = args
        use_cuda = args.use_cuda and torch.cuda.is_available()
        if not use_cuda:
            warnings.warn("Running in cpu mode which maybe very slow!", UserWarning)
        self.yolo_model = YoloPersonDetect(self.args)
        self.deepsort = build_tracker(cfg, args.sort, use_cuda=use_cuda)         # Deepsort with ReID
        imgsz = check_img_size(args.img_size, s=32)  # check img_size

        self.dataset_1 = LoadImages(path_in, img_size=imgsz)    # ok
        self.dataset_2 = LoadImages(path_out, img_size=imgsz)   # ok
        self.dataset_3 = LoadImages(path3, img_size=imgsz)      # todo: cam3.mp4视频本身有问题

        self.query_names = []
        self.query_feat = None

        exp_name = 'exp'
        project = ROOT / 'runs/tracks'
        save_dir = increment_path(Path(project) / exp_name, exist_ok=False) # 不允许同名目录存在，如果存在则新建一个
        # save_dir = increment_path(Path(project) / exp_name, exist_ok=True)  # 不允许同名目录存在，如果存在则新建一个
        save_dir = Path(save_dir)
        self.save_dir = save_dir
        self.save_dir_in = str(save_dir / 'in')
        makedir(self.save_dir)
        makedir(self.save_dir_in)

        # p1 = [0.31, 0.50] #
        # p2 = [0.36, 0.84] #
        p1 = [0.31, 0.74]
        p2 = [0.88, 0.62]
        # 0 means this camera is entering camera
        self.cam_in_tracker = VideoStreamTracker(self.yolo_model, self.deepsort, self.dataset_1, None, [],
                                                 self.save_dir_in, True, p1, p2, 0) # is_display = False
        # p2_1 = [0.52, 0.51] # [0.31, 0.50]
        # p2_2 = [0.52, 0.93] # [0.36, 0.84]
        p2_1 = [0.31, 0.50]
        p2_2 = [0.36, 0.84]

        # 3 means this camera in store, todo: 'in2' , 'in3' 变量化
        self.cam2_tracker = VideoStreamTracker(self.yolo_model, self.deepsort, self.dataset_2, None, [],
                                               str(save_dir / 'in2'), True, p2_1, p2_2, 3)


        # p3_1 = [0.31, 0.74] # [0.52, 0.51]
        # p3_2 = [0.88, 0.62] # [0.52, 0.93]
        p3_1 = [0.52, 0.51]
        p3_2 = [0.52, 0.93]
        self.cam3_tracker = VideoStreamTracker(self.yolo_model, self.deepsort, self.dataset_3, None, [],
                                               str(save_dir / 'in3'), True, p3_1, p3_2, 3)

        self._logger.info("args: ", self.args)

    def demo(self):
        self.query_feat, self.query_names = self.cam_in_tracker.tracking()
        # self.query_feat, self.query_names = self.feature_extract_from_project_dir()
        self.cam2_tracker.tracking(self.query_feat, self.query_names)
        self.cam3_tracker.tracking(self.query_feat, self.query_names)

    def feature_extract_from_project_dir(self):
        reid_feature = Reid_feature() # reid model
        names = []
        embs = np.ones((1, 512), dtype=np.int)
        for image_name in os.listdir(self.save_dir_in):
            img = cv2.imread(os.path.join(self.save_dir_in, image_name))
            feat = reid_feature(img)  # extract normlized feat
            pytorch_output = feat.numpy()
            embs = np.concatenate((pytorch_output, embs), axis=0)
            names.append(image_name[0:-4])  # 去除.jpg作为顾客的名字
        names = names[::-1]
        names.append("None")
        feat_path = os.path.join(str(self.save_dir), 'query_features')
        names_path = os.path.join(str(self.save_dir), 'names')
        np.save(feat_path, embs[:-1, :])
        np.save(names_path, names)  # save query

        # 从路径加载query todo: 这操作？
        path = '{}/query_features.npy'.format(str(self.save_dir))
        makedir(path)
        query = np.load(path)
        cos_sim = cosine_similarity(embs, query)
        max_idx = np.argmax(cos_sim, axis=1)
        maximum = np.max(cos_sim, axis=1)
        max_idx[maximum < 0.6] = -1
        self._logger.info("Succeed extracting features for ReID.")

        return query, names

# ********************************************************
if __name__ == '__main__':
    # initialize parameters
    args = parse_args()

    # initialize DeepSORT
    cfg = get_config()
    cfg.merge_from_file(args.config_deepsort)

    monitor = TrafficMonitor(cfg, args, path_in=args.video_path, path_out=args.video_out_path, path3=args.video3_path)
    with torch.no_grad():
        monitor.demo()
