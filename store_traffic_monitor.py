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
        self.deepsort = build_tracker(cfg, args.sort, use_cuda=use_cuda) # Deepsort with ReID
        imgsz = check_img_size(args.img_size, s=32)  # check img_size

        self.dataset_1 = LoadImages(path_in, img_size=imgsz)    # ok
        self.dataset_2 = LoadImages(path_out, img_size=imgsz)   # ok
        self.dataset_3 = LoadImages(path3, img_size=imgsz)      # ok

        self.reid_model = Reid_feature()
        self.cus_names = []
        self.cus_features = None

        exp_name = 'exp'
        project = ROOT / 'runs/tracks'
        save_dir = increment_path(Path(project) / exp_name, exist_ok=False) # 不允许同名目录存在，如果存在则另建一个名字不同的目录
        # save_dir = increment_path(Path(project) / exp_name, exist_ok=True)  # 允许同名目录存在，如果存在 不需要另建
        save_dir = Path(save_dir)
        self.save_dir = save_dir
        self.save_dir_in = str(save_dir / 'in')
        makedir(self.save_dir)
        makedir(self.save_dir_in)

        p1 = [0.31, 0.74]
        p2 = [1.00, 0.59]
        # 0 means this camera is entering camera
        self.cam_in_tracker = VideoStreamTracker(self.yolo_model, self.reid_model,
                                                 self.deepsort, self.dataset_1, None, [],
                                                 self.save_dir_in, True, p1, p2, 0)
        p2_1 = [0.31, 0.50]
        p2_2 = [0.36, 0.84]
        # 3 means this camera in store, todo: 'in2' , 'in3' 变量化
        self.cam2_tracker = VideoStreamTracker(self.yolo_model, self.reid_model,
                                               self.deepsort, self.dataset_2, None, [],
                                               str(save_dir / 'in2'), True, p2_1, p2_2, 3)
        p3_1 = [0.52, 0.51]
        p3_2 = [0.52, 0.93]
        self.cam3_tracker = VideoStreamTracker(self.yolo_model, self.reid_model,
                                               self.deepsort, self.dataset_3, None, [],
                                               str(save_dir / 'in3'), True, p3_1, p3_2, 3)

        self._logger.info("args: ", self.args)

    def demo(self):
        self.cus_features, self.cus_names = self.cam_in_tracker.tracking()
        self.cus_features, self.cus_names = self.feature_extract_from_in_dir()

        # self.cam2_tracker.update_reid_query(self.cus_features, self.cus_names)
        self.cam2_tracker.tracking(self.cus_features, self.cus_names)

        self.cam3_tracker.tracking(self.cus_features, self.cus_names)

        # name_idx = self.person_query('yoyo.jpg')  # 把需要查询的人物照片放在 self.save_dir，就可以通过函数查询
        # print("Query result: {}".format(self.cus_names[name_idx]))

# ----------------
    def feature_extract_from_in_dir(self):
        # reid = self.reid_model # reid model
        names = []
        embs = np.ones((1, 512), dtype=np.int)
        for image_name in os.listdir(self.save_dir_in):
            img = cv2.imread(os.path.join(self.save_dir_in, image_name))
            feat = self.reid_model(img)  # extract normlized feat
            pytorch_output = feat.numpy()
            embs = np.concatenate((pytorch_output, embs), axis=0)
            names.append(image_name[0:-4])  # 去除.jpg作为顾客的名字
        names = names[::-1]
        names.append("None")
        feat_path = os.path.join(str(self.save_dir), 'query_features')
        names_path = os.path.join(str(self.save_dir), 'names')
        np.save(feat_path, embs[:-1, :])
        np.save(names_path, names)  # save query

        path = '{}/query_features.npy'.format(str(self.save_dir))
        query = np.load(path)

        self._logger.info("query_features.npy & names.npy is created basing at images. path : ")

        return query, names

    # test: 新功能
    def person_query(self, query_image):
        '''
        Arguments:
            query_feature_vector: 要查询人物的特征向量
        Returns:
            idx : index of names, which is matched with query_image
        '''
        img = cv2.imread(str(self.save_dir / query_image))
        query_feature_vector = self.reid_model(img)

        cos_sim = cosine_similarity(self.cus_features, query_feature_vector)
        max_idx = np.argmax(cos_sim, axis=1) # 每行最大值的索引
        maximum = np.max(cos_sim, axis=1)
        max_idx[maximum < 0.6] = -1

        idx = np.argmax(max_idx)
        # print("RESULT: ", names[idx])
        return idx


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
