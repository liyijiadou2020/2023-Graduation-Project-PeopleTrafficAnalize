import os
import cv2
import numpy as np
import torch
import warnings
import argparse
from person_count_utils import \
    tlbr_midpoint, intersect, vector_angle, get_size_with_pil, compute_color_for_labels, \
    put_text_to_cv2_img_with_pil, draw_line, makedir, draw_statistics_to_frame, \
    draw_idx_frame, increment_path, ROOT
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size
from yolo_people_detect import YoloPersonDetect
from deep_sort import build_tracker, DeepReid
from utils.parser import get_config
from utils.log import get_logger
from sklearn.metrics.pairwise import cosine_similarity
from fast_reid.demo.person_bank import Reid_feature
from video_stream_tracker_2_lines import VideoStreamTracker_2_Lines
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser()
    # video source
    parser.add_argument("--video_path", default='./test_video/cam1.mp4', type=str)
    parser.add_argument("--video_out_path", default='./test_video/cam2.mp4', type=str)
    parser.add_argument("--video3_path", default='./test_video/cam3.mp4', type=str)

    parser.add_argument("--camera", action="store", dest="cam", type=int, default="-1")
    parser.add_argument('--device', default='cuda:0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument("--display", default=True, help='True: show window, False: not')
    parser.add_argument("--frame_interval", type=int, default=1)
    parser.add_argument("--cpu", dest="use_cuda", action="store_false", default=True)
    # yolov5
    parser.add_argument('--weights', nargs='+', type=str, default='./weights/yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('--img-size', type=int, default=540, help='inference size (pixels)')
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
    def __init__(self, cfg, args, path_1, path_2, path_3):
        self._logger = get_logger("root")
        self.args = args
        use_cuda = args.use_cuda and torch.cuda.is_available()
        if not use_cuda:
            warnings.warn("Running in cpu mode which maybe very slow!", UserWarning)
        # yolo + deepsort
        self.yolo_model = YoloPersonDetect(self.args)
        self.deepsort = build_tracker(cfg, args.sort, use_cuda=use_cuda) # Deepsort with ReID
        imgsz = check_img_size(args.img_size, s=32)  # check img_size
        self.dataset_1 = LoadImages(path_1, img_size=imgsz)
        self.dataset_2 = LoadImages(path_2, img_size=imgsz)
        self.dataset_3 = LoadImages(path_3, img_size=imgsz)
        # reid
        self.reid_model = Reid_feature()
        self.cus_names = []
        self.cus_features = None

        self.new_add_feats = None
        self.new_add_names = []

        exp_name = 'exp'
        project = ROOT / 'runs/tracks'
        save_dir = increment_path(Path(project) / exp_name, exist_ok=False) # 不允许同名目录存在，如果存在则另建一个名字不同的目录
        # save_dir = increment_path(Path(project) / exp_name, exist_ok=True)  # 允许同名目录存在，如果存在 不需要另建
        save_dir = Path(save_dir)
        self.save_dir = save_dir
        makedir(self.save_dir)
        # --------- cus_log -----------
        cam1_name = 'cam-1'
        cam2_name = 'cam-2'
        cam3_name = 'cam-3'
        self.cus_log = {}
        # ------------------------------------
        p1 = [0.65, 0.0]
        p2 = [0.65, 1.0]
        p3 = [0.51, 0]
        p4 = [0.51, 1.0]
        self.save_dir_in = str(save_dir / cam1_name)
        makedir(self.save_dir_in)
        # 0 means this camera is entering camera
        self.cam1_tracker = VideoStreamTracker_2_Lines(self.yolo_model, self.reid_model, self.deepsort, self.dataset_1,
                                                       None, [], cam1_name, self.save_dir_in, p1, p2, p3, p4, 0)
        p2_1 = [0.23, 0]
        p2_2 = [0.23, 1]
        p2_3 = [0.13, 0]
        p2_4 = [0.13, 1]
        # 3 means this camera in store
        self.cam2_tracker = VideoStreamTracker_2_Lines(self.yolo_model, self.reid_model, self.deepsort, self.dataset_2,
                                                       None, [], cam2_name, str(save_dir / cam2_name), p2_1, p2_2, p2_3,
                                                       p2_4, 3)
        p3_1 = [0.62, 0.51]
        p3_2 = [0.62, 1.0]
        p3_3 = [0.42, 0.51]
        p3_4 = [0.42, 1.0]
        self.cam3_tracker = VideoStreamTracker_2_Lines(self.yolo_model, self.reid_model, self.deepsort, self.dataset_3,
                                                       None, [], cam3_name, str(save_dir / cam3_name), p3_1, p3_2, p3_3,
                                                       p3_4, 3)
        # self._logger.info("args: ", self.args)

    def demo(self):
        # self.cus_features, self.cus_names = self.cam1_tracker.tracking()
        # -------- test --------
        self.cus_features, self.cus_names, self.cus_log = self.cam1_tracker.track()

        # self.cus_features, self.cus_names = self.feature_extract_from_in_dir()
        self.cus_features, self.cus_names, self.cus_log = self.cam2_tracker.track(self.cus_features, self.cus_names, self.cus_log)
        self.cus_features, self.cus_names, self.cus_log = self.cam3_tracker.track(self.cus_features, self.cus_names, self.cus_log)


        sav_txt = open(file="{}/cus_log.txt".format(self.save_dir), mode="w", encoding="utf-8")
        sav_txt.write(str(self.cus_log))
        sav_txt.close()
        # --------------- person search
        # name_idx = self.person_query('yoyo.jpg')  # 把需要查询的人物照片放在 self.save_dir，就可以通过函数查询
        # print("Query result: {}".format(self.cus_names[name_idx]))

# ----------------
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
        names = names[::-1] # 翻转
        names.append("None")
        feat_path = os.path.join(str(self.save_dir), 'query_features')
        names_path = os.path.join(str(self.save_dir), 'names')
        np.save(feat_path, embs[:-1, :]) # 取所有行，但不包括最后一行
        np.save(names_path, names)  # save query

        path = '{}/query_features.npy'.format(str(self.save_dir))
        query = np.load(path)

        self._logger.info("query_features.npy & names.npy is created basing at images. path : ")

        return query, names

# ********************************************************
if __name__ == '__main__':
    # initialize parameters
    args = parse_args()

    # initialize DeepSORT
    cfg = get_config()
    cfg.merge_from_file(args.config_deepsort)

    monitor = TrafficMonitor(cfg, args, path_1=args.video_path, path_2=args.video_out_path, path_3=args.video3_path)
    with torch.no_grad():
        monitor.demo()
