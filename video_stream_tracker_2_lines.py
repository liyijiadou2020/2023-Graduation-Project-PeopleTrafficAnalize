# import os
from collections import deque
import time
import cv2
import numpy as np
import os
import re
from person_count_utils import tlbr_midpoint, intersect, vector_angle, get_size_with_pil, compute_color_for_labels, \
    put_text_to_cv2_img_with_pil, draw_line, makedir, draw_statistics_to_frame, \
    draw_idx_frame, draw_newest_info_binary_lines
from utils.draw import draw_boxes_and_text, draw_reid_person, draw_boxes
from utils.torch_utils import select_device, load_classifier, time_synchronized
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path

# test -
def increment_person_name(person_name):
    # Increment person name, i.e. person1.jpg --> person1-1.jpg, person1-2.jpg etc.
    path = os.path.abspath(person_name)  # os-agnostic
    if (os.path.exists(path)):
        dirs = [d for d in os.listdir(os.path.dirname(path)) if
                re.match(rf"{os.path.splitext(os.path.basename(path))[0]}-\d+\.\w+", d)]
        i = [int(re.search(r"\d+", d).group()) for d in dirs]  # indices
        n = max(i) + 1 if i else 1  # increment number
        new_name = f"{os.path.splitext(os.path.basename(path))[0]}-{n}{os.path.splitext(os.path.basename(path))[1]}"
        return f"{os.path.dirname(path)}{os.path.sep}{new_name}"  # update path
    else:
        return str(path)

class VideoStreamTracker_2_Lines():
    def __init__(self, yolo_model,
                 reid_model,
                 deepsort_model,
                 dataset,
                 feats,
                 names,
                 camera_name,
                 output_people_img_path,
                 p1, p2, p3, p4,
                 tracker_type_number=-1, is_display=True, is_save_vid=True):
        '''
        todo: 默认参数：cus_features - None, cus_names - [],  is_display - True, is_save_vid - False
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
        self.p3_ratio = p3
        self.p4_ratio = p4
        # 2.处理tracks
        self._save_dir = output_people_img_path
        # 3.ReID
        self.reid_model = reid_model
        self.feats = feats
        self.names = names

        self.new_add_feats = np.ones((1, 512), dtype=np.int) # todo: 新增，如果发现该镜头下新出现了人，把它放到增加的feats中
        self.new_add_names = []
        # ------- 记录触线时间 -------
        self.camera_name = camera_name
        self.customers_log = {}
        # --------------------------
        # 4.绘制统计信息 & 绘制检测框 & 绘制帧数
        self.total_frame = 0
        # 5.展示图像，输出结果视频
        self.is_display = is_display
        self.is_save_vid = is_save_vid
        # 6.销毁窗口 & 打印log

    def track(self, query_feat=None, query_names=[], cus_log={}):
        # 如果不是入口摄像头，那么在处理之前要更新一下query_feat, cus_names
        if self.tracker_type_number != 0:
            self.update_reid_query(query_feat, query_names)
            self.customers_log = cus_log

        paths = {}  # 每一个track的行动轨迹
        last_track_id = -1
        # angle = -1
        is_in = False
        already_counted = deque(maxlen=100)  # temporary memory for storing counted IDs

        vid_path = None
        vid_writer = None

        '''
        img：当前帧的图像数据（numpy数组类型）,表示经过缩放或裁剪后的图像数据，这是为了满足YOLO目标检测算法输入图像的大小要求而进行的处理。
        ori_img：当前帧的原始图像数据（numpy数组类型）,表示没有进行任何处理的原始图像数据，它用于后续的跟踪器更新和结果可视化等操作。
            在这段代码中，img用于目标检测，ori_img用于跟踪和结果显示。
        vid_cap：视频的读取器对象（OpenCV中的VideoCapture类型）
        '''
        for video_path, img, ori_img, vid_cap in self.dataset:  # 获取视频帧
            self.idx_frame += 1
            start_time = time_synchronized()

            # yolo detection
            bbox_xywh, cls_conf, cls_ids, xy = self.yolo_model.detect(video_path, img, ori_img, vid_cap)
            if len(bbox_xywh) > 0: # 加上这句，如果没检测到人就直接跳过这一帧
                outputs, features = self.deepsort.update(bbox_xywh, cls_conf, ori_img)

            # 1.画黄线
            yellow_line = draw_line(self.p1_ratio, self.p2_ratio, ori_img)
            green_line = draw_line(self.p3_ratio, self.p4_ratio, ori_img, (0, 255, 0))
            # 2. 处理tracks
            for track in outputs:
                bbox = track[:4]
                track_id = track[-1]
                midpoint_1 = tlbr_midpoint(bbox)
                origi_midpoint = (midpoint_1[0],
                                   ori_img.shape[0] - midpoint_1[1])  # get midpoint_1 respective to bottom-left
                if track_id not in paths:
                    paths[track_id] = deque(maxlen=2)  # path保存了每个track的两个帧的midpoint（运动轨迹）
                    total_track = track_id
                paths[track_id].append(midpoint_1)
                midpoint_0 = paths[track_id][0]  # 此track前一帧的midpoint
                origin_previous_midpoint = (midpoint_0[0], ori_img.shape[0] - midpoint_0[1])

                if intersect(midpoint_1, midpoint_0, yellow_line[0], yellow_line[1]) \
                        and track_id not in already_counted: # 进入
                    is_in = True
                    self.total_counter += 1
                    last_track_id = track_id;  # 记录触线者的ID
                    cv2.line(ori_img, yellow_line[0], yellow_line[1], (0, 0, 255), 1)  # 触碰线的情况下画红线
                    already_counted.append(track_id)  # Set already counted for ID to true.
                    self.up_count += 1
                    self.save_photo_and_wirte_log_add_new_feats(bbox, ori_img, track_id,
                                                                self.tracker_type_number)  # 人流量统计的主要处理逻辑
                elif intersect(midpoint_1, midpoint_0, green_line[0], green_line[1]) \
                        and track_id not in already_counted: # 离开
                    is_in = False
                    self.total_counter += 1
                    last_track_id = track_id;  # 记录触线者的ID
                    cv2.line(ori_img, green_line[0], green_line[1], (0, 0, 255), 1)  # 触碰线的情况下画红线
                    already_counted.append(track_id)  # Set already counted for ID to true.
                    self.down_count += 1

            # 3.重识别结果 - Enter摄像头不需要管这个
            if self.tracker_type_number != 0:
                img, match_names = self.draw_reid_result_to_frame(features, ori_img, xy) #todo: bug

            # 4.绘制统计信息
            ori_img = self.draw_box_and_info_to_frame(is_in, last_track_id, ori_img, outputs, self.total_track)

            # 5.展示图像，输出结果视频
            if self.is_display:
                cv2.namedWindow("frame", 0) # 可以调整窗口大小
                # cv2.resizeWindow("frame", 1600, 900)  # 设置长和宽
                cv2.imshow("frame", ori_img)
                if cv2.waitKey(1) & 0xFF == 27:
                    break
            end_time = time_synchronized()
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
        cv2.destroyAllWindows()  ## 销毁所有opencv显示窗口

        if self.tracker_type_number == 0: # 如果这是入口摄像头，需要提取特征后续使用
            feats, names = self.feature_extract()
            self.save_feats_names_to_dir(feats, names)
            customer_logs = self.customers_log
            return feats, names, customer_logs
        else:
            # todo: 把这个镜头下新出现的feat和name也添加到query_features和names里去------
            # new_add_feats需要翻转
            self.new_add_feats = self.new_add_feats[:-1, :] # 取所有行，但不包括最后一行
            self.new_add_names = self.new_add_names[::-1] # 名字翻转
            self.new_add_names.append("None")

            self.feats = np.concatenate((self.feats, self.new_add_feats), axis=0)
            self.names = self.names[:-1] + self.new_add_names # 去掉names的最后一个"None"

            # ---------------
            feats, names, customer_logs = self.feats, self.names, self.customers_log
            return feats, names, customer_logs
            # ------------------------------------------------
        # vid_writer.release()

    def save_photo_and_wirte_log_add_new_feats(self, bbox, ori_img, track_id, tracker_type_number):
        # ------- reid 获取 person_name --------
        ROI_person = ori_img[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
        person_feature = self.reid_model(ROI_person)
        person_name = self.save_photo_and_get_person_name_by_reid(bbox, ori_img, track_id, tracker_type_number,
                                                                  self.camera_name)
        # ------------ 记录log ----------
        current_time = int(time.time())
        localtime = time.localtime(current_time)
        dt = time.strftime('%Y-%m-%d %H:%M:%S', localtime)
        if self.tracker_type_number == 0:  # 入口摄像机，记录所有进入的人
            self.customers_log[person_name] = {self.camera_name: dt}
        else:
            if person_name not in self.customers_log:
                self.customers_log[person_name] = {self.camera_name: dt}
                # ------- todo: 添加新的外观向量 + name -----------------
                self.new_add_feats = np.concatenate((person_feature, self.new_add_feats), axis=0) #todo: 最后需要翻转一下
                self.new_add_names.append(person_name)
                # ---------------------------------------------
            else:
                self.customers_log[person_name][self.camera_name] = dt

        # ------- print info to console 打印当前的时间 & 顾客入店信息 -------
        welcome_str = "[Customer came👏]" if self.tracker_type_number == 0 else "[ReID Result🙌]"
        print(" {} current customer💂‍♂️: {}, "
              " time⏰ : {}".format(
            welcome_str
            ,person_name
            ,dt
        ))


    def save_photo_and_get_person_name_by_reid(self, bbox, ori_img, track_id, tracker_tpye_number, cam_name):
        ROI_person = ori_img[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
        # person_name = '{}-{}'.format(cam_name, track_id)
        person_name = '{}-{}'.format('cus', track_id)
        path = str(self._save_dir + '/' + person_name + '.jpg')

        if tracker_tpye_number != 0:
            person_feature = self.reid_model(ROI_person)
            # ---- compare -----------
            cos_sim = cosine_similarity(self.feats, person_feature)
            max_idx = np.argmax(cos_sim, axis=1)  # 每行最大值的索引
            maximum = np.max(cos_sim, axis=1)
            print("[ReID DEBUG] maximum = ", max(maximum))
            if max(maximum) > 0.5:
                max_idx[maximum < 0.5] = -1
                idx = np.argmax(max_idx)
                person_name = self.names[idx]  # 搜寻得到的
            else:
                # person_name = '{}-{}'.format(cam_name, track_id)
                person_name = '{}-{}-{}'.format(cam_name,'cus', track_id)
            path = str(self._save_dir + '/' + person_name + '.jpg')
            if os.path.exists(path):
                path = increment_person_name(path)

        makedir(path)
        cv2.imwrite(path, ROI_person)
        return person_name

    def draw_box_and_info_to_frame(self, is_in, last_track_id, ori_img, outputs, total_track):
        ori_img = draw_statistics_to_frame(self.down_count, ori_img, self.total_counter, total_track, self.up_count)
        ori_img = draw_idx_frame(ori_img, self.idx_frame)
        if last_track_id >= 0:
            ori_img = draw_newest_info_binary_lines(is_in, last_track_id, ori_img)  # 打印撞线人的信息
        if len(outputs) > 0:  # 展示跟踪结果
            bbox_tlwh = []
            bbox_xyxy = outputs[:, :4]
            identities = outputs[:, -1]
            draw_boxes_and_text(ori_img, bbox_xyxy, identities)  # 给每个detection画框
            for bb_xyxy in bbox_xyxy:
                bbox_tlwh.append(self.deepsort._xyxy_to_tlwh(bb_xyxy))
        return ori_img

    def draw_reid_result_to_frame(self, features, ori_img, xy):
        person_cossim = cosine_similarity(features, self.feats)  # 计算features和query_features的余弦相似度
        max_idx = np.argmax(person_cossim, axis=1)
        maximum = np.max(person_cossim, axis=1)
        max_idx[maximum < 0.5] = -1
        score = maximum
        reid_results = max_idx
        img, match_names = draw_reid_person(ori_img, xy, reid_results, self.names)  # todo:bug
        return img, match_names

    def update_reid_query(self, features, names):
        self.feats = features
        self.names = names

    def feature_extract(self): # TODO: 并不是十分妥当！
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

        # ---------- 不需要的 --------------------
        # 代码读取一个.npy文件，该文件中包含了一个特征向量query，将另外一组特征向量embs与query计算余弦相似度
        # path = '{}.npy'.format(str(feat_path))
        # feats = np.load(path)

        # cos_sim = cosine_similarity(embs, query)
        # max_idx = np.argmax(cos_sim, axis=1)
        # maximum = np.max(cos_sim, axis=1)
        # max_idx[maximum < 0.6] = -1
        print("Succeed extracting features for ReID.")

        # return feats, names
        return embs, names # 就完全不影响

    def save_feats_names_to_dir(self, embs, names):
        feat_path = os.path.join(str(self._save_dir), '..', 'query_features')
        names_path = os.path.join(str(self._save_dir), '..', 'names')
        # 实际保存的是embs数组中除了最后一行以外的所有行
        np.save(feat_path, embs[:-1, :])  # 将numpy数组 embs 的第一维度的所有元素（除了最后一个元素）保存为二进制文件的操作，文件名为 feat_path
        np.save(names_path, names)  # save query
        print("[INFO] save query_features.npy & names.npy")

    def get_customer_logs(self):
        return self.customers_log