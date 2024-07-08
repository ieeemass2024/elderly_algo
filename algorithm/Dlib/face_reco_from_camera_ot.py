import dlib
import numpy as np
import cv2
import os
import pandas as pd
import time
import logging
import torch
from utils.file_oss_utils import oss_loader
from utils.mysql_utils.database import SessionLocal
from utils.mysql_utils.models import Event, Elderly

from algorithm.ExpressionRecognitionCNN.emo import emo
from datetime import datetime

# from algorithm.emotionKNN.detect import emotion

# 使用pytorch框架进行GPU加速
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Dlib 正向人脸检测器 / Use frontal face detector of Dlib
detector = dlib.get_frontal_face_detector()

# Dlib 人脸 landmark 特征点检测器 / Get face landmarks
predictor = dlib.shape_predictor('algorithm/Dlib/data/data_dlib/shape_predictor_68_face_landmarks.dat')

# Dlib Resnet 人脸识别模型, 提取 128D 的特征矢量 / Use Dlib resnet50 model to get 128D face descriptor
face_reco_model = dlib.face_recognition_model_v1(
    "algorithm/Dlib/data/data_dlib/dlib_face_recognition_resnet_model_v1.dat")


class Face_Recognizer:
    def __init__(self):
        self.hasUnknown = 0
        self.detect_img = cv2.imread("tip.jpg")
        self.font = cv2.FONT_ITALIC

        # FPS
        self.frame_time = 0
        self.frame_start_time = 0
        self.fps = 0
        self.fps_show = 0
        self.start_time = time.time()
        self.emo = {}

        # cnt for frame
        self.frame_cnt = 0

        # 情绪识别计时
        self.count_time = time.time()
        # 交互计时
        self.ia_count_time = time.time()
        # 报警计时
        self.alert_count_time = time.time()
        # 老人笑计时
        self.laugh_count_time = time.time()

        self.face = 0

        # 用来存放所有录入人脸特征的数组 / Save the features of faces in the database
        self.face_features_known_list = []
        # 存储录入人脸名字 / Save the name of faces in the database
        self.face_name_known_list = []

        # 用来存储上一帧和当前帧 ROI 的质心坐标 / List to save centroid positions of ROI in frame N-1 and N
        self.last_frame_face_centroid_list = []
        self.current_frame_face_centroid_list = []

        # 用来存储上一帧和当前帧检测出目标的名字 / List to save names of objects in frame N-1 and N
        self.last_frame_face_name_list = []
        self.current_frame_face_name_list = []

        # 上一帧和当前帧中人脸数的计数器 / cnt for faces in frame N-1 and N
        self.last_frame_face_cnt = 0
        self.current_frame_face_cnt = 0

        # 用来存放进行识别时候对比的欧氏距离 / Save the e-distance for faceX when recognizing
        self.current_frame_face_X_e_distance_list = []

        # 存储当前摄像头中捕获到的所有人脸的坐标名字 / Save the positions and names of current faces captured
        self.current_frame_face_position_list = []
        # 存储当前摄像头中捕获到的人脸特征 / Save the features of people in current frame
        self.current_frame_face_feature_list = []

        # e distance between centroid of ROI in last and current frame
        self.last_current_frame_centroid_e_distance = 0

        # 控制再识别的后续帧数 / Reclassify after 'reclassify_interval' frames
        # 如果识别出 "unknown" 的脸, 将在 reclassify_interval_cnt 计数到 reclassify_interval 后, 对于人脸进行重新识别
        self.reclassify_interval_cnt = 0
        self.reclassify_interval = 10

        self.stream_address = "0"
        self.event_location = "档案室"

    # 从 "features_all.csv" 读取录入人脸特征 / Get known faces from "features_all.csv"
    def get_face_database(self):
        if os.path.exists("algorithm/Dlib/data/features_all.csv"):
            path_features_known_csv = "algorithm/Dlib/data/features_all.csv"
            csv_rd = pd.read_csv(path_features_known_csv, header=None)
            for i in range(csv_rd.shape[0]):
                features_someone_arr = []
                self.face_name_known_list.append(csv_rd.iloc[i][0])
                for j in range(1, 129):
                    if csv_rd.iloc[i][j] == '':
                        features_someone_arr.append('0')
                    else:
                        features_someone_arr.append(csv_rd.iloc[i][j])
                self.face_features_known_list.append(features_someone_arr)
            logging.info("Faces in Database： %d", len(self.face_features_known_list))
            return 1
        else:
            logging.warning("'features_all.csv' not found!")
            logging.warning("Please run 'get_faces_from_camera.py' "
                            "and 'features_extraction_to_csv.py' before 'face_reco_from_camera.py'")
            return 0

    def update_fps(self):
        now = time.time()
        # 每秒刷新 fps / Refresh fps per second
        if str(self.start_time).split(".")[0] != str(now).split(".")[0]:
            self.fps_show = self.fps
        self.start_time = now
        self.frame_time = now - self.frame_start_time
        self.fps = 1.0 / self.frame_time
        self.frame_start_time = now

    @staticmethod
    # 计算两个128D向量间的欧式距离 / Compute the e-distance between two 128D features
    def return_euclidean_distance(feature_1, feature_2):

        feature_1 = torch.tensor(feature_1)
        feature_2 = torch.tensor(feature_2)
        dist = ((torch.pow((feature_1 - feature_2), 2)).sum()).sqrt()
        return dist

    # 使用质心追踪来识别人脸 / Use centroid tracker to link face_x in current frame with person_x in last frame
    def centroid_tracker(self):
        for i in range(len(self.current_frame_face_centroid_list)):
            e_distance_current_frame_person_x_list = []
            # 对于当前帧中的人脸1, 和上一帧中的 人脸1/2/3/4/.. 进行欧氏距离计算 / For object 1 in current_frame, compute e-distance with object 1/2/3/4/... in last frame
            for j in range(len(self.last_frame_face_centroid_list)):
                self.last_current_frame_centroid_e_distance = self.return_euclidean_distance(
                    self.current_frame_face_centroid_list[i], self.last_frame_face_centroid_list[j])

                e_distance_current_frame_person_x_list.append(
                    self.last_current_frame_centroid_e_distance)

            last_frame_num = e_distance_current_frame_person_x_list.index(
                min(e_distance_current_frame_person_x_list))
            # print("i: " + i)
            # print("last_frame_num: " + last_frame_num)
            self.current_frame_face_name_list[i] = self.last_frame_face_name_list[last_frame_num]

    # 生成的 cv2 window 上面添加说明文字 / putText on cv2 window
    def draw_note(self, img_rd):
        # 添加说明 / Add some info on windows
        # cv2.putText(img_rd, "Face Recognizer with OT", (20, 40), self.font, 1, (255, 255, 255), 1, cv2.LINE_AA)
        # cv2.putText(img_rd, "Frame:  " + str(self.frame_cnt), (20, 100), self.font, 0.8, (0, 255, 0), 1,
        #             cv2.LINE_AA)
        # cv2.putText(img_rd, "FPS:    " + str(self.fps.__round__(2)), (20, 130), self.font, 0.8, (0, 255, 0), 1,
        #             cv2.LINE_AA)
        cv2.putText(img_rd, "Faces:  " + str(self.current_frame_face_cnt), (20, 160), self.font, 0.8, (0, 255, 0), 1,
                    cv2.LINE_AA)
        # cv2.putText(img_rd, "Q: Quit", (20, 450), self.font, 0.8, (255, 255, 255), 1, cv2.LINE_AA)

        for i in range(len(self.current_frame_face_name_list)):
            img_rd = cv2.putText(img_rd, self.emo[i], tuple(
                [int(self.current_frame_face_centroid_list[i][0]), int(self.current_frame_face_centroid_list[i][1])]),
                                 self.font,
                                 0.8, (255, 190, 0),
                                 1,
                                 cv2.LINE_AA)

    # 处理获取的视频流, 进行人脸识别 / Face detection and recognition wit OT from input video stream
    def process(self, stream):
        # 1. 读取存放所有人脸特征的 csv / Get faces known from "features.all.csv"
        if self.get_face_database():
            while stream.isOpened():
                self.frame_cnt += 1
                logging.debug("Frame " + str(self.frame_cnt) + " starts")
                flag, img_rd = stream.read()
                kk = cv2.waitKey(1)

                # 2. 检测人脸 / Detect faces for frame X
                faces = detector(img_rd, 0)

                # 3. 更新人脸计数器 / Update cnt for faces in frames
                self.last_frame_face_cnt = self.current_frame_face_cnt
                self.current_frame_face_cnt = len(faces)

                # 将人脸图像切割，进行情绪识别
                # 存放预测结果
                t = time.time()
                if t - self.count_time >= 1 or self.face != self.current_frame_face_cnt:
                    self.emo = {}
                    self.face = self.current_frame_face_cnt
                    num = 0
                    for k, d in enumerate(faces):

                        # 计算矩形大小
                        # (x,y), (宽度width, 高度height)
                        pos_start = tuple([d.left(), d.top()])
                        pos_end = tuple([d.right(), d.bottom()])

                        # 计算矩形框大小
                        height = d.bottom() - d.top()
                        width = d.right() - d.left()

                        img_height, img_width, _ = img_rd.shape

                        # 根据人脸大小生成空的图像
                        img_blank = np.zeros((height, width, 3), np.uint8)

                        for i in range(height):
                            for j in range(width):
                                # 检查当前坐标是否超出图像范围
                                if 0 <= (d.top() + i) < img_height and (d.left() + j) >= 0 and (
                                        d.left() + j) < img_width:
                                    img_blank[i][j] = img_rd[d.top() + i][d.left() + j]
                                else:
                                    img_blank[i][j] = 0
                        self.emo[num] = emo(img_blank)
                        print(emo(img_blank))
                        # print(emotion(img_blank))
                        # cv2.imwrite("face/img_face_" + str(k + 1) + ".jpg", img_blank)
                        num += 1
                    self.count_time = t
                    # cv2.imshow("face_"+str(k+1), img_blank)

                # 4. 更新上一帧中的人脸列表 / Update the face name list in last frame
                self.last_frame_face_name_list = self.current_frame_face_name_list[:]

                # 5. 更新上一帧和当前帧的质心列表 / update frame centroid list
                self.last_frame_face_centroid_list = self.current_frame_face_centroid_list
                self.current_frame_face_centroid_list = []

                # 6.1 如果当前帧和上一帧人脸数没有变化 / if cnt not changes
                if (self.current_frame_face_cnt == self.last_frame_face_cnt) and (
                        self.reclassify_interval_cnt != self.reclassify_interval):
                    logging.debug \
                        ("scene 1: 当前帧和上一帧相比没有发生人脸数变化 / No face cnt changes in this frame!!!")

                    self.current_frame_face_position_list = []

                    if "unknown" in self.current_frame_face_name_list:
                        logging.debug("  有未知人脸, 开始进行 reclassify_interval_cnt 计数")
                        self.reclassify_interval_cnt += 1

                    if self.current_frame_face_cnt != 0:
                        for k, d in enumerate(faces):
                            self.current_frame_face_position_list.append(tuple(
                                [faces[k].left(), int(faces[k].bottom() + (faces[k].bottom() - faces[k].top()) / 4)]))
                            self.current_frame_face_centroid_list.append(
                                [int(faces[k].left() + faces[k].right()) / 2,
                                 int(faces[k].top() + faces[k].bottom()) / 2])

                            img_rd = cv2.rectangle(img_rd,
                                                   tuple([d.left(), d.top()]),
                                                   tuple([d.right(), d.bottom()]),
                                                   (255, 255, 255), 2)

                    # 如果当前帧中有多个人脸, 使用质心追踪 / Multi-faces in current frame, use centroid-tracker to track
                    if self.current_frame_face_cnt != 1:
                        self.centroid_tracker()

                    for i in range(self.current_frame_face_cnt):
                        # 6.2 Write names under ROI
                        img_rd = cv2.putText(img_rd, self.current_frame_face_name_list[i],
                                             self.current_frame_face_position_list[i], self.font, 0.8, (0, 255, 255), 1,
                                             cv2.LINE_AA)
                    self.draw_note(img_rd)

                # 6.2 如果当前帧和上一帧人脸数发生变化 / If cnt of faces changes, 0->1 or 1->0 or ...
                else:
                    logging.debug("scene 2: 当前帧和上一帧相比人脸数发生变化 / Faces cnt changes in this frame")
                    self.current_frame_face_position_list = []
                    self.current_frame_face_X_e_distance_list = []
                    self.current_frame_face_feature_list = []
                    self.reclassify_interval_cnt = 0

                    # 6.2.1 人脸数减少 / Face cnt decreases: 1->0, 2->1, ...
                    if self.current_frame_face_cnt == 0:
                        logging.debug("  scene 2.1 人脸消失, 当前帧中没有人脸 / No faces in this frame!!!")
                        # clear list of names and features
                        self.current_frame_face_name_list = []
                    # 6.2.2 人脸数增加 / Face cnt increase: 0->1, 0->2, ..., 1->2, ...
                    else:
                        logging.debug(
                            "  scene 2.2 出现人脸, 进行人脸识别 / Get faces in this frame and do face recognition")
                        self.current_frame_face_name_list = []
                        self.current_frame_face_feature_list = []
                        self.current_frame_face_centroid_list = []
                        self.current_frame_face_position_list = []
                        self.current_frame_face_X_e_distance_list = []

                        # 使用多线程处理每张人脸的特征提取和识别
                        import concurrent.futures

                        def process_face(i):
                            shape = predictor(img_rd, faces[i])
                            face_feature = face_reco_model.compute_face_descriptor(img_rd, shape)
                            face_name = "unknown"

                            # 计算人脸中心点
                            face_centroid = [int(faces[i].left() + faces[i].right()) / 2,
                                             int(faces[i].top() + faces[i].bottom()) / 2]
                            face_position = (
                            faces[i].left(), int(faces[i].bottom() + (faces[i].bottom() - faces[i].top()) / 4))

                            face_X_e_distance_list = []

                            for j in range(len(self.face_features_known_list)):
                                if str(self.face_features_known_list[j][0]) != '0.0':
                                    e_distance_tmp = self.return_euclidean_distance(face_feature,
                                                                                    self.face_features_known_list[j])
                                    logging.debug("      with person %d, the e-distance: %f", j + 1, e_distance_tmp)
                                    face_X_e_distance_list.append(e_distance_tmp)
                                else:
                                    face_X_e_distance_list.append(999999999)

                            similar_person_num = face_X_e_distance_list.index(min(face_X_e_distance_list))

                            if min(face_X_e_distance_list) < 0.4:
                                face_name = self.face_name_known_list[similar_person_num]
                                logging.debug("  Face recognition result: %s", face_name)
                            else:
                                logging.debug("  Face recognition result: Unknown person")

                            return face_feature, face_name, face_centroid, face_position, face_X_e_distance_list

                        with concurrent.futures.ThreadPoolExecutor() as executor:
                            results = list(executor.map(process_face, range(len(faces))))

                        for result in results:
                            face_feature, face_name, face_centroid, face_position, face_X_e_distance_list = result
                            self.current_frame_face_feature_list.append(face_feature)
                            self.current_frame_face_name_list.append(face_name)
                            self.current_frame_face_centroid_list.append(face_centroid)
                            self.current_frame_face_position_list.append(face_position)
                            self.current_frame_face_X_e_distance_list.append(face_X_e_distance_list)

                        for k, d in enumerate(faces):
                            img_rd = cv2.rectangle(img_rd,
                                                   tuple([d.left(), d.top()]),
                                                   tuple([d.right(), d.bottom()]),
                                                   (255, 255, 255), 2)

                        for i in range(len(self.current_frame_face_name_list)):
                            img_rd = cv2.putText(img_rd, self.current_frame_face_name_list[i],
                                                 self.current_frame_face_position_list[i], self.font, 0.8,
                                                 (0, 255, 255), 1, cv2.LINE_AA)
                    # else:
                    #     logging.debug \
                    #         ("  scene 2.2 出现人脸, 进行人脸识别 / Get faces in this frame and do face recognition")
                    #     self.current_frame_face_name_list = []
                    #     for i in range(len(faces)):
                    #         shape = predictor(img_rd, faces[i])
                    #         self.current_frame_face_feature_list.append(
                    #             face_reco_model.compute_face_descriptor(img_rd, shape))
                    #         self.current_frame_face_name_list.append("unknown")
                    #
                    #     # 6.2.2.1 遍历捕获到的图像中所有的人脸 / Traversal all the faces in the database
                    #     self.current_frame_face_centroid_list = []
                    #     self.current_frame_face_position_list = []
                    #     for k in range(len(faces)):
                    #         logging.debug("  For face %d in current frame:", k + 1)
                    #         self.current_frame_face_centroid_list.append(
                    #             [int(faces[k].left() + faces[k].right()) / 2,
                    #              int(faces[k].top() + faces[k].bottom()) / 2])
                    #
                    #         self.current_frame_face_X_e_distance_list = []
                    #
                    #         # 6.2.2.2 每个捕获人脸的名字坐标 / Positions of faces captured
                    #         self.current_frame_face_position_list.append(tuple(
                    #             [faces[k].left(), int(faces[k].bottom() + (faces[k].bottom() - faces[k].top()) / 4)]))
                    #
                    #         # 6.2.2.3 对于某张人脸, 遍历所有存储的人脸特征
                    #         # For every faces detected, compare the faces in the database
                    #         for i in range(len(self.face_features_known_list)):
                    #             # 如果 q 数据不为空
                    #             if str(self.face_features_known_list[i][0]) != '0.0':
                    #                 e_distance_tmp = self.return_euclidean_distance(
                    #                     self.current_frame_face_feature_list[k],
                    #                     self.face_features_known_list[i])
                    #                 logging.debug("      with person %d, the e-distance: %f", i + 1, e_distance_tmp)
                    #                 self.current_frame_face_X_e_distance_list.append(e_distance_tmp)
                    #             else:
                    #                 # 空数据 person_X
                    #                 self.current_frame_face_X_e_distance_list.append(999999999)
                    #
                    #         # 6.2.2.4 寻找出最小的欧式距离匹配 / Find the one with minimum e distance
                    #         similar_person_num = self.current_frame_face_X_e_distance_list.index(
                    #             min(self.current_frame_face_X_e_distance_list))
                    #
                    #         if min(self.current_frame_face_X_e_distance_list) < 0.4:
                    #             self.current_frame_face_name_list[k] = self.face_name_known_list[similar_person_num]
                    #             logging.debug("  Face recognition result: %s",
                    #                           self.face_name_known_list[similar_person_num])
                    #         else:
                    #             logging.debug("  Face recognition result: Unknown person")
                    #
                    #         self.current_frame_face_position_list = []
                    #         self.current_frame_face_centroid_list = []
                    #         for k, d in enumerate(faces):
                    #             self.current_frame_face_position_list.append(tuple(
                    #                 [faces[k].left(),
                    #                  int(faces[k].bottom() + (faces[k].bottom() - faces[k].top()) / 4)]))
                    #             self.current_frame_face_centroid_list.append(
                    #                 [int(faces[k].left() + faces[k].right()) / 2,
                    #                  int(faces[k].top() + faces[k].bottom()) / 2])
                    #
                    #             img_rd = cv2.rectangle(img_rd,
                    #                                    tuple([d.left(), d.top()]),
                    #                                    tuple([d.right(), d.bottom()]),
                    #                                    (255, 255, 255), 2)
                    #         for i in range(self.current_frame_face_cnt):
                    #             # 6.2 Write names under ROI
                    #             img_rd = cv2.putText(img_rd, self.current_frame_face_name_list[i],
                    #                                  self.current_frame_face_position_list[i], self.font, 0.8,
                    #                                  (0, 255, 255), 1,
                    #                                  cv2.LINE_AA)

                        # 7. 生成的窗口添加说明文字 / Add note on cv2 window
                        self.draw_note(img_rd)
                        # cv2.imwrite("debug/debug_" + str(self.frame_cnt) + ".png", img_rd) # Dump current frame image if needed
                t = time.time()
                if t - self.laugh_count_time >= 5:
                    self.laugh_count_time = t
                    for i, name in enumerate(self.current_frame_face_name_list):
                        role = name.split('_')[0]
                        # 老人笑
                        if role == "elderly" and self.emo[i] == 'happy':
                            print('老人笑')
                            timestamp = int(time.time())
                            time_str = time.strftime("%Y%m%d%H%M%S", time.localtime(timestamp))

                            event_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # 发生日期

                            # 要上传图片所在的文件夹
                            local_dir_path = 'utils/file_oss_utils/img/laugh'
                            # 图片对应的path，对应上传到的文件夹
                            elderly_path = "/img/" + "laugh"
                            img_path = \
                                'https://ai-care-system.oss-cn-beijing.aliyuncs.com/' \
                                'resources/smart_elderly_care/cv_file' \
                                + elderly_path + "/" + str(time_str) + '.jpg'

                            # 开启session
                            session = SessionLocal()

                            # 从数据库中查询老人数据
                            elderly_name = 'ljw'
                            elderly = session.query(Elderly).filter(Elderly.elderly_name == elderly_name).first()

                            # 向数据库插入事件数据
                            cv_event = Event(event_type=3,
                                             event_date=event_date,
                                             event_location=self.event_location,
                                             event_desc="检测到有老人笑了，老人是：" + elderly.elderly_name,
                                             event_img=img_path)
                            session.add(cv_event)

                            # 提交
                            session.flush()
                            cv2.imwrite('utils/file_oss_utils/img/laugh/' + str(time_str) + '.jpg', img_rd)
                            # 提交
                            session.commit()
                            # 关闭session
                            session.close()

                            # 上传到oss
                            # 最终路径为 local_dir_path +
                            oss_loader.upload_file_to_oss(local_dir_path=local_dir_path, elderly_path=elderly_path)

                            # 清空本地
                            for root, dirs, files in os.walk(local_dir_path):
                                # 删除每个文件
                                for file in files:
                                    file_path = os.path.join(root, file)
                                    os.remove(file_path)

                if 'unknown' in self.current_frame_face_name_list:
                    # 报警
                    t = time.time()
                    if t - self.alert_count_time >= 5:
                        self.hasUnknown = 1
                        self.alert_count_time = t
                        timestamp = int(t)
                        time_str = time.strftime("%Y%m%d%H%M%S", time.localtime(timestamp))
                        event_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # 发生日期
                        print(time_str)

                        # 要上传图片所在的文件夹
                        local_dir_path = 'utils/file_oss_utils/img/unknown'
                        # 图片对应的path，对应上传到的文件夹
                        elderly_path = "/img/" + "unknown"
                        img_path = \
                            'https://ai-care-system.oss-cn-beijing.aliyuncs.com/' \
                            'resources/smart_elderly_care/cv_file' \
                            + elderly_path + "/" + str(time_str) + '.jpg'

                        # 开启session
                        session = SessionLocal()
                        # 向数据库插入数据
                        cv_event = Event(event_type=2,
                                         event_date=event_date,
                                         event_location=self.event_location,
                                         event_desc="检测到有陌生人出现",
                                         event_img=img_path)
                        session.add(cv_event)

                        # 提交
                        session.flush()
                        cv2.imwrite('utils/file_oss_utils/img/unknown/' + str(time_str) + '.jpg', img_rd)
                        # 提交
                        session.commit()
                        # 关闭session
                        session.close()

                        # 上传到oss
                        # 最终路径为 local_dir_path +
                        oss_loader.upload_file_to_oss(local_dir_path=local_dir_path, elderly_path=elderly_path)

                        # 清空本地
                        for root, dirs, files in os.walk(local_dir_path):
                            # 删除每个文件
                            for file in files:
                                file_path = os.path.join(root, file)
                                os.remove(file_path)

                # 8. 按下 'q' 键退出 / Press 'q' to exit
                if kk == ord('q'):
                    break

                self.update_fps()
                # cv2.namedWindow("camera", 1)

                # 记录老人义工交互
                if self.current_frame_face_cnt >= 2:
                    hasElderly = False
                    hasVolunteer = False
                    elderlyNum = []
                    volunteerNum = []
                    for i, name in enumerate(self.current_frame_face_name_list):
                        role = name.split('_')[0]
                        if role == "elderly":
                            hasElderly = True
                            elderlyNum.append(i)
                        elif role == "volunteer":
                            hasVolunteer = True
                            volunteerNum.append(i)
                    if hasElderly and hasVolunteer:
                        for i in elderlyNum:
                            for j in volunteerNum:
                                pt1 = (int(self.current_frame_face_centroid_list[i][0]),
                                       int(self.current_frame_face_centroid_list[i][1]))
                                pt2 = (int(self.current_frame_face_centroid_list[j][0]),
                                       int(self.current_frame_face_centroid_list[j][1]))
                                distance = (pt1[0] - pt2[0]) ^ 2 + (pt2[1] + pt2[1]) ^ 2
                                if distance <= 750:
                                    cv2.line(img_rd, pt1, pt2, (255, 0, 0), 4, cv2.LINE_AA)
                                    t = time.time()
                                    if t - self.ia_count_time > 10:
                                        event_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # 发生日期
                                        timestamp = int(time.time())
                                        time_str = time.strftime("%Y%m%d%H%M%S", time.localtime(timestamp))
                                        # 要上传图片所在的文件夹
                                        local_dir_path = 'utils/file_oss_utils/img/interact'
                                        # 图片对应的path，对应上传到的文件夹
                                        elderly_path = "/img/" + "interact"
                                        img_path = \
                                            'https://ai-care-system.oss-cn-beijing.aliyuncs.com/' \
                                            'resources/smart_elderly_care/cv_file' \
                                            + elderly_path + "/" + str(time_str) + '.jpg'

                                        # 开启session
                                        session = SessionLocal()
                                        # 向数据库插入数据
                                        cv_event = Event(event_type=4,
                                                         event_date=event_date,
                                                         event_location=self.event_location,
                                                         event_desc="义工老人交互",
                                                         event_img=img_path)
                                        session.add(cv_event)
                                        # 提交
                                        session.flush()
                                        cv2.imwrite(
                                            'utils/file_oss_utils/img/interact/' + str(time_str) + '.jpg',
                                            img_rd)
                                        # 提交
                                        session.commit()
                                        # 关闭session
                                        session.close()
                                        # 上传到oss
                                        # 最终路径为 local_dir_path +
                                        oss_loader.upload_file_to_oss(local_dir_path=local_dir_path,
                                                                      elderly_path=elderly_path)

                                        # 清空本地
                                        for root, dirs, files in os.walk(local_dir_path):
                                            # 删除每个文件
                                            for file in files:
                                                file_path = os.path.join(root, file)
                                                os.remove(file_path)

                                        self.ia_count_time = t
                self.detect_img = img_rd
                logging.debug("Frame ends\n\n")
    # def run(self):
    #     cap = cv2.VideoCapture(".data/test/test.mp4")  # Get video stream from video file
    #     # cap = cv2.VideoCapture(0)              # Get video stream from camera
    #     self.process(cap)
    #
    #     cap.release()
    #     cv2.destroyAllWindows()

#
#
# def main():
# # logging.basicConfig(level=logging.DEBUG) # Set log level to 'logging.DEBUG' to print debug info of every frame
#     logging.basicConfig(level=logging.INFO)
#     Face_Recognizer_con = Face_Recognizer()
#     Face_Recognizer_con.run()

#
# if __name__ == '__main__':
#     main()
