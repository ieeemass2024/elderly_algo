import os
import time

import cv2
import numpy as np
import torch

from models.experimental import attempt_load
from utils.yolo_utils.datasets import LoadStreams
from utils.yolo_utils.general import non_max_suppression, scale_coords
from utils.yolo_utils.plots import plot_one_box
from utils.yolo_utils.torch_utils import select_device

from utils.file_oss_utils import oss_loader

from utils.mysql_utils.database import SessionLocal
from utils.mysql_utils.models import Event

from datetime import datetime

class yoloDetect:

    def __init__(self, url):
        self.device = select_device("cpu")#cpu改这里
        # self.model = attempt_load("models/pt/fall/best40.pt", map_location=self.device)  # 模型文件
        self.model = attempt_load("models/pt/fall/best.pt", map_location=self.device)  # 模型文件
        self.half = self.device.type != 'cpu'  # half precision only supported on CUDA
        self.conf_thres = 0.7  # 置信度
        self.iou_thres = 0.45  # iou阈值
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        # self.webcam = self.source.isnumeric() or self.source.endswith('.txt') or self.source.lower().startswith(
        #     ('rtsp://', 'rtmp://', 'http://'))
        self.img_det = cv2.imread("utils/file_oss_utils/img1/tip/tip.jpg")  # 加载过程的图片
        self.dataset = LoadStreams(url, img_size=640, stride=int(self.model.stride.max()))  # model stride)
        if self.half:
            self.model.half()  # to FP16
        if self.device.type != 'cpu':
            self.model(torch.zeros(1, 3, 640, 640).to(self.device).type_as(next(self.model.parameters())))  # run once
        # self.location = loc
        self.alert_count_time = time.time()
        self.warn = False
        self.event_loc = "307门口"

    # def preprocess(self, img1):
    #     img1 = np.ascontiguousarray(img1)
    #     img1 = torch.from_numpy(img1).to(self.device)
    #     img1 = img1.half() if self.half else img1.float()  # uint8 to fp16/32
    #     img1 /= 255.0  # 图像归一化
    #     if img1.ndimension() == 3:
    #         img1 = img1.unsqueeze(0)
    #     return img1

    def draw_box(self, im0, det):
        # view_img = check_imshow()
        # Print results
        pred_boxes = []
        for c in det[:, -1].unique():
            n = (det[:, -1] == c).sum()  # detections per class
            # s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

        for *xyxy, conf, cls_id in det:
            lbl = self.names[int(cls_id)]
            xyxy = torch.tensor(xyxy).view(1, 4).view(-1).tolist()
            score = round(conf.tolist(), 3)
            label = "{}: {}".format(lbl, score)
            x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
            pred_boxes.append((x1, y1, x2, y2, lbl, score))
            # if view_img:
            plot_one_box(xyxy, im0, color=(255, 0, 0), label=label)
            # frame = cv2.imencode('.jpg', im0)[1].tobytes()
            return im0

    def detect(self, stream):
        while True:
            for path, img, im0s, vid_cap in self.dataset:

                # #危险区域
                # for b in range(0, img1.shape[0]):
                #     mask = np.zeros([img1[b].shape[1], img1[b].shape[2]], dtype=np.uint8)  # 定义全零掩码mask
                #     # mask[round(img1[b].shape[1] * hl1):img1[b].shape[1], round(img1[b].shape[2] * wl1):img1[b].shape[2]] = 255
                #     pts = np.array([[int(img1[b].shape[2] * self.location[1]), int(img1[b].shape[1] * self.location[0])],  # pts1
                #                     [int(img1[b].shape[2] * self.location[3]), int(img1[b].shape[1] * self.location[2])],  # pts2
                #                     [int(img1[b].shape[2] * self.location[5]), int(img1[b].shape[1] * self.location[4])],  # pts3
                #                     [int(img1[b].shape[2] * self.location[7]), int(img1[b].shape[1] * self.location[6])]], np.int32)
                #     mask = cv2.fillPoly(mask, [pts], (255, 255, 255))  # 形成掩膜mask
                #     imgc = img1[b].transpose((1, 2, 0))
                #     imgc = cv2.add(imgc, np.zeros(np.shape(imgc), dtype=np.uint8), mask=mask)
                #     # cv2.imshow('1',imgc)
                #     img1[b] = imgc.transpose((2, 0, 1))

                # img1 = self.preprocess(img1)
                img = torch.from_numpy(img).to(self.device)
                img = img.half() if self.half else img.float()  # uint8 to fp16/32
                img /= 255.0  # 0 - 255 to 0.0 - 1.0
                if img.ndimension() == 3:
                    img = img.unsqueeze(0)
                # tensor_cpu = img1.to('cpu')
                # npy = tensor_cpu.numpy()
                # resize = np.expand_dims(npy, axis=0)
                # img1 = torch.from_numpy(resize)
                # # tensor_gpu = resize.to('gpu')
                pred = self.model(img, augment=False)[0]  # 0.22s
                pred = pred.float()
                pred = non_max_suppression(pred, self.conf_thres, self.iou_thres)
                for i, det in enumerate(pred):
                    p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), self.dataset.count
                    gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh

                    # # 危险区域画框
                    # p, s, im0, frame = path[i], f'{i}: ', im0s[i].copy(), self.dataset.count
                    # cv2.putText(im0, "dangerous area",
                    #             (int(im0.shape[1] * self.location[1] - 5), int(im0.shape[0] * self.location[0] - 5)),
                    #             cv2.FONT_HERSHEY_SIMPLEX,
                    #             1.0, (255, 255, 0), 2, cv2.LINE_AA)
                    #
                    # pts = np.array(
                    #     [[int(im0.shape[1] * self.location[1]), int(im0.shape[0] * self.location[0])],  # pts1
                    #      [int(im0.shape[1] * self.location[3]), int(im0.shape[0] * self.location[2])],  # pts2
                    #      [int(im0.shape[1] * self.location[5]), int(im0.shape[0] * self.location[4])],  # pts3
                    #      [int(im0.shape[1] * self.location[7]), int(im0.shape[0] * self.location[6])]], np.int32)
                    # # pts = pts.reshape((-1, 1, 2))
                    # zeros = np.zeros((im0.shape), dtype=np.uint8)
                    # mask = cv2.fillPoly(zeros, [pts], color=(0, 0, 100))
                    # im0 = cv2.addWeighted(im0, 1, mask, 0.2, 0)
                    # cv2.polylines(im0, [pts], True, (255, 255, 0), 3)
                    # # plot_one_box(dr, im0, label='Detection_Region', color=(0, 255, 0), line_thickness=2)
                    # colors = (255, 255, 255)

                    if len(det):
                        # alarm(self.names, det)
                        det[:, :4] = scale_coords(
                            img.shape[2:], det[:, :4], im0.shape).round()
                        # print(f'Done. ({time.time() - t0:.3f}s)')
                        im0 = self.draw_box(im0, det)
                        # cv2.imwrite(os.path.join(r"C:\Users\qys\Desktop\detectResult", det[]+'.jpg'), im0)
                        t = time.time()
                        if t - self.alert_count_time >= 5:
                            self.alert_count_time = t
                            timestamp = int(t)
                            time_str = time.strftime("%Y%m%d%H%M%S", time.localtime(timestamp))
                            print(time_str)

                            # 事件写入
                            event_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # 发生日期

                            # 上传到oss
                            # 要上传图片所在的文件夹
                            local_dir_path = 'utils/file_oss_utils/img/fall'
                            # 图片对应的path，对应上传到的文件夹
                            elderly_path = "/img/" + "fall"
                            img_path = \
                                'https://ai-care-system.oss-cn-beijing.aliyuncs.com/' \
                                'resources/smart_elderly_care/cv_file' \
                                + elderly_path + "/" + str(time_str) + '.jpg'

                            # 开启session
                            session = SessionLocal()
                            # 向数据库插入数据
                            cv_event = Event(event_type=5,
                                             event_date=event_date,
                                             event_location=self.event_loc,
                                             event_desc="检测到有老人跌倒",
                                             event_img=img_path)
                            session.add(cv_event)

                            # 提交
                            session.flush()
                            cv2.imwrite('utils/file_oss_utils/img/fall/' + str(time_str) + '.jpg', im0)
                            # 提交
                            session.commit()
                            # 关闭session
                            session.close()

                            oss_loader.upload_file_to_oss(local_dir_path=local_dir_path, elderly_path=elderly_path)
                            # 清空本地
                            for root, dirs, files in os.walk(local_dir_path):
                                # 删除每个文件
                                for file in files:
                                    file_path = os.path.join(root, file)
                                    os.remove(file_path)

                            # cv2.imwrite('utils/file_oss_utils/img1/fall/' + time_str + '.jpg', im0)
                            # cv2.imwrite(os.path.join(r"F:\detectResult", time_str + '.jpg'), im0)
                            print("保存截图")
                            self.warn = True
                    self.img_det = im0
