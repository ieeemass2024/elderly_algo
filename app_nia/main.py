import asyncio
import logging
import os
import socket
import time

import av
import cv2
import httpx
import threading

import utils.file_oss_utils.oss_loader as oss

from app_nia.dec_area import yoloDetect as yoloDetectArea
from app_nia.dec_fall import yoloDetect as yoloDetectFall

from fastapi import APIRouter, Body, Request, UploadFile, File
from fastapi.responses import StreamingResponse

from utils.mysql_utils.database import engine, Base, SessionLocal
from utils.mysql_utils.models import Elderly, Event, Algorithm, Camera
from datetime import datetime

from algorithm import integrate
from algorithm.Dlib.face_reco_from_camera_ot import Face_Recognizer

from typing import List
from pydantic import BaseModel


# 接收区域检测post请求体的类
class AreaRequest(BaseModel):
    loc: List[float]


detectArea = []

nia_app = APIRouter()

Base.metadata.create_all(bind=engine)


# 添加事件
@nia_app.get("/event")
async def add_event():
    # 开启session
    session = SessionLocal()

    # 检测异常，生成事件

    algorithm_name = "人脸识别"
    algorithm = session.query(Algorithm).filter(Algorithm.algorithm_name == algorithm_name).first()
    event_type = algorithm.algorithm_id  # 事件类型

    event_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # 发生日期
    event_location = '307门口'  # 发生地点
    event_desc = '测试1'  # 事件描述
    elderly_name = '毛景辉'
    # elderly_id = 1676623591240806402  # 老人id

    # 根据姓名搜索老人
    elderly = session.query(Event).filter(Elderly.elderly_name == elderly_name).first()
    elderly_id = elderly.elderly_id

    # 检测到后保存的文件
    file_name = "image_" + str(elderly_id) + "_" + event_date + ".jpg"

    # 上传样例
    # 要上传图片所在的文件夹
    local_dir_path = 'utils/file_oss_utils/img'
    # 老人图片对应的path，对应上传到的文件夹
    elderly_path = "/cv/" + str(elderly_id)
    # 最终路径为 local_dir_path +
    oss.upload_file_to_oss(local_dir_path=local_dir_path, elderly_path=elderly_path)

    # 向数据库插入数据
    cv_event = Event(event_type=event_type,
                     event_date=event_date,
                     event_location=event_location,
                     event_desc=event_desc,
                     elderly_id=elderly_id)
    session.add(cv_event)

    # 提交
    session.commit()
    # 关闭session
    session.close()
    return {"data": None, "msg": "已添加！", "code": "1"}


@nia_app.get("/")
async def root():
    return {"message": "Hello World"}


async def face_detect_iter(Face_Recognizer_con, request: Request):
    while True:
        cv2.destroyAllWindows()
        if await request.is_disconnected():
            print("Client disconnected")
            break

        # 获取图像
        img = Face_Recognizer_con.detect_img

        # 检查图像是否为空
        if img is None or img.size == 0:
            print("Empty frame detected, skipping...")
            # await asyncio.sleep(0.1)  # 避免过度占用 CPU
            continue

        # 检测陌生人并发送警报
        if Face_Recognizer_con.hasUnknown:
            async with httpx.AsyncClient() as client:
                url = "http://localhost:8090/api/v1/alert"
                payload = {"message": "这是一条预警消息", "alert_type": "检测到陌生人"}
                try:
                    response = await client.post(url, json=payload)
                    if response.status_code == 200:
                        print("Alert sent successfully")
                    else:
                        print(f"Failed to send alert, status code: {response.status_code}")
                except httpx.RequestError as e:
                    print(f"An error occurred while sending the alert: {e}")
            Face_Recognizer_con.hasUnknown = 0

        # 尝试编码图像
        try:
            frame = cv2.imencode('.jpg', img)[1].tobytes()
        except cv2.error as e:
            print(f"Error encoding image: {e}")
            continue

        # 生成视频流
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


# 人脸识别、情绪识别、交互检测
@nia_app.get('/face')
async def video_feed(request: Request):
    # 开启session
    session = SessionLocal()
    # video_url = 'rtmp://43.143.247.127:1935/hls/lym'
    # video_url = "rtmp://liteavapp.qcloud.com/live/liteavdemoplayerstreamid"
    video_url = 0
    # video_url = "rtmp://8.130.142.19:1935/hls/test"
    Face_Recognizer_con = Face_Recognizer()
    # 拿到对应的摄像头
    camera = session.query(Camera).filter(Camera.stream_address == str(video_url)).first()
    Face_Recognizer_con.stream_address = camera.stream_address
    Face_Recognizer_con.event_location = camera.camera_name
    # 关闭session
    session.close()
    integrate.detect(video_url, Face_Recognizer_con)
    return StreamingResponse(face_detect_iter(Face_Recognizer_con, request),
                             media_type='multipart/x-mixed-replace; boundary=frame')


def run(yolo, url):
    # cap = cv2.VideoCapture("./Dlib/data/test/test.mp4")  # Get video stream from video file
    # cap = cv2.VideoCapture(url)  # Get video stream from camera
    yolo.detect(url)
    # cap.release()
    # cv2.destroyAllWindows()


async def detect_iter(yolo, alertType, request: Request):
    while True:
        if await request.is_disconnected():
            print("Client disconnected")
            break
        img = yolo.img_det
        if yolo.warn:
            print("进来了-------------------------")
            async with httpx.AsyncClient() as client:
                url = "http://localhost:8090/api/v1/alert"
                if alertType == 'fall':
                    payload = {"message": "这是一条预警消息", "alert_type": "检测到有人摔倒"}
                elif alertType == 'area':
                    payload = {"message": "这是一条预警消息", "alert_type": "检测到禁区有人闯入"}
                else:
                    payload = {"message": "这是一条预警消息", "alert_type": "检测到异常"}
                await client.post(url, json=payload)
            yolo.warn = False
        frame = cv2.imencode('.jpg', img)[1].tobytes()
        yield b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n'


# 禁区入侵检测
@nia_app.post('/area')
async def cv_area(request: AreaRequest = Body(...)):  # 添加请求体参数
    global detectArea
    detectArea = request.loc
    return {"data": detectArea, "msg": "已添加！", "code": "1"}


@nia_app.get('/area')
async def cv_area(request: Request):
    global detectArea
    # url = "rtmp://liteavapp.qcloud.com/live/liteavdemoplayerstreamid"
    url = "rtmp://8.130.142.19:1935/hls/test"
    loc = detectArea
    if not loc:
        loc = [0, 0, 0, 0, 0, 0, 0, 0]
    session = SessionLocal()
    camera = session.query(Camera).filter(Camera.stream_address == str(url)).first()
    yl = yoloDetectArea(url, loc)  # 使用loc数组创建yoloDetectArea实例
    yl.event_location = camera.camera_name
    session.close()
    thread = threading.Thread(target=run, args=(yl, url))
    thread.start()
    return StreamingResponse(detect_iter(yl, alertType='area', request=request),
                             media_type='multipart/x-mixed-replace; boundary=frame')


# 摔倒检测
@nia_app.get('/fall')
async def cv_fall(request: Request):
    url = "rtmp://liteavapp.qcloud.com/live/liteavdemoplayerstreamid"
    url = "rtmp://8.130.142.19:1935/hls/test"
    session = SessionLocal()
    camera = session.query(Camera).filter(Camera.stream_address == str(url)).first()
    yl = yoloDetectFall(url)
    yl.event_loc = camera.camera_name
    session.close()
    thread = threading.Thread(target=run, args=(yl, url))
    thread.start()
    return StreamingResponse(detect_iter(yl, alertType='fall', request=request),
                             media_type='multipart/x-mixed-replace; boundary=frame')


# 未处理的视频流
@nia_app.get('/video')
async def video_stream(request: Request):
    url = "rtmp://liteavapp.qcloud.com/live/liteavdemoplayerstreamid"
    container = av.open(url, mode='r')
    stream = container.streams.video[0]

    async def video_stream_iter():
        try:
            for frame in container.decode(stream):
                if await request.is_disconnected():
                    logging.info("Client disconnected")
                    break

                img = frame.to_ndarray(format='bgr24')
                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
                result, encimg = cv2.imencode('.jpg', img, encode_param)
                if result:
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + encimg.tobytes() + b'\r\n')
                await asyncio.sleep(0.01)  # 控制帧率，防止占用过多CPU资源
        except asyncio.CancelledError:
            logging.info("Request cancelled by client")
        except socket.error as e:
            logging.error(f"Socket error in video_stream_iter: {e}")
        finally:
            container.close()
            logging.info("Video container closed")

    return StreamingResponse(video_stream_iter(),
                             media_type='multipart/x-mixed-replace; boundary=frame')
