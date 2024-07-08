# from time import sleep
#
# import cv2
# import httpx
# from fastapi import FastAPI
# from fastapi.responses import StreamingResponse
# from algorithm import integrate
# from algorithm.Dlib.face_reco_from_camera_ot import Face_Recognizer
#
# app = FastAPI()
#
#
# @app.get("/")
# async def root():
#     return {"message": "Hello World"}
#
#
# @app.get("/hello/{name}")
# async def say_hello(name: str):
#     return {"message": f"Hello {name}"}
#
#
# # video_url请求参数设置视频流地址
#
# async def detect_iter(Face_Recognizer_con):
#     while (True):
#         img1 = Face_Recognizer_con.detect_img
#         if Face_Recognizer_con.hasUnknown:
#             async with httpx.AsyncClient() as client:
#                 url = "http://43.143.247.127:8090/api/v1/alert"
#                 # 将需要发送的消息字典作为payload
#                 payload = {"message": "这是一条预警消息", "alert_type": "检测到陌生人"}
#                 # json格式发送
#                 await client.post(url, json=payload)
#             Face_Recognizer_con.hasUnknown = 0
#         cv2.imshow("camera", img1)
#         frame = cv2.imencode('.jpg', img1)[1].tobytes()
#         yield b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n'
#
#
# @app.get('/cv')
# async def video_feed():
#     # video_url = 'rtmp://43.143.247.127:1935/hls/lym'
#     video_url = 0
#     Face_Recognizer_con = Face_Recognizer()
#     integrate.detect(video_url, Face_Recognizer_con)
#     return StreamingResponse(detect_iter(Face_Recognizer_con),
#                              media_type='multipart/x-mixed-replace; boundary=frame')
