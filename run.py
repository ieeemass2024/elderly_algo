import uvicorn
from fastapi import FastAPI, Body, WebSocket, WebSocketDisconnect
from fastapi_socketio import SocketManager
from fastapi.middleware.cors import CORSMiddleware
import logging

from app_nia import nia_app
# from app_voice import voice_app
from app_info import info_app

app = FastAPI(
    title='FastAPI后端接口文档',
    description='使用Fastapi构建智慧养老系统的算法接口',
    version='1.0.0',
    docs_url='/api/v2/docs',
    redoc_url='/api/v2/redocs',
)

origins = [
    "http://localhost",
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    # allow_methods=["GET", "POST", "PUT", "DELETE", "WebSocket"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(nia_app, prefix='/api/v1/cv', tags=['cv应用'])
# app.include_router(voice_app, prefix='/api/v1/voice', tags=['voice应用'])
app.include_router(info_app, prefix='/api/v1/info', tags=['info应用'])

# fastapi_socketio挂载路径
socket_manager = SocketManager(app=app)

# 存储活动的 WebSocket 连接
active_connections = []


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    active_connections.append(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            logging.info(f"Message received: {data}")
    except WebSocketDisconnect:
        active_connections.remove(websocket)
        logging.info("Client disconnected")
    except Exception as e:
        logging.error(f"Connection error: {e}")
    finally:
        await websocket.close()


@app.post("/api/v1/alert",
          summary="报警信息推送接口",
          description="调用此接口即可使用Websocket向前端推送消息",
          tags=['socket应用'])
async def alert(message=Body(None)):
    # 向所有活动的 WebSocket 连接推送消息
    for connection in active_connections:
        await connection.send_json(message)
    return "预警信息发送成功"


if __name__ == '__main__':
    uvicorn.run('run:app', port=8090, reload=True, workers=1)
