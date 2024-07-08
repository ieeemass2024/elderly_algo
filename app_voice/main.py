import os
from fastapi import Request, APIRouter, UploadFile, Body

voice_app = APIRouter()


@voice_app.get("/chat", summary="语音接口", description="调用此接口可以对语音输入文件进行处理")
async def process_voice():
    return "ok"


@voice_app.post("/chat", summary="语音接口", description="调用此接口可以对语音输入文件进行处理")
async def process_voice(file: UploadFile):
    # 可以在这里进行文件的处理逻辑，比如保存到服务器或进行语音识别等操作
    # file 是上传的文件对象，可以根据需要进行处理

    # 从上传的文件对象中读取内容，并保存到服务器上
    contents = await file.read()
    directory = "./wav_file/"
    # os.makedirs(directory, exist_ok=True)
    with open(directory + 'test.silk', "wb") as f:
        f.write(contents)

    pilk.silk_to_wav(silk=directory + 'test.silk', wav=directory + 'test.wav')

    # 返回相应的处理结果
    return {"message": "录音文件已接收并处理成功"}

import pilk

@voice_app.post("/pilk")
async def my_pilk(data=Body(None)):
    directory = "./wav_file/"
    input_file = data["input_file"]
    output_file = directory + "test.wav"

    print(input_file)

    print(output_file)

    pilk.silk_to_wav(silk=input_file, wav=output_file)

    # 返回相应的处理结果
    return {"message": "okokokok"}
