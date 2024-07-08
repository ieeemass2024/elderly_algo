import os
import utils.file_oss_utils.oss_loader as oss

from fastapi import APIRouter, UploadFile, File, Form
from utils.mysql_utils.database import engine, Base

info_app = APIRouter()

Base.metadata.create_all(bind=engine)

@info_app.get("/")
async def nia():
    return "hello"

@info_app.post("/upload")
async def upload_file(file: UploadFile = File(...), id: str = Form(...), file_type: str = Form(...)):
    # 要上传图片所在的文件夹
    local_dir_path = "utils/file_oss_utils/img/avatar/"
    contents = await file.read()
    with open(local_dir_path + str(id) + ".jpg", "wb") as f:
        f.write(contents)

    # 图片对应的path，对应上传到的文件夹
    elderly_path = "/avatar/" + str(file_type)
    img_path = \
        'https://ai-care-system.oss-cn-beijing.aliyuncs.com/' \
        'resources/smart_elderly_care/cv_file' \
        + elderly_path + "/" + str(id) + '.jpg'
    # 最终路径为 local_dir_path +
    oss.upload_file_to_oss(local_dir_path=local_dir_path, elderly_path=elderly_path)

    # 清空本地
    for root, dirs, files in os.walk(local_dir_path):
        # 删除每个文件
        for file in files:
            file_path = os.path.join(root, file)
            os.remove(file_path)

    return {"data": img_path, "msg": "上传成功！", "code": "1"}
