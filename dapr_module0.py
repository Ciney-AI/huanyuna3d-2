from fastapi import Body, FastAPI, status
from dapr.ext.fastapi import DaprApp
import uvicorn
from dapr.clients import DaprClient
import json
from pydantic import BaseModel
import asyncio
import os
import time
import logging
import signal
import subprocess

logger = logging.getLogger()
logger.setLevel(logging.INFO)
fh = logging.FileHandler(filename='dapr_module0.log')
logger.addHandler(fh)   # 将日志输出至文件

from common_util.alg_util import AlgUtilIO

CLUSTER_NAME = str(os.environ['CLUSTER_NAME'])
COMPONENT_SUB = str(os.environ['COMPONENT_SUB'])
COMPONENT_PUB = str(os.environ['COMPONENT_PUB'])
COMPONENT_PUB_TOPIC = str(os.environ['COMPONENT_PUB_TOPIC'])
IMG_TO_3D_BOOT_TOPIC = str(os.environ['IMG_TO_3D_BOOT_TOPIC'])      # ${CLUSTER_NAME}-proc-img-to-3d

exit_count = 0
is_running = False

app = FastAPI()
dapr_app = DaprApp(app)
client = DaprClient()

def tryShutdown():
    global client
    for i in range(8):
        dp_resp = ''
        try:
            dp_resp = client.shutdown()
            if len(dp_resp.headers) != 0:   # shutdown接口调用失败
                print(f'error dapr client shutdown response: {dp_resp.headers}', flush=True)
                logger.error(f'error dapr client shutdown response: {dp_resp.headers}')
            else:
                print(f'dapr client shutdown succ', flush=True)
                break
        except Exception as e:
            print(f'shutdown error????\n--e {e}\n--resp {dp_resp}', flush=True)
            logger.error(f'shutdown error????\n--e {e}\n--resp {dp_resp}')
            time.sleep(2) 
    os.kill(os.getpid(), signal.SIGTERM)    # sys.exit(0)
    return

async def on_timer():
    global is_running, exit_count
    time_out = int(os.environ['MODULE_IDLE_TIMEOUT'])

    while True:
        if is_running == False:
            exit_count += 1
            if exit_count % 20 == 0:
                print(f'no task is running, exit_count: {exit_count} time_out {time_out}', flush=True)

        if exit_count >= time_out:
            print(f'no task is running, exit_count: {exit_count} time_out {time_out}', flush=True)
            print('no task is received, exit!', flush=True)
            tryShutdown()
        await asyncio.sleep(1)
    return

@app.on_event("startup")
async def app_start():
    print(f'hohohhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhh')
    asyncio.create_task(on_timer())

def tryPub(pubsub_name, topic_name, data_string, data_content_type) -> bool:
    global client
    resp = None
    for i in range(8):
        try:
            resp = client.publish_event(pubsub_name=pubsub_name, topic_name=topic_name,
                data=data_string, data_content_type=data_content_type)
        except Exception as e:
            print(f'pub error:\n--e {e}\n--resp {resp}', flush=True)
            time.sleep(2)
        if resp != None:
            break

    if resp == None:
        print('try publish module out msg error!', flush=True)
        logger.error('try publish module out msg error!')
        tryShutdown()
        return False
    return True


@dapr_app.subscribe(pubsub=COMPONENT_SUB, topic=IMG_TO_3D_BOOT_TOPIC)
async def module_subscriber(event_body = Body()):
    
    print(f'[MODULE] Subscriber received: message: {json.dumps(event_body)}', flush=True)
    event_data = event_body["data"]

    global is_running, exit_count

    if is_running:
        print('[MODULE11] bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbusy')
        return {"error": status.HTTP_406_NOT_ACCEPTABLE, "error_description": "busy! module is running"}, status.HTTP_406_NOT_ACCEPTABLE

    is_running = True
    exit_count = 0

    await mainProc(event_data)

    is_running = False
    exit_count = 0
    return {'error': status.HTTP_200_OK, "error_description": "everything is OK!"}, status.HTTP_200_OK

async def mainProc(module_in: dict):
    alg_io = None

    try:
        if len(module_in) == 0:
            return

        # 正式模式，直接来
        taskId = module_in['context']['taskId']
        context_param = module_in['context']['param']
        projectId = context_param['projectId']
        buildingId = module_in['context']['buildingId']
        s3_dir = f'_module_out/task-2To3/{projectId}/{buildingId}'               # 输出目录

        taskStamp = module_in['context']['taskStamp']   # 流程开始处理的时间
        generation_type = module_in['context']['generationType']
        
        alg_io = AlgUtilIO(taskId)
        if not alg_io.taskTick("start", taskStamp):
            print(f'task start tick error!', flush=True)
            return
        print(f'receive message, time {int(time.time() * 1000)}')

        # 下载数据文件
        assert generation_type == 1, f'GenTypeErr:{generation_type}'

        img_file = alg_io.download(context_param['img_file'], s3_dir, 1)
        
        glb_out = f'{s3_dir}/glb_model.glb'
        subprocess.run(f'python3 /src/predict_copy.py -- {img_file} {glb_out}', shell=True)
        # time.sleep(300)

        # 上传文件
        glb_file_url = alg_io.upload(glb_out, s3_dir, 1)

        # 生成对应的发布数据
        module_out = {
            "taskId": taskId,
            "taskStamp": taskStamp,
            "moduleName": "mdu2",
            "timeFinished": int(time.time() *1000),
            "depends": [],
            "to": f"{CLUSTER_NAME}-end",
            "dispatch": 0,
            "ttlInSeconds": 0,
            "isFinished": True,
            'result': {
                "filePaths": {
                    "model_glb": glb_file_url,
                }
            }
        }

        if not alg_io.taskTick("end", taskStamp):
            print(f'task end tick error!', flush=True)
            return
        
        ret = tryPub(COMPONENT_PUB, COMPONENT_PUB_TOPIC, json.dumps(module_out), 'application/json')
        alg_io.reportError(ret, "PUB_MESSAGE_FAILED")
        print(json.dumps(module_out), flush=True)  # Print the request

    except Exception as e:
        print(f'catch an error in dapr_app:\n--e {e}', flush=True)
        logger.error(f'catch an error in dapr_app:\n--e {e}')
        err_msg = ''
        if "CUDA out of memory" in str(e):
            err_msg = "CUDA_MEM_LACK"
        else:
            err_msg = str(e)
            err_msg = err_msg[0:min(21, len(err_msg))]

        if alg_io != None:
            alg_io.reportError(False, f"{err_msg}")
        tryShutdown()

    return

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)
