
import requests
import os


class AlgUtilIO:
    # 任务id，上传s3时的远端保存路径，本地下载时的保存路径
    def __init__(self, taskId) -> None:
        self.task_id = taskId
        self.http_api = str(os.environ['HTTP_BACKEND_API'])


    def taskTick(self, step:str, timeStamp:str) -> bool:
        try:
            # 任务打点
            headers = {"content-type": "application/json"}
            payload = {"step": step, "stamp": timeStamp}
            # json形式，参数用json
            result = requests.post(
                url=f'{self.http_api}/workflow/task/{self.task_id}/tick',
                params=payload, headers=headers
            )
            result_json = result.json()
            print(f'Post task {self.task_id} tick, step {step}, return {result_json}', flush=True)
            
            if not result_json['hasTask'] or result_json['status'] > 10:
                return False
            
        except Exception as e:
            print(f'Post task {self.task_id} tick failed: {e}', flush=True)
            return False
        
        return True


    # 算法库的文件均为Private(1)
    # 上传成功则返回上传后的路径，上传失败则返回None
    def upload(self, local_path:str, s3_dir:str, modifier = 1):
        response = None

        # 非调试模式，需使用后端接口获取对应的url和上传文件。
        # step-1: 生成一个文件的上传路径
        req = {
            "folder": s3_dir,
            "ext": local_path.split('.')[-1],
            "modifier": modifier,
        }
        result = requests.post(
            url=f'{self.http_api}/files/touch',
            params=req
        )
        result_json = result.json()
        if result.status_code != 200 or 'error' in result_json:   # failed
            print(f'[IO] error when touch file, msg: {result}')
            self.reportError(False, f'touch file failed')
            return response
        
        # step-2: 上传对应的文件
        # result_json      # {"fileIndexId", "filePath", "uploadUrl", "mime", "timeoutMills"}
        headers_put = {'Content-Type': result_json['mime']}
        result_put = requests.put(url=result_json['uploadUrl'], headers=headers_put, data=open(local_path, 'rb'))
        if result_put.status_code != 200:   # failed
            print(f'[IO] error when put file, msg: {result_put}')
            self.reportError(False, f'put file failed')
            return response

        # step-3: 上传之后，该文件的路径信息需通过dapr对外发布出去（最好是存redis中）
        return result_json['filePath']
    
    # modifier=1为私有存储桶
    # 下载成功则返回本地保存的路径，下载失败则返回None
    def download(self, remote_path: str, local_dir:str, modifier = 1):
        response = None

        # 生成一个可访问的下载地址
        result = requests.get(
            url=f'{self.http_api}/files/pre-sign',
            params={
                "filePath": remote_path,
                "modifier": modifier,
            }
        )
        result_json = result.json()
        if result.status_code != 200 or 'error' in result_json:   # failed
            print(f'[IO] error when pre-sign file, msg: {result}')
            self.reportError(False, f'pre-sign file failed')
            return response
        
        os.makedirs(local_dir, exist_ok=True)

        base_name = remote_path.split('/')[-1]
        r = requests.get(result_json['url'])
        open(f'{local_dir}/{base_name}', 'wb').write(r.content)

        # step-3: 上传之后，该文件的路径信息需通过dapr对外发布出去（最好是存redis中）
        return f'{local_dir}/{base_name}'
    
    
    # 向后端上报错误
    def reportError(self, succ: bool, error_code: str):

        if succ == False:
            headers = {"content-type":"application/json"}
            payload = { "taskId": self.task_id, "errorCode": error_code }
            # json形式，参数用json
            result = requests.post(
                url=f'{self.http_api}/workflow/report-error',
                json=payload, headers=headers
            )
            print(f'task {self.task_id} raise an error, msg: {error_code}')
            # raise ValueError      # 不在崩溃，而是正常退出
        return


    # 查询模型参数
    def queryActorModels(self, id: list):

        payload = {"id": id }
        result = requests.get(url=f'{self.http_api}/workflow/data/actor-models', params = payload)
        result_json = result.json()
        if result.status_code != 200 or 'error' in result_json:   # failed
            print(f'[IO] error when query actor-models, msg: {result}')
            self.reportError(False, f'query actor models failed')
            return result_json
        
        return result_json
