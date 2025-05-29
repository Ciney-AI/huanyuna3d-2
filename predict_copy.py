import os
weight_mount = os.environ['WEIGHT_MOUNT']
# weight_mount = '/s3-data'       # 这个与k8s配置的脚本有关，s3的挂载目录

# os.environ['HY3DGEN_MODELS'] = '/hunyuan3d/model/snapshots'
os.environ['HY3DGEN_MODELS'] = f'{weight_mount}/snapshots'
U2NET_PATH = f'{weight_mount}/u2net/'        # U2NET_PATH = os.path.join(CHECKPOINTS_PATH, ".u2net/")

import shutil
import subprocess
import time
import traceback
import sys

from PIL import Image
from torch import cuda, Generator

from hy3dgen.rembg import BackgroundRemover
from hy3dgen.shapegen import FaceReducer, FloaterRemover, DegenerateFaceRemover, MeshlibCleaner, Hunyuan3DDiTFlowMatchingPipeline
from hy3dgen.shapegen.models.autoencoders import SurfaceExtractors
from hy3dgen.shapegen.utils import logger
from hy3dgen.texgen import Hunyuan3DPaintPipeline

HUNYUAN3D_REPO = "andreca/hunyuan3d-2xet"
HUNYUAN3D_DIT_MODEL = "hunyuan3d-dit-v2-0-turbo"
HUNYUAN3D_PAINT_MODEL = "hunyuan3d-paint-v2-0"

class Predictor():
    def __init__(self):
        try:
            start = time.time()
            logger.info("Setup started")
            os.environ["OMP_NUM_THREADS"] = "1"
            os.environ['U2NET_HOME'] = U2NET_PATH

            mc_algo = 'dmc'
            use_delight = False
            use_super = False
            
            # download_if_not_exists(U2NET_URL, U2NET_PATH)
            self.i23d_worker = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
                HUNYUAN3D_REPO,
                subfolder=HUNYUAN3D_DIT_MODEL
            )
            self.i23d_worker.enable_flashvdm(mc_algo=mc_algo)
            self.i23d_worker.vae.surface_extractor = SurfaceExtractors[mc_algo]()
            self.texgen_worker = Hunyuan3DPaintPipeline.from_pretrained(
                HUNYUAN3D_REPO, 
                subfolder=HUNYUAN3D_PAINT_MODEL, 
                use_delight=use_delight, 
                use_super=use_super
            )
            self.floater_remove_worker = FloaterRemover()
            self.degenerate_face_remove_worker = DegenerateFaceRemover()
            self.face_reduce_worker = FaceReducer()
            self.rmbg_worker = BackgroundRemover()
            self.cleaner_worker = MeshlibCleaner()
            duration = time.time() - start
            logger.info(f"Setup took: {duration:.2f}s")
        except Exception as e:
            logger.error(f"Setup failed: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def _cleanup_gpu_memory(self):
        if cuda.is_available():
            cuda.empty_cache()
            cuda.ipc_collect()

    def _log_analytics_event(self, event_name, params=None):
        pass

    def predict(
        self,
        image:str,
        out_glb:str,
        steps=50,
        guidance_scale=5.5,
        max_facenum=40000,
        num_chunks=200000,
        seed=1234,
        octree_resolution=512,
        remove_background=True,
    ):
        start_time = time.time()
        
        self._log_analytics_event("predict_started", {
            "steps": steps,
            "guidance_scale": guidance_scale,
            "max_facenum": max_facenum,
            "num_chunks": num_chunks,
            "seed": seed,
            "octree_resolution": octree_resolution,
            "remove_background": remove_background
        })

        if os.path.exists("output"):
            shutil.rmtree("output")
        
        os.makedirs("output", exist_ok=True)

        self._cleanup_gpu_memory()

        generator = Generator()
        generator = generator.manual_seed(seed)

        if image is not None:
            input_image = Image.open(str(image))
            if remove_background or input_image.mode == "RGB":
                input_image = self.rmbg_worker(input_image.convert('RGB'))
                self._cleanup_gpu_memory()
        else:
            self._log_analytics_event("predict_error", {"error": "no_image_provided"})
            raise ValueError("Image must be provided")

        input_image.save("output/input.png")

        try:
            mesh = self.i23d_worker(
                image=input_image,
                num_inference_steps=steps,
                guidance_scale=guidance_scale,
                generator=generator,
                octree_resolution=octree_resolution,
                num_chunks=num_chunks
            )[0]
            self._cleanup_gpu_memory()

            mesh = self.floater_remove_worker(mesh)
            mesh = self.degenerate_face_remove_worker(mesh)
            mesh = self.cleaner_worker(mesh)
            mesh = self.face_reduce_worker(mesh, max_facenum=max_facenum)
            self._cleanup_gpu_memory()
            
            mesh = self.texgen_worker(mesh, input_image)
            self._cleanup_gpu_memory()
            
            mesh.export(out_glb, include_normals=True)

            if not os.path.isfile(out_glb):
                self._log_analytics_event("predict_error", {"error": "mesh_export_failed"})
                raise RuntimeError(f"Failed to generate mesh file at {out_glb}")

            duration = time.time() - start_time
            self._log_analytics_event("predict_completed", {
                "duration": duration,
                "final_face_count": len(mesh.faces),
                "success": True
            })

            return out_glb
        except Exception as e:
            logger.error(f"Predict failed: {str(e)}")
            logger.error(traceback.format_exc())
            self._log_analytics_event("predict_error", {
                "error": str(e),
                "error_type": type(e).__name__
            })
            raise


if __name__ == '__main__':
    # python3 /src/predict_copy.py -- /hunyuan3d/asset/hoho.jpg /hunyuan3d/output/demo_new.glb

    # 把 -- 之前的参数吃掉，我们从 '--' 后开始解析
    args = sys.argv

    if "--" in args:
        idx = args.index("--")
        script_args = args[idx + 1:]  # 获取 -- 后的所有参数
    else:
        script_args = []

    if len(script_args) >= 2:
        img_file = script_args[0]
        glb_file = script_args[1]

        print(f"start to process image: {img_file}")

        # 开始处理
        pred = Predictor()
        output_dir = os.path.dirname(glb_file)
        base_name = os.path.basename(glb_file)
        os.makedirs(output_dir, exist_ok=True)

        # glb_tmp_file = f'{output_dir}/temp_{base_name}'
        # pred.predict(img_file, glb_tmp_file)

        # print(f"start to refine model: {glb_tmp_file}")
        # subprocess.run(f'python3 /src/model_glb_re.py -- {glb_tmp_file} {glb_file}', shell=True)

    
        pred.predict(img_file, glb_file)

        print('OK')
    else:
        print("No input or output")