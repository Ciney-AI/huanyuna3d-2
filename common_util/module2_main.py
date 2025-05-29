import bpy
import subprocess
from pathlib import Path
import os
import imageio
import PIL
import numpy as np

def update_video_texture(material_name: str, new_video_path: str, frame_num: int):
    import bpy
    if material_name not in bpy.data.materials:
        print(f"Material {material_name} not found!")
        return

    material = bpy.data.materials[material_name]
    if not material.use_nodes:
        print("Material doesn't use nodes.")
        return

    nodes = material.node_tree.nodes
    if "Projection_Texture" not in nodes:
        print("Node 'Projection_Texture' not found.")
        return

    # 加载新的视频作为 image
    new_video = bpy.data.images.load(new_video_path)
    new_video.source = 'MOVIE'

    # 替换材质中 Projection_Texture 的 image
    nodes["Projection_Texture"].image = new_video
    nodes["Projection_Texture"].image_user.frame_start = 1
    nodes["Projection_Texture"].image_user.frame_duration = frame_num
    nodes["Projection_Texture"].image_user.use_auto_refresh = True

    print(f"Replaced video in material '{material_name}'")
    

def module_main(blend_path, video_new_path, fps:int, frame_start:int, frame_count:int, nodes_dir:str, output_dir:str):

    # 安装下插件
    addon_path = os.path.join(nodes_dir, 'addon', 'easy_rb_pro.zip')
    bpy.ops.preferences.addon_install(filepath=addon_path, overwrite=True)
    bpy.ops.preferences.addon_enable(module="easy_rb_pro")

    bpy.ops.wm.open_mainfile(filepath=blend_path)           # 加载文件


    cycles = bpy.context.preferences.addons["cycles"]
    # Use GPU acceleration if available.
    cycles.preferences.compute_device_type = "CUDA"
    bpy.context.scene.cycles.device = "GPU"
    # reload the devices to update the configuration
    cycles.preferences.get_devices()
    for device in cycles.preferences.devices:
        if device.type == "CUDA":
            device.use = True


    # 这两个位置的视频文件都需要更新
    frame_count_total = bpy.context.scene.frame_end
    update_video_texture('mesh', video_new_path, frame_num=frame_count_total)
    bpy.data.node_groups['Simulon_Backplate'].nodes['Backplate_Video'].clip = bpy.data.movieclips.load(video_new_path)

    out_fr_dir = f'{output_dir}/foreground'
    os.makedirs(out_fr_dir, exist_ok=True)

    render_step = 1
    output_path_list = []
    for i in range(0, frame_count, render_step):
        # 渲染的帧的id，加了1的：1 + frame_start + i
        
        output_path = f"{out_fr_dir}/output-{1 + frame_start + i:04d}.png"
        output_path_list.append(output_path)

        bpy.context.scene.render.filepath = output_path
        # bpy.context.scene.node_tree.nodes["file_shadow"].base_path = out_shadow_dir     # 这个地方比较特殊，不能设置文件名，只能指定文件夹

        bpy.context.scene.frame_set(1 + frame_start + i)        # 帧从1开始的
        bpy.ops.render.render(write_still=True)

        # if frame_start == 0:        # 只有第0个容器允许上报进度
        #     data = {
        #         "percent": round(i / float(frame_count) * 95),      # 进度 0-95，留5%给最后的上采样
        #         "status": "pending",                                # 可选值: "pending"、"failed"、"success"
        #     }

        #     if i == 0:
        #         jpg_path_url = modal_cloud.upload_local_file(output_path, 'render_cover')
        #         data["cover"] = jpg_path_url

        #     try:
        #         response = requests.post(webhook_url, json=data)        # 不管回复
        #         print(f'post url {data} response {response.json()}')
        #     except Exception as e:
        #         print(f'msg [{taskid}] post percent failed url {webhook_url} err {e} status failed')


    # 最后将图片帧合并为视频文件，水印放在最后一个模块添加
    out_mp4_path = f"{output_dir}/output_final.mp4"

    video_writer = imageio.get_writer(out_mp4_path, fps=fps)    # mask视频
    for i, image_file in enumerate(output_path_list):
        image = PIL.Image.open(image_file)
        image = image.rotate(-90, expand=True)                  # 旋转一下

        video_writer.append_data(np.array(image))               # 写入 
    video_writer.close()

    thumbnail = f"{out_fr_dir}/output-{1 + frame_start:04d}.png"
    return out_mp4_path, thumbnail