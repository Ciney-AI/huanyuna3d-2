import os
import bpy
from mathutils import Vector
import math
import sys

def adjust_postion(control_axes, frame_start, frame_end):
    mini_z = 0
    for frame in range(int(frame_start), int(frame_end) + 1):
        bpy.context.scene.frame_set(frame)
        print(frame)
        for child in control_axes.children_recursive:
            if child.type == 'MESH':
                bbox = [child.matrix_world @ Vector(corner)
                        for corner in child.bound_box]

                # 找到边界框的最低点的 Z 值
                min_z = min([v.z for v in bbox])
                mini_z = min(mini_z, min_z)

    # 将位移应用到Control Axes
    if mini_z < 0:
        control_axes.location.z -= mini_z
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=False)

def convert_glb_to_usdz(glb_file_path, glb_new_file):
    bpy.ops.wm.read_homefile(use_empty=True)
    print(f"import glb model: {glb_file_path}")
    bpy.ops.import_scene.gltf(filepath=glb_file_path)

    # 找出最顶层物体
    obj = bpy.context.object
    while obj.parent is not None:
        obj = obj.parent

    # 添加一个坐标控制模型的旋转和位移
    bpy.ops.object.empty_add(type='PLAIN_AXES', location=(0.0, 0.0, 0.0))
    control_axes = bpy.context.active_object
    control_axes.rename('Control Axes')
    obj.select_set(True)
    bpy.context.view_layer.objects.active = control_axes
    bpy.ops.object.parent_set(type='OBJECT', keep_transform=True)

    if len(bpy.data.actions) > 0:
        frame_start = bpy.data.actions[0].frame_range[0]
        frame_end = bpy.data.actions[0].frame_range[1]
    else:
        frame_start = 1
        frame_end = 1

    adjust_postion(control_axes, frame_start, frame_end)
    control_axes.rotation_mode = 'XYZ'
    control_axes.rotation_euler = (math.radians(-90), 0, 0)
    bpy.context.scene.frame_start = int(frame_start)
    bpy.context.scene.frame_end = int(frame_end)

    # 添加一个坐标保持与现有流程一致
    bpy.ops.object.empty_add(
        type='PLAIN_AXES', location=(0.0, 0.0, 0.0))
    render_axes = bpy.context.active_object
    render_axes.rename('Render Axes')
    control_axes.select_set(True)
    bpy.context.view_layer.objects.active = render_axes
    bpy.ops.object.parent_set(type='OBJECT', keep_transform=True)
    render_axes.rotation_mode = 'XYZ'
    render_axes.rotation_euler = (math.radians(90), 0, 0)

    # 导出usdz模型
    # bpy.ops.wm.usd_export(filepath=usdz_file_path, export_animation=True,
    #     export_materials=True, export_textures=True, convert_orientation=True,
    #     export_global_forward_selection='NEGATIVE_Z', export_global_up_selection='Y',
    #     root_prim_path='')

    # 导出为 GLB，包含动画
    bpy.ops.export_scene.gltf(
        filepath=glb_new_file,
        export_format='GLB',
        export_apply=True,
        export_animations=True,
        export_materials='EXPORT',
        export_texture_dir="",
        export_image_format='AUTO'
    )
    return

    
def apply_offset():

    for obj in list(bpy.data.objects):
        # 确保对象是网格类型
        if obj.type == 'MESH':

            bpy.ops.object.select_all(action='DESELECT')
            bpy.context.view_layer.objects.active = obj
            obj.select_set(True)        # 选中

            bpy.ops.object.mode_set(mode='EDIT')        # 进入编辑模式
            bpy.ops.object.mode_set(mode='OBJECT')      # 切换到对象模式以修改网格数据

            z_values = [vert.co.z for vert in obj.data.vertices]
            min_z = min(z_values)

            # 遍历顶点并修改坐标
            for vert in obj.data.vertices:
                vert.co.z -= min_z

            # 重新进入编辑模式
            bpy.ops.object.mode_set(mode='EDIT')
            bpy.ops.object.mode_set(mode='OBJECT')

            print("apply offset finished")
    return 


if __name__ == '__main__':
    
    # python3 /hunyuan3d-2/model_glb_re.py -- /code-hunyuan/output/demo.glb /code-hunyuan/output/demo-new.glb

    # Blender 会把 -- 之前的参数吃掉，我们从 '--' 后开始解析
    args = sys.argv

    if "--" in args:
        idx = args.index("--")
        script_args = args[idx + 1:]  # 获取 -- 后的所有参数
    else:
        script_args = []

    # 示例：读取输入输出参数
    if len(script_args) >= 2:
        glb_file = script_args[0]
        glb_new_file = script_args[1]
        print(f"Input: {glb_file} Output: {glb_new_file}")

        convert_glb_to_usdz(glb_file, glb_new_file)

        print('OK')

    else:
        print("No input or output")
