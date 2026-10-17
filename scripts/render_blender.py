import bpy 
import torch 
import smplx 
import numpy as np


def clear_material(material):
    if material.node_tree:
        material.node_tree.links.clear()
        material.node_tree.nodes.clear()

def colored_material(r, g, b, a=1, roughness=0.127451):
    materials = bpy.data.materials
    material = materials.new(name="body")
    clear_material(material)
    nodes = material.node_tree.nodes
    links = material.node_tree.links

    output = nodes.new(type='ShaderNodeOutputMaterial')
    diffuse = nodes.new(type='ShaderNodeBsdfDiffuse')

 
    diffuse.inputs["Color"].default_value = (r, g, b, a)
    diffuse.inputs["Roughness"].default_value = roughness
    links.new(diffuse.outputs['BSDF'], output.inputs['Surface'])
    return material


bm = smplx.create(gender='male', model_type='smplx', model_path='data/body_models', flat_hand_mean=True)

pose_1 = torch.zeros(1, 21*3)
pose_2 = torch.zeros(1, 21*3)

# left right hip 
pose_2[:, 2] = (38/180.0) * np.pi
pose_2[:, 5] = -(38/180.0) * np.pi

# left right shoulder 
pose_2[:, 47] = (20/180.0) * np.pi
pose_2[:, 48] = -(50/180.0) * np.pi
pose_2[:, 50] = -(20/180.0) * np.pi
pose_2[:, 51] = -(50/180.0) * np.pi



betas = torch.zeros(1,10) 


output_1 = bm(betas=betas)
output_2 = bm(body_pose=pose_2, betas=betas) 
pelvis_loc = output_1.joints[0,0].detach().cpu().numpy()


bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False)

 

# adjust the background color
bg = bpy.context.scene.world.node_tree.nodes['Background']
# orange background
bg.inputs[0].default_value = (1.0, 0.6, 0.2, 1.0)  
bg.inputs[1].default_value = 1.0 


# render in blender
mesh_1 = bpy.data.meshes.new("mesh_1")
obj_1 = bpy.data.objects.new("obj_1", mesh_1)
obj_1.active_material = colored_material(0.3, 0.8, 0.3)
bpy.context.collection.objects.link(obj_1)
verts_1 = output_1.vertices[0].detach().cpu().numpy()
faces_1 = bm.faces
 

mesh_1.from_pydata(verts_1, [], faces_1)
mesh_1.update()

mesh_2 = bpy.data.meshes.new("mesh_2")
obj_2 = bpy.data.objects.new("obj_2", mesh_2)
bpy.context.collection.objects.link(obj_2)
obj_2.active_material = colored_material(0.3, 0.8, 0.3)
# obj_2.active_material = colored_material(0.8, 0.3, 0.3)
verts_2 = output_2.vertices[0].detach().cpu().numpy()
faces_2 = bm.faces
mesh_2.from_pydata(verts_2, [], faces_2)
mesh_2.update()
 
# add sun 
sun_data = bpy.data.lights.new(name="Sun", type='SUN')
sun_object = bpy.data.objects.new(name="Sun", object_data=sun_data)
bpy.context.collection.objects.link(sun_object)
sun_object.location = (0.0, 0.0, 10.0)
# sun_object.rotation_euler = (0.7854, 0.0, 0.7854)
sun_object.rotation_euler = (0.7854, 0.0, 0)

# place camera
camera_data = bpy.data.cameras.new(name='Camera')
camera_object = bpy.data.objects.new('Camera', camera_data)
bpy.context.collection.objects.link(camera_object)
bpy.context.scene.camera = camera_object
 

camera_data.lens = 35
camera_object.location = (0.0, 0.0, 4.0) + pelvis_loc
camera_object.rotation_euler = (0.0, 0.0, 0.0)

 
bpy.ops.mesh.primitive_torus_add(
    major_radius=1.08,
    minor_radius=0.005,
    major_segments=1000,
    minor_segments=24,
    location=pelvis_loc
)
ring = bpy.context.object
bpy.ops.object.shade_smooth()

# Create a new material with nodes
mat = bpy.data.materials.new(name="RingMaterial")
ring.data.materials.append(mat)

# The material already has nodes by default; get the Principled BSDF
bsdf = mat.node_tree.nodes.get("Principled BSDF")
bsdf.inputs["Metallic"].default_value = 1.0
bsdf.inputs["Roughness"].default_value = 0.2
bsdf.inputs["Base Color"].default_value = (0.00, 0.00, 0.00, 1) 
 
# do the rendering
bpy.context.scene.render.filepath = '/home/enes/Desktop/codes/body-model-visualizer/smplx_render.png'

# resolution
bpy.context.scene.render.resolution_x = 1920
bpy.context.scene.render.resolution_y = 1080
bpy.context.scene.render.resolution_percentage = 100

bpy.ops.render.render(write_still=True)
 