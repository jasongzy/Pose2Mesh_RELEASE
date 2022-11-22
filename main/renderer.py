import os
os.environ['PYOPENGL_PLATFORM'] = 'egl'# 'osmesa'
import torch
from torchvision.utils import make_grid
import numpy as np
import pyrender
import trimesh
import math

from pyrender.constants import RenderFlags


class MyCamera(pyrender.Camera):
    def __init__(self,
                 znear=pyrender.camera.DEFAULT_Z_NEAR,
                 zfar=1000.,
                 name=None):
        super(MyCamera, self).__init__(
            znear=znear,
            zfar=zfar,
            name=name,
        )
    
    def get_projection_matrix(self, width=None, height=None):
        P = np.eye(4)
        return P

class Renderer:
    def __init__(self, faces, resolution=(224,224), orig_img=False, wireframe=False):
        self.resolution = resolution

        self.faces = faces
        self.orig_img = orig_img
        self.wireframe = wireframe
        self.renderer = pyrender.OffscreenRenderer(
            viewport_width=self.resolution[0],
            viewport_height=self.resolution[1],
            point_size=1.0
        )

        # set the scene
        self.scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0], ambient_light=(0.3, 0.3, 0.3))

        # light = pyrender.PointLight(color=[1.0, 1.0, 1.0], intensity=0.8)
        light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=0.8)

        light_pose = np.eye(4)
        light_pose[:3, 3] = [0, -1, 1]
        self.scene.add(light, pose=light_pose)

        light_pose[:3, 3] = [0, 1, 1]
        self.scene.add(light, pose=light_pose)

        light_pose[:3, 3] = [1, 1, 2]
        self.scene.add(light, pose=light_pose)

    def render(self, img, verts, color=(1.0, 1.0, 0.9)):
        Rx = trimesh.transformations.rotation_matrix(math.radians(180), [1, 0, 0])

        material = pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.0,
            alphaMode='OPAQUE',
            baseColorFactor=color
        )

        mesh_nodes = {}
        for side in ['right', 'left']:
            mesh = trimesh.Trimesh(vertices=verts[side], faces=self.faces[side], process=False)
            mesh.apply_transform(Rx)
            mesh = pyrender.Mesh.from_trimesh(mesh, material=material, smooth=False)
            mesh_node = self.scene.add(mesh)

            mesh_nodes[side] = mesh_node

        camera = MyCamera()
        camera_pose = np.eye(4)
        cam_node = self.scene.add(camera, pose=camera_pose)

        if self.wireframe:
            render_flags = RenderFlags.RGBA | RenderFlags.ALL_WIREFRAME
        else:
            render_flags = RenderFlags.RGBA

        rgb, depth = self.renderer.render(self.scene, flags=render_flags)
        valid_mask = (depth > 0)[:, :, np.newaxis]
        output_img = rgb[:, :, :3] * valid_mask + (1 - valid_mask) * img
        image = output_img.astype(np.uint8)

        self.scene.remove_node(mesh_nodes['right'])
        self.scene.remove_node(mesh_nodes['left'])
        self.scene.remove_node(cam_node)

        return image