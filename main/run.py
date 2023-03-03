import os
import torch
import torch.nn as nn
import torch.optim as optim
import cv2
import numpy as np
import colorsys
import json
import argparse
import pickle
import matplotlib.pyplot as plt

import __init_path
import models
from core.config import cfg
from aug_utils import j2d_processing
from coord_utils import get_bbox, process_bbox
from funcs_utils import load_checkpoint, save_obj
from graph_utils import build_coarse_graphs
from renderer import Renderer
from _mano import MANO
from smpl import SMPL


def vis_2d_keypoints(img, keypoints, threshold=0.5, alpha=1, show_keypoints=True):
    skeleton = [
        [15, 13],
        [13, 11],
        [16, 14],
        [14, 12],
        [11, 12],
        [5, 11],
        [6, 12],
        [5, 6],
        [5, 7],
        [6, 8],
        [7, 9],
        [8, 10],
        [1, 2],
        [0, 1],
        [0, 2],
        [1, 3],
        [2, 4],
        [3, 5],
        [4, 6],
        [15, 17],
        [15, 18],
        [15, 19],
        [16, 20],
        [16, 21],
        [16, 22],
        [91, 92],
        [92, 93],
        [93, 94],
        [94, 95],
        [91, 96],
        [96, 97],
        [97, 98],
        [98, 99],
        [91, 100],
        [100, 101],
        [101, 102],
        [102, 103],
        [91, 104],
        [104, 105],
        [105, 106],
        [106, 107],
        [91, 108],
        [108, 109],
        [109, 110],
        [110, 111],
        [112, 113],
        [113, 114],
        [114, 115],
        [115, 116],
        [112, 117],
        [117, 118],
        [118, 119],
        [119, 120],
        [112, 121],
        [121, 122],
        [122, 123],
        [123, 124],
        [112, 125],
        [125, 126],
        [126, 127],
        [127, 128],
        [112, 129],
        [129, 130],
        [130, 131],
        [131, 132],
    ]

    palette = np.array(
        [
            [255, 128, 0],
            [255, 153, 51],
            [255, 178, 102],
            [230, 230, 0],
            [255, 153, 255],
            [153, 204, 255],
            [255, 102, 255],
            [255, 51, 255],
            [102, 178, 255],
            [51, 153, 255],
            [255, 153, 153],
            [255, 102, 102],
            [255, 51, 51],
            [153, 255, 153],
            [102, 255, 102],
            [51, 255, 51],
            [0, 255, 0],
            [0, 0, 255],
            [255, 0, 0],
            [255, 255, 255],
        ],
        np.float64
    )

    pose_link_color = palette[
        [0, 0, 0, 0, 7, 7, 7, 9, 9, 9, 9, 9, 16, 16, 16, 16, 16, 16, 16]
        + [16, 16, 16, 16, 16, 16]
        + [0, 0, 0, 0, 4, 4, 4, 4, 8, 8, 8, 8, 12, 12, 12, 12, 16, 16, 16, 16]
        + [0, 0, 0, 0, 4, 4, 4, 4, 8, 8, 8, 8, 12, 12, 12, 12, 16, 16, 16, 16]
    ]

    kp_mask = np.copy(img)

    if show_keypoints:
        for l in range(len(keypoints)):
            p1 = keypoints[l, 0].astype(np.int32), keypoints[l, 1].astype(np.int32)
            if keypoints[l, 2] > threshold:
                cv2.circle(
                    kp_mask, p1, radius=3, color=(0.0,0.0,0.0), thickness=-1, lineType=cv2.LINE_AA)

    for l in range(len(skeleton)):
        i1 = skeleton[l][0]
        i2 = skeleton[l][1]
        p1 = keypoints[i1, 0].astype(np.int32), keypoints[i1, 1].astype(np.int32)
        p2 = keypoints[i2, 0].astype(np.int32), keypoints[i2, 1].astype(np.int32)
        if keypoints[i1, 2] > threshold and keypoints[i2, 2] > threshold:
            cv2.line(
                kp_mask, p1, p2, color=pose_link_color[l], thickness=2, lineType=cv2.LINE_AA)

    return cv2.addWeighted(img, 1.0 - alpha, kp_mask, alpha, 0)


def get_joint_setting(mesh_model):
    joint_regressor = mesh_model.joint_regressor
    joint_num = 21
    skeleton = ( (0,1), (1,2), (2,3), (3,4), (0,5), (5,6), (6,7), (7,8), (0,9), (9,10), (10,11), (11,12), (0,13), (13,14), (14,15), (15,16), (0,17), (17,18), (18,19), (19,20) )
    hori_conn = ((1, 5), (5, 9), (9, 13), (13, 17), (2, 6), (6, 10), (10, 14), (14, 18), (3, 7), (7, 11), (11, 15), (15, 19), (4, 8), (8, 12), (12, 16), (16, 20))
    graph_Adj, graph_L, graph_perm, graph_perm_reverse = build_coarse_graphs(mesh_model.face, joint_num, skeleton, hori_conn, levels=6)
    model_chk_path = './experiment/pose2mesh_manoJ_train_freihand/final.pth.tar'

    model = models.pose2mesh_net.get_model(joint_num, graph_L)
    checkpoint = load_checkpoint(load_dir=model_chk_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    return model, joint_regressor, joint_num, skeleton, graph_L, graph_perm_reverse


def optimize_cam_param(project_net, mesh_face, joint_input, crop_size):
    bbox = get_bbox(joint_input)
    bbox1 = process_bbox(bbox.copy(), aspect_ratio=1.0, scale=1.25)
    bbox2 = process_bbox(bbox.copy())
    proj_target_joint_img, trans = j2d_processing(joint_input.copy(), (crop_size, crop_size), bbox1, 0, 0, None)
    joint_img, _ = j2d_processing(joint_input.copy(), (cfg.MODEL.input_shape[1], cfg.MODEL.input_shape[0]), bbox2, 0, 0, None)

    joint_img = joint_img[:, :2]
    joint_img /= np.array([[cfg.MODEL.input_shape[1], cfg.MODEL.input_shape[0]]])
    mean, std = np.mean(joint_img, axis=0), np.std(joint_img, axis=0)
    joint_img = (joint_img.copy() - mean) / std
    joint_img = torch.Tensor(joint_img[None, :, :]).cuda()
    target_joint = torch.Tensor(proj_target_joint_img[None, :, :2]).cuda()

    # get optimization settings for projection
    criterion = nn.L1Loss()
    optimizer = optim.Adam(project_net.parameters(), lr=0.1)

    # estimate mesh, pose
    model.eval()
    pred_mesh, _ = model(joint_img)
    pred_mesh = pred_mesh[:, graph_perm_reverse[:mesh_face.max() + 1], :]
    pred_3d_joint = torch.matmul(joint_regressor, pred_mesh)

    out = {}
    # assume batch=1
    project_net.train()
    for j in range(0, 1500):
        # projection
        pred_2d_joint = project_net(pred_3d_joint.detach())

        loss = criterion(pred_2d_joint, target_joint[:, :21, :])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if j == 500:
            for param_group in optimizer.param_groups:
                param_group['lr'] = 0.05
        if j == 1000:
            for param_group in optimizer.param_groups:
                param_group['lr'] = 0.001

    out['mesh'] = pred_mesh[0].detach().cpu().numpy()
    out['cam_param'] = project_net.cam_param[0].detach().cpu().numpy()
    out['bbox'] = bbox1

    out['target'] = proj_target_joint_img
    out['pred_3d_joint'] = pred_3d_joint[0].detach().cpu().numpy()

    return out


def convert_crop_cam_to_orig_img(cam, bbox, img_width, img_height):
    x, y, w, h = bbox[:,0], bbox[:,1], bbox[:,2], bbox[:, 3]
    cx, cy, h = x + w/2, y + h/2, h
    hw, hh = img_width / 2., img_height / 2.
    sx = cam[:,0] * (1. / (img_width / h))
    sy = cam[:,0] * (1. / (img_height / h))
    tx = ((cx - hw) / hw / sx) + cam[:,1]
    ty = ((cy - hh) / hh / sy) + cam[:,2]
    orig_cam = np.stack([sx, sy, tx, ty]).T
    return orig_cam[0]


def render(joint_input, project_net, mesh_faces, img_size, color, virtual_crop_size):
    orig_width, orig_height = img_size
    orig_img = np.full((orig_height, orig_width,3), 255)

    pred_meshes = {}
    for side in ['right', 'left']:
        joint_2d = joint_input[side]
        if side == 'left':
            joint_2d[:,0] = orig_width - joint_2d[:,0]

        pred_result = optimize_cam_param(project_net, mesh_faces['right'], joint_2d, crop_size=virtual_crop_size)
        pred_mesh, pred_cam, bbox = pred_result['mesh'], pred_result['cam_param'][None, :], pred_result['bbox'][None, :]
        orig_cam = convert_crop_cam_to_orig_img(cam=pred_cam, bbox=bbox, img_width=orig_width, img_height=orig_height)

        pred_mesh[:,0] = (pred_mesh[:,0] + orig_cam[2]) * orig_cam[0]
        pred_mesh[:,1] = (pred_mesh[:,1] + orig_cam[3]) * orig_cam[1]
        pred_mesh[:,2] = -pred_mesh[:,2]
        if side == 'left':
            pred_mesh[:,0] = -pred_mesh[:,0]

        pred_meshes[side] = pred_mesh

    renderer = Renderer(mesh_faces, resolution=(orig_width, orig_height), orig_img=True, wireframe=False)
    renederd_img = renderer.render(
        orig_img,
        pred_meshes,
        color=color
    )

    return renederd_img


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Render Pose2Mesh output')
    parser.add_argument('--gpu', type=str, default='0', help='assign gpu number')
    parser.add_argument('--input_pose', type=str, default='.', help='path of input 2D pose')
    parser.add_argument('--begin_frame', type=int, default=0, help='begin frame of the video')
    parser.add_argument('--end_frame', type=int, default=0, help='end frame of the video. If begin_frame >= end_frame, then program will generate an image.')
    parser.add_argument('--fps', type=int, default=1, help='frame per second')

    # configure
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    input_path = args.input_pose
    begin_frame = args.begin_frame
    end_frame = args.end_frame
    output_path = './output/'
    cfg.DATASET.target_joint_set = 'mano'
    cfg.MODEL.posenet_pretrained = False
    if not (os.path.exists(output_path) and os.path.isdir(output_path)):
        os.mkdir(output_path)

    virtual_crop_size = 500
    img_size = (1920, 1080)
    color = (100/255, 123/255, 206/255) #BGR format

    # prepare model
    mesh_model_right = MANO('right')
    mesh_model_left = MANO('left')
    mesh_faces = {'right':mesh_model_right.face, 'left':mesh_model_left.face}
    model, joint_regressor, joint_num, skeleton, graph_L, graph_perm_reverse = get_joint_setting(mesh_model_right)
    model = model.cuda()
    joint_regressor = torch.Tensor(joint_regressor).cuda()
    project_net = models.project_net.get_model(crop_size=virtual_crop_size).cuda()
    
    # prepare 2d pose
    with open(args.input_pose, "rb") as f:
        data = pickle.load(f)
    joint_input = data["keypoints"]
    joint_input[:,:,:2] = joint_input[:,:,:2] * 2
    joint_input[:,:,0] -= (joint_input[0,5,0] + joint_input[0,6,0] + joint_input[0,11,0] + joint_input[0,12,0]) / 4 - img_size[0] * 0.5
    joint_input[:,:,1] -= (joint_input[0,5,1] + joint_input[0,6,1] + joint_input[0,11,1] + joint_input[0,12,1]) / 4 - img_size[1] * 0.6
    left_joint_input = joint_input[:,91:112,0:2].copy()
    right_joint_input = joint_input[:,112:133,0:2].copy()

    if begin_frame < end_frame:
        videowrite = cv2.VideoWriter(output_path + f'result_video.mp4', cv2.VideoWriter_fourcc(*'mp4v'), args.fps, img_size)

        for frame in range(begin_frame, min(end_frame,len(joint_input))):
            print("deal frame", frame)

            hand_joint_input = {'right':right_joint_input[frame], 'left':left_joint_input[frame]}
            pose_vis_img = render(hand_joint_input, project_net, mesh_faces, img_size, color, virtual_crop_size)
            pose_vis_img = vis_2d_keypoints(pose_vis_img, joint_input[frame], show_keypoints=True)

            videowrite.write(pose_vis_img)

        videowrite.release()
    else:
        hand_joint_input = {'right':right_joint_input[begin_frame], 'left':left_joint_input[begin_frame]}
        pose_vis_img = render(hand_joint_input, project_net, mesh_faces, img_size, color, virtual_crop_size)
        pose_vis_img = vis_2d_keypoints(pose_vis_img, joint_input[begin_frame], show_keypoints=True)

        cv2.imwrite(output_path + f'result_img.png', pose_vis_img)
