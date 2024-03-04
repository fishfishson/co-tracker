import os
from os.path import join
from glob import glob
import argparse
import numpy as np
import cv2
import trimesh
import subprocess
import torch
from pytorch3d.ops import sample_farthest_points
import matplotlib as mpl
from matplotlib import pyplot as plt
from tqdm import tqdm
from utils.ellipse_fit import EllipsoidTool
from scipy.spatial.transform import Rotation
    
def rigid_transform_3D(A, B, center_A, center_B):
    # https://github.com/nghiaho12/rigid_transform_3D/blob/843c4906fe2b22bec3b1c49f45683cb536fb054c/rigid_transform_3D.py#L10
    # Input: expects 3xN matrix of points
    # Returns R,t
    # R = 3x3 rotation matrix
    # t = 3x1 column vector

    assert A.shape == B.shape

    num_rows, num_cols = A.shape
    if num_rows != 3:
        raise Exception(f"matrix A is not 3xN, it is {num_rows}x{num_cols}")

    num_rows, num_cols = B.shape
    if num_rows != 3:
        raise Exception(f"matrix B is not 3xN, it is {num_rows}x{num_cols}")

    # find mean column wise
    centroid_A = np.array([center_A[0], center_A[1], 0])
    centroid_B = np.array([center_B[0], center_B[1], 0])

    # ensure centroids are 3x1
    centroid_A = centroid_A.reshape(-1, 1)
    centroid_B = centroid_B.reshape(-1, 1)

    # subtract mean
    Am = A - centroid_A
    Bm = B - centroid_B

    H = Am @ np.transpose(Bm)

    # sanity check
    if np.linalg.matrix_rank(H) < 3:
        return np.identity(3), 0

    # find rotation
    assert np.isnan(H).any() == False
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # special reflection case
    if np.linalg.det(R) < 0:
        print("det(R) < R, reflection detected!, correcting for it ...")
        print('ha')
        Vt[2,:] *= -1
        R = Vt.T @ U.T
        # R = np.identity(3)

    t = -R @ centroid_A + centroid_B

    return R, t

def guess_radius(tracks: np.ndarray, visibility: np.ndarray, max_iter: int = 1, outlier_scale: float = 1.7, pre_fps: bool = True):
    """
    tracks: T N 2
    visibility: T N
    """
    valid = visibility.copy()
    if pre_fps:
        N = tracks.shape[1]
        tracks, indices = sample_farthest_points(torch.from_numpy(tracks), K=min(int(0.1 * N), 256), random_start_point=True)
        tracks = tracks.numpy() # T K 2
        indices = indices.numpy() # T k
        # dummy = np.arange(tracks.shape[0])[..., None].repeat(tracks.shape[1], 1)
        valid = np.take_along_axis(valid, indices.astype('int'), axis=1)
    for _ in range(max_iter):
        centers = np.sum(valid[..., None] * tracks, axis=1) / np.sum(valid[..., None], axis=1) # T, 2
        r = np.linalg.norm(tracks - centers[:, None], ord=2, axis=-1) # T, N
        r_mean = np.sum(valid * r) / np.sum(valid)
        valid = r <= r_mean * outlier_scale
        if np.sum(~valid) == 0:
            break
    arrs = []
    for arr in np.split(centers, centers.shape[-1], axis=-1):
        arr = arr.squeeze(-1)
        arr = np.convolve(np.concatenate([arr[:1], arr, arr[-1:]]), [1/3, 1/3, 1/3], 'valid')
        arrs.append(arr)
    centers = np.stack(arrs, axis=-1)
    radiis = np.linalg.norm(tracks - centers[:, None], ord=2, axis=-1) # T, N
    radiis_mean = np.sum(valid * radiis) / np.sum(valid)
    return centers, radiis, radiis_mean

def geuss_radius_improved(tracks: np.ndarray, visibility: np.ndarray):
    """
    tracks: T N 2
    visibility: T N
    """
    valids = visibility.copy()
    centers = []
    radiis = []
    for i in tqdm(range(tracks.shape[0])):
        track = tracks[i]
        valid = valids[i]
        center, radius = cv2.minEnclosingCircle(track[valid])
        centers.append(center)
        radiis.append(radius)
    centers = np.stack(centers)
    radiis = np.stack(radiis)
    radiis_mean = np.mean(radiis)
    arrs = []
    for arr in np.split(centers, centers.shape[-1], axis=-1):
        arr = arr.squeeze(-1)
        arr = np.convolve(np.concatenate([arr[:1], arr, arr[-1:]]), [1/3, 1/3, 1/3], 'valid')
        arrs.append(arr)
    centers = np.stack(arrs, axis=-1)
    return centers, radiis, radiis_mean

def guess_depth(tracks: np.ndarray, centers: np.ndarray, radius: float):
    """
    tracks: T, N, 2
    centers: T, 2
    """
    depth = radius**2 - np.sum((tracks - centers[:, None])**2, axis=-1) # T, N
    depth = np.clip(depth, 0, None)
    depth = np.sqrt(depth)
    return depth

def geuss_rotation(points: np.ndarray, centers: np.ndarray, visibility: np.ndarray):
    """
    points: T N 3
    centers: T 2
    """
    Rs = []
    ts = []
    for i in tqdm(range(points.shape[0] - 1)):
        cur_points = points[i].T
        nxt_points = points[i + 1].T
        cur_center = centers[i]
        nxt_center = centers[i + 1]
        R, t = rigid_transform_3D(cur_points, nxt_points, cur_center, nxt_center)
        Rs.append(R)
        ts.append(t)
    Rs = np.stack(Rs)
    ts = np.stack(ts)
    return Rs, ts

def geuss_ellipse():
    pass

    # for i in range(args.max_iters):
        
    # 2. estimate depth 
    # 3. estimate rotation
    # 4. estimate missing
    # 5. estimate ellipse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str)
    # geuss configs
    parser.add_argument('--radius_mode', type=str, default='improved', choices=['improved', 'original'])
    parser.add_argument('--vis_radius', action='store_true', default=False)
    parser.add_argument('--vis_depth', action='store_true', default=False)
    parser.add_argument('--vis_points', action='store_true', default=False)
    args = parser.parse_args()
    
    vis_radius = args.vis_radius
    vis_depth = args.vis_depth
    vis_points = args.vis_points

    images_path = os.path.join(args.data_path, 'images')
    images = sorted(glob(os.path.join(images_path, '*.jpg')))

    data = np.load(join(args.data_path, 'tracks.npz'))
    tracks = data['tracks'] 
    visibility = data['visibility']

    print('Estimating radius')
    if args.radius_mode == 'improved':
        radius_scale = 1.1
        centers, radiis, radiis_mean = geuss_radius_improved(tracks, visibility)
    elif args.radius_mode == 'original':
        radius_iters = 1
        radius_scale = 2.0
        centers, radiis, radiis_mean = guess_radius(tracks, visibility, max_iter=args.radius_iters, pre_fps=False)
    else:
        raise NotImplementedError
    radii = radiis_mean * radius_scale
    if vis_radius:
        out_dir = join(args.data_path, 'radius')
        os.makedirs(out_dir, exist_ok=True)
        for i in range(len(centers)):
            img = cv2.imread(images[i])
            img = cv2.circle(img, (int(centers[i][0]), int(centers[i][1])), radius=int(radii), color=(0, 0, 255), thickness=1)
            cv2.imwrite(join(out_dir, os.path.basename(images[i])), img)
        cwd = os.getcwd()
        os.chdir(os.path.join(args.data_path))
        subprocess.call(f"ffmpeg -y -framerate 10 -pattern_type glob -i 'radius/*.jpg' -c:v h264 radius.mp4", shell=True)
        os.chdir(cwd)
    
    print('Estimating depth')
    depth = guess_depth(tracks, centers, radii)
    points = np.concatenate([tracks, depth[..., None]], axis=-1)
    Rs, ts = geuss_rotation(points, centers, visibility)

    if vis_points:
        os.makedirs(join(args.data_path, 'points'), exist_ok=True)
        for i in range(len(points)):
            _ = trimesh.PointCloud(points[i]).export(join(args.data_path, 'points', f'{i:06}.ply'))
        os.makedirs(join(args.data_path, 'pred_points'), exist_ok=True)
        for i in range(len(Rs)):
            pred_points = (Rs[i] @ points[i].T + ts[i]).T
            _ = trimesh.PointCloud(pred_points).export(join(args.data_path, 'pred_points', f'{i+1:06}.ply'))
    
    # os.makedirs(join(args.data_path, 'back'), exist_ok=True)
    # os.makedirs(join(args.data_path, 'forward'), exist_ok=True)
    Ts = [np.eye(4)]
    for i in range(len(Rs)):
        T = np.eye(4)
        T[:3, :3] = Rs[i]
        T[:3, 3:] = ts[i]
        T = T @ Ts[-1]
        Ts.append(T)
    Ts = np.stack(Ts)
    
    fuses = []
    for i in range(points.shape[0]):
        homo = np.ones_like(points[..., :1])
        homo_points = np.concatenate([points, homo], axis=-1)
        bwd_Ts = np.linalg.inv(Ts)
        fwd_T = Ts[i][None]
        T = fwd_T @ bwd_Ts
        T_points = (T @ homo_points.transpose(0, 2, 1)).transpose(0, 2, 1)[..., :3]
        T_points = T_points.mean(0)
        fuses.append(T_points)
        if vis_points:
            _ = trimesh.PointCloud(T_points).export(join(args.data_path, 'fuse', f'{i:06}.ply'))

        # homo = np.ones_like(points[0][..., :1])
        # homo_points = np.concatenate([points[0], homo], axis=-1)
        # forward = (T @ homo_points.T).T[..., :3]
        # _ = trimesh.PointCloud(forward).export(join(args.data_path, 'forward', f'{i+1:06}.ply'))

        # homo = np.ones_like(points[i+1][..., :1])
        # homo_points = np.concatenate([points[i+1], homo], axis=-1)
        # # back = (np.linalg.inv(T) @ homo_points.T).T[..., :3]
        # back = (np.eye(4) @ homo_points.T).T[..., :3]
        # _ = trimesh.PointCloud(back).export(join(args.data_path, 'back', f'{i+1:06}.ply'))
        # backs.append(back)
    fuses = np.stack(fuses)
    
    if vis_depth:
        os.makedirs(join(args.data_path, 'fuse'), exist_ok=True)
        cmap = mpl.colormaps['magma']
        colors = cmap(np.linspace(0, 1, points.shape[1]))
        for i in range(fuses.shape[0]):
            # set matplot 3d
            img = cv2.imread(images[i])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            H, W, _ = img.shape
            fig = plt.figure(figsize=(10, 10), dpi=100)
            ax = fig.add_subplot(111, projection='3d')
            ax.set_xlim(0, 512)
            ax.set_ylim(0, 512)
            ax.set_zlim(0, 512)
            # ax.scatter(T_points[:, 0] - H // 2, T_points[:, 1] - W // 2, T_points[:, 2], c=colors)
            ax.scatter(fuses[i, :, 0], fuses[i, :, 1], fuses[i, :, 2], c=colors)
            ax.view_init(elev=75, azim=45, roll=0)
            fig.canvas.draw()
            pcd = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            pcd = pcd.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            plt.close(fig)
            fig = plt.figure(figsize=(10, 10), dpi=100)
            ax = fig.add_subplot(111)
            ax.imshow(img)
            fig.canvas.draw()
            img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            plt.close(fig)
            img = np.concatenate([img, pcd], axis=1)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(join(args.data_path, 'fuse', os.path.basename(images[i])), img)
        cwd = os.getcwd()
        os.chdir(os.path.join(args.data_path))
        subprocess.call(f"ffmpeg -y -framerate 10 -pattern_type glob -i 'fuse/*.jpg' -c:v h264 fuse.mp4", shell=True)
        os.chdir(cwd)

    print('Estimating ellipse')
    ctrs = []
    rads = []
    rots = []
    euls = []
    ET = EllipsoidTool()
    for i in tqdm(range(fuses.shape[0])):
        ctr, rad, rot = ET.getMinVolEllipse(fuses[i], 0.01)
        ctrs.append(ctr)
        rads.append(rad)
        rots.append(rot)
        rot = Rotation.from_matrix(rot)
        eul = rot.as_euler('XYZ')
        euls.append(eul)
    ctrs = np.stack(ctrs)
    rads = np.stack(rads)
    rots = np.stack(rots)
    euls = np.stack(euls)
    os.makedirs(join(args.data_path, 'ellipse'), exist_ok=True)
    for i in range(ctrs.shape[0]):
        ctr = ctrs[i]
        rad = rads[i]
        rot = rots[i]
        ET.plotEllipsoid(ctr, rad, rot, path=join(args.data_path, 'ellipse', f'{i:06}.png'))
    cwd = os.getcwd()
    os.chdir(os.path.join(args.data_path))
    subprocess.call(f"ffmpeg -y -framerate 10 -pattern_type glob -i 'ellipse/*.png' -c:v h264 ellipse.mp4", shell=True)
    os.chdir(cwd)

    fig = plt.figure(figsize=(10, 10), dpi=100)
    plt.plot(euls[:, 0], label='X')
    plt.plot(euls[:, 1], label='Y')    
    plt.plot(euls[:, 2], label='Z')
    plt.legend()
    plt.savefig(join(args.data_path, 'euler.png'))
    plt.close(fig)

    os.makedirs(join(args.data_path, 'euler_velocity'), exist_ok=True)
    for i in range(euls.shape[0] - 1):
        image = cv2.imread(images[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        fig = plt.figure(figsize=(10, 20), dpi=100)
        ax = fig.add_subplot(121)
        ax.imshow(image)
        ax.set_axis_off()
        ax = fig.add_subplot(122)
        ax.plot(euls[1:, 0] - euls[:-1, 0], label='VelocityX')
        ax.plot(euls[1:, 1] - euls[:-1, 1], label='VelocityY')
        ax.plot(euls[1:, 2] - euls[:-1, 2], label='VelocityZ')
        ax.legend()
        ax.axvline(x=i, color='r', linestyle='--')
        plt.savefig(join(args.data_path, 'euler_velocity', f'{i:06}.jpg'))
        plt.close(fig)
    cwd = os.getcwd()
    os.chdir(os.path.join(args.data_path))
    subprocess.call(f"ffmpeg -y -framerate 10 -pattern_type glob -i 'euler_velocity/*.jpg' -c:v h264 euler.mp4", shell=True)
    os.chdir(cwd)