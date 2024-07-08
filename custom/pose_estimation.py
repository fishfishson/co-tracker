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
# import matplotlib as mpl
# mpl.use('agg')
from matplotlib import pyplot as plt
import matplotlib.animation as animation
from tqdm import tqdm
from utils.ellipse_fit import EllipsoidTool
from scipy.spatial.transform import Rotation
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.mixture import GaussianMixture
from skimage.filters import threshold_otsu


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
        print("rank(H) < 3, returning identity")
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


def geuss_radius_improved(tracks: np.ndarray, visibility: np.ndarray, tol_pixel: float = 30, smooth: str = True):
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
        points = track[valid]
        # points = np.concatenate([points, np.zeros_like(points[..., :1])], axis=-1)
        # pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
        # pcd.remove_statistical_outlier(nb_neighbors=50, std_ratio=0.01)
        # points = np.asarray(pcd.points)
        # points = points[..., :2].astype('float32')
        center, radius = cv2.minEnclosingCircle(points)
        _center = points.mean(axis=0)
        if np.linalg.norm(center - _center, axis=-1) > tol_pixel:
            print(f'outlier exist in {i}-th image!')
            dists = np.linalg.norm(points - _center[None], axis=-1)
            # dists = (dists - dists.min()) / (dists.max() - dists.min()) * 256
            # th = threshold_otsu(dists.reshape(-1, 1), 256)
            _dists = np.sort(dists)
            grad_dists = _dists[1:] - _dists[:-1] 
            th = np.max(grad_dists)
            index = np.where(grad_dists == th)[0]
            dist_th = _dists[index]
            valid = dists < dist_th
            center, radius = cv2.minEnclosingCircle(points[valid])
        centers.append(center)
        radiis.append(radius)
    centers = np.stack(centers)
    radiis = np.stack(radiis)
    radiis_mean = np.mean(radiis)
    if smooth:
        arrs = []
        for arr in np.split(centers, centers.shape[-1], axis=-1):
            arr = arr.squeeze(-1)
            arr = np.convolve(np.concatenate([arr[:1], arr, arr[-1:]]), [1/3, 1/3, 1/3], 'valid')
            arrs.append(arr)
        centers = np.stack(arrs, axis=-1)
    return centers, radiis, radiis_mean


def guess_radius_sem(masks, tol_pixel: float = 20, smooth: bool = False):
    masks = [((cv2.imread(x) > 0).any(-1) * 255).astype('uint8') for x in masks]
    centers = []
    radiis = []
    center_accum = np.zeros(2,)
    radius_accum = 0
    for i in tqdm(range(len(masks))):
        mask = masks[i]        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contours = np.concatenate(contours, axis=0)
        center, radius = cv2.minEnclosingCircle(contours)
        if i > 0 and abs(radius - radius_accum) > tol_pixel:
            print(f'outlier exist in {i}-th image!')
            center = center_accum
            radius = radius_accum
        centers.append(center)
        radiis.append(radius)
        center_accum = (center_accum * i + center) / (i + 1)
        radius_accum = (radius_accum * i + radius) / (i + 1)
    centers = np.stack(centers)
    radiis = np.stack(radiis)
    radiis_mean = np.mean(radiis)
    if smooth:
        arrs = []
        for arr in np.split(centers, centers.shape[-1], axis=-1):
            arr = arr.squeeze(-1)
            arr = np.convolve(np.concatenate([arr[:1], arr, arr[-1:]]), [1/3, 1/3, 1/3], 'valid')
            arrs.append(arr)
        centers = np.stack(arrs, axis=-1)
    return centers, radiis, radiis_mean


def filter_tracks(centers, radius, tracks, visibility):
    for i in tqdm(range(len(tracks))):
        track = tracks[i]
        center = centers[i]
        invalid = np.where(np.linalg.norm(track - center, axis=-1) > radius)[0]
        if len(invalid) > 0:
            visibility[i, invalid] = False
            print(f'{len(invalid)} invalid points in {i}-th image')
    return tracks, visibility


# def filter_tracks(path, images, tracks, visibility):
#     os.makedirs(join(path, 'filter'), exist_ok=True)
#     valids = visibility.copy()
#     for i in tqdm(range(len(images))):
#         image = cv2.imread(images[i])
#         track = tracks[i]
#         valid = valids[i]
#         points = track[valid]
#         gmm = GaussianMixture(n_components=2).fit(points)
#         # cluster = SpectralClustering(n_clusters=2, gamma=6.0, n_jobs=8).fit(points)
#         # label = cluster.labels_
#         label = gmm.predict(points)
#         points0 = points[label == 0]
#         points1 = points[label == 1]
#         for j in range(points0.shape[0]):
#             image = cv2.circle(image, (int(points0[j][0]), int(points0[j][1])), radius=1, color=(0, 255, 0), thickness=1)
#         for j in range(points1.shape[0]):
#             image = cv2.circle(image, (int(points1[j][0]), int(points1[j][1])), radius=1, color=(0, 0, 255), thickness=1)
#         cv2.imwrite(join(path, 'filter', f'{i:06}.jpg'), image)


def guess_depth(tracks: np.ndarray, centers: np.ndarray, radius: float):
    """
    tracks: T, N, 2
    centers: T, 2
    """
    depth = radius**2 - np.sum((tracks - centers[:, None])**2, axis=-1) # T, N
    depth = np.clip(depth, 0, None)
    depth = np.sqrt(depth)
    return depth


def scaled_icp(moving, fixed, weights=None):
    normals = moving - fixed
    b = np.sum(fixed * normals, axis=1)
    A1 = np.cross(moving, normals)
    A2 = normals
    A3 = np.expand_dims(np.sum(moving * normals, axis=1), axis=1)
    A = np.hstack((A1, A2, A3))

    if weights is not None:
        x = np.linalg.inv(A.T @ weights @ A) @ A.T @ weights @ b
    else:
        x = np.linalg.inv(A.T @ A) @ A.T @ b

    x[:3] /= x[6]

    R = np.eye(3)
    T = np.zeros(3)
    R[0, 0] = np.cos(x[2]) * np.cos(x[1])
    R[0, 1] = -np.sin(x[2]) * np.cos(x[0]) + np.cos(x[2]) * np.sin(x[1]) * np.sin(
        x[0]
    )
    R[0, 2] = np.sin(x[2]) * np.sin(x[0]) + np.cos(x[2]) * np.sin(x[1]) * np.cos(
        x[0]
    )
    R[1, 0] = np.sin(x[2]) * np.cos(x[1])
    R[1, 1] = np.cos(x[2]) * np.cos(x[0]) + np.sin(x[2]) * np.sin(x[1]) * np.sin(
        x[0]
    )
    R[1, 2] = -np.cos(x[2]) * np.sin(x[0]) + np.sin(x[2]) * np.sin(x[1]) * np.cos(
        x[0]
    )
    R[2, 0] = -np.sin(x[1])
    R[2, 1] = np.cos(x[1]) * np.sin(x[0])
    R[2, 2] = np.cos(x[1]) * np.cos(x[0])
    T[0] = x[3]
    T[1] = x[4]
    T[2] = x[5]
    c = x[6]

    transform = np.eye(4)
    transform[:3, :3] = R
    transform[:3, 3] = T
    scale = c

    # euler = np.rad2deg(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))
    # distance = np.sqrt(np.sum(T**2))

    return transform, scale


def unscaled_icp(moving, fixed, weights=None):
    normals = moving - fixed
    b1 = np.sum(fixed * normals, axis=1)
    b2 = np.sum(moving * normals, axis=1)
    b = np.expand_dims(b1 - b2, axis=1)
    A1 = np.cross(moving, normals)
    A2 = normals
    A = np.hstack((A1, A2))

    if weights is not None:
        x = np.linalg.inv(A.T @ weights @ A) @ A.T @ weights @ b
    else:
        x = np.linalg.inv(A.T @ A) @ A.T @ b

    R = np.eye(3)
    T = np.zeros(3)
    R[0, 0] = np.cos(x[2]) * np.cos(x[1])
    R[0, 1] = -np.sin(x[2]) * np.cos(x[0]) + np.cos(x[2]) * np.sin(x[1]) * np.sin(
        x[0]
    )
    R[0, 2] = np.sin(x[2]) * np.sin(x[0]) + np.cos(x[2]) * np.sin(x[1]) * np.cos(
        x[0]
    )
    R[1, 0] = np.sin(x[2]) * np.cos(x[1])
    R[1, 1] = np.cos(x[2]) * np.cos(x[0]) + np.sin(x[2]) * np.sin(x[1]) * np.sin(
        x[0]
    )
    R[1, 2] = -np.cos(x[2]) * np.sin(x[0]) + np.sin(x[2]) * np.sin(x[1]) * np.cos(
        x[0]
    )
    R[2, 0] = -np.sin(x[1])
    R[2, 1] = np.cos(x[1]) * np.sin(x[0])
    R[2, 2] = np.cos(x[1]) * np.cos(x[0])
    T[0] = x[3]
    T[1] = x[4]
    T[2] = x[5]

    transform = np.eye(4)
    transform[:3, :3] = R
    transform[:3, 3] = T

    # euler = np.rad2deg(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))
    # distance = np.sqrt(np.sum(T**2))

    return transform


def geuss_rotation(points: np.ndarray, centers: np.ndarray, visibility: np.ndarray, masses: np.ndarray, mode: str = 'icp'):
    """
    points: T N 3
    centers: T 2
    visibility: T N
    """
    Rs = []
    ts = []
    if mode == 'icp':
        num_iters = 10
        use_scaled = False
        for i in tqdm(range(points.shape[0] - 1)):
            cur_points = points[i]
            nxt_points = points[i + 1]
            cur_center = centers[i]
            nxt_center = centers[i + 1]
            cur_visibility = visibility[i]
            nxt_visibility = visibility[i + 1]
            valid = cur_visibility * nxt_visibility
            cur_points = cur_points[valid]
            nxt_points = nxt_points[valid]
            mass = np.diag(masses[valid]) if masses is not None else None
            mass = None
            cum_T = np.eye(4)
            _cur_points = cur_points.copy()
            for i in range(num_iters):
                if use_scaled:
                    curr_T, scale = scaled_icp(_cur_points, nxt_points, weights=mass)
                    cum_T = curr_T @ cum_T
                    _cur_points = (curr_T[:3, :3] @ _cur_points.T * scale + curr_T[:3, 3:]).T
                else:
                    curr_T = unscaled_icp(_cur_points, nxt_points, weights=mass)
                    cum_T = curr_T @ cum_T
                    _cur_points = (curr_T[:3, :3] @ _cur_points.T + curr_T[:3, 3:]).T
            R = cum_T[:3, :3]
            t = cum_T[:3, 3:]
            # cur_points = (R @ cur_points.T + t).T
            # print(np.linalg.norm(cur_points - nxt_points, axis=-1).mean())
            Rs.append(R)
            ts.append(t)
    else:
        for i in tqdm(range(points.shape[0] - 1)):
            cur_points = points[i].T
            nxt_points = points[i + 1].T
            cur_center = centers[i]
            nxt_center = centers[i + 1]
            cur_visibility = visibility[i]
            nxt_visibility = visibility[i + 1]
            large_mass = masses > np.quantile(masses, 0.75)
            valid = cur_visibility * nxt_visibility * large_mass
            cur_points = cur_points[:, valid]
            nxt_points = nxt_points[:, valid]
            R, t = rigid_transform_3D(cur_points, nxt_points, cur_center, nxt_center)
            # cur_points = R @ cur_points + t
            # print(np.linalg.norm(cur_points - nxt_points, axis=0).mean())
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
    parser.add_argument('--radius_mode', type=str, default='sem', choices=['improved', 'original', 'sem'])
    parser.add_argument('--smooth', action='store_true', default=False)
    parser.add_argument('--icp_mode', type=str, default='default', choices=['icp', 'default'])
    # visualize configs
    parser.add_argument('--vis_centers', action='store_true', default=False)
    parser.add_argument('--vis_depth', action='store_true', default=False)
    parser.add_argument('--vis_points', action='store_true', default=False)
    parser.add_argument('--vis_traj', action='store_true', default=False)
    args = parser.parse_args()
    
    vis_centers = args.vis_centers
    vis_depth = args.vis_depth
    vis_points = args.vis_points
    vis_traj = args.vis_traj

    images_path = os.path.join(args.data_path, 'images')
    images = sorted(glob(os.path.join(images_path, '*.jpg')))
    masks_path = os.path.join(args.data_path, 'images_masks')
    masks = sorted(glob(os.path.join(masks_path, '*.png')))
    data = np.load(join(args.data_path, 'tracks.npz'))
    tracks = data['tracks']
    visibility = data['visibility']
    masses = data['masses']

    print('Estimating radius')
    if args.radius_mode == 'improved':
        # if os.path.exists(join(args.data_path, 'mask.jpg')):
        #     mask = cv2.imread(join(args.data_path, 'mask.jpg'))
        #     h, w = np.where(mask[..., 0] == 0)
        #     hmax = h.max()
        #     hmin = h.min()
        #     wmax = w.max()
        #     wmin = w.min()
        #     invalid_x = np.logical_and(tracks[0, :, 0] > wmin, tracks[0, :, 0] < wmax)
        #     invalid_y = np.logical_and(tracks[0, :, 1] > hmin, tracks[0, :, 1] < hmax)
        #     valid = ~np.logical_and(invalid_x, invalid_y)
        #     tracks = tracks[:, valid]
        #     visibility = visibility[:, valid]
        radius_scale = 1.1
        centers, radiis, radiis_mean = geuss_radius_improved(tracks, visibility, smooth=args.smooth)
    elif args.radius_mode == 'original':
        radius_iters = 1
        radius_scale = 2.0
        centers, radiis, radiis_mean = guess_radius(tracks, visibility, max_iter=radius_iters, pre_fps=False)
    elif args.radius_mode == 'sem':
        radius_scale = 1.0
        centers, radiis, radiis_mean = guess_radius_sem(masks, smooth=args.smooth)
    else: raise NotImplementedError
    radii = radiis_mean * radius_scale
    if vis_centers:
        os.makedirs(join(args.data_path, 'centers'), exist_ok=True)
        for i in range(len(centers)):
            image = cv2.imread(images[i])
            image = cv2.circle(image, (int(centers[i][0]), int(centers[i][1])), radius=int(radii), color=(255, 0, 0), thickness=1)
            cv2.imwrite(join(args.data_path, 'centers', f'{i:06}.jpg'), image)
        cmd = ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-framerate', '10', '-pattern_type', 'glob', '-i', f'{args.data_path}/centers/*.jpg', '-c:v', 'libx264', '-crf', '23', '-pix_fmt', 'yuv420p', f'{args.data_path}/centers.mp4']
        subprocess.call(cmd)

    print('Filter tracks')
    tracks, visibility = filter_tracks(centers, radii, tracks, visibility)
    print('Estimating depth')
    depth = guess_depth(tracks, centers, radii)
    points = np.concatenate([tracks, depth[..., None]], axis=-1)
    # x, y, z = points[..., 0], points[..., 1], points[..., 2]
    # for i in tqdm(range(x.shape[1])):
    #     vis = visibility[:, i]
    #     _x = x[:, i]
    #     _x = np.convolve(np.concatenate([_x[:1], _x, _x[-1:]]), [1/3, 1/3, 1/3], 'valid')
    #     x[:, i] = _x
    #     _y = y[:, i]
    #     _y = np.convolve(np.concatenate([_y[:1], _y, _y[-1:]]), [1/3, 1/3, 1/3], 'valid')
    #     y[:, i] = _y
    #     _z = z[:, i]
    #     _z = np.convolve(np.concatenate([_z[:1], _z, _z[-1:]]), [1/3, 1/3, 1/3], 'valid')
    #     z[:, i] = _z
    # points = np.stack([x, y, z], axis=-1)

    print('Estimating rotation')
    Rs, ts = geuss_rotation(points, centers, visibility, masses, args.icp_mode)
    # Rs, ts = refine_rotation(Rs, ts, points, visibility, centers, radii)
    # if vis_points:
    #     os.makedirs(join(args.data_path, 'points'), exist_ok=True)
    #     for i in range(len(points)):
    #         _ = trimesh.PointCloud(points[i]).export(join(args.data_path, 'points', f'{i:06}.ply'))
    #     os.makedirs(join(args.data_path, 'pred_points'), exist_ok=True)
    #     for i in range(len(Rs)):
    #         pred_points = (Rs[i] @ points[i].T + ts[i]).T
    #         _ = trimesh.PointCloud(pred_points).export(join(args.data_path, 'pred_points', f'{i+1:06}.ply'))

    if vis_traj:
        # cmap = mpl.colormaps['winter']
        # N = points.shape[0]
        # t = np.linspace(0, 1, N)
        # colors = cmap(t)
        K = 5
        indices = visibility.sum(axis=0).argsort()[::-1][:K]
        _points = points[:, indices]
        _centers = np.concatenate([centers, np.zeros_like(centers[..., :1])], axis=-1)
        _points = (_points - _centers[:, None]) * 0.05
        x, y, z = _points[..., 0], _points[..., 1], _points[..., 2]
        x -= (np.max(x) + np.min(x)) / 2
        y -= (np.max(y) + np.min(y)) / 2
        z -= np.min(z)
        N = len(x)
        # _visibility = visibility[:, traj_indices]
        fig = plt.figure(figsize=(5, 5), dpi=100)
        ax = fig.add_subplot(111, projection='3d')
        for i in range(N - 1):
            for j in range(K):
                ax.plot(x[i:i+2,j], y[i:i+2,j], z[i:i+2,j], color=plt.cm.jet(i/N), alpha=1, linewidth=1)
        ax.set_xlim3d(x.min(), x.max())
        ax.set_ylim3d(y.min(), y.max())
        ax.set_zlim3d(z.min(), z.max())
        # ax.view_init(60, -110)
        ax.set_xlabel('X', fontsize=10)
        ax.set_ylabel('Y', fontsize=10)
        ax.set_zlabel('Z', fontsize=10)
        # ax.xaxis.set_tick_params(labelsize=30)
        # ax.yaxis.set_tick_params(labelsize=30)
        # ax.zaxis.set_tick_params(labelsize=30)
        ax.set_title('3D Trajectory')
        anim = animation.FuncAnimation(fig, lambda i: ax.view_init(elev=90, azim=i), frames=np.arange(0, 362, 2), interval=100)
        anim.save(join(args.data_path, 'traj.mp4'), writer='ffmpeg')
        plt.close(fig)

    Ts = [np.eye(4)]
    for i in range(len(Rs)):
        T = np.eye(4)
        T[:3, :3] = Rs[i]
        T[:3, 3:] = ts[i]
        T = T @ Ts[-1]
        Ts.append(T)
    Ts = np.stack(Ts)
    
    if vis_depth:
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
        os.makedirs(join(args.data_path, 'fuse'), exist_ok=True)
        cmap = mpl.colormaps['magma']
        colors = cmap(np.linspace(0, 1, points.shape[1]))
        for i in range(fuses.shape[0]):
            img = cv2.imread(images[i])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            fig = plt.figure(figsize=(20, 10), dpi=100)
            ax = fig.add_subplot(121)
            ax.imshow(img)
            ax.set_axis_off()

            # set matplot 3d
            ax = fig.add_subplot(122, projection='3d')
            ax.set_xlim(0, 512)
            ax.set_ylim(0, 512)
            ax.set_zlim(0, 512)
            # ax.scatter(T_points[:, 0] - H // 2, T_points[:, 1] - W // 2, T_points[:, 2], c=colors)
            ax.scatter(fuses[i, :, 0], fuses[i, :, 1], fuses[i, :, 2], c=colors)
            ax.view_init(elev=75, azim=45, roll=0)
            # fig.canvas.draw()
            # pcd = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            # pcd = pcd.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            # plt.close(fig)
            # fig = plt.figure(figsize=(10, 10), dpi=100)
            # fig.canvas.draw()
            # img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            # img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            plt.savefig(join(args.data_path, 'fuse', os.path.basename(images[i])))
            plt.close(fig)
            # img = np.concatenate([img, pcd], axis=1)
            # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        cwd = os.getcwd()
        os.chdir(os.path.join(args.data_path))
        subprocess.call(f"ffmpeg -y -framerate 10 -pattern_type glob -i 'fuse/*.jpg' -c:v libxh264 -pix_fmt yuv420p fuse.mp4", shell=True)
        os.chdir(cwd)

    # print('Estimating ellipse')
    # ET = EllipsoidTool()
    # ctr, rad, rot = ET.getMinVolEllipse(fuses[0], 0.01)
    # ctrs = []
    # rads = []
    # rots = []
        
    # euls = []
    # for i in tqdm(range(fuses.shape[0])):
    #     # ctr, rad, rot = ET.getMinVolEllipse(fuses[i], 0.01)
    #     # ctrs.append(ctr)
    #     # rads.append(rad)
    #     # rots.append(rot)
    #     eul = Rotation.from_matrix(Ts[i][:3, :3] @ rot)
    #     eul = eul.as_euler('xyz')
    #     euls.append(eul)
    # ctrs = np.stack(ctrs)
    # rads = np.stack(rads)
    # rots = np.stack(rots)
    # euls = np.stack(euls)
    # os.makedirs(join(args.data_path, 'ellipse'), exist_ok=True)
    # for i in range(ctrs.shape[0]):
    #     ctr = ctrs[i]
    #     rad = rads[i]
    #     rot = rots[i]
    #     ET.plotEllipsoid(ctr, rad, rot, path=join(args.data_path, 'ellipse', f'{i:06}.png'))
    # cwd = os.getcwd()
    # os.chdir(os.path.join(args.data_path))
    # subprocess.call(f"ffmpeg -y -framerate 10 -pattern_type glob -i 'ellipse/*.png' -c:v wmv2 ellipse.wmv", shell=True)
    # os.chdir(cwd)

    # fig = plt.figure(figsize=(10, 10), dpi=100)
    # plt.plot(euls[:, 0], label='X')
    # plt.plot(euls[:, 1], label='Y')    
    # plt.plot(euls[:, 2], label='Z')
    # plt.legend()
    # plt.savefig(join(args.data_path, 'euler.png'))
    # plt.close(fig)

    # os.makedirs(join(args.data_path, 'euler_velocity'), exist_ok=True)
    # for i in range(euls.shape[0] - 1):
    #     image = cv2.imread(images[i])
    #     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #     fig = plt.figure(figsize=(10, 10), dpi=50)
        
    #     ax = fig.add_subplot(221)
    #     ax.imshow(image)
    #     ax.set_axis_off()
        
    #     ax = fig.add_subplot(222)
    #     ax.plot(np.abs(euls[1:, 0] - euls[:-1, 0]))
    #     ax.set_title('VelocityX')
    #     ax.axvline(x=i, color='r', linestyle='--')

    #     ax = fig.add_subplot(223)
    #     ax.plot(np.abs(euls[1:, 1] - euls[:-1, 1]))
    #     ax.set_title('VelocityY')
    #     ax.axvline(x=i, color='r', linestyle='--')

    #     ax = fig.add_subplot(224)        
    #     ax.plot(np.abs(euls[1:, 2] - euls[:-1, 2]))
    #     ax.set_title('VelocityZ')
    #     ax.axvline(x=i, color='r', linestyle='--')

    #     plt.savefig(join(args.data_path, 'euler_velocity', f'{i:06}.jpg'))
    #     plt.close(fig)
        
    print('Estimating angular velocity')
    os.makedirs(join(args.data_path, 'rotation'), exist_ok=True)
    abs_rotation = False
    rotations = [0]
    euler_angles = [(0, 0, 0)]
    rotation_axis = np.array([0, 0, 1])
    theta = 0
    for i in range(Rs.shape[0]):
        rot = Rotation.from_matrix(Rs[i])
        rotvec = rot.as_rotvec(degrees=True)
        euler = rot.as_euler('zxy', degrees=True)
        if abs_rotation:
            theta += np.abs(np.linalg.norm(rotvec))
        else:
            if rotation_axis @ rotvec > 0:
                theta += np.abs(np.linalg.norm(rotvec))
            else:
                theta -= np.abs(np.linalg.norm(rotvec))
        rotations.append(theta)
        euler_angles.append(euler)
    rotations = np.array(rotations)
    euler_angles = np.array(euler_angles)

    for i in range(len(images)):
        fig = plt.figure(figsize=(20, 10), dpi=100)
        image = cv2.imread(images[i])
        image = cv2.circle(image, (int(centers[i][0]), int(centers[i][1])), radius=int(radii), color=(255, 0, 0), thickness=1)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        gs = fig.add_gridspec(1, 2)

        ax1 = fig.add_subplot(gs[0, 0])
        ax1.imshow(image)
        ax1.set_axis_off()

        gssub = gs[0, 1].subgridspec(nrows=2, ncols=2, wspace=0.2)

        ax2a = fig.add_subplot(gssub[0, 0])
        ax2a.plot(rotations)
        ax2a.axvline(x=i, color='r', linestyle='--')
        # ax2a.axhline(y=0, color='g')
        ax2a.set_title('theta')

        ax2b = fig.add_subplot(gssub[0, 1])
        ax2b.plot(euler_angles[:, 1])
        ax2b.axvline(x=i, color='r', linestyle='--')
        ax2b.set_title('euler-x')

        ax2c = fig.add_subplot(gssub[1, 0])
        ax2c.plot(euler_angles[:, 2])
        ax2c.axvline(x=i, color='r', linestyle='--')
        ax2c.set_title('euler-y')

        ax2d = fig.add_subplot(gssub[1, 1])
        ax2d.plot(euler_angles[:, 0])
        ax2d.axvline(x=i, color='r', linestyle='--')
        ax2d.set_title('euler-z')

        plt.savefig(join(args.data_path, 'rotation', f'{i:06}.jpg'))
        plt.close(fig)
    
    cmd = ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-framerate', '10', '-pattern_type', 'glob', '-i', f'{args.data_path}/rotation/*.jpg', '-c:v', 'libx264', '-crf', '23', '-pix_fmt', 'yuv420p', f'{args.data_path}/rotation.mp4']
    subprocess.call(cmd)
