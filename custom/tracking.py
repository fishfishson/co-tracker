import os
import argparse
import numpy as np
import cv2
from tqdm import tqdm
from matplotlib import pyplot as plt
import matplotlib as mpl
from glob import glob
import shutil
import pandas as pd
import subprocess
import h5py
from itertools import combinations

import torch
from cotracker.predictor import CoTrackerPredictor
# from cotracker.utils.visualizer import Visualizer

def discrete_cmap(N, base_cmap=None):
    """Create an N-bin discrete colormap from the specified input map"""

    # Note that if base_cmap is a string or None, you can simply do
    #    return plt.cm.get_cmap(base_cmap, N)
    # The following works for string, None, or a colormap instance:

    base = plt.cm.get_cmap(base_cmap)
    color_list = base(np.linspace(0, 1, N, 0))
    return plt.cm.colors.ListedColormap(color_list, color_list, N)

def main(args):
    device = "cuda:0"
    model = torch.hub.load("facebookresearch/co-tracker", "cotracker2").to(device)
    # model = CoTrackerPredictor(checkpoint="../checkpoints/cotracker2.pth").to(device)

    images_path = os.path.join(args.data_path, 'images')
    images = glob(os.path.join(images_path, '*.jpg'))
    images.sort()

    locates_path = os.path.join(args.data_path, 'locate')
    locates = glob(os.path.join(locates_path, '*.npz'))
    locates.sort()

    video = []
    max_len = len(images)
    start_time = max(0, args.start_time)
    if args.end_time is None: end_time = max_len
    else: end_time = min(max_len, args.end_time) 
    duratoin = end_time - start_time
    for image in images[start_time:end_time]:
        img = cv2.imread(image)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # img = cv2.resize(img, (int(img.shape[1] * 0.5), int(img.shape[0] * 0.5)), cv2.INTER_AREA)
        video.append(img)
    video = np.array(video)
    video = torch.from_numpy(video).float().to(device)
    video = video.permute(0, 3, 1, 2)[None]

    mannual_path = os.path.join(args.data_path, args.mannual_path)
    if os.path.exists(mannual_path):
        mannual = pd.read_pickle(mannual_path)
        particles = sorted([x for x in np.unique(mannual['particle']) if not np.isnan(x)])
        gt_tracks = np.zeros((duratoin, len(particles), 2))
        gt_visibility = np.zeros((duratoin, len(particles)))
        for pid, p in enumerate(particles):
            valid = mannual['particle'] == p
            cx = mannual['x'][valid]
            cy = mannual['y'][valid]
            frame = mannual['frame'][valid]
            valid = np.logical_and(frame >= start_time, frame < end_time)
            cx = cx[valid]
            cy = cy[valid]
            frame = frame[valid]
            frame = [int(x) - start_time for x in frame]
            gt_tracks[frame, pid, 0] = cx * args.mannual_scale
            gt_tracks[frame, pid, 1] = cy * args.mannual_scale
            gt_visibility[frame, pid] = 1.
    else:
        gt_tracks = None
        gt_visibility = None

    # pred_tracks = gt_tracks[None]
    # pred_visibility = gt_visibility[None]

    queries = []
    masses = []
    if args.mode == 'only_first':
        for idx, locate in enumerate(locates[start_time:start_time+1]):
            loc = np.load(locate)
            cx, cy = loc['cx'], loc['cy']
            time = np.ones_like(cx) * idx
            queries.append(np.stack([time, cx, cy], axis=1))
    elif args.mode == 'full':
        for idx, locate in enumerate(locates[start_time:end_time]):
            loc = np.load(locate)
            cx, cy, mass = loc['cx'], loc['cy'], loc['mass']
            time = np.ones_like(cx) * idx
            queries.append(np.stack([time, cx, cy], axis=1))
            masses.append(mass)
    queries = np.concatenate(queries, axis=0)
    queries = torch.from_numpy(queries).float().to(device)
    with torch.no_grad():
        pred_tracks, pred_visibility = model(video, queries=queries[None], backward_tracking=True)
    pred_tracks = pred_tracks[0].cpu().numpy()
    pred_visibility = pred_visibility[0].cpu().numpy()
    count = pred_visibility.sum(axis=0)
    count_thresh = int(pred_visibility.shape[0] * args.vis_threshold)
    valid = count > count_thresh
    pred_tracks = pred_tracks[:, valid]
    pred_visibility = pred_visibility[:, valid]
    masses = np.concatenate(masses, axis=0)[valid]
    np.savez_compressed(os.path.join(args.data_path, 'tracks.npz'), tracks=pred_tracks, visibility=pred_visibility, masses=masses)
    
    os.makedirs(os.path.join(args.data_path, 'sparse'), exist_ok=True)
    image_names = [x for x in images[start_time:end_time]]
    
    pairs = [x for x in combinations([os.path.basename(x) for x in image_names], 2)]
    with open(os.path.join(args.data_path, 'sparse', 'pairs.txt'), 'w') as f:
        for pair in pairs:
            f.write(f"{pair[0]} {pair[1]}\n")

    for idx, image_name in enumerate(image_names):
        img = cv2.imread(image_name)
        name = os.path.basename(image_name)
        with h5py.File(os.path.join(args.data_path, 'sparse', 'keypoints.h5'), 'a', libver='latest') as fd:
            if name in fd:
                del fd[name]
            grp = fd.create_group(name)
            grp.create_dataset('image_size', data=img.shape[:2][::-1])
            grp.create_dataset('keypoints', data=pred_tracks[idx])
            grp.create_dataset('scores', data=pred_visibility[idx].astype('float32'))
        
        with h5py.File(os.path.join(args.data_path, 'sparse', 'matches.h5'), 'a', libver='latest') as fd:
            if name in fd:
                del fd[name]
            
            fd.create_dataset(name, data=img)

    video = video[0].permute(0, 2, 3, 1).cpu().numpy()
    
    gt_images = []
    # os.makedirs(os.path.join(args.data_path, 'gt_tracks'), exist_ok=True)
    for i in tqdm(range(duratoin)):
        img = video[i].astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        if gt_tracks is not None and gt_visibility is not None:
            track = gt_tracks[i]
            vis = gt_visibility[i]
            for j in range(track.shape[0]):
                if vis[j] > 0:
                    cv2.putText(img, f'{j}', (int(track[j, 0]), int(track[j, 1])), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 1)
        gt_images.append(img)
        # cv2.imwrite(os.path.join(args.data_path, 'gt_tracks', f'{i:06}.jpg'), img)
    
    N = pred_visibility.shape[1]
    # cmap = discrete_cmap(N)
    cmap = mpl.colormaps['gist_rainbow']
    # indices = torch.randperm(N)
    # pred_tracks = pred_tracks[:, indices]
    # pred_visibility = pred_visibility[:, indices]
    pred_images = []
    # os.makedirs(os.path.join(args.data_path, 'pred_tracks'), exist_ok=True)
    for i in tqdm(range(duratoin)):
        img = video[i].astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        track = pred_tracks[i]
        vis = pred_visibility[i]
        for j in range(track.shape[0]):
            color = (np.array((cmap(j/N)))[:3] * 255.0).astype('uint8').tolist()
            if vis[j] > 0:
                cv2.circle(img, (int(track[j, 0]), int(track[j, 1])), 2, color, thickness=-1)
        pred_images.append(img)

    if os.path.exists(os.path.join(args.data_path, 'tracks')):
        shutil.rmtree(os.path.join(args.data_path, 'tracks'))
    os.makedirs(os.path.join(args.data_path, 'tracks'), exist_ok=True)
    for i, (gt, pred) in enumerate(zip(gt_images, pred_images)):
        img = np.hstack([gt, pred])
        cv2.imwrite(os.path.join(args.data_path, 'tracks', f'{i:06}.jpg'), img)

    os.chdir(os.path.join(args.data_path))
    subprocess.call(f"ffmpeg -y -framerate 10 -pattern_type glob -i 'tracks/*.jpg' -c:v h264 tracking_s{start_time}_e{end_time}.mp4", shell=True)
    # subprocess.call("ffmpeg -y -framerate 10 -pattern_type glob -i 'pred_tracks/*.jpg' -c:v copy pred_tracking.mp4", shell=True)

    # vis = Visualizer(save_dir=os.path.join(args.data_path), pad_value=0, fps=5, linewidth=1, tracks_leave_trace=5, mode='cool', show_first_frame=5)
    # vis.visualize(video=video, tracks=pred_tracks, visibility=pred_visibility, filename=f'gt_tracking')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str)
    # mannual
    parser.add_argument('--mannual_scale', type=int, default=2)
    parser.add_argument('--mannual_path', type=str, default='2d.pkl')
    # track
    parser.add_argument('--start_time', type=int, default=0)
    parser.add_argument('--end_time', type=int, default=None)
    parser.add_argument('--mode', type=str, default='full', choices=['full', 'only_first'])
    parser.add_argument('--vis_threshold', type=float, default=0.75)

    args = parser.parse_args()
    main(args)
