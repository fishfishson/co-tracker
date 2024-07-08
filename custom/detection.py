import os
import argparse
import pims
import numpy as np
import cv2
from tqdm import tqdm
import trackpy as tp
from matplotlib import pyplot as plt
from glob import glob
import subprocess
import h5py


def main(args):
    os.makedirs(os.path.join(args.data_path, 'locate'), exist_ok=True)
    os.makedirs(os.path.join(args.data_path, 'images'), exist_ok=True)
    os.makedirs(os.path.join(args.data_path, 'sparse'), exist_ok=True)

    video_path = glob(os.path.join(args.data_path, '*.tif'))
    assert len(video_path) == 1
    video = pims.open(video_path[0])
    assert len(video.shape) == 3
    n_frames = video.shape[0]
    images = np.array(video)
    images = (images - images.min()) / (images.max() - images.min())
    # images = images / images.max()

    for i in tqdm(range(n_frames)):
        img = cv2.resize(images[i], (images[i].shape[1] * args.scale, images[i].shape[0] * args.scale), cv2.INTER_LINEAR)
        vis = (img * 255).astype(np.uint8)
        vis = cv2.GaussianBlur(vis, (3, 3), 1)
        # img = cv2.equalizeHist(img)
        vis = cv2.applyColorMap(vis, cv2.COLORMAP_VIRIDIS)
        cv2.imwrite(os.path.join(args.data_path, 'images', f'{i:06}.jpg'), vis)

        ret = tp.locate(img, args.diameter, minmass=args.minmass, invert=False)
        cx, cy = np.array(ret['x']), np.array(ret['y'])
        out = {'cx': cx, 'cy': cy, 'size': np.array(ret['size']), 'ecc': np.array(ret['ecc']), 'mass': np.array(ret['mass'])}
        np.savez_compressed(os.path.join(args.data_path, 'locate', f'{i:06}.npz'), **out)
        for ii in range(len(cx)):
            vis_locate = cv2.circle(vis, (int(cx[ii]), int(cy[ii])), 8, (0, 0, 255), 2)
        cv2.imwrite(os.path.join(args.data_path, 'locate', f'{i:06}.jpg'), vis_locate)

        # with h5py.File(os.path.join(args.data_path, 'sparse', 'keypoints.h5'), 'a', libver='latest') as fd:
        #     name = f'{i:06}.jpg'
        #     if name in fd:
        #         del fd[name]
        #     grp = fd.create_group(name)
        #     grp.create_dataset('image_size', data=vis.shape[:2][::-1])
        #     grp.create_dataset('keypoints', data=np.array([cx, cy]).reshape(-1, 2))
            
    os.chdir(os.path.join(args.data_path))
    subprocess.call(f"ffmpeg -y -framerate 30 -pattern_type glob -i 'images/*.jpg' -c:v copy {args.vid_name}", shell=True)
    subprocess.call(f"ffmpeg -y -framerate 30 -pattern_type glob -i 'locate/*.jpg' -c:v copy {args.out_name}", shell=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str)

    # vis
    parser.add_argument('--scale', type=int, default=2)
    parser.add_argument('--vid_name', type=str, default='images.mp4')
    parser.add_argument('--out_name', type=str, default='locate.mp4')

    # location
    parser.add_argument('--diameter', type=int, default=11)
    parser.add_argument('--minmass', type=float, default=1.0)

    args = parser.parse_args()
    main(args)