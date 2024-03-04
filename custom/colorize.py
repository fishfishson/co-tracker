import os
import argparse
import pims
import numpy as np
import cv2
from tqdm import tqdm
import trackpy as tp
from matplotlib import pyplot as plt
import subprocess


def main(args):
    dir_path = os.path.dirname(args.data_path)
    videos = pims.open(args.data_path)
    assert len(videos.shape) == 3
    n_frames = videos.shape[0]
    images = np.array(videos)
    images = (images - images.min()) / (images.max() - images.min())
    for i in tqdm(range(n_frames)):
        # preprocess
        img = (images[i] * 255).astype(np.uint8)
        # img = cv2.GaussianBlur(img, (3, 3), 0)
        # img = cv2.equalizeHist(img)
        img = cv2.resize(img, (img.shape[1] * args.scale, img.shape[0] * args.scale), cv2.INTER_LANCZOS4)
        img = cv2.applyColorMap(img, cv2.COLORMAP_VIRIDIS)
        os.makedirs(os.path.join(dir_path, 'images'), exist_ok=True)
        cv2.imwrite(os.path.join(dir_path, 'images', f'{i:06}.jpg'), img)
    os.chdir(os.path.join(dir_path))
    subprocess.call("ffmpeg -y -framerate 30 -pattern_type glob -i 'images/*.jpg' -pix_fmt yuv420p color.mp4", shell=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--scale', type=int, default=2)
    args = parser.parse_args()
    main(args)