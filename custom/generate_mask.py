import numpy as np
import cv2
import os
import argparse


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Generate mask for image')
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--xy', nargs=2, type=int, default=None)
    parser.add_argument('--wh', nargs=2, type=int, default=None)
    args = parser.parse_args()

    # Read image
    image = cv2.imread(os.path.join(args.data_path, 'images', '000000.jpg'))
    mask = np.ones_like(image[..., 0]) * 255.0
    x = args.xy[0]
    y = args.xy[1]
    mask[x:, y:] = 0
    cv2.imwrite(os.path.join(args.data_path, 'mask.jpg'), mask.astype('uint8'))