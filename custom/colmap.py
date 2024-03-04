import os
from os.path import join
import argparse
from pathlib import Path
import pycolmap
import subprocess


def main(args):
    output_path = Path(args.data_root)
    database_path = output_path / "database.db"
    images_path = output_path / args.image_dir
    sparse_path = output_path / "sparse"
    sparse_path.mkdir(exist_ok=True, parents=True)

    # pycolmap.extract_features(database_path, args.image_dir)
    # pycolmap.match_exhaustive(database_path)
    # maps = pycolmap.incremental_mapping(database_path, args.image_dir, output_path)
    # maps[0].write(output_path)

    cmd = ['colmap', 'feature_extractor', '--database_path', database_path, '--image_path', images_path, 
        #    '--ImageReader.camera_model', 'OPENCV', 
           '--ImageReader.single_camera', '1']
    subprocess.call(cmd)
    cmd = ['colmap', 'sequential_matcher', '--database_path', database_path,
           '--SequentialMatching.overlap', '10',]
    subprocess.call(cmd)
    cmd = ['colmap', 'mapper', '--database_path', database_path, '--image_path', images_path, '--output_path', sparse_path,
           '--Mapper.tri_complete_max_reproj_error', '8', 
           '--Mapper.tri_merge_max_reproj_error', '8',
           '--Mapper.filter_max_reproj_error', '8',
           '--Mapper.init_max_error', '16']
    subprocess.call(cmd)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--image_dir", type=str, default="images")
    args = parser.parse_args()
    main(args)