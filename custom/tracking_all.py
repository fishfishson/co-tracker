import os
import argparse
import subprocess


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str)
    args = parser.parse_args()
    if 'YW' in args.data_path:
        cmd = ['python3', 'tracking.py', '--data_path', f'{args.data_path}',
               '--start_time', '0', '--duration', '200', '--mode', 'full', '--vis_threshold', '0.1']
        subprocess.call(cmd)
    else:
        for seq in os.listdir(args.data_path):
            for dirs in os.listdir(os.path.join(args.data_path, seq)):
                if dirs.startswith('Before') or dirs.startswith('After'):
                    print(os.path.join(args.data_path, seq, dirs))
                    cmd = ['python3', 'tracking.py', '--data_path', f'{os.path.join(args.data_path, seq, dirs)}', 
                           '--start_time', '0', '--duration', '200', '--mode', 'full', '--vis_threshold', '0.1']
                    subprocess.call(cmd)
