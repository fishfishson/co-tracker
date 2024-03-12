import os
import argparse
import subprocess


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str)
    args = parser.parse_args()
    if 'YW' in args.data_path:
        cmd = ['python3', 'detection.py', '--data_path', f'{args.data_path}']
        subprocess.call(cmd)
    else:
        for seq in os.listdir(args.data_path):
            for dirs in os.listdir(os.path.join(args.data_path, seq)):
                if dirs.startswith('Before') or dirs.startswith('After'):
                    print(os.path.join(args.data_path, seq, dirs))
                    cmd = ['python3', 'detection.py', '--data_path', f'{os.path.join(args.data_path, seq, dirs)}']
                    subprocess.call(cmd)
