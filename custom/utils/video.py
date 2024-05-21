import os


def generate_video(result_str: str,
                   output: str,
                   verbose: bool = False,
                   fps: int = 30,
                   crf: int = 17,
                   cqv: int = 19,
                   lookahead: int = 20,
                   hwaccel: str = 'cuda',
                   preset: str = 'p7',
                   tag: str = 'hvc1',
                   vcodec: str = 'hevc_nvenc',
                   pix_fmt: str = 'yuv420p',  # chrome friendly
                   ):
    cmd = [
        'ffmpeg',
        '-hwaccel', hwaccel,
    ] + ([
        '-hide_banner',
        '-loglevel', 'error',
    ] if not verbose else []) + ([
        '-framerate', fps,
    ] if fps > 0 else []) + ([
        '-f', 'image2',
        '-pattern_type', 'glob',
    ] if '*' in result_str else []) + ([
        '-r', fps,
    ] if fps > 0 else []) + [
        '-nostdin',  # otherwise you cannot chain commands together
        '-y',
        '-i', result_str,
        '-c:v', vcodec,
        '-preset', preset,
        '-cq:v', cqv,
        '-rc:v', 'vbr',
        '-tag:v', tag,
        '-crf', crf,
        '-pix_fmt', pix_fmt,
        '-rc-lookahead', lookahead,
        '-vf', '"pad=ceil(iw/2)*2:ceil(ih/2)*2"',  # avoid yuv420p odd number bug
        output,
    ]
    cmd = ' '.join(list(map(str, cmd)))
    print(f'running ffmpeg with cmd: {cmd}')
    os.system(cmd)
    return output