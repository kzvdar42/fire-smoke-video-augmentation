import os
import subprocess


def convert_mov2webm(input_path, out_path, y=False):
    print(f'Converting {filename}')
    arguments = [
        'ffmpeg',
        '-i',
        input_path,
        '-c:v',
        'libvpx',
        *'-pix_fmt yuva420p -auto-alt-ref 0 -filter:v scale=720:-1'.split(' '),
        out_path,
    ]
    if y:
        arguments.append('-y')
    command = subprocess.Popen(arguments, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, universal_newlines=True)
    # Print the command and results.
    print(' '.join(command.args))
    output = command.communicate()[0]
    print(output)


def extract_alpha(input_path, y=False):
    input_path = os.path.abspath(input_path)
    out_path = os.path.splitext(input_path)[0] + '_alpha.mp4'
    print(f'Extracting alpha from {input_path} to {out_path}')
    arguments = [
        'ffmpeg',
        '-vcodec',
        'libvpx',
        '-i',
        input_path,
        '-vf',
        'alphaextract',
        out_path,
    ]
    if y:
        arguments.append('-y')
    command = subprocess.Popen(arguments, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, universal_newlines=True)
    # Print the command and results.
    print(' '.join(command.args))
    output = command.communicate()[0]
    print(output)
