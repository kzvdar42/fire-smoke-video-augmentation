import os
import shutil
from glob import glob
import numpy as np
import json
import argparse
import threading
from dataclasses import dataclass

import cv2
from tqdm import tqdm

from augment import Augmentations
from writer import COCO_writer
from reader import VideoEffectReader, ImageEffectReader
from bbox_utils import get_scale_ratio, resize_by_max_side, ImagesReader


def get_int_from_str(s):
  # Extract only digit characters
  s = ''.join([ch for ch in s if ch.isdigit()])
  return int(s if len(s) else '0')

def sort_by_digits_in_name(path):
  return get_int_from_str(os.path.splitext(os.path.split(path)[1])[0])


def process_image(frame, augmentations, writer=None, frame_num=None):
    for augment in augmentations:
        frame, debug_frame = augment.augment(frame, writer, frame_num)
    return frame, debug_frame if debug_frame is not None else frame


def _process_image(image, augmentations, writer, image_num, out_ipath, write_debug):
        image, debug_image = process_image(image, augmentations, writer, image_num)
        out_img = debug_image if write_debug else image
        threading.Thread(target=cv2.imwrite, args=(out_ipath, out_img)).start()
        return debug_image


def draw_debug(debug_frame):
    debug_frame = cv2.resize(debug_frame, (1280, 720))
    cv2.imshow('Debug', debug_frame)
    return cv2.waitKey(1) & 0xFF == ord('q')


def process_video(in_video_path, augmentations, out_path,
                  writer=None, show_debug=False, write_debug=False):
    in_stream = cv2.VideoCapture(in_video_path)

    # Create writer
    frame_width = int(in_stream.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(in_stream.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_rate = int(in_stream.get(cv2.CAP_PROP_FPS))
    total_frames = int(in_stream.get(cv2.CAP_PROP_FRAME_COUNT))
    fourcc = cv2.VideoWriter_fourcc(*'MPEG')
    out_stream = cv2.VideoWriter(out_path, fourcc, frame_rate,
                                 (frame_width, frame_height))

    pbar = tqdm(total=total_frames)
    frame_num = None
    try:
        while in_stream.isOpened():
            _, frame = in_stream.read()
            if frame is None:
                print("No image in the stream, stopping.")
                break

            if writer:
                frame_num, _ = writer.add_frame(*frame.shape[:2])
            frame, debug_frame = process_image(frame, augmentations, writer, frame_num)
            out_stream.write(debug_frame if write_debug else frame)
            pbar.update(1)
            frame_num += 1

            if show_debug and draw_debug(debug_frame):
                break
    except KeyboardInterrupt:
        tqdm.write('Exited.')
    finally:
        # Close streams.
        in_stream.release()
        out_stream.release()
        pbar.close()
        cv2.destroyAllWindows()


def get_coco_writer():
    return COCO_writer([
        {
            'name': 'person',
            'supercategory': '',
            'id': 0,
        },
        {
            'name': 'vehicle',
            'supercategory': '',
            'id': 1,
        },
        {
            'name': 'fire',
            'supercategory': '',
            'id': 2,
        },
        {
            'name': 'animal',
            'supercategory': '',
            'id': 3,
        },
        {
            'name': 'smoke',
            'supercategory': '',
            'id': 4,
        },
    ],
    synonyms={
        'vehicle': ['Car']
    }
    )


def process_images(image_paths, augmentations, out_path,
                   writer=None, show_debug=False, write_debug=False, n_workers=None):
    buffer_size = 32
    image_reader = ImagesReader(image_paths, buffer_size=buffer_size)
    pbar = tqdm(total=len(image_paths))
    # threads = []
    image_num = None
    try:
        for image_path, image in image_reader:
            image_name = os.path.split(image_path)[1]
            out_ipath = os.path.join(out_path, image_name)
            if writer:
                image_num, _ = writer.add_frame(*image.shape[:2], image_name)
            args = (image, augmentations, writer,
                    image_num, out_ipath, write_debug)
            debug_image = _process_image(*args)
            # while len(threads) > n_workers:
            #     threads[0].join()
            #     threads = [thread for thread in threads if thread.is_alive()]
            # thread = threading.Thread(target=_process_image, args=args)
            # thread.start()
            # threads.append(thread)
            pbar.update(1)

            if show_debug and draw_debug(debug_image):
                break
    except KeyboardInterrupt:
        tqdm.write('Exited.')
    finally:
        # Close streams.
        pbar.close()
        cv2.destroyAllWindows()

def clean_folder_content(path):
    for root, dirs, files in os.walk(path):
        for f in files:
            os.unlink(os.path.join(root, f))
        for d in dirs:
            shutil.rmtree(os.path.join(root, d))

def get_default_reader_kwargs():
    return {
        'probability': 1,
        'use_alpha': True,
        'preload': False,
    }


def get_args():
    parser = argparse.ArgumentParser(description='Video augmentation.')

    parser.add_argument('in_path', help='path to the input')
    parser.add_argument('out_path', help='path for the output images')
    parser.add_argument('--in_extention', default='jpg',
                        help='in file extention, support png/jpg/mp4/avi')
    parser.add_argument('--show_debug', action='store_true',
                        help='show debug window')
    parser.add_argument('--write_debug', action='store_true',
                        help='write debug info to output')
    parser.add_argument('--write_annotations',
                        action='store_true', help='Write the coco annotations')
    parser.add_argument('--clean_out',
                        action='store_true', help='Clean out folder before starting')
    parser.add_argument('--n_workers', default=None,
                        type=int, help='number of threads to use')
    parser.add_argument('--e_config', default=None,
                        help='path to the config file for augmentations')
    parser.add_argument('--e_paths', default=None, nargs='+',
                        help='path for the effects')
    parser.add_argument('--use_alpha', default=None,
                        help='use alpha channel. binary in format `0,1,0,1`')
    parser.add_argument('--preload', default=None,
                        help='preload image effects. binary in format `0,1,0,1`')
    parser.add_argument('--probability', default=None,
                        help='probability of choosing this kind of effects. number in format `1,2,3,4`')
    args = parser.parse_args()

    args.kwargs = []
    args.out_path = os.path.join(args.out_path, 'images')
    if args.use_alpha:
        args.kwargs.append(['use_alpha'] + [bool(int(a)) for a in args.use_alpha.split(',')])
    if args.preload:
        args.kwargs.append(['preload'] + [bool(int(a)) for a in args.preload.split(',')])
    if args.probability:
        args.kwargs.append(['probability'] + [int(a) for a in args.probability.split(',')])
    args.kwargs = list(zip(*args.kwargs))
    return args

if __name__ == "__main__":
    args = get_args()
    e_video_exts = ['webm']
    video_exts = ['mp4', 'avi']
    image_exts = ['jpg', 'png']
    e_kwargs = [dict() for _ in range(len(args.e_paths))]
    if args.kwargs:
        keys = args.kwargs[0]
        for i, values in enumerate(args.kwargs[1:len(e_kwargs) + 1]):
            e_kwargs[i] = {k:v for k, v in zip(keys, values)}

    print('OpenCV is optimized:', cv2.useOptimized())
    # sys.setcheckinterval
    if args.write_annotations:
        annot_out_path = os.path.join(os.path.split(args.out_path)[0], 'annotations', 'instances_default.json')
        print(f'Writing annotations to {annot_out_path}')
        coco_writer = get_coco_writer()
    else:
        print('Not writing annotaions!')
        coco_writer = None

    # Get path for effects
    e_readers = []
    for path, kwargs in zip(args.e_paths, e_kwargs):
        files = [os.path.join(path, f) for f in os.listdir(path)]
        files = [f for f in files if os.path.isfile(f)]
        image_files, video_files = [], []
        for file_path in files:
            ext = os.path.splitext(file_path)[1][1:]
            if ext in image_exts:
                image_files.append(file_path)
            elif ext in e_video_exts:
                video_files.append(file_path)
        
        if len(image_files):
            e_readers.append(ImageEffectReader(image_files, **kwargs))
        if len(video_files):
            e_readers.append(VideoEffectReader(video_files, **kwargs))

    # Augmentations
    augment = Augmentations(
        e_readers,
        config_path=args.e_config,
    )

    augmentations = [
        augment
    ]
    
    try:
        os.makedirs(args.out_path)
    except OSError:
        if args.clean_out:
            clean_folder_content(args.out_path)

    if args.in_extention in video_exts:
        out_path = os.path.split(args.out_path)[0]
        out_path = os.path.join(out_path, 'out.mp4')
        process_video(args.in_path, augmentations, out_path,
                      coco_writer, args.show_debug, args.write_debug)
    elif args.in_extention in image_exts:
        image_paths = glob(os.path.join(args.in_path, f'*.{args.in_extention}'))
        image_paths.sort(key=sort_by_digits_in_name)
        process_images(image_paths, augmentations, args.out_path, coco_writer,
                       args.show_debug, args.write_debug, args.n_workers)
    else:
        raise ValueError(f'Unsupported extention! ({args.in_extention})')

    # Write annotations.
    if args.write_annotations:
        os.makedirs(os.path.split(annot_out_path)[0], exist_ok=True)
        coco_writer.write_result(annot_out_path)
