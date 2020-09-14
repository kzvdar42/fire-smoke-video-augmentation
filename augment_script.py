import os
import shutil
from glob import glob
import numpy as np
import json
import argparse
import threading
import traceback
from dataclasses import dataclass

import cv2
from tqdm import tqdm

from augment import Augmentations
from writer import COCO_writer
from reader import VideoEffectReader, ImageEffectReader, ThreadsHandler, ThreadedImagesReader
from bbox_utils import get_scale_ratio, resize_by_max_side


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


def write_image_and_test_out(path, image):
    cv2.imwrite(path, image)
    if not (os.path.isfile(path) and os.path.getsize(path)) > 0:
        print(f"[ERROR] Failed to save augmented image to {path}")


def process_and_write_image(image, augmentations, writer, image_num, out_ipath, write_debug):
        image, debug_image = process_image(image, augmentations, writer, image_num)
        out_img = debug_image if write_debug else image
        thread = threading.Thread(target=write_image_and_test_out, args=(out_ipath, out_img))
        thread.start()
        return debug_image, thread


def draw_debug(debug_frame):
    debug_frame = cv2.resize(debug_frame, (1280, 720))
    cv2.imshow('Debug', debug_frame)
    return cv2.waitKey(1) & 0xFF == ord('q')


def process_video(in_video_path, augmentations, out_path,
                  writer=None, show_debug=False, write_debug=False):
    in_stream = cv2.VideoCapture(in_video_path)
    is_exit = False

    # Create writer
    frame_width = int(in_stream.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(in_stream.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_rate = int(in_stream.get(cv2.CAP_PROP_FPS))
    total_frames = int(in_stream.get(cv2.CAP_PROP_FRAME_COUNT))
    fourcc = cv2.VideoWriter_fourcc(*'MPEG')

    pbar = tqdm(total=total_frames, desc=f'Processing {in_video_path}')
    image_num = None
    frame_num = 0
    write_threads = ThreadsHandler()
    try:
        while in_stream.isOpened():
            _, image = in_stream.read()
            if image is None:
                tqdm.write("No image in the stream, stopping.")
                break

            if writer:
                image_num, image_name = writer.add_frame(*image.shape[:2])
            else:
                frame_num += 1
                image_name = f'{frame_num:0>7}.jpg'
            out_ipath = os.path.join(out_path, image_name)
            debug_image, thread = process_and_write_image(
                image, augmentations, writer,
                image_num, out_ipath, write_debug
            )
            write_threads.append(thread)
            pbar.update(1)

            if show_debug and draw_debug(debug_frame):
                break
    except KeyboardInterrupt:
        tqdm.write('Exited.')
        is_exit = True
    finally:
        # Close streams.
        pbar.close()
        tqdm.write('Closing in_stream')
        in_stream.release()
        tqdm.write('Closed')
        tqdm.write('Closing out streams')
        write_threads.join_threads()
        cv2.destroyAllWindows()
        return is_exit


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
    is_exit = False
    image_reader = ThreadedImagesReader(image_paths, buffer_size=buffer_size)
    pbar = tqdm(total=len(image_paths))
    image_num = None
    write_threads = ThreadsHandler()
    try:
        for image_path, image in image_reader:
            image_name = os.path.split(image_path)[1]
            out_ipath = os.path.join(out_path, image_name)
            if writer:
                image_num, _ = writer.add_frame(*image.shape[:2], out_ipath)
            debug_image, thread = process_and_write_image(
                image, augmentations, writer,
                image_num, out_ipath, write_debug
            )
            write_threads.append(thread)
            pbar.update(1)

            if show_debug and draw_debug(debug_image):
                break
    except KeyboardInterrupt:
        tqdm.write('Exited.')
        is_exit = True
    except Exception:
        tqdm.write('Error!')
        traceback.print_exc()
    finally:
        # Close streams.
        pbar.close()
        write_threads.join_threads()
        cv2.destroyAllWindows()
        return is_exit

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
    parser.add_argument('--skip_annotated', action='store_true',
                        help='skip annotated files')
    parser.add_argument('--probability', default=None,
                        help='probability of choosing this kind of effects. number in format `1,2,3,4`')
    args = parser.parse_args()

    args.kwargs = []
    if args.use_alpha:
        args.kwargs.append(['use_alpha'] + [bool(int(a)) for a in args.use_alpha.split(',')])
    if args.preload:
        args.kwargs.append(['preload'] + [bool(int(a)) for a in args.preload.split(',')])
    if args.probability:
        args.kwargs.append(['probability'] + [int(a) for a in args.probability.split(',')])
    args.kwargs = list(zip(*args.kwargs))
    return args

def create_folder(path, clean_out):
    try:
        os.makedirs(path)
    except OSError:
        if clean_out:
            clean_folder_content(path)


def get_subfolders_with_files(path, file_ext):
    for dp, dn, fn in os.walk(path):
        file_paths = [os.path.join(dp, f) for f in fn if f.endswith(file_ext)]
        if len(file_paths):
            if file_ext in video_exts:
                for file_path in file_paths:
                    yield [file_path]
            else:
                yield file_paths


if __name__ == "__main__":
    args = get_args()
    coco_writer = None
    e_video_exts = ['webm']
    video_exts = ['mp4', 'avi']
    image_exts = ['jpg', 'png']
    e_kwargs = [dict() for _ in range(len(args.e_paths))]
    if args.kwargs:
        keys = args.kwargs[0]
        for i, values in enumerate(args.kwargs[1:len(e_kwargs) + 1]):
            e_kwargs[i] = {k:v for k, v in zip(keys, values)}

    print('OpenCV is optimized:', cv2.useOptimized())
    if not args.write_annotations:
        print('Not writing annotaions!')

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
    subfolders = list(get_subfolders_with_files(args.in_path, args.in_extention))
    if len(subfolders) == 0:
        print(f'No files with ext `.{args.in_extention}` found!')
        exit(0)
    fold_pbar = tqdm(subfolders)
    for file_paths in fold_pbar:
        file_paths.sort(key=sort_by_digits_in_name)
        folder_path = os.path.relpath(os.path.split(file_paths[0])[0], args.in_path)
        fold_pbar.set_description(f'Processing {folder_path}')
        if os.path.split(folder_path)[1] == 'images':
            folder_path = os.path.split(folder_path)[0]
        first_file_name = os.path.split(file_paths[0])[1]
        out_path = os.path.join(args.out_path, folder_path)
        if args.in_extention in video_exts:
            out_path = os.path.join(out_path, first_file_name)
        annot_out_path = os.path.join(out_path, 'annotations', 'instances_default.json')
        # Skip already annotated
        if args.skip_annotated and os.path.isfile(annot_out_path):
            tqdm.write(f"Skipping {annot_out_path} as it's already annotated")
            continue
        if args.write_annotations:
            coco_writer = get_coco_writer()
        create_folder(out_path, args.clean_out)
        if args.in_extention in video_exts:
            assert len(file_paths) == 1, "Input for video is one file_path!"
            file_paths = file_paths[0]
            process_fn = process_video
        elif args.in_extention in image_exts:
            process_fn = process_images
        else:
            raise ValueError(f'Unsupported extention! ({args.in_extention})')
        data_out_path = os.path.join(out_path, 'images')
        create_folder(data_out_path, args.clean_out)
        is_exit = process_fn(file_paths, augmentations, data_out_path,
                             coco_writer, args.show_debug, args.write_debug)
        # Write annotations
        if args.write_annotations:
            create_folder(os.path.split(annot_out_path)[0], args.clean_out)
            tqdm.write(f'Writing annotations to {annot_out_path}')
            coco_writer.write_result(annot_out_path)
        if is_exit:
            break
    fold_pbar.close()
