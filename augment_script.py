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
from pycocotools.coco import COCO

from core.augment import Augmentations, AugmentationConfig
from core.writer import get_coco_writer
from core.reader import VideoEffectReader, ImageEffectReader, ThreadedImagesReader, ThreadPoolHelper
from utils.bbox_utils import get_scale_ratio, resize_by_max_side, convert_xywh_xyxy


def get_int_from_str(s):
    # Extract only digit characters
    s = ''.join([ch for ch in s if ch.isdigit()])
    return int(s if len(s) else '0')


def sort_by_digits_in_name(path):
    return get_int_from_str(os.path.splitext(os.path.split(path)[1])[0])


def process_image(frame, augmentations, f_box_cats=None, writer=None, frame_num=None):
    if f_box_cats is None:
        f_box_cats = ([], [])
    for augment in augmentations:
        frame, debug_frame = augment(
            frame, f_box_cats=f_box_cats, writer=writer, frame_num=frame_num)
    return frame, debug_frame if debug_frame is not None else frame


def write_image_and_test_out(path, image):
    try:
        cv2.imwrite(path, image)
        if not (os.path.isfile(path) and os.path.getsize(path)) > 0:
            raise ValueError
    except:
        return (False, f"[ERROR] Failed to save augmented image to {path}")
    return (True, None)


def draw_debug(debug_frame):
    debug_frame = cv2.resize(debug_frame, (1280, 720))
    cv2.imshow('Debug', debug_frame)
    return cv2.waitKey(0) & 0xFF == ord('q')


def process_video(in_video_path, augmentations, out_path,
                  writer=None, show_debug=False, write_debug=False, in_annots=None, max_workers=None):
    if in_annots is not None:
        raise NotImplementedError('Not yet implemented for video sources.')
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
    write_pool = ThreadPoolHelper(max_workers=max_workers)
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
            image, debug_image = process_image(image, augmentations,
                                               writer=writer, frame_num=image_num)
            image = debug_image if write_debug else image
            write_pool.submit(write_image_and_test_out, out_ipath, image)
            pbar.update(1)

            if show_debug and draw_debug(debug_image):
                raise KeyboardInterrupt()
    except KeyboardInterrupt:
        tqdm.write('Exited.')
        is_exit = True
    except Exception:
        tqdm.write('Error!')
        traceback.print_exc()
    finally:
        # Close streams.
        pbar.close()
        tqdm.write('Closing streams...')
        in_stream.release()
        write_pool.shutdown(wait=True)
        tqdm.write('Closed.')
        cv2.destroyAllWindows()
        return is_exit


def process_images(image_paths, augmentations, out_path,
                   writer=None, show_debug=False, write_debug=False,
                   in_annots=None, max_workers=None):
    is_exit = False
    image_reader = ThreadedImagesReader(
        image_paths,
        buffer_size=32,
        max_workers=max_workers
    )
    write_pool = ThreadPoolHelper(max_workers=max_workers)
    pbar = tqdm(total=len(image_paths))
    futures = []
    skipped_counter = 0
    image_num = None
    f_box_cats = None
    try:
        for i, (image_path, image) in enumerate(image_reader):
            pbar.set_description(f'Skipped {skipped_counter} - {image_path}')
            image_name = os.path.split(image_path)[1]
            out_ipath = os.path.join(out_path, image_name)
            if in_annots:
                height, width = image.shape[:2]
                rel_path = os.path.relpath(image_path, in_annots.root_path).replace('\\', '/')
                # Get annotations for the frame, otherwise skip the image
                try:
                    f_ann_ids = in_annots.getAnnIds(imgIds=in_annots.path2img_id[rel_path], iscrowd=None)
                    f_objects = in_annots.loadAnns(f_ann_ids)
                    f_boxes = [convert_xywh_xyxy(obj['bbox'], width, height) for obj in f_objects]
                    f_cats = [in_annots.cats[obj['category_id']]['name'] for obj in f_objects]
                    f_box_cats = (f_boxes, f_cats)
                except KeyError:
                    skipped_counter += 1
                    # pbar.update(1)
                    continue
            if writer:
                image_num, _ = writer.add_frame(*image.shape[:2], out_ipath)
            image, debug_image = process_image(image, augmentations, f_box_cats=f_box_cats,
                                               writer=writer, frame_num=image_num)
            image = debug_image if write_debug else image
            write_pool.submit(write_image_and_test_out, out_ipath, image)
            pbar.update(1)

            if show_debug and draw_debug(debug_image):
                raise KeyboardInterrupt()
    except KeyboardInterrupt:
        tqdm.write('Exited.')
        is_exit = True
    except Exception:
        tqdm.write('Error!')
        traceback.print_exc()
    finally:
        # Close streams.
        pbar.close()
        tqdm.write('Closing streams...')
        write_pool.shutdown(wait=True)
        tqdm.write('Closed.')
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
    parser.add_argument('--in_annotations', default=None,
                        help='in files coco annotation file')
    parser.add_argument('--show_debug', action='store_true',
                        help='show debug window')
    parser.add_argument('--write_debug', action='store_true',
                        help='write debug info to output')
    parser.add_argument('--skip_annotations',
                        action='store_true', help='Do not write the coco annotations')
    parser.add_argument('--clean_out',
                        action='store_true', help='Clean out folder before starting')
    parser.add_argument('--skip_augmented', action='store_true',
                        help='skip augmented files')
    parser.add_argument('--max_workers', default=None,
                        type=int, help='number of threads to use')
    parser.add_argument('--e_configs', default=None, nargs='+',
                        help='paths to the config files for augmentations')
    parser.add_argument('--e_paths', default=None, nargs='+',
                        help='path for the effects')
    parser.add_argument('--use_alpha', default=None,
                        help='use alpha channel. binary in format `0,1,0,1`')
    parser.add_argument('--preload', default=None,
                        help='preload image effects. binary in format `0,1,0,1`')
    parser.add_argument('--probability', default=None,
                        help='probability of choosing this kind of effects. number in format `1,2,3,4`')
    parser.add_argument('--min_n_objects', type=int, default=1,
                        help='min amount of objects in one frame')
    parser.add_argument('--max_n_objects', type=int, default=1,
                        help='max amount of objects in one frame')
    parser.add_argument('--gen_prob', type=int, default=1,
                        help='Generation probability of 1 object: 1 / gen_prob')
    parser.add_argument('--next_gen_prob', type=int, default=0,
                        help="Gen prob of (n + 1)'th object: 1 / (gen_prob + (n - 1) * next_gen_prob)")
    parser.add_argument('--debug_level', type=int, default=0,
                        help="Debug level")

    args = parser.parse_args()

    args.kwargs = []
    if args.use_alpha:
        args.kwargs.append(['use_alpha']
                    + [bool(int(a)) for a in args.use_alpha.split(',')])
    if args.preload:
        args.kwargs.append(['preload']
                    + [bool(int(a)) for a in args.preload.split(',')])
    if args.probability:
        args.kwargs.append(['probability']
                    + [int(a) for a in args.probability.split(',')])
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
    in_annots = None
    e_video_exts = ['webm']
    video_exts = ['mp4', 'avi']
    image_exts = ['jpg', 'png']
    e_kwargs = [dict() for _ in range(len(args.e_paths))]
    if args.kwargs:
        keys = args.kwargs[0]
        for i, values in enumerate(args.kwargs[1:len(e_kwargs) + 1]):
            e_kwargs[i] = {k: v for k, v in zip(keys, values)}

    print('OpenCV is optimized:', cv2.useOptimized())
    if args.skip_annotations:
        print('Not writing annotaions!')

    if args.in_annotations is not None:
        in_annots = COCO(args.in_annotations)
        print('Building path to img id mapping...')
        path2img_id = dict()
        for img in in_annots.imgs.values():
            path2img_id[img['file_name']] = img['id']
        in_annots.__dict__['root_path'] = args.in_path
        in_annots.__dict__['path2img_id'] = path2img_id

    # Get path for effects
    e_readers, e_cfgs = [], []
    for i, (path, kwargs) in enumerate(zip(args.e_paths, e_kwargs)):
        files = [os.path.join(path, f) for f in os.listdir(path)]
        files = [f for f in files if os.path.isfile(f)]
        image_files, video_files = [], []
        for file_path in files:
            ext = os.path.splitext(file_path)[1][1:]
            if ext in image_exts:
                image_files.append(file_path)
            elif ext in e_video_exts:
                video_files.append(file_path)

        config_path = args.e_configs[i] if len(args.e_configs) > i else args.e_configs[0]
        config_path = os.path.join('configs', config_path + '.yaml')
        cfg = AugmentationConfig(config_path)

        if len(image_files):
            e_readers.append(ImageEffectReader(image_files, **kwargs))
            e_cfgs.append(cfg)
        if len(video_files):
            e_readers.append(VideoEffectReader(video_files, **kwargs))
            e_cfgs.append(cfg)

    # Augmentations
    augment = Augmentations(
        e_readers,
        configs=e_cfgs,
        min_n_objects=args.min_n_objects,
        max_n_objects=args.max_n_objects,
        debug_level=args.debug_level,
        gen_prob=args.gen_prob,
        next_gen_prob=args.next_gen_prob,
    )

    augmentations = [
        augment
    ]

    # Get the list of subfolders with such files
    subfolders = list(get_subfolders_with_files(args.in_path, args.in_extention))
    if len(subfolders) == 0:
        print(f'No files with ext `.{args.in_extention}` found!')
        exit(0)
    # Start augmenting files in each subfolder
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
        if args.skip_augmented and os.path.isfile(annot_out_path):
            tqdm.write(f"Skipping {annot_out_path} as it's already augmented")
            continue
        if not args.skip_annotations:
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
        is_exit = process_fn(
            file_paths, augmentations, data_out_path,
            coco_writer,
            show_debug=args.show_debug,
            write_debug=args.write_debug,
            in_annots=in_annots,
            max_workers=args.max_workers,
        )
        # Write annotations
        if not args.skip_annotations:
            create_folder(os.path.split(annot_out_path)[0], args.clean_out)
            tqdm.write(f'Writing annotations to {annot_out_path}')
            coco_writer.write_result(annot_out_path)
        if is_exit:
            break
    fold_pbar.close()
