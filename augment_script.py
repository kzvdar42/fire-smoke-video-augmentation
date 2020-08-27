import os
from glob import glob
import numpy as np
import json
import argparse

import cv2
from tqdm import tqdm

from augment import Augmentations
from writer import COCO_writer
from bbox_utils import get_scale_ratio, resize_by_max_side


def process_video(in_video_path, augmentations, out_path,
                  writer=None, show_debug=False):
    in_stream = cv2.VideoCapture(in_video_path)

    # Create writer
    frame_width = int(in_stream.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(in_stream.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_rate = int(in_stream.get(cv2.CAP_PROP_FPS))
    total_frames = int(in_stream.get(cv2.CAP_PROP_FRAME_COUNT))
    fourcc = cv2.VideoWriter_fourcc(*'MPEG')
    out_stream = cv2.VideoWriter(out_path, fourcc, frame_rate, (frame_width, frame_height))

    pbar = tqdm(total=total_frames)
    frame_num = 0
    try:
        while in_stream.isOpened():
            _, frame = in_stream.read()
            if frame is None:
                print("No image in the stream, stopping.")
                break

            image_name = f'image_{10:06d}.jpg'
            if writer:
                writer.add_frame(*frame.shape[:2], image_name)
            for augment in augmentations:
                frame = augment.augment(frame, frame_num, writer)
            out_stream.write(frame)
            pbar.update(1)
            frame_num += 1

            if show_debug:
                frame = cv2.resize(frame, (1280, 720))
                cv2.imshow('annotated', frame)
                # Exit on key `q`.
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
    except KeyboardInterrupt:
        print('Exited.')
    finally:
        # Close streams.
        in_stream.release()
        out_stream.release()
        cv2.destroyAllWindows()


def get_coco_writer():
    return COCO_writer([
        {
            'name': 'Fire',
            'supercategory': '',
            'id': 2,
        },
        {
            'name': 'Smoke',
            'supercategory': '',
            'id': 4,
        },
    ])


def process_images(image_paths, augmentations, out_path,
                   writer=None, show_debug=False):
    pbar = tqdm(total=len(image_paths))
    try:
        for image_num, image_path in enumerate(image_paths):
            image_name = os.path.split(image_path)[1]
            image = cv2.imread(image_path)
            # Write image info.
            if writer:
                writer.add_frame(*image.shape[:2], image_name)
            # Augment
            for augment in augmentations:
                image = augment.augment(image, image_num, writer)

            # Write result
            out_image_path = os.path.join(out_path, image_name)
            cv2.imwrite(out_image_path, image)
            pbar.update(1)

            if show_debug:
                image = cv2.resize(image, (1280, 720))
                cv2.imshow('annotated', image)
                # Exit on key `q`.
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
    except KeyboardInterrupt:
        print('Exited.')
    finally:
        # Close streams.
        cv2.destroyAllWindows()


def get_args():
    parser = argparse.ArgumentParser(description='Video augmentation.')

    parser.add_argument('in_path', help='path to the input')
    parser.add_argument('out_path', help='path for the output images')
    parser.add_argument('--video', action='store_true',
                        help='flag for video input')
    parser.add_argument('--show_debug', action='store_true',
                        help='show debug window')
    parser.add_argument('--e_png_path', default=None,
                        help='path for the png effects')
    parser.add_argument('--e_mov_path', default=None,
                        help='path for the mov effects')
    parser.add_argument('--write_annotations',
                        action='store_true', help='Write the coco annotations')
    parser.add_argument('--e_config', default=None,
                        help='path to the config file for augmentations')
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    coco_writer = get_coco_writer() if args.write_annotations else None

    # Get path for effects
    e_png = glob(os.path.join(args.e_png_path, '*.png')) if args.e_png_path else []
    e_mov = glob(os.path.join(args.e_mov_path, '*.webm'))  if args.e_mov_path else []

    # Augmentations
    augment = Augmentations(
        e_png,
        e_mov,
        config_path=args.e_config,
    )

    augmentations = [
        augment
    ]

    if args.video:
        os.makedirs(os.path.split(args.out_path)[0], exist_ok=True)
        process_video(args.in_path, augmentations, args.out_path, coco_writer, args.show_debug)
    else:
        os.makedirs(args.out_path, exist_ok=True)
        images = glob(os.path.join(args.in_path, '*.jpg'))
        process_images(images, augmentations, args.out_path, coco_writer, args.show_debug)

    # Write annotations.
    if args.write_annotations:
        annot_out_path = os.path.join(os.path.split(args.out_path)[0], 'annotations', 'instances_default.json')
        os.makedirs(os.path.split(annot_out_path)[0], exist_ok=True)
        coco_writer.write_result(annot_out_path)
