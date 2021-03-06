from collections import defaultdict
from dataclasses import dataclass
import os
from threading import Thread, Lock
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

import cv2
import numpy as np
from pycocotools.coco import COCO
from utils.get_image_info import get_image_size
from utils.bbox_utils import (convert_xywh_xyxy, get_corners, prepare_image)


def get_bboxes_from_obj(obj):
    bbox = convert_xywh_xyxy(obj['bbox'], width, height)


def get_segments_and_cats_from_obj(obj, cats):
    # If no segmentation, get box corners
    if obj['segmentation'] is not None and len(obj['segmentation']) == 0:
        bbox = convert_xywh_xyxy(obj['bbox'], width, height)
        segment = get_corners(np.array(bbox)).reshape(-1, 1, 2)
    else:
        segments = []
        for segment in obj['segmentation']:
            segment = np.array(segment, dtype=np.int32).reshape(-1, 1, 2)
            segment = cv2.approxPolyDP(segment, 3, True)
            # XXX: Collecting all polygons into one. If you want to use them afterwards, change
            segments.extend(segment)
        segments = np.array(segments)
    cat = cats[obj['category_id']]['name']
    return segments, cat


@dataclass(init=False)
class VideoEffectReader:

    probability: int = 1
    use_alpha: bool = True
    ck_start: int = 10
    ck_range: int = 20

    def __init__(self, paths, **kwargs):
        self.paths = paths
        # Override values by kwargs
        self.__dict__.update(kwargs)
        self.total_frames = []
        self.frame_shapes = []
        self.loaded, self.annotations = [], []
        # Load mov effects.
        for mov_path in self.paths:
            cap = cv2.VideoCapture(mov_path)
            # Get width and height
            width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
            height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
            self.frame_shapes.append((height, width))

            # XXX: This should work, but in my case it fails
            # https://github.com/opencv/opencv/pull/13395
            # print(cap.set(cv2.CAP_PROP_CONVERT_RGB, 0))
            # print(cap.get(cv2.CAP_PROP_CONVERT_RGB))

            # Because of that we read alpha channel individually.
            alpha_path = os.path.splitext(mov_path)[0] + '_alpha.mp4'
            alpha_cap = cv2.VideoCapture(alpha_path)
            self.loaded.append((cap, alpha_cap))
            self.total_frames.append(cap.get(7))

            # Load Coco annotations
            file_path, filename = os.path.split(mov_path)
            filename = os.path.splitext(filename)[0]
            annot_path = os.path.join(file_path, 'annotations', filename + '.json')
            if os.path.isfile(annot_path):
                self.annotations.append(COCO(annot_path))
            else:
                self.annotations.append(None)

    def get_frame_shape(self, video_idx):
        return self.frame_shapes[video_idx]

    def get_frame(self, e_info, use_alpha=None, read_annot=False):
        use_alpha = use_alpha if use_alpha is not None else self.use_alpha
        segments, e_cats = [], []
        cap, alpha_cap = self.loaded[e_info.idx]
        cap_pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        if cap_pos != e_info.cur_dur:
            cap.set(cv2.CAP_PROP_POS_FRAMES, e_info.cur_dur)
        e_image = cap.read()[1]
        # If no frame is read, return
        if e_image is None:
            return None, segments, e_cats
        if use_alpha:
            cap_pos = int(alpha_cap.get(cv2.CAP_PROP_POS_FRAMES))
            if cap_pos != e_info.cur_dur:
                alpha_cap.set(cv2.CAP_PROP_POS_FRAMES, e_info.cur_dur)
            e_alpha = alpha_cap.read()[1]
            e_alpha = e_alpha[:, :, :1]
            # rgb_sum = np.expand_dims(np.sum(e_alpha, axis=2), -1)
            # e_alpha = np.clip(rgb_sum, 0, 255, dtype=np.uint8)
        else:
            # Color key black with everything < ck_start = 0
            # and everything > ck_start + ck_range = 255
            # (value - ck_start) / ck_range
            hsv_e_image = cv2.cvtColor(e_image, cv2.COLOR_BGR2HSV)
            v_e_image = hsv_e_image[:, :, 2:3].astype(np.int32)
            e_alpha = np.divide((v_e_image - self.ck_start) * 255, self.ck_range)
            e_alpha = np.clip(e_alpha, 0, 255).astype(np.uint8)
        # Concat with alpha channel
        e_image = np.concatenate((e_image, e_alpha), -1)
        # Get bboxes
        if read_annot:
            annot = self.annotations[e_info.idx]
            if annot is not None:
                height, width = e_image.shape[:2]
                ann_ids = annot.getAnnIds(imgIds=e_info.cur_dur + 1, iscrowd=None)
                for obj in annot.loadAnns(ann_ids):
                    segment, cat = get_segments_and_cats_from_obj(obj, annot.cats)
                    segments.append(segment)
                    e_cats.append(cat)
        return e_image, segments, e_cats


@dataclass(init=False)
class ImageEffectReader:

    probability: int = 1
    annot_type: str = 'coco'
    preload: bool = False

    def __init__(self, paths, **kwargs):
        self.paths = paths
        # Override values by kwargs
        self.__dict__.update(kwargs)
        self.loaded, self.annotations = [], []
        # Load image annotations.
        load_func = {
            'coco': self.load_coco_annotations,
            'csv': self.load_csv_annotations,
            None: lambda: {os.path.split(p)[1]: None for p in paths},
        }[self.annot_type]
        annotations = load_func()
        # Load images.
        for png_path in self.paths:
            if self.preload:
                e_image = self.load_image(png_path)
                self.loaded.append(e_image)
            else:
                if os.path.isfile(png_path):
                    self.loaded.append(png_path)
            png_file_name = os.path.split(png_path)[1]
            self.annotations.append(annotations[png_file_name])

    def get_frame_shape(self, frame_idx):
        e_image = self.loaded[frame_idx]
        if isinstance(e_image, str):
            shape = get_image_size(e_image)
        else:
            shape = e_image.shape[:2]
        return shape

    def load_image(self, path):
        e_image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        e_image = prepare_image(e_image)
        # Apply alpha
        e_image[:, :, :3] = e_image[:, :, :3] * (e_image[:, :, 3:] / 255)
        return e_image

    def load_coco_annotations(self):
        img_folder_path = os.path.split(self.paths[0])[0]
        root_path, img_folder_name = os.path.split(img_folder_path)
        coco = COCO(os.path.join(root_path, f'{img_folder_name}.json'))
        self.cats = coco.cats
        annotations = defaultdict(list)
        for img_id, img_info in coco.imgs.items():
            ann_ids = coco.getAnnIds(imgIds=img_id, iscrowd=None)
            for obj in coco.loadAnns(ann_ids):
                annotations[img_info['file_name']].append(obj)
        return annotations

    def load_csv_annotations(self):
        folder_path = os.path.split(self.paths[0])[0]
        annot_path = os.path.join(folder_path, 'annotations.csv')
        annotations = defaultdict(list)
        if os.path.isfile(annot_path):
            with open(annot_path) as in_file:
                for line in in_file.readlines():
                    category, x1, y1, x2, y2, img_name = line.split(',')[:6]
                    annotations[img_name].append([*map(int, [x1, y1, x2, y2]), category])
            for k, v in annotations.items():
                annotations[k] = np.array(v)
        return annotations

    def get_frame(self, e_info, read_annot):
        e_image = self.loaded[e_info.idx]
        if isinstance(e_image, str):
            e_image = self.load_image(e_image)
        else:
            e_image = e_image.copy()
        segments, e_cats = [], []
        if read_annot:
            if self.annot_type == 'csv':
                segments = get_corners(self.annotations[e_info.idx][:, :4])
                e_cats = self.annotations[e_info.idx][:, 4]
            elif self.annot_type == 'coco':
                for obj in self.annotations[e_info.idx]:
                    segment, cat = get_segments_and_cats_from_obj(obj, self.cats)
                    segments.append(segment)
                    e_cats.append(cat)
            else:
                segments, e_cats = [], []
        return e_image, segments, e_cats


class ThreadPoolHelper:

    def __init__(self, max_workers=None, write_out=True):
        self.pool = ThreadPoolExecutor(max_workers=max_workers)
        self.write_out = write_out
        self.futures = []

    def submit(self, func, *args, **kwargs):
        future = self.pool.submit(func, *args, **kwargs)
        self.futures.append(future)
        self.check_status()
        return future

    def shutdown(self, wait=True):
        return self.pool.shutdown(wait)

    def check_status(self):
        futures, self.futures = self.futures, []
        for future in futures:
            if future.done():
                code, res = future.result()
                # If Error, print it
                if code == False:
                    print(res)
            else:
                self.futures.append(future)


class ThreadedImagesReader:
    """Input image reader."""

    def __init__(self, images, buffer_size=32, max_workers=None):
        self.images = [image.replace('\\', '/') for image in images]
        self.total = len(images)
        self.buffer_size = buffer_size
        self.last_id = -1
        self.pool = ThreadPoolExecutor(max_workers=max_workers)
        self.futures = []
        self._lock = Lock()
        self._fill_buffer()

    def is_open(self):
        return (
            self.__is_open() or
            len(self.futures)
        )

    def __is_open(self):
        return self.last_id < self.total

    def __get_next_id(self):
        with self._lock:
            self.last_id += 1
            return self.last_id

    def __read_to_buffer(self):
        try:
            image_path = self.images[self.__get_next_id()]
        except IndexError:
            return None
        image = cv2.imread(image_path)
        if image is not None:
            return (image_path, image)
        else:
            return "[ERROR] Coudn't read image with path {image_path}"

    def start_new_thread(self):
        self.futures.append(self.pool.submit(self.__read_to_buffer))

    def _fill_buffer(self):
        with self._lock:
            if self.__is_open():
                items_left = self.total - self.last_id
                n_fill = min(self.buffer_size - len(self.futures), items_left)
                for _ in range(n_fill):
                    self.start_new_thread()

    def __iter__(self):
        return self

    def __next__(self):
        if not self.is_open():
            raise StopIteration
        while self.is_open():
            if self.__is_open():
                self.start_new_thread()
            if len(self.futures):
                future = self.futures.pop(0)
                res = future.result()
                if isinstance(res, str):
                    print(res)
                elif res is not None:
                    return res
        raise StopIteration
