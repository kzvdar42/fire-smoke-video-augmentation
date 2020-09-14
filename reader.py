from collections import defaultdict
from dataclasses import dataclass
import os
from threading import Thread, Lock

import cv2
import numpy as np
from pycocotools.coco import COCO
from bbox_utils import (convert_xywh_xyxy, get_corners, prepare_image)

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
        self.loaded, self.annotations = [], []
        # Load mov effects.
        for mov_path in self.paths:
            cap = cv2.VideoCapture(mov_path)
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
        load_func = (self.load_csv_annotations if self.annot_type == 'csv'
                     else self.load_coco_annotations)
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
            else:
                for obj in self.annotations[e_info.idx]:
                    segment, cat = get_segments_and_cats_from_obj(obj, self.cats)
                    segments.append(segment)
                    e_cats.append(cat)
        return e_image, segments, e_cats


class ThreadsHandler:

    def __init__(self):
        self.threads = []
        self._lock = Lock()
    
    def add(self, target, args=None, name=None):
        thread = Thread(target=target, args=args, name=name)
        thread.start()
        self.threads.append(thread)
    
    def append(self, thread):
        self.threads.append(thread)

    def clean_threads(self):
        with self._lock:
            threads, self.threads = self.threads, []
            threads = [thread for thread in threads if not thread.is_alive()]
            self.threads.extend(threads)

    def join_threads(self):
        with self._lock:
            threads, self.threads = self.threads, []
            for thread in threads:
                thread.join()
    
    def __len__(self):
        return len(self.threads)

class ThreadedImagesReader:
    """Input image reader."""

    def __init__(self, images, buffer_size=32):
        self.images = images
        self.total = len(images)
        self.buffer_size = buffer_size
        self.next_id = -1
        self.buffer = []
        self.threads = ThreadsHandler()
        self._lock = Lock()
        self._fill_buffer()

    def is_open(self):
        self.threads.clean_threads()
        return (
            self.__is_open() or
            len(self.buffer) or
            len(self.threads)
        )

    def __is_open(self):
        return self.next_id < self.total

    def __get_next_id(self):
        with self._lock:
            self.next_id += 1
            return self.next_id

    def __read_to_buffer(self):
        try:
            image_path = self.images[self.__get_next_id()]
        except IndexError:
            return
        image = cv2.imread(image_path)
        if image is not None:
            self.buffer.append((image_path, image))
        else:
            print(f"[ERROR] Coudn't read image with path {image_path}")

    def start_new_thread(self):
        self.threads.add(target=self.__read_to_buffer, args=(), name=str(self.next_id))

    def _fill_buffer(self):
        with self._lock:
            if self.__is_open():
                items_left = self.total - self.next_id
                n_fill = min(self.buffer_size - len(self.buffer), items_left)
                for _ in range(n_fill):
                    self.start_new_thread()

    def __iter__(self):
        return self

    def __next__(self):
        if not self.is_open():
            raise StopIteration
        n_tries = 0
        while self.is_open():
            if len(self.buffer):
                if self.__is_open():
                    self.start_new_thread()
                return self.buffer.pop(0)
            n_tries += 1
            if n_tries % 50 == 0:
                self.threads.join_threads()
        raise StopIteration