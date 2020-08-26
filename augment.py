from collections import defaultdict
from dataclasses import dataclass
import os

import cv2
import numpy as np
from pycocotools.coco import COCO
from bbox_utils import (rotate_im, get_corners, rotate_box,
                        get_enclosing_box, from_ratio_to_pixel,
                        convert_xywh_xyxy, convert_xyxy_xywh,
em                        resize, rotate, flip, gamma_correction)


@dataclass
class Effect:
    type: str
    idx: int
    track_id: int
    size: int
    offset: tuple
    angle: float
    is_flip: bool
    gain: float # contrast 
    bias: float # brightness
    gamma: float # gamma correction
    duration: int
    cur_dur: int  # Current duration


class Augmentations:

    def __init__(self, png_effects, mov_effects, cat_id,
                 do_resize=True, do_flip=True, do_rotate=True,
                 do_brightness=True, do_gamma=True, debug_level=0):
        self.png_effects = png_effects
        self.mov_effects = mov_effects
        self.cat_id = cat_id
        self.do_resize, self.do_flip = do_resize, do_flip
        self.do_rotate, self.do_brightness = do_rotate, do_brightness
        self.do_gamma = do_gamma
        self.debug_level = debug_level
        self.objects = []
        self.min_size = 60
        self.max_size = 400
        self.max_angle = 30
        # Duration limits for `png` effects.
        self.min_duration = 1 * 30
        self.max_duration = 5 * 30
        self.last_object_id = -1
        self.max_n_objects = 5

        # Preload effect images and videos.
        self.loaded_png = []
        self.loaded_mov = []
        self.png_annotations = []
        self.mov_annotations = []

        # Load image annotations.
        folder_path = os.path.split(self.png_effects[0])[0]
        png_annot_path = os.path.join(folder_path, 'annotations.csv')
        png_annotations = defaultdict(list)
        with open(png_annot_path) as in_file:
            for line in in_file.readlines():
                category, x1, y1, x2, y2, img_name = line.split(',')[:6]
                png_annotations[img_name].append([*map(int, [x1, y1, x2, y2])])

        for k, v in png_annotations.items():
            png_annotations[k] = np.array(v)

        # Load images.
        for png_path in self.png_effects:
            effect = cv2.imread(png_path, cv2.IMREAD_UNCHANGED)
            self.loaded_png.append(effect)
            png_file_name = os.path.split(png_path)[1]
            self.png_annotations.append(png_annotations[png_file_name])
        
        # Load mov effects.
        for mov_path in self.mov_effects:
            cap = cv2.VideoCapture(mov_path)
            # XXX: This should work, but in my case it fails
            # https://github.com/opencv/opencv/pull/13395
            # print(cap.set(cv2.CAP_PROP_CONVERT_RGB, 0))
            # print(cap.get(cv2.CAP_PROP_CONVERT_RGB))

            # Because of that we read alpha channel individually.
            alpha_path = os.path.splitext(mov_path)[0] + '_alpha.mp4'
            alpha_cap = cv2.VideoCapture(mov_path)
            self.loaded_mov.append((cap, alpha_cap))

            # Load Coco annotations
            file_path, filename = os.path.split(mov_path)
            filename = os.path.splitext(filename)[0]
            annot_path = os.path.join(file_path, 'annotations', filename + '.json')
            if os.path.isfile(annot_path):
                self.mov_annotations.append(COCO(annot_path))
            else:
                self.mov_annotations.append(None)

    def merge_images(self, img1, img2, offset=None):
        offset = offset if offset is not None else (0, 0)
        img1 = img1.copy()
        # Get application mask
        mask = img2[:, :, 3:] / 255
        # Left top coordinates
        x1, y1 = np.clip(offset, 0, None)
        img2_x1, img2_y1 = - np.clip(offset, None, 0)
        # Down right coordinates
        height, width = img1[y1: y1 + img2.shape[0] -
                             img2_y1, x1: x1 + img2.shape[1] - img2_x1, 0].shape
        x2, y2 = x1 + width, y1 + height
        img2_x2, img2_y2 = img2_x1 + width, img2_y1 + height
        # Merge using mask
        mask = mask[img2_y1:img2_y2, img2_x1:img2_x2]
        img1[y1: y2, x1: x2] = (img2[img2_y1:img2_y2, img2_x1:img2_x2, :3] * mask
                                + img1[y1:y2, x1: x2] * (1 - mask))
        img1[y1: y2, x1: x2] = cv2.convertScaleAbs(img1[y1: y2, x1: x2])
        return img1

    def create_effect(self, frame):
        if np.random.randint(2):
            e_type = 'png'
            idx = np.random.randint(len(self.png_effects))
            duration = np.random.randint(self.min_duration, self.max_duration)
        else:
            e_type = 'mov'
            idx = np.random.randint(len(self.mov_effects))
            cap = cv2.VideoCapture(self.mov_effects[idx])
            total_frames = cap.get(7)
            duration = np.random.randint(min(self.min_duration, total_frames) - 1, total_frames)
            cap.release()
            del cap
        size = np.random.randint(self.min_size, self.max_size)
        # XXX: as video effects are only a part of the frame, increase the size.
        if e_type == 'mov':
            size = np.clip(size * 4, 0, int(self.max_size * 1.2))

        angle = np.random.uniform(-self.max_angle, self.max_angle)
        offset = (np.random.randint(-size // 2 + self.min_size, frame.shape[1] - self.min_size),
                  np.random.randint(self.min_size, frame.shape[0] - self.min_size))
        is_flip = np.random.randint(2) > 0
        gain = np.random.normal(loc=1, scale=0.15) # ~ from: 0.40 to: 1.6  mean: 1.00
        bias = np.random.normal(loc=0, scale=10)   # ~ from: -40  to: 40   mean: 0
        gamma = np.random.uniform(0.25, 2)
        # Add object to our dict.
        self.last_object_id += 1
        self.objects.append(Effect(e_type, idx, self.last_object_id,
                                   size, offset, angle, is_flip, 
                                   gain, bias, gamma, duration, 0))


    def draw_effect_info(self, frame, e_info):
        effects = self.png_effects if e_info.type == 'png' else self.mov_effects
        effect_filename = os.path.split(effects[e_info.idx])[1]

        text = [f'{effect_filename}', f'offset: {e_info.offset}']
        if self.do_flip:
            text.append(f'flip: {e_info.is_flip:1d}')
        if self.do_resize:
            text.append(f'size: {e_info.size}')
        if self.do_brightness:
            text.extend([f'gain: {e_info.gain:.2f}', f'bias: {e_info.bias:.2f}'])
        if self.do_gamma:
            text.append(f'gamma: {e_info.gamma:.2f}')
        if self.do_rotate:
            text.append(f'angle: {e_info.angle:.2f}')
        
        for j in range(0, len(text) + 1, 2):
            cv2.putText(frame, '-'.join(text[j:j + 2]),
                (e_info.offset[0], e_info.offset[1] + j * 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 0),
                4)
            
            cv2.putText(frame, '-'.join(text[j:j + 2]),
                (e_info.offset[0], e_info.offset[1] + j * 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2)

    def augment(self, frame, frame_num, writer=None):
        # Add new effect
        if (len(self.objects) < self.max_n_objects
                and np.random.randint(30 + len(self.objects) * 50) == 0):
            self.create_effect(frame)

        # Display effects
        eff_to_delete = []
        for i,  e_info in enumerate(self.objects):
            # Get image
            if e_info.type == 'png':
                e_image = self.loaded_png[e_info.idx]
                bboxes = self.png_annotations[e_info.idx]
            elif e_info.type == 'mov':
                cap, alpha_cap = self.loaded_mov[e_info.idx]
                annot = self.mov_annotations[e_info.idx]
                cap.set(cv2.CAP_PROP_POS_FRAMES, e_info.cur_dur)
                e_image = cap.read()[1]
                # Color key black with everything < start = 0
                # and everything > start + range = 255
                # (value - start) / range
                hsv_e_image = cv2.cvtColor(e_image, cv2.COLOR_BGR2HSV)
                v_e_image = hsv_e_image[:, :, 2:3].astype(np.int32)
                e_alpha = np.clip((v_e_image - 10) / 20, 0, 1) * 255
                e_alpha = e_alpha.astype(np.uint8)
                # Concat with alpha channel
                e_image = np.concatenate((e_image, e_alpha), -1)
                # Get bboxes
                bboxes = []
                if annot is not None:
                    height, width = e_image.shape[:2]
                    ann_ids = annot.getAnnIds(
                        imgIds=e_info.cur_dur + 1, iscrowd=None)
                    for obj in annot.loadAnns(ann_ids):
                        bboxes.append(convert_xywh_xyxy(obj['bbox'], width, height))
            
            bboxes = np.array(bboxes).astype(np.float64)
            if len(bboxes) == 0:
                bboxes = None
            
            # Resize image
            if self.do_resize:
                e_image, bboxes = resize(e_image, e_info.size, bboxes)

            # Flip image
            if self.do_flip and e_info.is_flip:
                e_image, bboxes = flip(e_image, bboxes)

            # Contrast & Brightness
            if self.do_brightness:
                e_image[:, :, :3] = cv2.convertScaleAbs(e_image[:, :, :3], alpha=e_info.gain, beta=e_info.bias)

            # Gamma correction
            if self.do_gamma:
                e_image[:, :, :3] = gamma_correction(e_image[:, :, :3], e_info.gamma)

            # Rotate image
            if self.do_rotate and e_info.angle:
                e_image, bboxes = rotate(e_image, e_info.angle, bboxes)

            offset = e_info.offset
            offset = (offset[0] - e_image.shape[1] //
                      2, offset[1] - e_image.shape[0])

            # Apply effect on image
            frame = self.merge_images(frame, e_image, offset)

            # Move bboxes to offset position
            if bboxes is not None:
                bboxes += np.hstack((offset, offset))
                bboxes = bboxes.astype(np.int32)
                for bbox in bboxes:
                    # Show annotations
                    if self.debug_level > 0:
                        cv2.rectangle(frame, tuple(bbox[:2]), tuple(bbox[2:4]), (0, 0, 255), 3)
                    # Write annotations
                    if writer is not None:
                        bbox = convert_xyxy_xywh(bbox)
                        writer.add_annotation(frame_num, bbox, e_info.track_id, self.cat_id)

            # draw a point there offset is
            if self.debug_level > 0:
                cv2.circle(frame, e_info.offset, 10, (0, 255, 0), 2)
            # Add effect info
            if self.debug_level > 1:
                self.draw_effect_info(frame, e_info)

            # Delete or update cur_dur
            if e_info.cur_dur >= e_info.duration:
                eff_to_delete.append(i)
            else:
                e_info.cur_dur += 1

        # Delete expired effects
        for i in sorted(eff_to_delete, reverse=True):
            del self.objects[i]

        return frame
