from dataclasses import dataclass
import os
import yaml

import cv2
from tqdm import tqdm
import numpy as np
from reader import VideoEffectReader, ImageEffectReader
from bbox_utils import (convert_xywh_xyxy, convert_xyxy_xywh,
                        blur_contour, resize, rotate, flip,
                        gamma_correction)


@dataclass
class Effect:
    reader_id: int
    idx: int
    track_id: int
    size: int
    offset: tuple
    angle: float
    is_flip: bool
    transparency: float
    gain: float  # contrast
    bias: float  # brightness
    gamma: float  # gamma correction
    duration: int
    cur_dur: int # Current duration
    image: np.ndarray = None
    segments: list = None
    e_cats: list = None
    c_offset: tuple = None # Centered offset


@dataclass(init=None)
class AugmentationConfig:
    do_resize: bool = True
    min_size_far: float = 0.05
    min_size_close: float = 0.2
    size_jitter: float = 0.1

    do_flip: bool = True
    flip_chance: int = 2

    do_rotate: bool = True
    max_angle: bool = 30

    do_brightness: bool = True
    gain_loc: float = 1
    gain_scale: float = 0.075
    bias_loc: float = 0
    bias_scale: float = 5

    do_gamma: bool = True
    gamma_from: float = 0.5
    gamma_to: float = 1.5

    do_blur: bool = True
    blur_radius: int = 5
    contour_radius: int = 5

    min_transparency: int = 80
    max_transparency: int = 100

    min_duration: int = 30
    max_duration: int = 150
    min_bbox_size: int = 30

    def __init__(self, config_path=None):
        # Load config
        if config_path:
            with open(config_path) as in_file:
                config = yaml.load(in_file, Loader=yaml.FullLoader)
                self.__dict__.update(config)


@dataclass(init=False)
class Augmentations:

    min_n_objects: int = 1
    max_n_objects: int = 5
    gen_prob: int = 30
    next_gen_prob: int = 50
    debug_level: int = 0

    def __init__(self, e_readers, configs, **kwargs):
        self.e_readers = [r for r in e_readers if len(r.loaded)]
        self.e_cfgs = configs
        assert len(self.e_readers), "At least one effect!"
        assert len(configs) == len(self.e_readers), "Need config for each e_reader"
        self.objects = []
        self.probabilities = [r.probability for r in e_readers]
        self.probabilities = np.array(self.probabilities) / sum(self.probabilities)
        self.last_object_id = -1
        # Override values by kwargs
        self.__dict__.update(kwargs)

    def create_effect(self, frame):
        reader_id = np.random.choice(range(len(self.e_readers)), p=self.probabilities)
        reader = self.e_readers[reader_id]
        cfg = self.e_cfgs[reader_id]
        idx = np.random.randint(len(reader.loaded))
        start_dur = 0
        duration = np.random.randint(cfg.min_duration, cfg.max_duration + 1)
        if isinstance(reader, VideoEffectReader):
            total_frames = reader.total_frames[idx]
            start_dur = np.random.randint(0, max(1, total_frames - cfg.min_duration))
            duration = min(total_frames - 1, duration + start_dur)
        angle = np.float32(np.random.uniform(-cfg.max_angle, cfg.max_angle))
        # Make offset such that at least `min_size` of object is still visible.
        max_side = max(frame.shape[:2])
        min_size = int(cfg.min_size_far * max_side)
        y_offset = np.random.randint(min_size, max(frame.shape[0] - min_size, min_size + 1))
        min_size = cfg.min_size_far * frame.shape[0] + (cfg.min_size_close - cfg.min_size_far) * y_offset
        size = np.random.randint(int(min_size), int(min_size * (1 + cfg.size_jitter)) + 1)
        low_x = -size // 2 + min_size
        offset = [np.random.randint(low_x, max(frame.shape[1] - min_size, low_x + 1)),
                  y_offset]
        is_flip = np.random.randint(cfg.flip_chance) == 0
        transparency = np.float32(np.random.uniform(cfg.min_transparency, cfg.max_transparency) / 100)
        gain = np.float32(np.random.normal(loc=cfg.gain_loc, scale=cfg.gain_scale))
        bias = np.float32(np.random.normal(loc=cfg.bias_loc, scale=cfg.bias_scale))
        gamma = np.float32(np.random.uniform(cfg.gamma_from, cfg.gamma_to))
        # Add object to our dict.
        self.last_object_id += 1
        self.add_effect(Effect(reader_id, idx, self.last_object_id,
                               size, offset, angle, is_flip, transparency,
                               gain, bias, gamma, duration=duration, cur_dur=start_dur))

    def add_effect(self, effect):
        self.objects.append(effect)

    def merge_images(self, img1, img2, offset=None):
        offset = offset if offset is not None else (0, 0)
        # Left top coordinates
        def get_coords(x): return (x, 0) if x >= 0 else (0, -x)
        (x1, img2_x1), (y1, img2_y1) = map(get_coords, offset)
        # Down right coordinates
        height, width = img1[y1: y1 + img2.shape[0] - img2_y1,
                             x1: x1 + img2.shape[1] - img2_x1, 0].shape
        x2, y2 = x1 + width, y1 + height
        img2_x2, img2_y2 = img2_x1 + width, img2_y1 + height
        # Merge using mask
        mask = img2[img2_y1:img2_y2, img2_x1:img2_x2, 3:] / 255
        orig_img, effect = img1[y1:y2, x1: x2], img2[img2_y1:img2_y2, img2_x1:img2_x2, :3]
        # a, b, mask = np.asfortranarray(a), np.asfortranarray(b), np.asfortranarray(mask)
        # a, b, mask = a.reshape(a.shape, order='F'), b.reshape(b.shape, order='F'), mask.reshape(mask.shape, order='F')
        img1[y1: y2, x1: x2] = np.clip(effect * mask + orig_img * (1 - mask), 0, 255)
        # img1[y1: y2, x1: x2] = cv2.convertScaleAbs(effect * mask + orig_img * (1 - mask))
        return img1


    def get_image(self, e_info, read_annot=True):
        e_reader = self.e_readers[e_info.reader_id]
        if isinstance(e_reader, ImageEffectReader) and e_info.image is not None:
            e_image, segments, e_cats = e_info.image, e_info.segments, e_info.e_cats
        else:
            e_image, segments, e_cats = e_reader.get_frame(e_info, read_annot=read_annot)
            segments = np.array([np.array(segment, dtype=np.float32) for segment in segments], dtype=object) if segments else None
            e_info.image, e_info.segments, e_info.e_cats = None, None, None
            # e_info.centered_offset = None
        return e_image, segments, e_cats

    def transform_effect(self, e_image, e_info, segments):
        cfg = self.e_cfgs[e_info.reader_id]
        # Transparency
        e_image[:, :, 3:] = e_image[:, :, 3:] * e_info.transparency

        # Flip image
        if cfg.do_flip and e_info.is_flip:
            e_image, _, segments = flip(e_image, segments=segments)

        # Contrast & Brightness
        if cfg.do_brightness:
            e_image[:, :, :3] = cv2.convertScaleAbs(
                e_image[:, :, :3], alpha=e_info.gain, beta=e_info.bias)

        # Gamma correction
        if cfg.do_gamma:
            e_image[:, :, :3] = gamma_correction(
                e_image[:, :, :3], e_info.gamma)

        # Rotate image
        if cfg.do_rotate and e_info.angle:
            e_image, _, segments = rotate(e_image, e_info.angle, segments=segments)
        
        if cfg.do_blur:
            e_image = blur_contour(e_image,
                cfg.blur_radius, cfg.contour_radius)

        # Resize image
        if cfg.do_resize:
            e_image, _, segments = resize(e_image, e_info.size, segments=segments)
        return e_image, segments
    
    def __call__(self, frame, writer=None, frame_num=None):
        # Add new effects
        while len(self.objects) < self.min_n_objects:
            self.create_effect(frame)
        gen_prob = np.random.randint(self.gen_prob + len(self.objects) * self.next_gen_prob)
        if len(self.objects) < self.max_n_objects and gen_prob == 0:
            self.create_effect(frame)
        return self.augment(frame, writer, frame_num)


    def augment(self, frame, writer=None, frame_num=None):
        # Display effects
        eff_to_delete = []
        for i,  e_info in enumerate(self.objects):
            cfg = self.e_cfgs[e_info.reader_id]
            # Get image
            e_image, segments, e_info.e_cats = self.get_image(e_info,
                read_annot=writer is not None or self.debug_level > 1
            )
            # If no image skip and delete
            if e_image is None:
                eff_to_delete.append(i)
                continue
            
            if e_info.image is None:
                e_info.image, e_info.segments = \
                e_image, segments = self.transform_effect(e_image, e_info, segments)

            # If image is grayed (night), convert effect to gray.
            hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            if (hsv_frame[:, :, 1] < 5).all():
                e_info.image = \
                e_image[:, :, :3] = (0.299 * e_image[:, :, 2:3] +
                                     0.587 * e_image[:, :, 1:2] +
                                     0.114 * e_image[:, :, :1])

            if e_info.c_offset is None:
                # Correct the offset if it's wrong.
                min_x_pos = -e_image.shape[1] // (2 * int(cfg.min_size_far * max(frame.shape[:2])))
                if min_x_pos > e_info.offset[0]:
                    e_info.offset = (min_x_pos, e_info.offset[1])
                
                # Center offset
                e_info.c_offset = (e_info.offset[0] - e_image.shape[1] // 2,
                                   e_info.offset[1] - e_image.shape[0])


            # Apply effect on image
            frame = self.merge_images(frame, e_image, e_info.c_offset)
            
            # Update/Delete cur_dur
            e_info.cur_dur += 1
            if e_info.cur_dur >= e_info.duration:
                eff_to_delete.append(i)
        
        # Write annotations and debug info
        debug_frame = self.write_and_debug(frame, writer=writer)

        # Delete expired effects
        for i in sorted(eff_to_delete, reverse=True):
            del self.objects[i]

        return frame, debug_frame


    # Debug info
    def write_and_debug(self, frame, writer=None, frame_num=None):
        # Write debug info to independent image
        debug_frame = frame.copy() if self.debug_level > 0 else None
        for e_info in self.objects:
            cfg = self.e_cfgs[e_info.reader_id]
            segments = e_info.segments
            # Create bboxes from segments
            # Move bboxes to offset position
            if segments is not None:
                h, w = frame.shape[:2]
                for si, poly in enumerate(segments):
                    poly += e_info.c_offset
                    poly[:, :, 0] = np.clip(poly[:, :, 0], 0, w - 1)
                    poly[:, :, 1] = np.clip(poly[:, :, 1], 0, h - 1)
                    segments[si] = poly
                    if self.debug_level > 2:
                        cv2.drawContours(debug_frame, poly.astype(np.int32), -1, (0, 0, 255), 3)
                for poly, cat in zip(segments, e_info.e_cats):
                    bbox = cv2.boundingRect(poly.astype(np.int32))
                    min_side_size = min(bbox[2:])
                    # Show annotations
                    if self.debug_level > 1:
                        b = convert_xywh_xyxy(bbox, w, h)
                        clr = (0, 255, 0) if min_side_size >= cfg.min_bbox_size else (0, 0, 255)
                        cv2.rectangle(debug_frame, tuple(b[:2]), tuple(b[2:4]), clr, 2)
                    if self.debug_level > 1:
                        self.put_text(debug_frame, cat, tuple(bbox[:2]))
                    # Write annotations
                    if writer is not None and min_side_size >= cfg.min_bbox_size:
                        writer.add_annotation(frame_num, bbox, e_info.track_id,
                                                 writer.get_cat_id(cat))
            # draw a point there offset is
            if self.debug_level > 0:
                cv2.circle(debug_frame, e_info.offset, 4, (0, 255, 0), 2)
            # Add effect info
            if self.debug_level > 3:
                self.draw_effect_info(debug_frame, e_info)
        return debug_frame


    def put_text(self, frame, text, position):
        """Draw white text with black outline on the given frame."""
        cv2.putText(frame, text, position,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 0, 0), 4)
        cv2.putText(frame, text, position,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1, (255, 255, 255), 2)

    def draw_effect_info(self, frame, e_info):
        e_reader = self.e_readers[e_info.reader_id]
        cfg = self.e_cfgs[e_info.reader_id]
        effect_filename = os.path.split(e_reader.paths[e_info.idx])[1]

        text = [f'{effect_filename}', f'offset: {e_info.offset}',
                f'alpha: {e_info.transparency:.2f}']
        if cfg.do_flip:
            text.append(f'flip: {e_info.is_flip:1d}')
        if cfg.do_resize:
            text.append(f'size: {e_info.size}')
        if cfg.do_brightness:
            text.extend([f'gain: {e_info.gain:.2f}',
                         f'bias: {e_info.bias:.2f}'])
        if cfg.do_gamma:
            text.append(f'gamma: {e_info.gamma:.2f}')
        if cfg.do_rotate:
            text.append(f'angle: {e_info.angle:.2f}')

        for j in range(0, len(text) + 1, 2):
            position = (e_info.offset[0], e_info.offset[1] + j * 15)
            self.put_text(frame, '-'.join(text[j:j + 2]), position)
