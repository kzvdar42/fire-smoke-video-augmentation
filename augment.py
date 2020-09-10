from dataclasses import dataclass
import os
import yaml

import cv2
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
    cur_dur: int  # Current duration


@dataclass(init=False)
class Augmentations:

    do_resize: bool = True
    png_min_size: int = 60
    png_max_size: int = 200
    mov_min_size: int = 180
    mov_max_size: int = 600

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

    min_transparency: int = 50
    max_transparency: int = 100

    png_min_duration: int = 30
    png_max_duration: int = 150
    mov_min_duration: int = 30
    mov_max_duration: int = 150

    min_n_objects: int = 1
    max_n_objects: int = 5
    gen_prob: int = 30
    next_gen_prob: int = 50

    debug_level: int = 0
    ck_start: int = 10
    ck_range: int = 20

    def __init__(self, e_readers,
                 config_path=None, **kwargs):
        self.e_readers = [r for r in e_readers if len(r.loaded)]
        assert len(self.e_readers), "At least one effect!"
        self.objects = []
        self.probabilities = [r.probability for r in e_readers]
        self.probabilities = np.array(self.probabilities) / sum(self.probabilities)
        self.last_object_id = -1
        # Load config
        if config_path:
            with open(config_path) as in_file:
                config = yaml.load(in_file, Loader=yaml.FullLoader)
                self.__dict__.update(config)
        # Override values by kwargs
        self.__dict__.update(kwargs)

    def create_effect(self, frame):
        reader_id = np.random.choice(range(len(self.e_readers)), p=self.probabilities)
        reader = self.e_readers[reader_id]

        if isinstance(reader, ImageEffectReader):
            min_size, max_size = self.png_min_size, self.png_max_size
            idx = np.random.randint(len(reader.loaded))
            start_dur = 0
            duration = np.random.randint(self.png_min_duration, self.png_max_duration + 1)
        elif isinstance(reader, VideoEffectReader):
            min_size, max_size = self.mov_min_size, self.mov_max_size
            idx = np.random.randint(len(reader.loaded))
            total_frames = reader.total_frames[idx]
            start_dur = np.random.randint(0, max(1, total_frames - self.mov_min_duration))
            duration = np.random.randint(min(self.mov_min_duration + start_dur, total_frames) - 1,
                                         min(total_frames, self.mov_max_duration + start_dur))
        size = np.random.randint(min_size, max_size + 1)
        angle = np.float32(np.random.uniform(-self.max_angle, self.max_angle))
        # Make offset such that at least `min_size` of object is still visible.
        low_x = -size // 2 + min_size
        offset = (np.random.randint(low_x, max(frame.shape[1] - min_size, low_x + 1)),
                  np.random.randint(min_size, max(frame.shape[0] - min_size, min_size + 1)))
        is_flip = np.random.randint(self.flip_chance) == 0
        transparency = np.float32(np.random.uniform(self.min_transparency, self.max_transparency) / 100)
        gain = np.float32(np.random.normal(loc=self.gain_loc, scale=self.gain_scale))
        bias = np.float32(np.random.normal(loc=self.bias_loc, scale=self.bias_scale))
        gamma = np.float32(np.random.uniform(self.gamma_from, self.gamma_to))
        # Add object to our dict.
        self.last_object_id += 1
        self.objects.append(Effect(reader_id, idx, self.last_object_id,
                                   size, offset, angle, is_flip, transparency,
                                   gain, bias, gamma, duration, start_dur))

    def merge_images(self, img1, img2, offset=None):
        offset = offset if offset is not None else (0, 0)
        # Left top coordinates
        def get_coords(x): return (x, 0) if x >= 0 else (0, -x)
        (x1, img2_x1), (y1, img2_y1) = map(get_coords, offset)
        # Down right coordinates
        height, width = img1[y1: y1 + img2.shape[0] -
                             img2_y1, x1: x1 + img2.shape[1] - img2_x1, 0].shape
        x2, y2 = x1 + width, y1 + height
        img2_x2, img2_y2 = img2_x1 + width, img2_y1 + height
        # Merge using mask
        mask = img2[img2_y1:img2_y2, img2_x1:img2_x2, 3:].astype(np.float32) / 255
        orig_img, effect = img1[y1:y2, x1: x2], img2[img2_y1:img2_y2, img2_x1:img2_x2, :3]
        # a, b, mask = np.asfortranarray(a), np.asfortranarray(b), np.asfortranarray(mask)
        # a, b, mask = a.reshape(a.shape, order='F'), b.reshape(b.shape, order='F'), mask.reshape(mask.shape, order='F')
        img1[y1: y2, x1: x2] = effect * mask + orig_img * (1 - mask)
        img1[y1: y2, x1: x2] = cv2.convertScaleAbs(img1[y1: y2, x1: x2])
        return img1


    def get_image(self, e_info, read_annot=True):
        e_image, segments, e_cats = self.e_readers[e_info.reader_id].get_frame(e_info, read_annot=read_annot)
        segments = np.array([np.array(segment, dtype=np.float32) for segment in segments]) if segments else None
        return e_image, segments, e_cats
    
    def transform_effect(self, e_image, e_info, segments):
        # Transparency
        e_image[:, :, 3:] = e_image[:, :, 3:] * e_info.transparency

        # Flip image
        if self.do_flip and e_info.is_flip:
            e_image, _, segments = flip(e_image, segments=segments)

        # Contrast & Brightness
        if self.do_brightness:
            e_image[:, :, :3] = cv2.convertScaleAbs(
                e_image[:, :, :3], alpha=e_info.gain, beta=e_info.bias)

        # Gamma correction
        if self.do_gamma:
            e_image[:, :, :3] = gamma_correction(
                e_image[:, :, :3], e_info.gamma)

        # Rotate image
        if self.do_rotate and e_info.angle:
            e_image, _, segments = rotate(e_image, e_info.angle, segments=segments)
        
        if self.do_blur:
            e_image = blur_contour(e_image,
                self.blur_radius, self.contour_radius)

        # Resize image
        if self.do_resize:
            e_image, _, segments = resize(e_image, e_info.size, segments=segments)
        return e_image, segments

    def augment(self, frame, writer=None, frame_num=None):
        # Add new effects
        while len(self.objects) < self.min_n_objects:
            self.create_effect(frame)
        gen_prob = np.random.randint(self.gen_prob + len(self.objects) * self.next_gen_prob)
        if len(self.objects) < self.max_n_objects and gen_prob == 0:
            self.create_effect(frame)

        # Display effects
        eff_to_delete = []
        debug_frame = None
        for i,  e_info in enumerate(self.objects):
            # Get image
            e_image, segments, e_cats = self.get_image(e_info,
                read_annot=writer is not None
            )
            # If no image skip and delete
            if e_image is None:
                eff_to_delete.append(i)
                continue
            
            e_image, segments = self.transform_effect(e_image, e_info, segments)

            
            # If image is grayed (night), convert effect to gray.
            hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            if (hsv_frame[:, :, 1] < 5).all():
                e_image[:, :, :3] = 0.299 * e_image[:, :, 2:3] + 0.587 * e_image[:, :, 1:2] + 0.114 * e_image[:, :, :1]

            # Center offset
            offset = e_info.offset
            offset = (offset[0] - e_image.shape[1] // 2,
                      offset[1] - e_image.shape[0])

            # Apply effect on image
            frame = self.merge_images(frame, e_image, offset)

            # Write debug info to independent image
            if self.debug_level > 0:
                debug_frame = frame.copy()
            
            # Create bboxes from segments
            # Move bboxes to offset position
            if segments is not None:
                h, w = frame.shape[:2]
                for si, poly in enumerate(segments):
                    poly += offset
                    poly = poly.astype(np.int32)
                    poly[:, :, 0] = np.clip(poly[:, :, 0], 0, w - 1)
                    poly[:, :, 1] = np.clip(poly[:, :, 1], 0, h - 1)
                    segments[si] = poly
                    if self.debug_level > 0:
                        cv2.drawContours(debug_frame, poly, -1, (0, 0, 255), 3)
                for poly, cat in zip(segments, e_cats):
                    bbox = cv2.boundingRect(poly)
                    # Show annotations
                    if self.debug_level > 0:
                        b = convert_xywh_xyxy(bbox, w, h)
                        cv2.rectangle(debug_frame, tuple(b[:2]), tuple(b[2:4]), (0, 0, 255), 3)
                    # Write annotations
                    if writer is not None:
                        cat_id = writer.get_cat_id(cat)
                        writer.add_annotation(frame_num, bbox, e_info.track_id,
                                              cat_id)
                        if self.debug_level > 1:
                            self.put_text(debug_frame, cat, tuple(bbox[:2]))

            # draw a point there offset is
            if self.debug_level > 0:
                cv2.circle(debug_frame, e_info.offset, 10, (0, 255, 0), 2)
            # Add effect info
            if self.debug_level > 1:
                self.draw_effect_info(debug_frame, e_info)
            
            # Update/Delete cur_dur
            e_info.cur_dur += 1
            if e_info.cur_dur >= e_info.duration:
                eff_to_delete.append(i)

        # Delete expired effects
        for i in sorted(eff_to_delete, reverse=True):
            del self.objects[i]

        return frame, debug_frame

    # Debug info
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
        effect_filename = os.path.split(e_reader.paths[e_info.idx])[1]

        text = [f'{effect_filename}', f'offset: {e_info.offset}',
                f'alpha: {e_info.transparency:.2f}']
        if self.do_flip:
            text.append(f'flip: {e_info.is_flip:1d}')
        if self.do_resize:
            text.append(f'size: {e_info.size}')
        if self.do_brightness:
            text.extend([f'gain: {e_info.gain:.2f}',
                         f'bias: {e_info.bias:.2f}'])
        if self.do_gamma:
            text.append(f'gamma: {e_info.gamma:.2f}')
        if self.do_rotate:
            text.append(f'angle: {e_info.angle:.2f}')

        for j in range(0, len(text) + 1, 2):
            position = (e_info.offset[0], e_info.offset[1] + j * 15)
            self.put_text(frame, '-'.join(text[j:j + 2]), position)
