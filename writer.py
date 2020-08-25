import json

import numpy as np

from bbox_utils import get_corners


class COCO_writer:

    def __init__(self, categories=None):
        categories = [] if categories is None else categories
        self.categories = categories
        self.annotations = []
        self.images = []
    
    def get_cat_id(self, cat_name):
        for cat in self.categories:
            if cat['name'] == cat_name:
                return cat['id']
        return ValueError(f'Unknown category ({cat_name})')

    def add_category(name, supercategory, cat_id=None):
        existing_ids = [x['id'] for x in self.categories]
        if cat_id is not None and any([x == cat_id for x in existing_ids]):
            raise ValueError(f"Category with id {cat_id} already exists")

        if cat_id is None:
            cat_id = max(existing_ids) + 1 if len(existing_ids) else 1

        self.categories.append({
            'name': name,
            'supercategory': supercategory,
            'id': cat_id,
        })

    def add_frame(self, height, width, filename):
        self.images.append({
            'height': height,
            'date_captured': 0,
            'id': len(self.images) + 1,
            'file_name': filename,
            'flickr_url': "",
            'coco_url': "",
            'width': width,
            'license': 0,
        })

    def add_annotation(self, image_id, bbox, track_id, category_id):
        area = int(bbox[1] * bbox[3])
        segmentation = get_corners(np.expand_dims(bbox, axis=0))

        self.annotations.append({
            'image_id': image_id,
            'segmentation': segmentation.astype(int).tolist(),
            'iscrowd': 0,
            'bbox': bbox.astype(int).tolist(),
            'attributes': {
                'occluded': False,
                'keyframe': True,
                'track_id': track_id,
            },
            'area': area,
            'id': len(self.annotations) + 1,
            'category_id': category_id,
        })

    def write_result(self, save_path):
        result = dict()

        result['annotations'] = self.annotations
        result['categories'] = self.categories
        result['images'] = self.images
        result['licenses'] = [{
            'id': 0,
            'url': '',
            'name': '',
        }]
        result['info'] = {
            'date_created': '',
            'version': '',
            'contributor': '',
            'description': '',
            'year': '',
            'url': '',
        }
        with open(save_path, 'w') as out_file:
            json.dump(result, out_file)
