import json

import numpy as np


class NumpyEncoder(json.JSONEncoder):
    """Class to encode the numpy arrays in json."""

    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


class COCO_writer:

    def __init__(self, categories=None, synonyms=None):
        categories = [] if categories is None else categories
        self.categories = categories
        self.annotations = []
        self.images = []
        self.cat_to_id = dict()
        for cat in self.categories:
            cat_id = cat['id']
            self.cat_to_id[cat['name'].lower()] = cat_id
            # Add synonyms
            if synonyms is not None:
                for s in synonyms.get(cat['name'], []):
                    self.cat_to_id[s.lower()] = cat_id

    def get_cat_id(self, cat_name):
        cat_id = self.cat_to_id.get(cat_name.lower(), None)
        if cat_id is None:
            raise ValueError(f'Unknown category ({cat_name})')
        else:
            return cat_id

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

    def add_frame(self, height, width, filename=None, file_ext=None):
        image_id = len(self.images) + 1
        if filename is None:
            file_ext = file_ext if file_ext is not None else 'jpg'
            filename = f'{image_id:0>7}.{file_ext}'
        filename = filename.replace('\\', '/')
        self.images.append({
            'height': height,
            'date_captured': None,
            'dataset': 'Roadar',
            'id': image_id,
            'file_name': filename,
            'image': filename,
            'flickr_url': None,
            'coco_url': None,
            'width': width,
            'license': None,
        })
        return image_id, filename

    def add_annotation(self, image_id, bbox, track_id, category_id, segmentation=None):
        area = int(bbox[1] * bbox[3])

        assert image_id is not None

        self.annotations.append({
            'image_id': image_id,
            'segmentation': segmentation,
            'iscrowd': 0,
            'bbox': bbox,
            'attributes': {},
            'area': area,
            'is_occluded': False,
            'id': len(self.annotations) + 1,
            'category_id': category_id,
        })

    def write_result(self, save_path):
        result = dict()

        result['annotations'] = self.annotations
        result['categories'] = self.categories
        result['images'] = self.images
        result['licenses'] = None
        result['info'] = None
        with open(save_path, 'w') as out_file:
            json.dump(
                result,
                out_file,
                indent=2,
                ensure_ascii=False,
                cls=NumpyEncoder,
            )


def get_coco_writer():
    return COCO_writer([
        {
            'name': 'person',
            'supercategory': '',
            'id': 1,
        },
        {
            'name': 'vehicle',
            'supercategory': '',
            'id': 2,
        },
        {
            'name': 'fire',
            'supercategory': '',
            'id': 3,
        },
        {
            'name': 'animal',
            'supercategory': '',
            'id': 4,
        },
        {
            'name': 'smoke',
            'supercategory': '',
            'id': 5,
        },
    ],
    synonyms={
        'vehicle': ['Car']
    }
    )
