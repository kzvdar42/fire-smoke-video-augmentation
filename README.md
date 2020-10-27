# Image/Video effect augmentations

## How to run

### Preparation

#### Effect Images
Collect all effect images in one folder, better to trim empty pixels and resize to 720p.

Store annotations in coco json file with the images folder with the name `{folder_name}.json`

#### Effect Videos
Collect videos, compress to `.wemp`, extract alpha channel to `*_alpha.mp4` file. (better to resize to 720p and reduce bitrate)

#### Config
Before running, create augmentation config, based on the provided configs in the `configs` folder.

### Run script
Example for images:
```bash
python augment_script.py in_img_path out_img_path --in_extention jpg --use_alpha 0,1 --probability 20,1 --e_config animals cars --e_paths e_path_1 e_path_2
```

Example for videos:
```bash
python augment_script.py in_img_path out_img_path --in_extention mp4 --use_alpha 0 --e_config fire_smoke --e_paths e_path_1
```

#### Flags
Script supports a list of flags:
* `--in_annotations` - use, if there is a need to not collide with already annotated objects. Need provide a path to the coco annotations json file
* `--skip_annotations` - do not write the annotations (choose if just need augmentation, without corresponding annotation file)
* `--clean_out` - clean output folder before starting writing in them
* `--skip_augmented` - skip already annotated files
* `--use_alpha` - usage of alpha channel for video effects. Binary format (`0,1,0,1`), default is 1
* `--probability` - probability of choosing this kind of effects. Number in format `1,2,3,4`, default is 1
* `--max_workers` - number of threads to use
* `--preload` - preload image effects. Binary format (`0,1,0,1`), default is 0
* `--min_n_objects` - minimum amount of objects in one frame, default is 1
* `--max_n_objects` - maximum amount of objects in one frame, default is 1
* `--gen_prob` - probability of generation of 1 object: `1 / gen_prob`, default is 1
* `--next_gen_prob` - probability of generation of (n + 1)'th object: `1 / (gen_prob + (n - 1) * next_gen_prob)`, default is 0

#### Debug info
Also, there are debug flags. You can change `--debug_level` flag to next levels:
* 0 - no debug info
* 1 - offset point
* 2 - also show bboxes and category
* 3 - also show segmentation
* 2 - also show effect info (filename, rotation angle, gain, bias, gamma, etc.)

To see debug window, add `--show_debug` flag.
If you want to write debug info to output, add `--write_debug` flag.
