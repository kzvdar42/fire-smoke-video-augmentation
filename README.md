# fire-smoke-video-augmentation



## How to run

### Preparation

#### Effect Images
Collect all effect images in one folder, better to trim empty pixels and resize to 720p.

Store annotations in coco json file with the images folder with the name `{folder_name}.json`

#### Effect Videos
Collect videos, compress to `.wemp`, extract alpha channel to `*_alpha.mp4` file. (better to resize to 720p and reduce bitrate)

#### Config
Before running, create augmentation config, based on the provided `augment_config.yaml` file.

### Run script
Example for images:
```bash
python augment_script.py in_img_path out_img_path --in_extention jpg --use_alpha 0,1 --probability 20,1 --e_config augment_config.yaml --e_paths e_path_1 e_path_2
```

Example for videos:
```bash
python augment_script.py in_img_path out_img_path --in_extention jpg --use_alpha 0,1 --probability 20,1 --e_config augment_config.yaml --e_paths e_path_1 e_path_2
```

#### Debug info
You can change `debug_level` in config file to next levels:
* 0 - no debug info
* 1 - show bboxes and offset point
* 2 - also show effect info (filename, rotation angle, gain, bias, gamma, etc.)

To see debug window, add `--show_debug` flag.
If you want to write debug info to output, add `--write_debug` flag.