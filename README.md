# fire-smoke-video-augmentation



## How to run

### Preparation

#### Images
Collect all effect images in one folder, better to trim empty pixels and resize to 720p.

#### Videos
Collect videos, compress to `.wemp`, extract alpha channel to `*_alpha.mp4` file. (better to resize to 720p and reduce bitrate)

#### Config
Before running, create augmentation config, based on the provided `augment_config.yaml` file.

### Run script
Example for images:
```bash
python augment_script.py in_img_path out_img_path --in_extention jpg --e_png_path png_effects_path --e_mov_path mov_effects_path --e_config augment_config.yaml
```

Example for videos:
```bash
python augment_script.py in_vid_path out_vid_path --in_extention mp4 --e_png_path png_effects_path --e_mov_path mov_effects_path --e_config augment_config.yaml
```
#### Debug info
You can change `debug_level` in config file to next levels:
* 0 - no debug info
* 1 - show bboxes and offset point
* 2 - also show effect info (filename, rotation angle, gain, bias, gamma, etc.)

To see debug window, add `--show_debug` flag.
If you want to write debug info to output, add `--write_debug` flag.