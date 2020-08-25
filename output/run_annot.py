import os
import cv2
import json
import sys
import argparse


parser = argparse.ArgumentParser(description='test annotation')
parser.add_argument('--folder_path', type=str, required=True, help="path to folder with video and unzipped annotations")

args = parser.parse_args()
annotation_path = args.folder_path + os.sep + 'annotations' + os.sep + 'instances_default.json'
folders = os.listdir(args.folder_path)
mp4state = False

for element in folders:
    if ".mp4" in element:
        video_file_path = args.folder_path + os.sep + element
        mp4state = True
        
if mp4state is False:
    sys.exit("no video file found")

dct = dict()
rect_color = (252, 3, 169)
class_color = (3, 115, 252)
track_color = (252, 148, 3)

with open(annotation_path) as json_file:
    data = json.load(json_file)

print(data.keys())
print(len(data['images']))
print(len(data['annotations']))
print(data['annotations'][0])

print(data['categories'])
class_mapper = {e['id']: e['name'] for e in data['categories']}
dct = {e['id']: [] for e in data['images']}

for annot in data['annotations']:
    if annot['image_id'] in dct.keys():
        track_id = annot['attributes']['track_id']
        class_id = annot['category_id']
        bbox = annot['bbox']
        bbox[2] = bbox[0] + bbox[2]
        bbox[3] =  bbox[1] + bbox[3]
        element = {'track_id': track_id, 'bbox': bbox, 'class_id': class_id}
        dct[annot['image_id']].append(element)
    else:
        continue

cap = cv2.VideoCapture(video_file_path)
counter = 0
while True:
    ret, frame = cap.read()
    if ret is False:
        break

    if counter in dct.keys():
        annots = dct[counter]
        for annot in annots:
            dets = [int(x) for x in annot['bbox']]
            cv2.rectangle(frame, (dets[0], dets[1]), (dets[2], dets[3]), rect_color, 2)

            track_id = str(annot['track_id'])
            cx = dets[0] + 12
            cy = dets[1] - 10
            cv2.putText(frame, f"track_id - {track_id}", (cx, cy), cv2.FONT_HERSHEY_DUPLEX, 1, track_color, 2)
            class_id = class_mapper[annot['class_id']]
            cx = dets[0] + 30
            cy = dets[1] - 40
            cv2.putText(frame, f"class - {class_id}", (cx, cy), cv2.FONT_HERSHEY_DUPLEX, 1, class_color, 2)

    
    frame = cv2.resize(frame, (720, 480))
    cv2.imshow('window', frame)

    key = cv2.waitKey(0) & 0xFF

    if key == ord('q'):
        break
    elif key == ord('w'):
        counter -= 1
        cap.set(1, counter)
    else:
        counter += 1
