import io
import json
import os.path
from collections import defaultdict
import glob

from skimage.segmentation import felzenszwalb
from skimage.io import imread
from skimage.measure import regionprops

import multiprocessing as mp

import argparse

from functools import partial


def process_image(path, scale):
    centroids_dict = defaultdict(set)
    prompt_dict = {}
    img_id = int(path.split('/')[-1].split('.')[0])
    img = imread(path)
    h, w = img.shape[:2]

    seg = felzenszwalb(img, scale=scale)
    for prop in regionprops(seg):
        x, y = prop.centroid
        x /= float(h)
        y /= float(w)
        centroids_dict[img_id].add((y, x))

    for k in centroids_dict.keys():
        s = centroids_dict[k]
        l = list(s)
        l.sort(key=lambda item: item[0])
        l.sort(key=lambda item: item[1])
        prompt_dict[k] = l
    return prompt_dict


def launch(args):
    img_paths = glob.glob(os.path.join(args.image_dir,'*.jpg'))
    with mp.Pool(mp.cpu_count()) as p:
        dicts = p.map(
            partial(process_image, scale=args.scale), img_paths[:])
    final_prompt_dict = {}
    for d in dicts:
        final_prompt_dict = {**final_prompt_dict, **d}

    with io.open(args.prompt_path, 'w', encoding='utf-8') as f:
        str_ = json.dumps(final_prompt_dict, indent=4, sort_keys=True, separators=(',', ': '), ensure_ascii=False)
        f.write(str_)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Script for generating prompts based on Spx object prior.')
    parser.add_argument('image_dir', type=str, 
                        help="Path to image directory, e.g., COCO's train2017 folder.")
    parser.add_argument('prompt_path', type=str,
                        help='Path to outfile (.json) for generated prompts.')
    parser.add_argument('--scale', dest='scale', default=10000,
                        help='Maximum number of prompts per image. Parameter S in the paper.')

    args = parser.parse_args()
    launch(args)
