from numpy.random import choice
import os.path
import numpy as np

import multiprocessing as mp
import io
import json
import glob

import argparse

from functools import partial


def process_image(path, center_aggregator, num_samples, prune_size, min_score):
    object_prior = np.load(center_aggregator)
    object_prior = object_prior.astype(float)
    object_prior = object_prior / np.max(object_prior)
    print(object_prior.shape)

    img_id = int(path.split('/')[-1].split('.')[0])
    object_prior[object_prior < min_score] = 0
    h, w = object_prior.shape

    xs = []
    ys = []
    samples = np.zeros_like(object_prior)
    for i in range(num_samples):
        flat_object_prior = object_prior.flatten().astype(float)
        if np.sum(flat_object_prior) == 0:
            break
        flat_object_prior /= np.sum(flat_object_prior)

        flat_indices = choice(range(len(flat_object_prior)), 1, False, flat_object_prior)
        x, y = np.unravel_index(flat_indices, object_prior.shape)
        x = int(x)
        y = int(y)
        xs.append(x / float(h))
        ys.append(y / float(w))
        samples[x, y] = 1
        object_prior[max(0, x - prune_size):min(x + prune_size + 1, h),
        max(0, y - prune_size):min(y + prune_size + 1, w)] = 0

    coordinates = zip(ys, xs)
    l = list(coordinates)
    l.sort(key=lambda item: item[0])
    l.sort(key=lambda item: item[1])
    prompt_dict = {img_id: l}
    return prompt_dict


def launch(args):
    img_paths = glob.glob(os.path.join(args.image_dir,'*.jpg'))
    with mp.Pool(mp.cpu_count()) as p:
        dicts = p.map(
            partial(process_image, center_aggregator=args.center_aggregator, num_samples=args.num_samples,
                    prune_size=args.prune_size, min_score=args.min_score), img_paths[:])
    final_prompt_dict = {}
    for d in dicts:
        final_prompt_dict = {**final_prompt_dict, **d}

    with io.open(args.prompt_path, 'w', encoding='utf-8') as f:
        str_ = json.dumps(final_prompt_dict, indent=4, sort_keys=True, separators=(',', ': '), ensure_ascii=False)
        f.write(str_)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Script for generating prompts based on Dist object prior.')
    parser.add_argument('image_dir', type=str, required=True,
                        help="Path to image directory, e.g., COCO's train2017 folder.")
    parser.add_argument('center_aggregator', type=str, required=True,
                        help='Probability distribution of object centers from known classes across provided images as '
                             'array in npy file.')
    parser.add_argument('prompt_path', type=str, required=True,
                        help='Path to outfile (.json) for generated prompts.')
    parser.add_argument('--samples', dest='num_samples', default=50,
                        help='Maximum number of prompts per image. Parameter S in the paper.')
    parser.add_argument('--prune', dest='prune_size', default=0,
                        help='Size of pruned region around each extracted prompt coordinate. Parameter N in the paper.')
    parser.add_argument('--score', dest='min_score', default=0.4,
                        help='Minimum value of an object prior pixel to be considered a prompt.')

    args = parser.parse_args()
    launch(args)
