import os.path

from numpy.random import choice
import numpy as np

import multiprocessing as mp
import io
import json
import glob

from torchvision import models
from torchvision import transforms

from captum.attr import LayerGradCam
from captum.attr._utils.attribution import LayerAttribution

from PIL import Image

import torch
import torch.nn.functional as F

import argparse

from functools import partial


def process_image(image_path, num_samples, prune_size, min_score, cls_score):
    model = models.resnet50(pretrained=True)
    model = model.eval()

    transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor()])
    transform_normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    img = Image.open(image_path)
    img = img.convert('RGB')
    transformed_img = transform(img)

    input_img = transform_normalize(transformed_img)
    input_img = input_img.unsqueeze(0)

    output = model(input_img)
    output = F.softmax(output, dim=1)
    prediction_score, pred_label_idx = torch.topk(output, 1)

    pred_label_idx.squeeze_()
    prediction_score.squeeze_()

    cam = LayerGradCam(model, model.layer4[2].conv3)
    overall_cam_result = np.zeros(input_img.shape[2:])
    if prediction_score > cls_score:
        attributions_ig = LayerAttribution.interpolate(cam.attribute(input_img, pred_label_idx), input_img.shape[2:],
                                                       interpolate_mode='bilinear')
        res = np.maximum(attributions_ig.detach().numpy()[0][0], 0)
        overall_cam_result = np.maximum(overall_cam_result, res)
    else:
        return {}
    overall_cam_result = overall_cam_result - np.min(overall_cam_result)
    object_prior = overall_cam_result / np.max(overall_cam_result)

    object_prior = object_prior.astype(float)
    object_prior = object_prior / np.max(object_prior)

    img_id = int(image_path.split('/')[-1].split('.')[0])
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
    img_paths = glob.glob(os.path.join(args.image_dir, '*.jpg'))
    with mp.Pool(mp.cpu_count()) as p:
        dicts = p.map(
            partial(process_image, num_samples=args.num_samples, prune_size=args.prune_size, min_score=args.min_score,
                    cls_score=args.cls_score), img_paths[:])
    final_prompt_dict = {}
    for d in dicts:
        final_prompt_dict = {**final_prompt_dict, **d}

    with io.open(args.prompt_path, 'w', encoding='utf-8') as f:
        str_ = json.dumps(final_prompt_dict, indent=4, sort_keys=True, separators=(',', ': '), ensure_ascii=False)
        f.write(str_)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Script for generating prompts based on CAM object prior.')
    parser.add_argument('image_dir', type=str, required=True,
                        help="Path to image directory, e.g., COCO's train2017 folder.")
    parser.add_argument('prompt_path', type=str, required=True,
                        help='Path to outfile (.json) for generated prompts.')
    parser.add_argument('--samples', dest='num_samples', default=50,
                        help='Maximum number of prompts per image. Parameter S in the paper.')
    parser.add_argument('--prune', dest='prune_size', default=4,
                        help='Size of pruned region around each extracted prompt coordinate. Parameter N in the paper.')
    parser.add_argument('--score', dest='min_score', default=0.7,
                        help='Minimum value of an object prior pixel to be considered a prompt.')
    parser.add_argument('--cls_score', dest='cls_score', default=0.2,
                        help='Minimum class logit of the most likely class predicted by the CNN that is necessary to '
                             'generate prompts.')

    args = parser.parse_args()
    launch(args)
