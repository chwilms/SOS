import io
import json
import numpy as np
from collections import defaultdict
from pycocotools import mask as maskUtils
from skimage.measure import regionprops
import multiprocessing as mp
from itertools import chain
import argparse
from functools import partial


def ann2RLE(img_dict, ann):
    h, w = img_dict[ann['image_id']]
    segm = ann['segmentation']
    if type(segm) == list:
        rles = maskUtils.frPyObjects(segm, h, w)
        rle = maskUtils.merge(rles)
    elif type(segm['counts']) == list:
        rle = maskUtils.frPyObjects(segm, h, w)
    else:
        rle = ann['segmentation']
    return rle


def process_image(img_id, results_dict, anns_dict, max_iou_threshold, sam_score_threshold, num_new_gt):
    img_result = results_dict[img_id]
    img_result.sort(key=lambda item: item['score'])
    img_result = img_result[::-1]
    new_gts = []
    h, w = img_result[0]['segmentation']['size']
    for r in img_result:
        if len(new_gts) == num_new_gt:
            break
        if r['score'] < sam_score_threshold:
            continue

        max_iou = 0
        for nGTs in new_gts:
            iou = maskUtils.iou([r['segmentation']], [nGTs['rle']], [False])[0]
            max_iou = max(max_iou, iou)
            if iou > max_iou_threshold:
                break
        if max_iou > max_iou_threshold:
            continue

        max_iou = 0
        for ann in anns_dict[img_id]:
            iou = maskUtils.iou([r['segmentation']], [ann['rle']], [False])[0]
            max_iou = max(max_iou, iou)
            if iou > max_iou_threshold:
                break
        if max_iou > max_iou_threshold:
            continue
        new_ann = {'iscrowd': False, 'category_id': 1, 'image_id': img_id}
        mask = maskUtils.decode(r['segmentation'])
        if np.sum(mask) < 1:
            continue
        new_ann['segmentation'] = r['segmentation']
        new_ann['rle'] = r['segmentation']
        props = regionprops(mask.astype(int))[0]
        x1, y1, x2, y2 = props.bbox
        bbox_h = x2 - x1
        bbox_w = y2 - y1
        new_ann['bbox'] = [y1, x1, bbox_w, bbox_h]
        new_ann['SAM_score'] = r['score']
        new_gts.append(new_ann)
    return new_gts


def launch(args):
    with open(args.base_annotations) as f:
        data = json.load(f)

    with open(args.SAM_segments) as f:
        results = json.load(f)

    anns_dict = defaultdict(list)
    img_dict = {}
    max_ann_ID = 0

    for img in data['images']:
        img_dict[img['id']] = ((img['height'], img['width']))

    for ann in data['annotations']:
        max_ann_ID = max(max_ann_ID, ann['id'])
        ann['rle'] = ann2RLE(img_dict, ann)
        ann['rle']["counts"] = ann['rle']["counts"].decode("utf-8")
        anns_dict[ann['image_id']].append(ann)

    results_dict = defaultdict(list)
    for result in results[:]:
        results_dict[result['image_id']].append(result)

    with mp.Pool(mp.cpu_count()//2) as p:
        new_anns = list(p.map(partial(process_image, results_dict=results_dict, anns_dict=anns_dict, max_iou_threshold=args.max_iou,
                                      sam_score_threshold=args.sam_score, num_new_gt=args.num_new_gt), results_dict.keys()))

    for newAnn in chain(*new_anns):
        max_ann_ID += 1
        newAnn['id'] = max_ann_ID
        data['annotations'].append(newAnn)

    with io.open(args.new_annotations, 'w', encoding='utf-8') as f:
        str_ = json.dumps(data, indent=4, sort_keys=True, separators=(',', ': '), ensure_ascii=False)
        f.write(str_)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script for combining the original annotations of the known classes "
                                                 "with SOS's pseudo-annotations generated from SAM segments based on "
                                                 "prompts from object priors.")
    parser.add_argument('base_annotations', type=str, required=True,
                        help='Path to original annotation file (json) of known classes.')
    parser.add_argument('SAM_segments', type=str, required=True,
                        help='Path to json file with segments generated with SAM as part of SOS.')
    parser.add_argument('new_annotations', type=str, required=True,
                        help='Path to output annotation file (.json) for merged original and pseudo annotations.')
    parser.add_argument('--iou', dest='max_iou', default=0.2, type=float,
                        help='Maximum IoU value between each pair of annotations (original or pseudo). Others will be '
                             'suppressed. Parameter tau_NMS in the paper.')
    parser.add_argument('--score', dest='sam_score', default=0.9, type=float,
                        help='Minimum value for the confidence score related to a SAM segment. Segments with lower '
                             'confidence will be removed. Parameter tau_conf in the paper.')
    parser.add_argument('--numGT', dest='num_new_gt', default=10, type=float,
                        help='Maximum number of pseudo annotations per image.')

    args = parser.parse_args()
    launch(args)
