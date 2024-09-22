# SOS: Segment Object System for Open-World Instance Segmentation With Object Priors

[Christian Wilms](https://www.inf.uni-hamburg.de/en/inst/ab/cv/people/wilms.html), [Tim Rolff](https://www.inf.uni-hamburg.de/en/inst/ab/cv/people/rolff.html), Maris Hillemann, [Robert Johanson](https://github.com/RbtJhs), [Simone Frintrop](https://www.inf.uni-hamburg.de/en/inst/ab/cv/people/frintrop.html)

This repository contains the code of our ECCV'24 paper **SOS: Segment Object System for Open-World Instance Segmentation With Object Priors** including the SOS system and the study on object-focused SAM. For the **results and pre-trained models**, check the tables below.

[[Paper](https://fiona.uni-hamburg.de/a3c1f3ec/wilms-etal-eccv-2024.pdf)], [[Supplementary Material](https://fiona.uni-hamburg.de/a3c1f3ec/wilms-etal-eccv-2024-supp-mat.pdf)], [[Video](https://youtu.be/ASN9UI9M3NU)]

The **Segment Object System (SOS)** is an open-world instance segmentation system capable of segmenting arbitrary objects in scenes. It utilizes rich pre-trained DINO self-attention maps as object priors to roughly localize unannotated objects in a training dataset. Subsequently, it applies the modern Segment Anything Model (SAM) to produce pseudo annotations from these rough localizations. Finally, a vanilla Mask R-CNN system is trained on original and pseudo annotations to provide strong generalization ability to unannotated objects. Note that a key difference to vanilla SAM is the focus of SOS on objects and not all coherent regions. 

![Object segmentation results of SAM and SOS](segment_objects.png)

Overall, SOS produces new state-of-the-art results on several open-world instance segmentation setups, showing strong generalization from annotated objects in training to unannotated objects during testing.

![OWIS results of Mask R-CNN, GGN, and SOS](owis_results.png)

## Installation

First, clone this repository with the ```--recursive``` option

```
git clone --recursive https://github.com/chwilms/SOS.git
cd SOS
git config -f .gitmodules submodule.SOS_SAM.branch main
git config -f .gitmodules submodule.SOS_MASKRCNN.branch main
git submodule update --recursive --remote
```

Depending on the parts of SOS that are needed, different installation requirements exist. If only the final Mask R-CNN in SOS is trained or tested with pre-trained weights, follow the installation instructions in the linked detectron2 repo. If SOS's Pseduo Annotation Creator is of interest, install the linked SAM repo and the ```requirements.txt``` in this repo. Similarly, the packages in the ```requirements.txt``` are needed to generate the prompts from the object priors. Note that only [generate_CAM_prompts.py](prompt_generation/generate_CAM_prompts.py) needs GPU support as well as ```torch```, ```torchvision```, and ```captum```. However, further repositories are needed to create object priors like [*Contour*](https://github.com/pdollar/edges), [*VOCUS2*](https://github.com/GeeeG/VOCUS2/tree/master), [*DeepGaze*](https://github.com/matthias-k/DeepGaze),  or [*DINO*](https://github.com/facebookresearch/dino).

## Usage
The entire SOS pipeline consists of five steps. Given the intermediate results for most steps ([see below](https://github.com/chwilms/SOS/tree/main#Prompts,-Models,-and-Results)), it's possible to start at an almost arbitrary step. The pipeline starts with the object prior generation (step 1) with the subsequent creation of the object-focused point prompts (step 2), known as Object Localization Module in the paper. This is followed by the segment generation with SAM based on the prompts (step 3), the filtering of the segment, and the creation of the final annotations combining pseudo annotations and original annotations (step 4). The final step is to train and test Mask R-CNN based on the new annotations/pre-trained weights.

In the subsequent steps, we assume the COCO dataset with VOC classes as original annotations and enrich these annotations with pseudo annotations based on the *DINO* object prior.

### Step 1: Object Priors

Generate the respective object priors, e.g., [*Contour*](https://github.com/pdollar/edges), [*VOCUS2*](https://github.com/GeeeG/VOCUS2/tree/master), [*DeepGaze*](https://github.com/matthias-k/DeepGaze),  or [*DINO*](https://github.com/facebookresearch/dino). For the object priors *Dist*, *Spx*, and *CAM*, the step is part of the prompt generation scripts (see step 2). For the *Grid* prior, directly move to step 3.

The result of this step is the object prior per training image in a suitable format.

### Step 2: Prompt

Once the object priors are generated, pick the respective script from the [```./prompt_generation```](https://github.com/chwilms/SOS/tree/main/prompt_generation) and set the parameters accordingly. For instance, to generate the DINO-based prompts with default parameters given DINO self-attention maps in ```/data/SOS/dino_attMaps``` and the images in ```/data/SOS/coco/train2017```, call 

```
python generate_DINO_prompts.py /data/SOS/coco/train2017 /data/SOS/dino_attMaps /data/SOS/prompts/prompts_DINO.json
```

The result of this step is a file with the object-focused point prompts based on a given object prior.

### Step 3: Segments

Based on the generated prompts, apply SAM from the [linked sub-repo](https://github.com/chwilms/SOS_segment-anything) on the training images by calling ```applySAM.py``` with an appropriate SAM checkpoint and generating the output segments.

```
python applySAM.py /data/SOS/coco/train2017 /data/SOS/SAM_checkpoints/sam_vit_h_4b8939.pth /data/SOS/prompts/prompts_DINO.json /data/SOS/segments/segments_DINO.json
```

If the *Grid* object prior is used, directly call ```applySAM_grid.py``` without providing a prompt file 

```
python applySAM_grid.py /data/SOS/coco/train2017 /data/SOS/SAM_checkpoints/sam_vit_h_4b8939.pth /data/SOS/segments/segments_Grid.json
```

The result of this step is a file with the object segments based on a given object prior.

### Step 4: Annotations

Given the segments generated by SAM and the original annotations of the known classes like the VOC classes from the COCO ```train2017``` dataset, this step creates the merged annotations by filtering the segments yielding pseudo annotations. Call ```combineAnnotations.py``` with paths to the original annotation, the SAM segments, and the output annotation file path as well as optional parameters

```
python combineAnnotations.py /data/SOS/coco/annotations/instances_train2017_voc.json /data/SOS/segments/segments_DINO.json /data/SOS/coco/annotations/instances_train2017_voc_SOS_DINO.json
```

The result of this step is a file with the merged annotations.

### Step 5: Training/Testing Mask R-CNN

Using the merged annotations, this step trains a class-agnostic Mask R-CNN, resulting in the final SOS open-world instance segmentation system. To train Mask R-CNN in a class-agnostic manner and the merged annotation in ```/data/SOS/coco/annotations/instances_train2017_voc_SOS_DINO.json```, use the [linked detectron2 sub-repo](https://github.com/chwilms/SOS_detectron2/tree/main) and first provide the base directory of the data followed by calling the training script.

```
export DETECTRON2_DATASETS=/data/SOS/
./tools/train_net.py --config-file ./configs/COCO-OpenWorldInstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml --num-gpus 8
```

If necessary, change the annotation file in Mask R-CNN's configuration file or through the command line call. As described in the sub repo's readme, annotation files following the above naming convention will be registered automatically.

To test SOS's Mask R-CNN with pre-trained weights, call the training script with the ```--eval-only``` and a respective file for the pre-trained weights. Note that this will default to a test on the COCO ```val2017``` dataset. For evaluation in this setup (cross-category, see paper), we use the code provided by [Saito et al.](https://ksaito-ut.github.io/openworld_ldet/).

```
./tools/train_net.py --config-file ./configs/COCO-OpenWorldInstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml --num-gpus 8 --eval-only MODEL.WEIGHTS /data/SOS/maskrcnn_weights/SOS_DINO_coco_voc.pth
```

## Prompts, Models, and Results

This section provides the intermediate results of SOS and our object prior study, including pre-trained models for the final Mask R-CNN system in SOS based on pseudo annotations and original annotations.

### Study Results

Training dataset COCO ```train2017``` dataset with original annotations for VOC classes (20 classes), test dataset COCO ```val2017``` dataset with original annotations of non-VOC classes (60 classes). Note that we only use 1/4 of the full schedule for training Mask R-CNN here.

Obejct Prior  | AP | AR | F $_1$ | | 
------------- | ------------- |  ------------- | ------------- |  ------------- |
SOS+*Grid* | 3.8 | 36.5 | 6.9 | [final merged annotations](https://fiona.uni-hamburg.de/a3c1f3ec/instancestrain2017vocsosgrid.zip), [pre-trained Mask R-CNN](https://fiona.uni-hamburg.de/a3c1f3ec/studysosgrid.pth), [OWIS detections](https://fiona.uni-hamburg.de/a3c1f3ec/val2017studysosgrid.json)|
SOS+*Dist* | 3.4 | 27.4 | 6.0 | [prompts](https://fiona.uni-hamburg.de/a3c1f3ec/promptsdist.json), [final merged annotations](https://fiona.uni-hamburg.de/a3c1f3ec/instancestrain2017vocsosdist.zip), [pre-trained Mask R-CNN](https://fiona.uni-hamburg.de/a3c1f3ec/studysosdist.pth), [OWIS detections](https://fiona.uni-hamburg.de/a3c1f3ec/val2017studysosdist.json)|
SOS+*Spx* | 5.6 | 34.8 | 9.6 | [prompts](https://fiona.uni-hamburg.de/a3c1f3ec/promptsspx.json), [final merged annotations](https://fiona.uni-hamburg.de/a3c1f3ec/instancestrain2017vocsosspx.zip), [pre-trained Mask R-CNN](https://fiona.uni-hamburg.de/a3c1f3ec/studysosspx.pth), [OWIS detections](https://fiona.uni-hamburg.de/a3c1f3ec/val2017studysosspx.json)|
SOS+*Contour* | 5.6 | 36.6 | 9.7 | [prompts](https://fiona.uni-hamburg.de/a3c1f3ec/promptscontour.json), [final merged annotations](https://fiona.uni-hamburg.de/a3c1f3ec/instancestrain2017vocsoscontour.zip), [pre-trained Mask R-CNN](https://fiona.uni-hamburg.de/a3c1f3ec/studysoscontour.pth), [OWIS detections](https://fiona.uni-hamburg.de/a3c1f3ec/val2017studysoscontour.json)|
SOS+*VOCUS2* | 6.1 | 37.7 | 10.5 | [prompts](https://fiona.uni-hamburg.de/a3c1f3ec/promptsvocus.json), [final merged annotations](https://fiona.uni-hamburg.de/a3c1f3ec/instancestrain2017vocsosvocus.zip), [pre-trained Mask R-CNN](https://fiona.uni-hamburg.de/a3c1f3ec/studysosvocus.pth), [OWIS detections](https://fiona.uni-hamburg.de/a3c1f3ec/val2017studysosvocus.json)|
SOS+*DeepGaze* | 5.4 | 35.9 | 9.4 | [prompts](https://fiona.uni-hamburg.de/a3c1f3ec/promptsdeepgaze.json), [final merged annotations](https://fiona.uni-hamburg.de/a3c1f3ec/instancestrain2017vocsosdeepgaze.zip), [pre-trained Mask R-CNN](https://fiona.uni-hamburg.de/a3c1f3ec/studysosdeepgaze.pth), [OWIS detections](https://fiona.uni-hamburg.de/a3c1f3ec/val2017studysosdeepgaze.json)|
SOS+*CAM* | 5.4 | 36.7 | 9.4 | [prompts](https://fiona.uni-hamburg.de/a3c1f3ec/promptscam.json), [final merged annotations](https://fiona.uni-hamburg.de/a3c1f3ec/instancestrain2017vocsoscam.zip), [pre-trained Mask R-CNN](https://fiona.uni-hamburg.de/a3c1f3ec/studysoscam.pth), [OWIS detections](https://fiona.uni-hamburg.de/a3c1f3ec/val2017studysoscam.json)|
SOS+*DINO* | 8.9 | 38.1 | 14.4 | [prompts](https://fiona.uni-hamburg.de/a3c1f3ec/promptsdino.json), [final merged annotations](https://fiona.uni-hamburg.de/a3c1f3ec/instancestrain2017vocsosdino.zip), [pre-trained Mask R-CNN](https://fiona.uni-hamburg.de/a3c1f3ec/studysosdino.pth), [OWIS detections](https://fiona.uni-hamburg.de/a3c1f3ec/val2017studysosdino.json)|
SOS+*U-Net* | 7.3 | 37.3 | 12.2 | [prompts](https://fiona.uni-hamburg.de/a3c1f3ec/promptsunet.json), [final merged annotations](https://fiona.uni-hamburg.de/a3c1f3ec/instancestrain2017vocsosunet.zip), [pre-trained Mask R-CNN](https://fiona.uni-hamburg.de/a3c1f3ec/studysosunet.pth), [OWIS detections](https://fiona.uni-hamburg.de/a3c1f3ec/val2017studysosunet.json)|


### OWIS: Cross-category COCO (VOC) -> COCO (non-VOC)

Training dataset is COCO ```train2017``` dataset with original annotations for VOC classes (20 classes), test dataset is COCO ```val2017``` dataset with original annotations of non-VOC classes (60 classes).

Obejct Prior  | AP | AR | F $_1$ | | 
------------- | ------------- |  ------------- | ------------- |  ------------- |
[Mask R-CNN](https://openaccess.thecvf.com/content_ICCV_2017/papers/He_Mask_R-CNN_ICCV_2017_paper.pdf) | 1.0 | 8.2 | 1.8 | [code](https://github.com/facebookresearch/detectron2)|
[SAM](https://arxiv.org/abs/2304.02643) | 3.6 | 48.1 | 6.7 | [code](https://github.com/facebookresearch/segment-anything)|
[OLN](https://arxiv.org/abs/2108.06753) | 4.2 | 28.4 | 7.3 | [code](https://github.com/mcahny/object_localization_network)|
[LDET](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136840265.pdf) | 4.3 | 24.8 | 7.3 | [code](https://ksaito-ut.github.io/openworld_ldet/)|
[GGN](https://openaccess.thecvf.com/content/CVPR2022/papers/Wang_Open-World_Instance_Segmentation_Exploiting_Pseudo_Ground_Truth_From_Learned_Pairwise_CVPR_2022_paper.pdf) | 4.9 | 28.3 | 8.4 | [code](https://github.com/facebookresearch/Generic-Grouping)|
[SWORD](https://openaccess.thecvf.com/content/ICCV2023/papers/Wu_Exploring_Transformers_for_Open-world_Instance_Segmentation_ICCV_2023_paper.pdf) | 4.8 | 30.2 | 8.3 | |
[UDOS](https://openaccess.thecvf.com/content/CVPR2024W/L3D-IVU/papers/Kalluri_Open-world_Instance_Segmentation_Top-down_Learning_with_Bottom-up_Supervision_CVPRW_2024_paper.pdf) | 2.9 | 34.3 | 5.3 | [code](https://tarun005.github.io/UDOS/)|
**SOS (ours)** | 8.9 | 29.3 | 14.5 | [final merged annotations](https://fiona.uni-hamburg.de/a3c1f3ec/promptsdino.json), [pre-trained Mask R-CNN](https://fiona.uni-hamburg.de/a3c1f3ec/sosdinococovoc.pth), [OWIS detections](https://fiona.uni-hamburg.de/a3c1f3ec/val2017sosdinococovoc2cocononvoc.json)|

### OWIS: Cross-dataset COCO -> LVIS

Training dataset is COCO ```train2017``` dataset with all original annotations (80 classes), test dataset is LVIS validation dataset with all original annotations.

Obejct Prior  | AP | AR | F $_1$ | | 
------------- | ------------- |  ------------- | ------------- |  ------------- |
[Mask R-CNN](https://openaccess.thecvf.com/content_ICCV_2017/papers/He_Mask_R-CNN_ICCV_2017_paper.pdf) | 7.5 | 23.6 | 11.4 | [code](https://github.com/facebookresearch/detectron2)|
[SAM](https://arxiv.org/abs/2304.02643) | 6.8 | 45.1 | 11.8 | [code](https://github.com/facebookresearch/segment-anything)|
[LDET](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136840265.pdf) | 6.7 | 24.8 | 10.5 | [code](https://ksaito-ut.github.io/openworld_ldet/)|
[GGN](https://openaccess.thecvf.com/content/CVPR2022/papers/Wang_Open-World_Instance_Segmentation_Exploiting_Pseudo_Ground_Truth_From_Learned_Pairwise_CVPR_2022_paper.pdf) | 6.5 | 27.0 | 10.5 | [code](https://github.com/facebookresearch/Generic-Grouping)|
[SOIS](https://arxiv.org/abs/2208.09023) | - | 25.2 | - | |
[OpenInst](https://arxiv.org/abs/2303.15859) | - | 29.3 | - | |
[UDOS](https://openaccess.thecvf.com/content/CVPR2024W/L3D-IVU/papers/Kalluri_Open-world_Instance_Segmentation_Top-down_Learning_with_Bottom-up_Supervision_CVPRW_2024_paper.pdf) | 3.9 | 24.9 | 6.7 | [code](https://tarun005.github.io/UDOS/)|
**SOS (ours)** | 8.1 | 33.3 | 13.3 | [final merged annotations](https://fiona.uni-hamburg.de/a3c1f3ec/instancestrain2017sosdino.zip), [pre-trained Mask R-CNN](https://fiona.uni-hamburg.de/a3c1f3ec/sosdinococo.pth), [OWIS detections](https://fiona.uni-hamburg.de/a3c1f3ec/lvissosdinococo2lvis.json)|

### OWIS: Cross-dataset COCO -> ADE20k

Training dataset is COCO ```train2017``` dataset with all original annotations (80 classes), test dataset is ADE20k validation dataset with all original annotations.

Obejct Prior  | AP | AR | F $_1$ | | 
------------- | ------------- |  ------------- | ------------- |  ------------- |
[Mask R-CNN](https://openaccess.thecvf.com/content_ICCV_2017/papers/He_Mask_R-CNN_ICCV_2017_paper.pdf) | 6.9 | 11.9 | 8.7 | [code](https://github.com/facebookresearch/detectron2), OWIS detections|
[OLN](https://arxiv.org/abs/2108.06753) | - | 20.4 | - | [code](https://github.com/mcahny/object_localization_network)|
[LDET](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136840265.pdf) | 9.5 | 18.5 | 12.6 | [code](https://ksaito-ut.github.io/openworld_ldet/)|
[GGN](https://openaccess.thecvf.com/content/CVPR2022/papers/Wang_Open-World_Instance_Segmentation_Exploiting_Pseudo_Ground_Truth_From_Learned_Pairwise_CVPR_2022_paper.pdf) | 9.7 | 21.0 | 13.3 | [code](https://github.com/facebookresearch/Generic-Grouping)|
[UDOS](https://openaccess.thecvf.com/content/CVPR2024W/L3D-IVU/papers/Kalluri_Open-world_Instance_Segmentation_Top-down_Learning_with_Bottom-up_Supervision_CVPRW_2024_paper.pdf) | 7.6 | 22.9 | 11.4 | [code](https://tarun005.github.io/UDOS/)|
**SOS (ours)** | 12.5 | 26.5 | 17.0 | [final merged annotations](https://fiona.uni-hamburg.de/a3c1f3ec/instancestrain2017sosdino.zip), [pre-trained Mask R-CNN](https://fiona.uni-hamburg.de/a3c1f3ec/sosdinococo.pth), [OWIS detections](https://fiona.uni-hamburg.de/a3c1f3ec/ade20ksosdinococo2ade20k.json)|

### OWIS: Cross-dataset COCO -> UVO

Training dataset is COCO ```train2017``` dataset with all original annotations (80 classes), test dataset is UVO sparse dataset with all original annotations.

Obejct Prior  | AP | AR | F $_1$ | | 
------------- | ------------- |  ------------- | ------------- |  ------------- |
[Mask R-CNN](https://openaccess.thecvf.com/content_ICCV_2017/papers/He_Mask_R-CNN_ICCV_2017_paper.pdf) | 20.7 | 36.7 | 26.5 | [code](https://github.com/facebookresearch/detectron2)|
[SAM](https://arxiv.org/abs/2304.02643) | 11.3 | 50.1 | 18.4 | [code](https://github.com/facebookresearch/segment-anything)|
[OLN](https://arxiv.org/abs/2108.06753) | - | 41.4 | - | [code](https://github.com/mcahny/object_localization_network)|
[LDET](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136840265.pdf) | 22.0 | 40.4 | 28.5 | [code](https://ksaito-ut.github.io/openworld_ldet/)|
[GGN](https://openaccess.thecvf.com/content/CVPR2022/papers/Wang_Open-World_Instance_Segmentation_Exploiting_Pseudo_Ground_Truth_From_Learned_Pairwise_CVPR_2022_paper.pdf) | 20.3 | 43.4 | 27.7 | [code](https://github.com/facebookresearch/Generic-Grouping)|
[UDOS](https://openaccess.thecvf.com/content/CVPR2024W/L3D-IVU/papers/Kalluri_Open-world_Instance_Segmentation_Top-down_Learning_with_Bottom-up_Supervision_CVPRW_2024_paper.pdf) | 10.6 | 43.1 | 17.0 | [code](https://tarun005.github.io/UDOS/)|
**SOS (ours)** | 20.9 | 42.3 | 28.0 | [final merged annotations](https://fiona.uni-hamburg.de/a3c1f3ec/instancestrain2017sosdino.zip), [pre-trained Mask R-CNN](https://fiona.uni-hamburg.de/a3c1f3ec/sosdinococo.pth), [OWIS detections](https://fiona.uni-hamburg.de/a3c1f3ec/uvososdinococo2uvo.json)|

## Cite SOS

If you use SOS or the study on the object priors to focus prompts in SAM, cite our paper:

```
@inproceedings{WilmsEtAlECCV2024,
  title = {{SOS}: Segment Object System for Open-World Instance Segmentation With Object Priors},
  author = {Christian Wilms and Tim Rolff and Maris Hillemann and Robert Johanson and Simone Frintrop},
  booktitle = {European Conference on Computer Vision (ECCV)},
  year = {2024}
}
```
