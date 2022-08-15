#!/usr/bin/env pyhton3
# %%
from pathlib import Path
import sys
from typing import Dict, Tuple
if (p := './RetCCL') not in sys.path:
    sys.path = [p] + sys.path

import ResNet
import torch.nn as nn
import torch
from torchvision import transforms
import math
import os
from concurrent import futures
from matplotlib import pyplot as plt
import openslide
import PIL
from tqdm import tqdm
import numpy as np
from fastai.vision.all import load_learner
import pandas as pd

# use all the threads
torch.set_num_threads(os.cpu_count())
torch.set_num_interop_threads(os.cpu_count())


slide_dir = Path('/mnt/Sirius_02_empty/CPTAC_IMGS_PATH/CPTAC_BRCA')

def _load_tile(
    slide: openslide.OpenSlide, pos: Tuple[int, int], stride: Tuple[int, int], target_size: Tuple[int, int]
) -> np.ndarray:
    # Loads part of a WSI. Used for parallelization with ThreadPoolExecutor
    tile = slide.read_region(pos, 0, stride).convert('RGB').resize(target_size)
    return np.array(tile)


def load_slide(slide: openslide.OpenSlide, target_mpp: float = 256/224) -> np.ndarray:
    """Loads a slide into a numpy array."""
    # We load the slides in tiles to
    #  1. parallelize the loading process
    #  2. not use too much data when then scaling down the tiles from their
    #     initial size
    steps = 8
    stride = np.ceil(np.array(slide.dimensions)/steps).astype(int)
    slide_mpp = float(slide.properties[openslide.PROPERTY_NAME_MPP_X])
    tile_target_size = np.round(stride*slide_mpp/target_mpp).astype(int)

    with futures.ThreadPoolExecutor(min(32, os.cpu_count())) as executor:
        # map from future to its (row, col) index
        future_coords: Dict[futures.Future, Tuple[int, int]] = {}
        for i in range(steps):  # row
            for j in range(steps):  # column
                future = executor.submit(
                    _load_tile, slide, (stride*(j, i)), stride, tile_target_size)
                future_coords[future] = (i, j)

        # write the loaded tiles into an image as soon as they are loaded
        im = np.zeros((*(tile_target_size*steps)[::-1], 3), dtype=np.uint8)
        for tile_future in tqdm(futures.as_completed(future_coords), total=steps*steps):
            i, j = future_coords[tile_future]
            tile = tile_future.result()
            x, y = tile_target_size * (j, i)
            im[y:y+tile.shape[0], x:x+tile.shape[1], :] = tile

    return im


def linear_to_conv2d(linear):
    """Converts a fully connected layer to a 1x1 Conv2d layer with the same weights."""
    conv = nn.Conv2d(in_channels=linear.in_features,
                     out_channels=linear.out_features, kernel_size=1)
    conv.load_state_dict({
        "weight": linear.weight.view(conv.weight.shape),
        "bias": linear.bias.view(conv.bias.shape),
    })
    return conv


tfms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# load base fully convolutional model (w/o pooling / flattening or head)
base_model = ResNet.resnet50(num_classes=128, mlp=False,
                             two_branch=False, normlinear=True)
pretext_model = torch.load('./xiyue-wang.pth')
base_model.avgpool = nn.Identity()
base_model.flatten = nn.Identity()
base_model.fc = nn.Identity()
base_model.load_state_dict(pretext_model, strict=True)
base_model.eval()


for csv in Path('/home/mvantreeck/Downloads/BRCA').glob('*.csv'):
    target = csv.name.split('-', 1)[0]
    df = pd.read_csv(csv).nsmallest(5, columns='loss')
    for _, (patient_name, preds_path) in df[['patient', 'path']].iterrows():
        fold = Path(preds_path).parent.name
        for slide_path in tqdm([slide for slide in slide_dir.glob('*.svs') if patient_name[1:] in slide.name]):
            print(slide_path)
            slide = openslide.OpenSlide(str(slide_path))
            outdir = Path('/mnt/Sirius_02_empty/Oliver_TCGA_runs/BRCA/val_mil_multitarget/')/target/fold/csv.name/slide_path.stem
            outdir.mkdir(parents=True, exist_ok=True)
            print(f'saving to {outdir}')

            # transform MIL model into fully convolutional equivalent
            learn = load_learner(Path('/mnt/Sirius_02_empty/Oliver_TCGA_runs/BRCA/mil_multitarget')/target/fold/'export.pkl')
            att = nn.Sequential(
                nn.AvgPool2d(7, 1),
                linear_to_conv2d(learn.encoder[0]),
                nn.ReLU(),
                linear_to_conv2d(learn.attention[0]),
                nn.Tanh(),
                linear_to_conv2d(learn.attention[2]),
            )
            score = nn.Sequential(
                nn.AvgPool2d(7, 1),
                linear_to_conv2d(learn.encoder[0]),
                nn.ReLU(),
                linear_to_conv2d(learn.head[3]),
            )

            # load WSI as one image
            slide_t = load_slide(slide)
            PIL.Image.fromarray(slide_t).save(outdir/f'{slide_path.stem}.jpg')

            # pass the WSI through the fully convolutional network
            # since our RAM is still too small, we do this in two steps
            # (if you run out of RAM, try upping the number of slices)
            no_slices = 2
            step = slide_t.shape[1]//no_slices
            slices = []
            for slice_i in range(no_slices):
                x = tfms(slide_t[:, slice_i*step:(slice_i+1)*step, :])
                with torch.inference_mode():
                    slices.append(base_model(x.unsqueeze(0)))
            feat_t = torch.concat(slices, 3).squeeze()
            # save the features (large)
            torch.save(feat_t, outdir/'feats.pkl')

            # calculate the attentions / scores according to the MIL model
            with torch.inference_mode():
                att_map = att(feat_t).squeeze()
                lower = torch.quantile(att_map, .01)
                att_map = att_map.where(att_map > lower, lower)
                att_map -= att_map.min()
                att_map /= att_map.max()

                score_map = score(feat_t).squeeze()
                score_map = torch.softmax(score_map, 0)

            att_map_im = PIL.Image.fromarray(
                np.uint8(plt.get_cmap('coolwarm')(att_map)*255.)).convert('RGB')
            att_map_im.save(outdir/'attention.png')
            slide_im = PIL.Image.fromarray(slide_t)
            PIL.Image.blend(slide_im, att_map_im.resize(slide_im.size),
                            0.75).save(outdir/'attention_overlayed.jpg')
            im = plt.get_cmap('coolwarm')(score_map[1])
            im[:, :, 3] = att_map * .8
            map_im = PIL.Image.fromarray(np.uint8(im*255.))
            map_im.save(outdir/'map.png')
            map_im = map_im.resize(slide_im.size, PIL.Image.Resampling.NEAREST)
            x = slide_im.copy().convert('RGBA')
            x.paste(map_im, mask=map_im)
            x.convert('RGB').save(outdir/'map_overlayed.jpg')
