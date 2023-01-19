# VinVL+L: Enriching Visual Representation with Location Context in VQA
![Examples](./docs/intro.gif)

## Introduction
We describe a novel method - VinVL+L - that enriches the visual representations (i.e. object tags and region features)
of the State-of-the-Art Vision and Language (VL) method - VinVL - with Location information. To verify the importance
of such metadata for VL models, we:

1. trained a Swin-B model on the Places365 dataset and obtained additional sets of visual and tag features; both were
made public to allow reproducibility and further experiments,

2. did an architectural update to the existing VinVL method to include the new feature sets,

3. provide a qualitative and quantitative evaluation.

By including just binary location metadata, the VinVL+L method provides incremental improvement to the State-of-the-Art
VinVL in Visual Question Answering (VQA). The VinVL+L achieved an accuracy of **64.85%** and increased the performance
by **+0.32%** in terms of accuracy on the GQA dataset; the statistical significance of the new representations is
verified via Approximate Randomization.

## Download
### Location features
<table>
    <thead>
        <th>Swin-Base</th>
        <th>Accuracy<sub>C</sub></th>
        <th>Accuracy<sub>IO</sub></th>
        <th>Scene tags</th>
        <th>Other Links</th>
    </thead>
    <tbody>
        <tr>
            <td>Best accuracy</td>
            <td align="center">56.0%</td>
            <td align="center">96.1%</td>
            <td align="center">
                <a href="https://drive.google.com/file/d/1KzmcIfWSThaaLlusKXPhEnm5i9jTvMiw">C</a> |
                <a href="https://drive.google.com/file/d/14LP4xXHr3Wg42iXr-1Sd77I81UIhmLVH">IO</a>
            </td>
            <td align="center">
                <a href="https://drive.google.com/file/d/19wLz7sDAYZ183D3Aiu5FsjRTuqhqzGDU">model</a> |
                <a href="https://drive.google.com/file/d/1pr5-5ZJfZNKt_Ilw6ToOS3580E-Igaj1">scene features</a>
            </td>
        </tr>
        <tr>
            <td>Best loss / Final epoch</td>
            <td align="center">53.3%</td>
            <td align="center">95.5%</td>
            <td align="center">
                <a href="https://drive.google.com/file/d/1XR6uMoMlDUaEvPYV5kfAeXRfUkKBijq8">C</a> |
                <a href="https://drive.google.com/file/d/1HH920NL_Ek5kLPFnNSj2NULH8_itLS0_">IO</a>
            </td>
            <td align="center">
                <a href="https://drive.google.com/file/d/1ZHnaFhJHZmyCN-98eojotFkHzBTn0HS1">model</a> |
                <a href="https://drive.google.com/file/d/1yIhZ65mPuuUQfr1nreFPggQqX4p6EXzL">scene features</a>
            </td>
        </tr>
    </tbody>
</table>

<sub>_Notes:_
<br/>
_#1: **C** refers to 365 location categories, and **IO** to their indoors/outdoors supercategories._
<br/>
_#2: The listed results are on the <a href="http://places2.csail.mit.edu/download.html">Places365</a> validation
     dataset; **C &rarr; IO** can be found
     <a href="https://docs.google.com/spreadsheets/d/1H7ADoEIGgbF_eXh9kcJjCs5j_r3VJwke4nebhkdzksg">here</a>._
</sub>

### VinVL+L

<table>
    <thead>
        <th>Scene tags</th>
        <th>Accuracy</th>
        <th>Binary</th>
        <th>Open</th>
        <th>Links</th>
    </thead>
    <tbody>
        <tr>
            <td>C</td>
            <td align="center">64.85%</td>
            <td align="center">82.59%</td>
            <td align="center">49.19%</td>
            <td align="center">
                <a href="https://drive.google.com/drive/folders/1A8rtOMPUXOyJ07-S_qEY4Xweo9Ygd_u9">model</a> |
                <a href="https://drive.google.com/file/d/1DcAdQ90s4Z2mgoGoIu-1ohKm58TSiSN3">server results</a>
            </td>
        </tr>
        <tr>
            <td>C+IO</td>
            <td align="center">64.71%</td>
            <td align="center">82.38%</td>
            <td align="center">49.12%</td>
            <td align="center">
                <a href="https://drive.google.com/drive/folders/1r6CKLseDVJEKrF2gJ3peJnYu6HYx-w0P">model</a> |
                <a href="https://drive.google.com/file/d/1m9WFq8l7uatCYQYBZajg0Q2ip0EnLZPF">server results</a>
            </td>
        </tr>
        <tr>
            <td>IO</td>
            <td align="center">64.65%</td>
            <td align="center">82.44%</td>
            <td align="center">48.94%</td>
            <td align="center">
                <a href="https://drive.google.com/drive/folders/1--kAAxFA-JRA0-l86rQAxgtaTUDgn5wN">model</a> |
                <a href="https://drive.google.com/file/d/1GdVyRzgqBEkvBf6SWxJZaRD29oTSqczY">server results</a>
            </td>
        </tr>
        <tr>
            <td>— <i>(reproduced VinVL)</i></td>
            <td align="center">64.53%</td>
            <td align="center">82.36%</td>
            <td align="center">48.79%</td>
            <td align="center">
                <a href="https://drive.google.com/drive/folders/1r01ruXXJXrJUCveXObqXMycRPo54jinw">model</a> |
                <a href="https://drive.google.com/file/d/1BZy5mQ_8pu9gmac6LSf4UYOnbLLyWIMH">server results</a>
            </td>
        </tr>
    </tbody>
</table>

<sub>_Notes:_
<br/>
_#1: The listed models do not use scene features. In the previous section are links to all features._
<br/>
_#2: The listed results are on the <a href="https://cs.stanford.edu/people/dorarad/gqa/about.html">GQA</a> test2019
     dataset; an official leaderboard can be found
     <a href="https://eval.ai/web/challenges/challenge-page/225/leaderboard/733/Accuracy">here</a>._
</sub>

## Usage

Follow [Oscar's instructions](https://github.com/microsoft/Oscar/blob/4788a7425cd0f9861ea80fed79528abbb72eb169/INSTALL.md)
for installation and download their data for GQA dataset
([Oscar](https://github.com/microsoft/Oscar/blob/4788a7425cd0f9861ea80fed79528abbb72eb169/DOWNLOAD.md#datasets)/[VinVL](https://github.com/microsoft/Oscar/blob/4788a7425cd0f9861ea80fed79528abbb72eb169//VinVL_DOWNLOAD.md#datasets)).
Additionally, you will need the [Timm library](https://github.com/rwightman/pytorch-image-models) in case of using our
Swin-B model, and `h5py` if you want to use the scene features.

To run VinVL+L script, you can follow the [original usage (see section GQA)](https://github.com/microsoft/Oscar/blob/4788a7425cd0f9861ea80fed79528abbb72eb169/VinVL_MODEL_ZOO.md#gqa);
you must type `python run_gqa_places.py`, then continue in using the same arguments and consider using the following
ones:
- `--wandb_entity "name"` in case you want to log your run in [WandB](https://wandb.ai/site), where `"name"` refers to
  your profile name. Do not forget that you will need to login during the first run via console.
- `--places_io_json "path_to_io_json"` to use indoors/outdoors (IO) scene tags.
- `--places_c_json "path_to_c_json"` to use 365 location categories (C) scene tags.
- `--places_feats_hdf5 "path_to_hdf5"` to use scene features.

If you want to use the Swin-B model provided by us, you can type the following code:

```python
# required packages
import timm
import torch
import torch.nn as nn

# predefined constants/arguments
ckpt_path = "./swin_base_model.pth" # checkpoints
target_size = 365 # of location categories

...

# get the timm model
model = timm.create_model(f"swin_base_patch4_window7_224_in22k", pretrained=True)

# get output layer and size of the previous one
last_layer = model.default_cfg['classifier']
num_ftrs = getattr(model, last_layer).in_features

# set the target size of the output layer
setattr(model, last_layer, nn.Linear(num_ftrs, target_size))
    
# set the checkpoints
model.load_state_dict(torch.load(ckpt_path))
```

To generate scene tags for Oscar/VinVL models, you need to save them in the json:
```json
{
  "<img_id>": "<scene_tag>",
  ...
}
```
where `"<img_id>"` is the id of the image (this is the key that even the original Oscar/VinVL scripts follow); for the
GQA dataset, the id is an image name without extension. The `"<scene_tag>"` is the location category - the first letter
always begins with a capital letter, e.g., "Living room". You can combine both 365 location categories (C) and
indoors/outdoors (IO) by typing them one after the other, e.g., "Hospital room Indoor" (C must be followed by the IO).
The same applies to scene features stored in hdf5, only the value will be a 2054-long vector.

## Possible issues

- `StopIteration: Caught StopIteration in replica 0 on device 0`.
  <details>
    <summary>Fix</summary>
    
    in [`./Oscar/oscar/modeling/modeling_bert.py`](https://github.com/microsoft/Oscar/blob/4788a7425cd0f9861ea80fed79528abbb72eb169/oscar/modeling/modeling_bert.py#L225) rewrite line 225 from:
    
    ```python
    extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
    ```
    to:
    ```python
    extended_attention_mask = extended_attention_mask.to(dtype=torch.float32) # fp16 compatibility
    ```
  </details>
    
  <details>
    <summary>Full exception</summary>
    
    ```commandline
    Traceback (most recent call last):
      File "run_gqa_places.py", line 1236, in <module>
        main()
      File "run_gqa_places.py", line 1154, in main
        global_step, tr_loss = train(args, train_dataset, eval_dataset, model, tokenizer)
      File "run_gqa_places.py", line 538, in train
        outputs = model(**inputs)
      File "/storage/brno2/home/vyskocj/.conda/envs/VinVL-g/lib/python3.7/site-packages/torch/nn/modules/module.py", line 1110, in _call_impl
        return forward_call(*input, **kwargs)
      File "/storage/brno2/home/vyskocj/.conda/envs/VinVL-g/lib/python3.7/site-packages/torch/nn/parallel/data_parallel.py", line 168, in forward
        outputs = self.parallel_apply(replicas, inputs, kwargs)
      File "/storage/brno2/home/vyskocj/.conda/envs/VinVL-g/lib/python3.7/site-packages/torch/nn/parallel/data_parallel.py", line 178, in parallel_apply
        return parallel_apply(replicas, inputs, kwargs, self.device_ids[:len(replicas)])
      File "/storage/brno2/home/vyskocj/.conda/envs/VinVL-g/lib/python3.7/site-packages/torch/nn/parallel/parallel_apply.py", line 86, in parallel_apply
        output.reraise()
      File "/storage/brno2/home/vyskocj/.conda/envs/VinVL-g/lib/python3.7/site-packages/torch/_utils.py", line 457, in reraise
        raise exception
    StopIteration: Caught StopIteration in replica 0 on device 0.
    Original Traceback (most recent call last):
      File "/storage/brno2/home/vyskocj/.conda/envs/VinVL-g/lib/python3.7/site-packages/torch/nn/parallel/parallel_apply.py", line 61, in _worker
        output = module(*input, **kwargs)
      File "/storage/brno2/home/vyskocj/.conda/envs/VinVL-g/lib/python3.7/site-packages/torch/nn/modules/module.py", line 1110, in _call_impl
        return forward_call(*input, **kwargs)
      File "./Oscar/oscar/modeling/modeling_bert.py", line 328, in forward
        attention_mask=attention_mask, head_mask=head_mask, img_feats=img_feats)
      File "/storage/brno2/home/vyskocj/.conda/envs/VinVL-g/lib/python3.7/site-packages/torch/nn/modules/module.py", line 1110, in _call_impl
        return forward_call(*input, **kwargs)
      File "./Oscar/oscar/modeling/modeling_bert.py", line 225, in forward
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
    StopIteration
    ```
  </details>

## Citation
Please consider citing the following paper in case you use this code or provided data:

```BibTeX
@inproceedings{vyskocil2023VinVL+L,
  title     = {VinVL+L: Enriching Visual Representation with Location Context in VQA},
  author    = {Vysko{\v{c}}il, Ji{\v{r}}{\'\i} and Picek, Luk{\'a}{\v{s}}},
  year      = {2023},
  booktitle = {Computer Vision Winter Workshop},
  series    = {{CEUR} Workshop Proceedings},
  month     = {February 15-17},
  address   = {Krems an der Donau, Austria},
}
```
