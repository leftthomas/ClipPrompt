# ClipPrompt

A PyTorch implementation of ClipPrompt based on CVPR 2023 paper
[CLIP for All Things Zero-Shot Sketch-Based Image Retrieval, Fine-Grained or Not](https://openaccess.thecvf.com/content/CVPR2023/html/Sain_CLIP_for_All_Things_Zero-Shot_Sketch-Based_Image_Retrieval_Fine-Grained_or_CVPR_2023_paper.html).

![Network Architecture](result/arch.png)

## Requirements

- [Anaconda](https://www.anaconda.com/download/)
- [PyTorch](https://pytorch.org)

```
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

- [TorchMetrics](https://lightning.ai/docs/torchmetrics/stable/)

```
conda install -c conda-forge torchmetrics
```

- [CLIP](https://github.com/openai/CLIP)

```
pip install git+https://github.com/openai/CLIP.git
```

## Dataset

[Sketchy Extended](http://sketchy.eye.gatech.edu) and
[TU-Berlin Extended](http://cybertron.cg.tu-berlin.de/eitz/projects/classifysketch/) datasets are used in this repo, you
could download these datasets from official websites, or download them from
[Google Drive](https://drive.google.com/drive/folders/1lce41k7cGNUOwzt-eswCeahDLWG6Cdk0?usp=sharing). The data directory
structure is shown as follows:

 ```
├──sketchy
   ├── train
       ├── sketch
           ├── airplane
               ├── n02691156_58-1.jpg
               └── ...
           ...
       ├── photo
           same structure as sketch
   ├── val
      same structure as train
      ...
├──tuberlin
   same structure as sketchy
   ...
```

## Usage

To train a model on `Sketchy Extended` dataset, run:

```
python main.py --mode train --data_name sketchy
```

To test a model on `Sketchy Extended` dataset, run:

```
python main.py --mode test --data_name sketchy --query_name <query image path>
```

common arguments:

```
--data_root                   Datasets root path [default value is '/home/data']
--data_name                   Dataset name [default value is 'sketchy'](choices=['sketchy', 'tuberlin'])
--prompt_num                  Number of prompt embedding [default value is 3]
--save_root                   Result saved root path [default value is 'result']
--mode                        Mode of the script [default value is 'train'](choices=['train', 'test'])
```

train arguments:

```
--batch_size                  Number of images in each mini-batch [default value is 64]
--epochs                      Number of epochs over the model to train [default value is 60]
--triplet_margin              Margin of triplet loss [default value is 0.3]
--encoder_lr                  Learning rate of encoder [default value is 1e-4]
--prompt_lr                   Learning rate of prompt embedding [default value is 1e-3]
--cls_weight                  Weight of classification loss [default value is 0.5]
--seed                        Random seed (-1 for no manual seed) [default value is -1]
```

test arguments:

```
--query_name                  Query image path [default value is '/home/data/sketchy/val/sketch/cow/n01887787_591-14.jpg']
--retrieval_num               Number of retrieved images [default value is 8]
```

## Benchmarks

The models are trained on one NVIDIA GeForce RTX 3090 (24G) GPU. `seed` is `42`, `prompt_lr` is `1e-3`
and `distance function` is `1.0 - F.cosine_similarity(x, y)`, the other hyperparameters are the default values.

<table>
<thead>
  <tr>
    <th rowspan="3">Dataset</th>
    <th rowspan="3">Prompt Num</th>
    <th rowspan="3">mAP@200</th>
    <th rowspan="3">mAP@all</th>
    <th rowspan="3">P@100</th>
    <th rowspan="3">P@200</th>
    <th rowspan="3">Download</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td align="center">Sketchy Extended</td>
    <td align="center">3</td>
    <td align="center">71.9</td>
    <td align="center">64.3</td>
    <td align="center">70.8</td>
    <td align="center">68.1</td>
    <td align="center"><a href="https://mega.nz/folder/vXpAiQAI#AM3WSis-wE_r2-QQHzfjSA">MEGA</a></td>
  </tr>
  <tr>
    <td align="center">TU-Berlin Extended</td>
    <td align="center">3</td>
    <td align="center">75.3</td>
    <td align="center">66.0</td>
    <td align="center">73.9</td>
    <td align="center">69.7</td>
    <td align="center"><a href="https://mega.nz/folder/uHRQBZYA#uqCSuW-WZRBMVDu44srBmw">MEGA</a></td>
  </tr>
</tbody>
</table>

## Results

![vis](result/vis.png)
