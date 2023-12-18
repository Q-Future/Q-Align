# Q-Align

![](fig/radar.png)


## Installation

If you only need to infer (or evaluate):

```shell
git clone https://github.com/Q-Future/Q-Align.git
cd Q-Align
pip install -e .
```

For training, you need to install additional dependencies as follows:

```shell
pip install -e ".[train]"
pip install flash_attn --no-build-isolation
```

## Quick Start: Scoring *Single* Image

### Quality Scorer

- CLI Interface

```shell
export DEFAULT_IMG_PATH=fig/singapore_flyer.jpg
python scorer.py --img_path $DEFAULT_IMG_PATH
```

- Python API

```python
from scorer import QAlignScorer
from PIL import Image

scorer = QAlignScorer()
img_list = [Image.open("fig/singapore_flyer.jpg")] # can be multiple images
print(scorer(img_list).tolist())
```

### Aesthetic Scorer

- CLI Interface

```shell
export DEFAULT_IMG_PATH=fig/singapore_flyer.jpg
python scorer.py --img_path $DEFAULT_IMG_PATH --aesthetic
```

- Python API

```python
from scorer import QAlignAestheticScorer
from PIL import Image

scorer = QAlignAestheticScorer()
img_list = [Image.open("fig/singapore_flyer.jpg")] # can be multiple images
print(scorer(img_list).tolist())
```

## Training & Evaluation

### Get Datasets

Download all datasets needed together.

```python
import os, glob
from huggingface_hub import snapshot_download


snapshot_download("q-future/q-align-datasets", repo_type="dataset", local_dir="./playground/data", local_dir_use_symlinks=False)

gz_files = glob.glob("playground/data/*.tar")

for gz_file in gz_files:
    print(gz_file)
    os.system("tar -xf {} -C ./playground/data/".format(gz_file))
```

### Evaluation

After preparing the datasets, you can evaluate pre-trained Q-Align as follows:

- Image Quality Assessment (IQA)

```shell
python iqa_eval.py --model_path q-future/q-align-koniq-spaq-v0 --device cuda:0
```

- Image Aesthetic Assessment (IAA)

```shell
python iaa_eval.py --model_path q-future/q-align-aesthetic --device cuda:0
```

- Video Quality Assessment (VQA)

```shell
python vqa_eval.py --model_path q-future/q-align-koniq-spaq-v0 --device cuda:0
```


### Training

#### L1: *Single* Image Quality Assessment

- Training Q-Align with KonIQ-10k:

```shell
sh scripts/l1_koniq.sh
```

- Training Q-Align with mixture of KonIQ-10k and SPAQ:

```shell
sh scripts/l1_koniq_spaq_mix.sh
```

- Training Q-Align Aesthetic Predictor with AVA dataset:

```shell
sh scripts/l1_ava.sh
```

At least 4\*A6000 GPUs or 2\*A100 GPUs will be enough for the training.


#### L2: ????

Coming soon.


