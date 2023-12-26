<div align="center">
  <h1>Model Zoo</h1> 
</div>

To support open-source research, we release multiple checkpoints we have trained, with the respective training scripts. They should result in almost the same (or the same) performance with the reported ones in our paper.

While most pre-trained weights as mentioned in paper (*except single-dataset checkpoints for IQA and few-shot checkpoints, which are also easy to be reproduced in less than 10 minutes*) are released to facilitate future research, we strongly suggest in-practical usages to be based on **OneAlign**, our most capable and robust checkpoint co-trained on IQA, IAA, and VQA.

*You should comply with licenses from LLaMA-2 if you consider to use the checkpoints commercially.*
    
## *OneAlign*

- Huggingface Path

[q-future/one-align](https://huggingface.co/q-future/one-align)

- Training Dataset

KonIQ, SPAQ, KADID, AVA, LSVQ.

- Training JSON

[train_all.json](../playground/data/training_sft/train_all.json)

- Training script

[all_.sh](../scripts/all_.sh)

## Multi-task Checkpoints

### IQA + VQA

- Huggingface Path

[q-future/q-align-quality](https://huggingface.co/q-future/q-align-quality)

- Training Dataset

KonIQ, SPAQ, KADID, LSVQ.

- Training JSON

[train_iqa_vqa.json](../playground/data/training_sft/train_iqa_vqa.json)

- Training script

[iqa_vqa.sh](../scripts/iqa_vqa.sh)


### IQA + IAA

- Huggingface Path

[q-future/q-align-image](https://huggingface.co/q-future/q-align-image)

- Training Dataset

KonIQ, SPAQ, KADID, AVA.

- Training JSON

[train_iqa_iaa.json](../playground/data/training_sft/train_iqa_iaa.json)

- Training script

[iqa_iaa.sh](../scripts/iqa_iaa.sh)

### IAA + VQA

- Huggingface Path

[q-future/q-align-iaa-vqa](https://huggingface.co/q-future/q-align-iaa-vqa)

- Training Dataset

AVA, LSVQ.

- Training JSON

[train_ava_lsvq.json](../playground/data/training_sft/train_ava_lsvq.json)

- Training script

[vqa_iaa.sh](../scripts/vqa_iaa.sh)

## Single-task Checkpoints

### IQA

- Huggingface Path

[q-future/q-align-iqa](https://huggingface.co/q-future/q-align-iqa)

- Training Dataset

KonIQ, SPAQ, KADID.

- Training JSON

[train_koniq_spaq_kadid.json](../playground/data/training_sft/train_koniq_spaq_kadid.json)

- Training script

[iqa_mix.sh](../scripts/iqa_mix.sh)


### IAA

- Huggingface Path

[q-future/q-align-aesthetic](https://huggingface.co/q-future/q-align-aesthetic)

- Training Dataset

AVA.

- Training JSON

[train_ava.json](../playground/data/training_sft/train_ava.json)

- Training script

[l1_ava.sh](../scripts/l1_ava.sh)

### VQA

- Huggingface Path

[q-future/q-align-aesthetic](https://huggingface.co/q-future/q-align-vqa)

- Training Dataset

LSVQ.

- Training JSON

[train_lsvq.json](../playground/data/training_sft/train_lsvq.json)

- Training script

[l1_lsvq.sh](../scripts/l1_lsvq.sh)



