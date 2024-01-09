# Model Zoo

<div align="center">
  <h1>Model Zoo</h1> 
</div>

### Q-Align Models

To support open-source research, we release multiple checkpoints we have trained, with the respective training scripts. They should result in almost the same (or the same) performance with the reported ones in our paper.

While most pre-trained weights as mentioned in paper (*except single-dataset checkpoints for IQA and few-shot checkpoints, which are also easy to be reproduced in less than 10 minutes*) are released to facilitate future research, we strongly suggest in-practical usages to be based on ***OneAlign***, our most capable and robust checkpoint co-trained on IQA, IAA, and VQA.

*You should comply with licenses from LLaMA-2 if you consider to use the checkpoints commercially.*
    

| Checkpoint Type   | Huggingface Path | Training Dataset         | Training Labels                                     | Training script             |
|-------------------|------------------|--------------------------|---------------------------------------------------|-----------------------------|
| ***OneAlign***      | [q-future/one-align](https://huggingface.co/q-future/one-align) | KonIQ, SPAQ, KADID, AVA, LSVQ | [train_all.json](../playground/data/training_sft/train_all.json) | [all_.sh](../scripts/all_.sh) |
| **IQA + VQA**     | [q-future/q-align-quality](https://huggingface.co/q-future/q-align-quality) | KonIQ, SPAQ, KADID, LSVQ | [train_iqa_vqa.json](../playground/data/training_sft/train_iqa_vqa.json) | [iqa_vqa.sh](../scripts/iqa_vqa.sh) |
| **IQA + IAA**     | [q-future/q-align-image](https://huggingface.co/q-future/q-align-image) | KonIQ, SPAQ, KADID, AVA | [train_iqa_iaa.json](../playground/data/training_sft/train_iqa_iaa.json) | [iqa_iaa.sh](../scripts/iqa_iaa.sh) |
| **IAA + VQA**     | [q-future/q-align-iaa-vqa](https://huggingface.co/q-future/q-align-iaa-vqa) | AVA, LSVQ | [train_ava_lsvq.json](../playground/data/training_sft/train_ava_lsvq.json) | [vqa_iaa.sh](../scripts/vqa_iaa.sh) |
| **IQA**           | [q-future/q-align-iqa](https://huggingface.co/q-future/q-align-iqa) | KonIQ, SPAQ, KADID | [train_koniq_spaq_kadid.json](../playground/data/training_sft/train_koniq_spaq_kadid.json) | [iqa_mix.sh](../scripts/iqa_mix.sh) |
| **IAA**           | [q-future/q-align-aesthetic](https://huggingface.co/q-future/q-align-aesthetic) | AVA | [train_ava.json](../playground/data/training_sft/train_ava.json) | [l1_ava.sh](../scripts/l1_ava.sh) |
| **VQA**           | [q-future/q-align-aesthetic](https://huggingface.co/q-future/q-align-vqa) | LSVQ | [train_lsvq.json](../playground/data/training_sft/train_lsvq.json) | [l1_lsvq.sh](../scripts/l1_lsvq.sh) |


### Q-Instruct+Q-Align Models

This is only the preview checkpoint. Though improving Q-Bench-MCQ accuracy a bit (*70.44->71.82\% dev / 71.10->72.58\% test now*), this combination will slightly degrade the visual scoring accuracy and thus is only a **preview version**. We are still working on a better combination which does not degrade visual scoring accuracy.

| Checkpoint Type   | Huggingface Path | Training Dataset         | Training Labels                                     | Training script             |
|-------------------|------------------|--------------------------|---------------------------------------------------|-----------------------------|
| **Q-Instruct+*OneAlign***      | [teowu/q-instruct-plus-one-align-preview-v0.3](https://huggingface.co/teowu/q-instruct-plus-one-align-preview-v0.3) | Q-Instruct, KonIQ, SPAQ, KADID, AVA, LSVQ | [qinstruct_.json (click to download)](https://huggingface.co/datasets/teowu/Q-Instruct/resolve/main/qinstruct_qalign.json) | [qinstruct_qalign.sh](../scripts/qinstruct_qalign.sh) |

The current preview checkpoint's visual scoring results:

```
konvid srcc:0.8723 plcc:0.8768
maxwell_test srcc:0.7579 plcc:0.7544
test_koniq srcc:0.9439 plcc:0.9530
test_lsvq_1080p srcc:0.7874 plcc:0.8202
test_kadid srcc:0.9171 plcc:0.9181
csiq srcc:0.8685 plcc:0.8911
agi srcc:0.7869 plcc:0.8403
test_lsvq srcc:0.8732 plcc:0.8668
live srcc:0.8934 plcc:0.8620
test_ava srcc:0.8077 plcc:0.8070
livec srcc:0.8661 plcc:0.8765
test_spaq srcc:0.9404 plcc:0.9404
```



