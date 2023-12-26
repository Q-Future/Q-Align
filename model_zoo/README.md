<div align="center">
  <h1>Model Zoo</h1> 
</div>

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



