## LoRA Fine-tuning

Training Q-Align / OneAlign is resource-consuming. While it has shown good performance on lots of datasets, if new datasets come, we will still need to adapt it to newer ones. 

*Can we make this add-on adaptation more efficient?*

**Yes, we can.**

![](q_align_lora.png)

We propose a more efficient LoRA (tunes less parameter than LLaVA-style default LoRA), which only needs to tune 149M parameters (1.8\% compared with full version Q-Align), and requires only **2 RTX3090 GPUs** (available to many independent researches). To do this, simply run

```shell
sh scripts/${YOUR_DATASET}_lora.sh
```

The available dataset options are `agi` (for AGIQA-3K), `cgi` (for CGIQA-6K), `live` (for LIVE) and `csiq` (for CSIQ), though we discourage fine-tuning on datasets that are very similar with the original training corpus of **OneAlign** (will make your adapted model less robust to be applied).


To evaluate, please refer to the code below:

```shell
python q_align/evaluate/iqa_eval_lora_split.py --model-path ${YOUR_MODEL_PATH} --model-base q-future/one-align
```

By default (if `YOUR_MODEL_PATH` is not specified), if will automatically evaluate on the test set of AGIQA-3K (*split 1*).

We will update more datasets and a complete performance report on this feature soon.

