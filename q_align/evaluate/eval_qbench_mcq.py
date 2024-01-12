import argparse
import torch

from q_align.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from q_align.conversation import conv_templates, SeparatorStyle
from q_align.model.builder import load_pretrained_model
from q_align.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

from PIL import Image

import requests
from PIL import Image
from io import BytesIO
from transformers import TextStreamer

import json
from tqdm import tqdm

import os
os.makedirs("results/co-instruct/", exist_ok=True)

def disable_torch_init():
    """
    Disable the redundant torch default initialization to accelerate model creation.
    """
    import torch
    setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
    setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)


def load_image(image_file):
    if image_file.startswith('http://') or image_file.startswith('https://'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image


def main(args):
    # Model
    disable_torch_init()

    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name, args.load_8bit, args.load_4bit, device=args.device)
    
    correct = 0
    with open(args.questions_file.format(args.split)) as f:
        llvqa_data = json.load(f)  
        
    pbar = tqdm(total=len(llvqa_data))
    
    if args.split == "test":
        print("This will cause error if you are not from the Q-Future team.")
        if args.lang == "zh":
            zh_split = "测试集"
            with open(f"/home/ps/Downloads/datasets/质衡-问答.json") as f:
                answer_data = json.load(f)
            for i, llddata in enumerate(llvqa_data):
                llddata["correct_ans"] = answer_data[2*i]["answers"][0]
        else:
            with open(f"/home/ps/Downloads/datasets/LLVQA/llvisionqa_3k_qbench_c1.json") as f:
                answer_data =  json.load(f)
            for i, llddata in enumerate(llvqa_data):
                llddata["correct_ans"] = answer_data[2*i]["answers"][0]


    conv_mode = "mplug_owl2"

    if args.conv_mode is not None and conv_mode != args.conv_mode:
        print('[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}'.format(conv_mode, args.conv_mode, args.conv_mode))
    else:
        args.conv_mode = conv_mode

    conv = conv_templates[args.conv_mode].copy()
    roles = conv.roles

    for i, llddata in enumerate((llvqa_data)):
        filename = llddata["img_path"]
        
        message = llddata["question"] + "\n"
        for choice, ans in zip(["A.", "B.", "C.", "D."], llddata["candidates"]):
            message += f"{choice} {ans}\n"
            if "correct_ans" in llddata and ans == llddata["correct_ans"]:
                correct_choice = choice[0]
        if args.lang == "en":
            message = message + "Answer with the option's letter from the given choices directly.\n"
        elif args.lang == "zh":
            message = message + "请直接回答正确选项的字母\n"
        else:
            raise NotImplementedError("Q-Bench does not support languages other than English (en) and Chinese (zh) yet. Contact us (https://github.com/Q-Future/Q-Bench/) to convert Q-Bench into more languages.")
            
        inp = message
        
        conv = conv_templates[args.conv_mode].copy()
        inp = "The input image:" + DEFAULT_IMAGE_TOKEN + inp
        conv.append_message(conv.roles[0], inp)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        
        print(prompt)
        
        image = load_image(args.image_folder + filename)
        image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'].half().to(model.device)
        

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(model.device)
        stop_str = conv.sep if conv.sep_style not in [SeparatorStyle.TWO, SeparatorStyle.TWO_NO_SYS] else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
        streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor,
                do_sample=False,
                temperature=args.temperature,
                max_new_tokens=args.max_new_tokens,
                num_beams=1,
                streamer=streamer,
                use_cache=True,
                stopping_criteria=[stopping_criteria])

        outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
        llddata["response"] = outputs
        
        if correct_choice in outputs: 
            correct += 1
        
        pbar.update(1)
        pbar.set_description("[Running Accuracy]: {:.4f},[Response]: {}, [Correct Ans]: {}, , [Prog]: {}".format(correct/(i+1), outputs, llddata.get("correct_ans", -1), i+1))
        
        with open(args.answers_file.format(args.split), "a") as wf:
            json.dump(llddata, wf)

        if args.debug:
            print("\n", {"prompt": prompt, "outputs": outputs}, "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="q-future/co-instruct-preview")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="/home/ps/Downloads/datasets/LLVQA/images/")
    parser.add_argument("--questions-file", type=str, default="/home/ps/Downloads/datasets/LLVQA/llvisionqa_{}.json")
    parser.add_argument("--answers-file", type=str, default="results/co-instruct/qbench_a1_{}.jsonl")
    parser.add_argument("--split", type=str, default="dev")
    parser.add_argument("--lang", type=str, default="en")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--image-aspect-ratio", type=str, default='pad')
    args = parser.parse_args()
    main(args)