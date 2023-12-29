from PIL import Image

import torch.nn as nn
import torch

from typing import List

from q_align.model.builder import load_pretrained_model

from q_align.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from q_align.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

def load_video(video_file):
    from decord import VideoReader
    vr = VideoReader(video_file)

    # Get video frame rate
    fps = vr.get_avg_fps()

    # Calculate frame indices for 1fps
    frame_indices = [int(fps * i) for i in range(int(len(vr) / fps))]
    frames = vr.get_batch(frame_indices).asnumpy()
    return [Image.fromarray(frames[i]) for i in range(int(len(vr) / fps))]


class QAlignScorer(nn.Module):
    def __init__(self, pretrained="q-future/one-align", device="cuda:0"):
        super().__init__()
        tokenizer, model, image_processor, _ = load_pretrained_model(pretrained, None, "mplug_owl2", device=device)
        prompt = "USER: How would you rate the quality of this image?\n<|image|>\nASSISTANT: The quality of the image is"
        
        self.preferential_ids_ = [id_[1] for id_ in tokenizer(["excellent","good","fair","poor","bad"])["input_ids"]]
        self.weight_tensor = torch.Tensor([1,0.75,0.5,0.25,0.]).half().to(model.device)
    
        self.tokenizer = tokenizer
        self.model = model
        self.image_processor = image_processor
        self.input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(model.device)
        
    def expand2square(self, pil_img, background_color):
        width, height = pil_img.size
        if width == height:
            return pil_img
        elif width > height:
            result = Image.new(pil_img.mode, (width, width), background_color)
            result.paste(pil_img, (0, (width - height) // 2))
            return result
        else:
            result = Image.new(pil_img.mode, (height, height), background_color)
            result.paste(pil_img, ((height - width) // 2, 0))
            return result
        
    def forward(self, image: List[Image.Image]):
        image = [self.expand2square(img, tuple(int(x*255) for x in self.image_processor.image_mean)) for img in image]
        with torch.inference_mode():
            image_tensor = self.image_processor.preprocess(image, return_tensors="pt")["pixel_values"].half().to(self.model.device)
            output_logits = self.model(self.input_ids.repeat(image_tensor.shape[0], 1),
                        images=image_tensor)["logits"][:,-1, self.preferential_ids_]

            return torch.softmax(output_logits, -1) @ self.weight_tensor
        
        
class QAlignAestheticScorer(nn.Module):
    def __init__(self, pretrained="q-future/one-align", device="cuda:0"):
        super().__init__()
        tokenizer, model, image_processor, _ = load_pretrained_model(pretrained, None, "mplug_owl2", device=device)
        prompt = "USER: How would you rate the aesthetics of this image?\n<|image|>\nASSISTANT: The aesthetics of the image is"
        
        self.preferential_ids_ = [id_[1] for id_ in tokenizer(["excellent","good","fair","poor","bad"])["input_ids"]]
        self.weight_tensor = torch.Tensor([1,0.75,0.5,0.25,0.]).half().to(model.device)
    
        self.tokenizer = tokenizer
        self.model = model
        self.image_processor = image_processor
        self.input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(model.device)
        
    def expand2square(self, pil_img, background_color):
        width, height = pil_img.size
        if width == height:
            return pil_img
        elif width > height:
            result = Image.new(pil_img.mode, (width, width), background_color)
            result.paste(pil_img, (0, (width - height) // 2))
            return result
        else:
            result = Image.new(pil_img.mode, (height, height), background_color)
            result.paste(pil_img, ((height - width) // 2, 0))
            return result
        
    def forward(self, image: List[Image.Image]):
        image = [self.expand2square(img, tuple(int(x*255) for x in self.image_processor.image_mean)) for img in image]
        with torch.inference_mode():
            image_tensor = self.image_processor.preprocess(image, return_tensors="pt")["pixel_values"].half().to(self.model.device)
            output_logits = self.model(self.input_ids.repeat(image_tensor.shape[0], 1),
                        images=image_tensor)["logits"][:,-1, self.preferential_ids_]

            return torch.softmax(output_logits, -1) @ self.weight_tensor
        
class QAlignVideoScorer(nn.Module):
    def __init__(self, pretrained="q-future/one-align", device="cuda:0"):
        super().__init__()
        tokenizer, model, image_processor, _ = load_pretrained_model(pretrained, None, "mplug_owl2", device=device)
        prompt = "USER: How would you rate the quality of this video?\n<|image|>\nASSISTANT: The quality of the video is"
        
        self.preferential_ids_ = [id_[1] for id_ in tokenizer(["excellent","good","fair","poor","bad"])["input_ids"]]
        self.weight_tensor = torch.Tensor([1,0.75,0.5,0.25,0.]).half().to(model.device)
    
        self.tokenizer = tokenizer
        self.model = model
        self.image_processor = image_processor
        self.input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(model.device)
        
    def expand2square(self, pil_img, background_color):
        width, height = pil_img.size
        if width == height:
            return pil_img
        elif width > height:
            result = Image.new(pil_img.mode, (width, width), background_color)
            result.paste(pil_img, (0, (width - height) // 2))
            return result
        else:
            result = Image.new(pil_img.mode, (height, height), background_color)
            result.paste(pil_img, ((height - width) // 2, 0))
            return result
        
    def forward(self, video: List[List[Image.Image]]):
        video = [[self.expand2square(frame, tuple(int(x*255) for x in self.image_processor.image_mean)) for frame in vid] for vid in video]
        with torch.inference_mode():
            video_tensors = [self.image_processor.preprocess(vid, return_tensors="pt")["pixel_values"].half().to(self.model.device) for vid in video]
            output_logits = self.model(self.input_ids.repeat(len(video_tensors), 1),
                        images=video_tensors)["logits"][:,-1, self.preferential_ids_]
            return torch.softmax(output_logits, -1) @ self.weight_tensor
    

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="q-future/one-align")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--img_path", type=str, default="fig/singapore_flyer.jpg")
    parser.add_argument("--aesthetic", action="store_true")
    parser.add_argument("--video", action="store_true")
    args = parser.parse_args()

    if args.video:
        scorer = QAlignVideoScorer(pretrained=args.model_path, device=args.device)
        print(scorer([load_video(args.img_path)]).tolist())
    else:
        scorer = QAlignScorer(pretrained=args.model_path, device=args.device) if not args.aesthetic else QAlignAestheticScorer(pretrained=args.model_path, device=args.device) 
        print(scorer([Image.open(args.img_path)]).tolist())

