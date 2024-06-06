import os
import gc
import random
from glob import glob
from pathlib import Path
import logging
import torch
import torchvision
import numpy as np
from einops import rearrange
from omegaconf import OmegaConf
from PIL import Image
from folder_paths import add_model_folder_path, get_folder_paths, models_dir, get_full_path
from diffusers import (AutoencoderKL, DDIMScheduler,
                       DPMSolverMultistepScheduler,
                       EulerAncestralDiscreteScheduler, EulerDiscreteScheduler,
                       PNDMScheduler)
from transformers import T5EncoderModel, T5Tokenizer
from .easyanimate.models.autoencoder_magvit import AutoencoderKLMagvit
from .easyanimate.models.transformer3d import Transformer3DModel
from .easyanimate.pipeline.pipeline_easyanimate import EasyAnimatePipeline
from .easyanimate.utils.lora_utils import merge_lora, unmerge_lora
from diffusers.utils.import_utils import is_xformers_available

_CATEGORY = "FrankChieng/EasyAnimate"
_MAPPING = "FrankChiengEasyAnimate"
scheduler_dict = {
    "Euler": EulerDiscreteScheduler,
    "Euler A": EulerAncestralDiscreteScheduler,
    "DPM++": DPMSolverMultistepScheduler, 
    "PNDM": PNDMScheduler,
    "DDIM": DDIMScheduler,
}
add_model_folder_path("EasyAnimate", str(Path(models_dir) / "EasyAnimate"))
diffusion_transformer_dir  = os.path.join(Path(get_folder_paths("EasyAnimate")[0]), "Diffusion_Transformer")
diffusion_transformer_list = glob(os.path.join(diffusion_transformer_dir, "*/"))
personalized_model_dir     = os.path.join(Path(get_folder_paths("EasyAnimate")[0]), "Personalized_Model")
personalized_model_list = glob(os.path.join(personalized_model_dir, "*.safetensors"))
personalized_model_dict = {
    "none": "none"
}
for count,ele in enumerate(personalized_model_list):
    personalized_model_dict.update({os.path.basename(ele) : ele})

device = "cuda" if torch.cuda.is_available() else "cpu"

class EasyAnimateController:
    def __init__(self):
        # config dirs
        self.basedir                    = os.path.dirname(os.path.abspath(__file__))
        self.config_dir                 = os.path.join(self.basedir, "config")
        
        self.diffusion_transformer_dir  = os.path.join(self.basedir, "models", "Diffusion_Transformer")
        self.motion_module_dir          = os.path.join(self.basedir, "models", "Motion_Module")
        self.personalized_model_dir     = os.path.join(self.basedir, "models", "Personalized_Model")
        self.edition                    = "v2"
        self.inference_config           = OmegaConf.load(os.path.join(self.config_dir, "easyanimate_video_magvit_motion_module_v2.yaml"))

        # config models
        self.tokenizer             = None
        self.text_encoder          = None
        self.vae                   = None
        self.transformer           = None
        self.pipeline              = None
        
        self.weight_dtype = torch.bfloat16

    def load_diffusion_transformer(self, diffusion_transformer):
        logging.info("diffusion transformer kick off!")
        if OmegaConf.to_container(self.inference_config['vae_kwargs'])['enable_magvit']:
            Choosen_AutoencoderKL = AutoencoderKLMagvit
        else:
            Choosen_AutoencoderKL = AutoencoderKL
        self.vae = Choosen_AutoencoderKL.from_pretrained(diffusion_transformer, subfolder="vae",).to(self.weight_dtype)
        self.transformer = Transformer3DModel.from_pretrained_2d(diffusion_transformer, subfolder="transformer", transformer_additional_kwargs=OmegaConf.to_container(self.inference_config.transformer_additional_kwargs)).to(self.weight_dtype)
        self.tokenizer = T5Tokenizer.from_pretrained(diffusion_transformer, subfolder="tokenizer")
        self.text_encoder = T5EncoderModel.from_pretrained(diffusion_transformer, subfolder="text_encoder", torch_dtype=self.weight_dtype)

        # Get pipeline
        self.pipeline = EasyAnimatePipeline(
                vae=self.vae, 
                text_encoder=self.text_encoder, 
                tokenizer=self.tokenizer, 
                transformer=self.transformer,
                scheduler=scheduler_dict["Euler"](**OmegaConf.to_container(self.inference_config.noise_scheduler_kwargs))
            )
        self.pipeline.enable_model_cpu_offload()
        logging.info("load diffusion transformer")
    
    def videos_transform(self, videos: torch.Tensor, rescale=False, n_rows=6):
        videos = rearrange(videos, "b c t h w -> t b c h w")
        outputs = []
        for x in videos:
            x = torchvision.utils.make_grid(x, nrow=n_rows)
            x = x.transpose(0, 1).transpose(1, 2).squeeze(-1)
            if rescale:
                x = (x + 1.0) / 2.0  # -1,1 -> 0,1
            x = (x * 255).numpy().astype(np.uint8)
            outputs.append(Image.fromarray(x))
        images = np.array(outputs).astype(np.float32) / 255.0
        images = torch.from_numpy(images)
        return images

class Generator:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING",{"forceInput": True}),
                "negative_prompt": ("STRING", {"multiline": True, "default":"The video is not of a high quality, it has a low resolution, and the audio quality is not clear. Strange motion trajectory, a poor composition and deformed video, low resolution, duplicate and ugly, strange body structure, long and strange neck, bad teeth, bad eyes, bad limbs, bad hands, rotating camera, blurry camera, shaking camera. Deformation, low-resolution, blurry, ugly, distortion."}),
                "pretrained_model": (diffusion_transformer_list,),
                #"lora_model": (personalized_model_list,),
                "lora_model": (list(personalized_model_dict.keys()),),
                "lora_alpha": ("FLOAT", {"default": 0.55, "min": 0, "max": 2, "step": 0.01}),
                "sampling_method": (list(scheduler_dict.keys()), {"default":list(scheduler_dict.keys())[0]}),
                "sampling_steps": ("INT", {"default": 50, "min": 10, "max": 100, "step": 1}),
                "width": ("INT", {"default": 672, "min": 128, "max": 1280, "step": 16}),
                "height": ("INT", {"default": 384, "min": 128, "max": 1280, "step": 16}), 
                "animation_length": ("INT", {"default": 144, "min": 9, "max": 144, "step": 9}), 
                "cfg": ("INT", {"default": 7, "min": 0, "max": 20, "step": 1}),     
                "seed": ("INT", {"default": random.randint(1, 1e8)}),                                            
            },
        }

    CATEGORY = _CATEGORY
    FUNCTION = "generate"
    OUTPUT_NODE = True
    RETURN_NAMES = ("images",)
    RETURN_TYPES = ("IMAGE",)

    def generate(self, **kwargs):
        controller = EasyAnimateController()

        controller.load_diffusion_transformer(kwargs['pretrained_model'])
        if is_xformers_available(): controller.transformer.enable_xformers_memory_efficient_attention()
        controller.pipeline.scheduler = scheduler_dict[kwargs['sampling_method']](**OmegaConf.to_container(controller.inference_config.noise_scheduler_kwargs))
        logging.info(f"personalized model dir: {personalized_model_dir}")
        lora_model = personalized_model_dict[kwargs['lora_model']]
        logging.info(f"lora model: {lora_model}")       
        
        if lora_model!= "none":
            # lora part
            controller.pipeline = merge_lora(controller.pipeline, lora_model, multiplier=kwargs['lora_alpha'])

        controller.pipeline.to(device)

        if int(kwargs['seed']) != -1 and kwargs['seed'] != "": torch.manual_seed(int(kwargs['seed']))
        else: kwargs['seed'] = np.random.randint(0, 1e10)
        generator = torch.Generator(device=device).manual_seed(int(kwargs['seed']))
        
        try:
            sample = controller.pipeline(
                kwargs['prompt'],
                negative_prompt     = kwargs['negative_prompt'],
                num_inference_steps = kwargs['sampling_steps'],
                guidance_scale      = kwargs['cfg'],
                width               = kwargs['width'],
                height              = kwargs['height'],
                video_length        = kwargs['animation_length'],
                generator           = generator
            ).videos
        except Exception as e:
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            logging.info(f"sample error: {e}")

            if lora_model!= "none":
                controller.pipeline = unmerge_lora(controller.pipeline, lora_model, multiplier=kwargs['lora_alpha'])

        # lora part 
        if lora_model!= "none":
            controller.pipeline = unmerge_lora(controller.pipeline, lora_model, multiplier=kwargs['lora_alpha'])
        

        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
 
        images = controller.videos_transform(sample)
        return (images,)

NODE_CLASS_MAPPINGS = {
    f"{_MAPPING}Generator": Generator,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    f"{_MAPPING}Generator": "EasyAnimateGenerator",
}
