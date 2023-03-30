import os
import torch
from tqdm import tqdm
from PIL import Image
from typing import Union, Optional, Dict, List
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler

class Generator():
    def __init__(self, model_id:str, device:str, hparams:Dict, tokenizer:Optional[CLIPTokenizer]=None, text_encoder:Optional[CLIPTextModel]=None, seed:int=42) -> None:
        super(Generator, self).__init__()
        self.seed = seed
        self.device = device
        self.hparams = hparams
    # set up the components
        # vae
        self.vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae").to(device)
        # scheduler 
        self.scheduler = PNDMScheduler.from_pretrained(model_id, subfolder="scheduler")
        # UNet
        self.unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet").to(device)
        # tokenizer
        if tokenizer is not None: self.tokenizer = tokenizer
        else: self.tokenizer = CLIPTokenizer.from_pretrained(model_id,subfolder="tokenizer")
        # text_encoder
        if text_encoder is not None: self.text_encoder = text_encoder
        else: self.text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder").to(device)

    def get_clip_text_features(self, prompt:List[str], max_length:Union[int,None]=None) -> torch.Tensor:
        text_tokens_ids = self.tokenizer(prompt, 
                                         padding="max_length",  
                                         max_length= max_length if max_length is not None else self.tokenizer.model_max_length,  
                                         truncation=True,  
                                         return_tensors="pt").input_ids
        return self.text_encoder(text_tokens_ids.to(self.device))[0], text_tokens_ids.size(-1)
            
    def generate_conditioned_latents(self, prompt:Union[List[str],None], text_embeddings:Union[torch.Tensor,None]=None, max_length:Union[int,None]=None,negative_prompt:Union[str,None]=None) -> torch.Tensor:
    # CLIP embeddings
        # condition
        if text_embeddings is None:
            text_embeddings,max_length  = self.get_clip_text_features(prompt)
        # uncondition
        uncond_prompt = [""] if negative_prompt is None else [negative_prompt] * self.hparams['batch_size']
        uncond_embeddings,_ = self.get_clip_text_features(uncond_prompt,max_length) 

        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

        # generation
        height, width = self.hparams['height'], self.hparams['width']
        generator = torch.manual_seed(self.seed)
        latents = torch.randn((self.hparams['batch_size'], 
                            self.unet.in_channels, 
                            height // 8,
                            width // 8),
                            generator=generator)
        latents = latents.to(self.device)

        self.scheduler.set_timesteps(self.hparams['num_inference_steps'])
        latents = latents * self.scheduler.init_noise_sigma

        for t in tqdm(self.scheduler.timesteps):
            # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, timestep=t)

            # predict the noise residual
            with torch.no_grad():
                noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

            # perform guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + self.hparams['guidance_scale'] * (noise_pred_text - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample

        return 1 / 0.18215 * latents
    
    def decode_latents(self, latents:List[torch.Tensor]) -> torch.Tensor:
        with torch.no_grad():
            sample = self.vae.decode(latents.to(self.device)).sample
        return sample
    
    def save_image(self, sample:torch.Tensor, save_dir:str, names:List[str]) -> None:
        sample = (sample / 2 + 0.5).clamp(0, 1)
        image = sample.detach().cpu().permute(0, 2, 3, 1).numpy()
        images = (image * 255).round().astype("uint8")
        pil_images = [Image.fromarray(image) for image in images]
        for i,img in enumerate(pil_images):
            file_name = os.path.join(save_dir,names[i])
            img.save(f"{file_name}.png")


       
        
        
    