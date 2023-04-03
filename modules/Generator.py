import os
import math
import torch
import itertools
from tqdm import tqdm
from PIL import Image
import torch.nn.functional as F
from accelerate import Accelerator
from torch.utils.data import DataLoader
from accelerate.logging import get_logger
from typing import Union, Optional, Dict, List
from transformers import CLIPTextModel, CLIPTokenizer
from .TextualInversionDataset import TextualInversionDataset
from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler, DDPMScheduler, StableDiffusionPipeline

class Generator():
    def __init__(self, model_id:str, device:str, hparams:Dict, 
                 tokenizer:Optional[CLIPTokenizer]=None, 
                 text_encoder:Optional[CLIPTextModel]=None, 
                 vae:Optional[AutoencoderKL]=None,
                 scheduler:Optional[Union[PNDMScheduler,DDPMScheduler]]=None,
                 unet:Optional[UNet2DConditionModel]=None,
                 seed:int=42) -> None:
        super(Generator, self).__init__()
        self.seed = seed
        self.device = device
        self.hparams = hparams
    # set up the components
        # vae
        if vae is not None: self.vae=vae
        else: self.vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae").to(device)
        # scheduler 
        if scheduler is not None: self.scheduler = scheduler
        else: self.scheduler = PNDMScheduler.from_pretrained(model_id, subfolder="scheduler")
        # UNet
        if unet is not None:self.unet=unet
        else: self.unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet").to(device)
        # tokenizer
        if tokenizer is not None: self.tokenizer = tokenizer
        else: self.tokenizer = CLIPTokenizer.from_pretrained(model_id,subfolder="tokenizer")
        # text_encoder
        if text_encoder is not None: self.text_encoder = text_encoder
        else: self.text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder").to(device)
    
    def freeze_params(self,params):
        for param in params:
            param.requires_grad = False

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

    def save_progress(self, placeholder_token, placeholder_token_id, accelerator, save_path):
        self.logger.info("Saving embeddings")
        learned_embeds = accelerator.unwrap_model(self.text_encoder).get_input_embeddings().weight[placeholder_token_id]
        learned_embeds_dict = {placeholder_token: learned_embeds.detach().cpu()}
        torch.save(learned_embeds_dict, save_path)

    def training_function(self, train_hparams, placeholder_token, placeholder_token_id, what_to_teach, image_path:str):
        self.logger = get_logger(__name__)
        # Freeze vae and unet
        self.freeze_params(self.vae.parameters())
        self.freeze_params(self.unet.parameters())
        # Freeze all parameters except for the token embeddings in text encoder
        params_to_freeze = itertools.chain(
            self.text_encoder.text_model.encoder.parameters(),
            self.text_encoder.text_model.final_layer_norm.parameters(),
            self.text_encoder.text_model.embeddings.position_embedding.parameters(),
        )
        self.freeze_params(params_to_freeze)

        train_dataset = TextualInversionDataset(
            data_root=image_path,
            tokenizer=self.tokenizer,
            size=self.vae.sample_size,
            placeholder_token=placeholder_token,
            repeats=100,
            learnable_property=what_to_teach, #Option selected above between object and style
            center_crop=False,
            set="train"
        )
        train_dataloader = DataLoader(train_dataset, 
                                      batch_size=train_hparams['train_batch_size'], 
                                      shuffle=True)
        
        train_batch_size = train_hparams["train_batch_size"]
        gradient_accumulation_steps = train_hparams["gradient_accumulation_steps"]
        learning_rate = train_hparams["learning_rate"]
        max_train_steps = train_hparams["max_train_steps"]
        output_dir = train_hparams["output_dir"]
        gradient_checkpointing = train_hparams["gradient_checkpointing"]

        accelerator = Accelerator(
            gradient_accumulation_steps=gradient_accumulation_steps,
            mixed_precision=train_hparams["mixed_precision"]
        )

        if gradient_checkpointing:
            self.text_encoder.gradient_checkpointing_enable()
            self.unet.enable_gradient_checkpointing()

        train_dataloader = DataLoader(train_dataset, 
                                      batch_size=train_batch_size, 
                                      shuffle=True)

        if train_hparams["scale_lr"]:
            learning_rate = (
                learning_rate * gradient_accumulation_steps * train_batch_size * accelerator.num_processes
            )

        # Initialize the optimizer
        optimizer = torch.optim.AdamW(self.text_encoder.get_input_embeddings().parameters(),  # only optimize the embeddings
                                      lr=learning_rate)

        text_encoder, optimizer, train_dataloader = accelerator.prepare(self.text_encoder, 
                                                                        optimizer, 
                                                                        train_dataloader)

        weight_dtype = torch.float32
        if accelerator.mixed_precision == "fp16":
            weight_dtype = torch.float16
        elif accelerator.mixed_precision == "bf16":
            weight_dtype = torch.bfloat16

        # Move vae and unet to device
        self.vae.to(accelerator.device, dtype=weight_dtype)
        self.unet.to(accelerator.device, dtype=weight_dtype)

        # Keep vae in eval mode as we don't train it
        self.vae.eval()
        # Keep unet in train mode to enable gradient checkpointing
        self.unet.train()

        # We need to recalculate our total training steps as the size of the training dataloader may have changed.
        num_update_steps_per_epoch = math.ceil(len(train_dataloader) / gradient_accumulation_steps)
        num_train_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)

        # Train!
        total_batch_size = train_batch_size * accelerator.num_processes * gradient_accumulation_steps

        self.logger.info("***** Running training *****")
        self.logger.info(f"  Num examples = {len(train_dataset)}")
        self.logger.info(f"  Instantaneous batch size per device = {train_batch_size}")
        self.logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        self.logger.info(f"  Gradient Accumulation steps = {gradient_accumulation_steps}")
        self.logger.info(f"  Total optimization steps = {max_train_steps}")
        # Only show the progress bar once on each machine.
        progress_bar = tqdm(range(max_train_steps), disable=not accelerator.is_local_main_process)
        progress_bar.set_description("Steps")
        global_step = 0

        for epoch in range(num_train_epochs):
            text_encoder.train()
            for step, batch in enumerate(train_dataloader):
                with accelerator.accumulate(text_encoder):
                    # Convert images to latent space
                    latents = self.vae.encode(batch["pixel_values"].to(dtype=weight_dtype)).latent_dist.sample().detach()
                    latents = latents * 0.18215

                    # Sample noise that we'll add to the latents
                    noise = torch.randn_like(latents)
                    bsz = latents.shape[0]
                    # Sample a random timestep for each image
                    timesteps = torch.randint(0, self.scheduler.num_train_timesteps, (bsz,), device=latents.device).long()

                    # Add noise to the latents according to the noise magnitude at each timestep
                    # (this is the forward diffusion process)
                    noisy_latents = self.scheduler.add_noise(latents, noise, timesteps)

                    # Get the text embedding for conditioning
                    encoder_hidden_states = text_encoder(batch["input_ids"])[0]

                    # Predict the noise residual
                    noise_pred = self.unet(noisy_latents, timesteps, encoder_hidden_states.to(weight_dtype)).sample

                    # Get the target for loss depending on the prediction type
                    if self.scheduler.config.prediction_type == "epsilon":
                        target = noise
                    elif self.scheduler.config.prediction_type == "v_prediction":
                        target = self.scheduler.get_velocity(latents, noise, timesteps)
                    else:
                        raise ValueError(f"Unknown prediction type {self.scheduler.config.prediction_type}")

                    loss = F.mse_loss(noise_pred, target, reduction="none").mean([1, 2, 3]).mean()
                    accelerator.backward(loss)

                    # Zero out the gradients for all token embeddings except the newly added
                    # embeddings for the concept, as we only want to optimize the concept embeddings
                    if accelerator.num_processes > 1:
                        grads = text_encoder.module.get_input_embeddings().weight.grad
                    else:
                        grads = text_encoder.get_input_embeddings().weight.grad
                    # Get the index for tokens that we want to zero the grads for
                    index_grads_to_zero = torch.arange(len(self.tokenizer)) != placeholder_token_id
                    grads.data[index_grads_to_zero, :] = grads.data[index_grads_to_zero, :].fill_(0)

                    optimizer.step()
                    optimizer.zero_grad()

                # Checks if the accelerator has performed an optimization step behind the scenes
                if accelerator.sync_gradients:
                    progress_bar.update(1)
                    global_step += 1
                    if global_step % train_hparams["save_steps"] == 0:
                        save_path = os.path.join(output_dir, f"learned_embeds-step-{global_step}.bin")
                        self.save_progress(text_encoder, placeholder_token_id, accelerator, save_path)

                logs = {"loss": loss.detach().item()}
                progress_bar.set_postfix(**logs)
                if global_step >= max_train_steps:
                    break
            accelerator.wait_for_everyone()

        # Create the pipeline using using the trained modules and save it.
        if accelerator.is_main_process:
            pipeline = StableDiffusionPipeline.from_pretrained(
                self.model_id,
                text_encoder=accelerator.unwrap_model(text_encoder),
                tokenizer=self.tokenizer,
                vae=self.vae,
                unet=self.unet,
            )
            pipeline.save_pretrained(output_dir)
            # Also save the newly trained embeddings
            save_path = os.path.join(output_dir, f"learned_embeds.bin")
            self.save_progress(text_encoder, placeholder_token_id, accelerator, save_path)
            
            
        