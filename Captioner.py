import torch
import open_clip
from PIL import Image
from typing import Union
from transformers import pipeline

class Captioner():
    def __init__(self, model_id:str, device:str) -> None:
        super(Captioner, self).__init__()
        if model_id.split('/')[0]=='laion':
            self.captioner, _, self.transform = open_clip.create_model_and_transforms("coca_ViT-L-14",
                                                                        pretrained="mscoco_finetuned_laion2B-s13B-b90k")
            self.use_open_clip=True
            self.captioner.to(device)
        else:
            self.captioner = pipeline("image-to-text", model=model_id)
            self.use_open_clip=False

        self.device=device
    
    def open_clip_inference_caption(self, image:Image.Image, decoding_method:str="Beam search", rep_penalty:float=1.2, top_p:float=0.5, min_seq_len:int=5, seq_len:int=20) -> str:
        im = self.transform(image).unsqueeze(0).to(self.device)
        generation_type = "beam_search" if decoding_method == "Beam search" else "top_p"
        with torch.no_grad():
            generated = self.captioner.generate(
                im, 
                generation_type=generation_type,
                top_p=float(top_p), 
                min_seq_len=min_seq_len, 
                seq_len=seq_len, 
                repetition_penalty=float(rep_penalty)
            )
        return open_clip.decode(generated[0].detach()).split("<end_of_text>")[0].replace("<start_of_text>", "")
            
    def get_caption(self, image: Union[str,Image.Image]) -> str: 
        if type(image)==str:image = Image.oepn(image)
        if self.use_open_clip: caption = self.open_clip_inference_caption(image)
        else: caption = self.captioner(image)[0]['generated_text']
        return caption