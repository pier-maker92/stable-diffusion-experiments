import os
import shutil
import torch
from typing import Optional, Dict
from huggingface_hub import hf_hub_download
from transformers import CLIPTextModel, CLIPTokenizer

class Conceptualizer():
    def __init__(self, model_id:str, device:str, downloaded_embedding_folder:str="./downloaded_embedding") -> None:
        super(Conceptualizer, self).__init__()
        if not os.path.exists(downloaded_embedding_folder):
            os.mkdir(downloaded_embedding_folder)
        self.downloaded_embedding_folder = downloaded_embedding_folder
        # tokenizer
        self.tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
        # text encoder
        self.text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder").to(device)

    def load_learned_embed_in_clip(self, learned_embeds_path, token=None):
        loaded_learned_embeds = torch.load(learned_embeds_path, map_location="cpu") 

        # separate token and the embeds
        trained_token = list(loaded_learned_embeds.keys())[0]
        embeds = loaded_learned_embeds[trained_token]

        # cast to dtype of text_encoder
        dtype = self.text_encoder.get_input_embeddings().weight.dtype
        embeds.to(dtype)

        # add the token in tokenizer
        token = token if token is not None else trained_token
        num_added_tokens = self.tokenizer.add_tokens(token)
        if num_added_tokens == 0:
            raise ValueError(f"The tokenizer already contains the token {token}. Please pass a different `token` that is not already in the tokenizer.")
        
        # resize the token embeddings
        self.text_encoder.resize_token_embeddings(len(self.tokenizer))
        
        # get the id for the token and assign the embeds
        token_id = self.tokenizer.convert_tokens_to_ids(token)
        self.text_encoder.get_input_embeddings().weight.data[token_id] = embeds

    def load_concepts(self, concepts:Dict):
        for k,repo in concepts['repo'].items():
            embeds_path = hf_hub_download(repo_id=repo, filename="learned_embeds.bin")
            token_path = hf_hub_download(repo_id=repo, filename="token_identifier.txt")
            shutil.copy(embeds_path,self.downloaded_embedding_folder)
            shutil.copy(token_path,self.downloaded_embedding_folder)
            with open(f'{self.downloaded_embedding_folder}/token_identifier.txt', 'r') as file:
                placeholder_token_string = file.read()
                # store proper placeholder for 'from' and 'to' prompt
                concepts['placeholders'][k] = placeholder_token_string
            # load concept embeddins in clip
            learned_embeds_path = f"{self.downloaded_embedding_folder}/learned_embeds.bin"
            self.load_learned_embed_in_clip(learned_embeds_path)
        return self.tokenizer, self.text_encoder
