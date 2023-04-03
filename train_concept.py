import os
import torch
import argparse
import accelerate
from modules.Generator import Generator
from modules.Conceptualizer import Conceptualizer

# Create the parser and add arguments
parser = argparse.ArgumentParser()
parser.add_argument('--model_id', default="CompVis/stable-diffusion-v1-4", type=str,
                    help="The model checpoint you want to use")
parser.add_argument('-f','--from_path', default="./image_concepts", type=str,
                    help="The model checpoint you want to use")
parser.add_argument('--what_to_teach', required=True, choices=['object','style'],type=str,
                    help="`object` enables you to teach the model a new object to be used, `style` allows you to teach the model a new style one can use.")
parser.add_argument('--placeholder_token', required=True, type=str,
                    help="`placeholder_token` is the token you are going to use to represent your new concept")
parser.add_argument('--initializer_token', required=True, type=str,
                    help="`initializer_token` is a word that can summarise what your new concept is, to be used as a starting point")
parser.add_argument('--max_train_steps', default=2000, type=int,
                    help="max_train_steps to use")
parser.add_argument('-b','--batch_size', default=1, type=int,
                    help="batch_size to use")
parser.add_argument('--gradient_accumulation_steps', default=1, type=int,
                    help="gradient_accumulation_steps to use")
parser.add_argument('--lr', default=5e-04, type=float,
                    help="learning rate to use")
parser.add_argument('--save_steps', default=10, type=int,
                    help="save_steps to use")
parser.add_argument('--output_dir', default='sd-concept-output', type=str,
                    help="The output dir where to store learned concepts")
parser.add_argument('-s','--seed', default=42, type=int,
                    help="random seed to use")

parser.add_argument('--gradient_checkpointing', action=argparse.BooleanOptionalAction)
parser.add_argument('--scale_lr', action=argparse.BooleanOptionalAction)

parser.add_argument('--mps', action=argparse.BooleanOptionalAction,
                    help="Flag that set mps as gpu device")

if __name__=='__main__':
    # get args
    args = parser.parse_args()
    #login()
    if not args.mps:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device='mps'
    seed = args.seed
    model_id = args.model_id
    train_hparams = {
        "learning_rate": args.lr,
        "scale_lr": args.scale_lr, #True
        "max_train_steps": args.max_train_steps,
        "save_steps": args.save_steps,
        "train_batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "gradient_checkpointing": args.gradient_checkpointing, #True
        "mixed_precision": "no" if device=='mps' else 'fp16', #fp16 in cuda
        "seed": seed,
        "output_dir": args.output_dir
    }
    hparams = {
        'width': 512,
        'height': 512,
        'batch_size': args.batch_size,
        'guidance_scale': 7.5,
        'num_inference_steps': 50
    }

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    # get tokenizer and text encoder for new concepts
    conceptualizer = Conceptualizer(model_id, device)
    placeholder_token_id, tokenizer, text_encoder = conceptualizer.learn_concept(placeholder_token=args.placeholder_token, 
                                                                                 initializer_token=args.initializer_token)
                            
    generator = Generator(model_id, device, hparams, tokenizer, text_encoder, seed=seed)
    generator.training_function(train_hparams, 
                                args.placeholder_token, 
                                placeholder_token_id, 
                                args.what_to_teach, 
                                args.from_path)

