import os
import sys
import torch
import shutil
import argparse
from modules.Generator import Generator
from modules.Conceptualizer import Conceptualizer

# Create the parser and add arguments
parser = argparse.ArgumentParser()
parser.add_argument('--model_id', default="CompVis/stable-diffusion-v1-4", type=str,
                    help="The model checpoint you want to use")
parser.add_argument('--from_file', action=argparse.BooleanOptionalAction,
                    help="load arguments from file")
parser.add_argument('-p', '--prompt_file_path', default='prompt.txt', type=str, required= '--from_file' in sys.argv,
                    help="path file where to read prompt")
parser.add_argument('-s','--seed', default=42, type=int,
                    help="Set the random seed")
parser.add_argument('--from_concept_repo',required= not '--from_file' in sys.argv, type=str,
                    help="The start concept you want to use. (Provide a hugginface concept repo)")
parser.add_argument('--to_concept_repo',required= not '--from_file' in sys.argv, type=str,
                    help="The end concept you want to use. (Provide a hugginface concept repo)")
parser.add_argument('--from_prompt', required= not '--from_file' in sys.argv, type=str,
                    help="Start prompt you want to use")
parser.add_argument('--to_prompt', required= not '--from_file' in sys.argv, type=str,
                    help="End prompt you want to use")
parser.add_argument('--num_inference_steps', default=50, type=int,
                    help="Number of inference step.")
parser.add_argument('--guidance_scale', default=7.5, type=float,
                    help="The guidance scale value to set.")
parser.add_argument('--width', default=512, type=int,
                    help="Canvas width of generated image.")
parser.add_argument('--height', default=512, type=int,
                    help="Canvas height of generated image.")
parser.add_argument('--use_negative_prompt', action=argparse.BooleanOptionalAction,
                    help="flag to use negative prompt stored in negative_prompt.txt")
parser.add_argument('-b','--batch_size', default=1, type=int,
                    help="Batch size to use")
parser.add_argument('--mps', action=argparse.BooleanOptionalAction,
                    help="Flag that set mps as gpu device")
parser.add_argument('-i','--interpolation', default='semantic', choices=['semantic','visual'], type=str,
                    help="Choose the type of the interpolation. Options: semantic | visual. Default = semantic")

if __name__=='__main__':
    # get args
    args = parser.parse_args()
    #login()
    if not args.mps:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device='mps'
    print(f'running on {device}')
    if args.from_file:
        f = open(args.prompt_file_path); prompt_file = f.read()
        from_concept_repo, to_concept_repo, from_prompt, to_prompt = prompt_file.split('\n')
        from_concept_repo = from_concept_repo.replace('--from_concept_repo ','')
        to_concept_repo = to_concept_repo.replace('--to_concept_repo ','')
        from_prompt = from_prompt.replace('--from_prompt ','')
        to_prompt = to_prompt.replace('--to_prompt ','')
    else:
        from_concept_repo, to_concept_repo, from_prompt, to_prompt = args.from_concept_repo, args.to_concept_repo, args.from_prompt, args.to_prompt
    
    print(f"\nfrom_concept_repo: {from_concept_repo}\nto_concept_repo: {to_concept_repo}\nfrom_prompt: {from_prompt}\nto_prompt: {to_prompt}\n")
        

    # set up the components
    seed = args.seed
    model_id = args.model_id
    hparams = {
        'width': args.width,
        'height': args.height,
        'batch_size': args.batch_size,
        'guidance_scale': args.guidance_scale,
        'num_inference_steps': args.num_inference_steps
    }
    # load concepts
    concepts = {
        'repo': {
            'to': to_concept_repo,
            'from': from_concept_repo
        },
        'placeholders':{},
        'learned_embeds_path':{}
    }
    # get tokenizer and text encoder for new concepts
    conceptualizer = Conceptualizer(model_id, device)
    tokenizer, text_encoder = conceptualizer.load_concepts(concepts)

    # load generator
    generator = Generator(model_id, device, hparams, tokenizer, text_encoder, seed=seed)
    # get prompt
    start_prompt = from_prompt.replace('<concept>',concepts['placeholders']['from'])
    end_prompt = to_prompt.replace('<concept>',concepts['placeholders']['to'])
    latents_list = []
    if args.use_negative_prompt:
        f = open('negative_prompt.txt'); negative_prompt = f.read()
    else: negative_prompt=None
    for prompt in [start_prompt,end_prompt]:
        latents = generator.generate_conditioned_latents([prompt],negative_prompt=negative_prompt)
        latents_list.append(latents)
    
    # decode in 20 frames
    save_dir = 'frames'
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir); 
        os.mkdir(save_dir)
    else:os.mkdir(save_dir)

    for frame in range(20):
        latents = torch.lerp(latents_list[0].detach().cpu(), latents_list[1].detach().cpu(), frame/20)
        image = generator.decode_latents(latents)
        generator.save_image(image, save_dir=save_dir, names=[f'{frame:02}'])

    save_video_dir = 'gen_videos'
    if not os.path.exists(save_video_dir):
        os.mkdir(save_video_dir)
    
    start_prompt=start_prompt.replace('<',"");start_prompt=start_prompt.replace('>',"")
    end_prompt=end_prompt.replace('<',"");end_prompt=end_prompt.replace('>',"")
    # make video

    cmd = f"ffmpeg -framerate 10 -pattern_type glob -i '{save_dir}/*.png' -c:v libx264 -pix_fmt yuv420p\
        {save_video_dir}/from_prompt_{'-'.join(start_prompt.split(' '))}_to_prompt_{'-'.join(end_prompt.split(' '))}_seed_{args.seed}.mp4"
    os.system(cmd)
