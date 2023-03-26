
# Stable diffusion experiments

This is a repo providing same stable diffusion experiments, regarding textual inversion task and captioning task.

## Installation

Clone the repo, then create a conda envirnoment from `envirnoment.yml` and install the dependecies.

```bash
  conda env create --file=environment.yml
  conda activate sd
  pip install -r requirements.txt
```

## Textual inversion

The textual inversion experiment creates a video of 20 frames out of the generation of two images that starts from different concepts provided by the user.

It is possible to load concepts giving a valid Huggin Face 🤗 concept repo:
https://huggingface.co/spaces/sd-concepts-library/stable-diffusion-conceptualizer

#### Usage

```python
  --model_id MODEL_ID   The s.d. model checpoint you want to use
  --from_file, --no-from_file
                        load arguments from file
  -p PROMPT_FILE_PATH, --prompt_file_path PROMPT_FILE_PATH
                        path file where to read prompt
  -s SEED, --seed SEED  Set the random seed
  --from_concept_repo FROM_CONCEPT_REPO
                        The start concept you want to use. (Provide a hugginface concept repo)
  --to_concept_repo TO_CONCEPT_REPO
                        The end concept you want to use. (Provide a hugginface concept repo)
  --from_prompt FROM_PROMPT
                        Start prompt you want to use
  --to_prompt TO_PROMPT
                        End prompt you want to use
  --num_inference_steps NUM_INFERENCE_STEPS
                        Number of inference step.
  --guidance_scale GUIDANCE_SCALE
                        The guidance scale value to set.
  --width WIDTH         Canvas width of generated image.
  --height HEIGHT       Canvas height of generated image.
  --use_negative_prompt, --no-use_negative_prompt
                        flag to use negative prompt stored in negative_prompt.txt
  -b BATCH_SIZE, --batch_size BATCH_SIZE
                        Batch size to use
  --mps, --no-mps       Set the device to 'mps' (M1 Apple)
```
#### example
```python
python textual_inversion.py --from_file -p "prompt_close_up.txt" --mps --num_inference_steps 50
```


## img -> caption -> img

This is more an evaluation across different models to perform image-to-text, providing caption to use as s.d. prompt for recreate the original image. 
It has been designed as an investigation task, so I used the notebook `captioning_task.ipynb` to conduct experiments.

There are 3 different models for image2caption wich have been evaluated
```python
mscoco_finetuned_CoCa-ViT-L-14-laion2B-s13B-b90k
vit-gpt2-image-captioning
blip-image-captioning-base
```
And then there is a comparison with a image2prompt model, the `CLIP-Interrogator`
```python
pharma/CLIP-Interrogator
```