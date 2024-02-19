import PIL
import torch
from PIL import Image
from diffusers import StableDiffusionInpaintPipeline

import os

def pair_images_with_masks(directory):
    pairs = []
    for filename in os.listdir(directory):
        if filename.endswith("_mask.png"):
            infos = filename.split("_")
            original = infos[0] + ".png"
            pairs.append((original, filename))
    return pairs



if __name__ == "__main__":
    """Example usage:
    python stablediffusion.py \ """

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        dtype = torch.float16
    else:
        dtype = torch.float32

    pipe = StableDiffusionInpaintPipeline.from_pretrained(
         "runwayml/stable-diffusion-inpainting", torch_dtype=dtype
     )
    pipe = pipe.to(device)
    directory = "./orig_mask_depth"
    image_pairs = pair_images_with_masks(directory)
    prompt = "Floor and Wall"
    for num, (original, mask) in enumerate(image_pairs): 
        orig_img = Image.open("./orig_mask_depth/" + original)
        mask = Image.open("./orig_mask_depth/" + mask)
        
        image = pipe(prompt=prompt, image=orig_img, mask_image=mask).images[0]
        fp = original.split(".")[0]
        image.save(f"{fp}_output.png")
        print(f"done with {fp}")
            





    

