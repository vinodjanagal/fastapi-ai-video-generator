import os
from diffusers import StableDiffusionPipeline
import torch

def main():
    print("✅ Loading Stable Diffusion v1.5 model (CPU mode)...")
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float32
    ).to("cpu")  # Running fully on CPU

    prompt = "a serene lake at dawn, mist on water, golden sunrise, ultra realistic, 35mm photography"
    print(f"\n🎨 Generating image for prompt:\n   {prompt}\n")

    image = pipe(
        prompt,
        num_inference_steps=30,
        guidance_scale=7.5
    ).images[0]

    output_dir = "static/previews/sd_test"
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, "sd_preview.png")
    image.save(output_path)

    print(f"\n✅ Image saved at: {output_path}")

if __name__ == "__main__":
    main()
