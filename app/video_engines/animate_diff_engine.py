import argparse
import torch
import os
import logging
from diffusers import AnimateDiffPipeline, MotionAdapter, EulerDiscreteScheduler
from diffusers.utils import export_to_gif

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("phoenix.engine")

def run(args):
    print(f">> ENGINE START: {args.width}x{args.height} | Steps: {args.num_steps}")
    
    if not torch.cuda.is_available():
        raise RuntimeError("CRITICAL: CUDA GPU NOT DETECTED.")
        
    device = "cuda"
    torch.use_deterministic_algorithms(False)
    
    adapter = MotionAdapter.from_pretrained("guoyww/animatediff-motion-adapter-v1-5-2", torch_dtype=torch.float16)
    pipe = AnimateDiffPipeline.from_pretrained(
        "Lykon/dreamshaper-7",
        motion_adapter=adapter,
        torch_dtype=torch.float16
    ).to(device)
    
    pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing", beta_schedule="linear")
    pipe.enable_vae_slicing() 
    
    generator = torch.Generator("cpu").manual_seed(args.seed)
    
    output = pipe(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        num_frames=args.num_frames,
        guidance_scale=args.guidance_scale,
        num_inference_steps=args.num_steps,
        width=args.width,
        height=args.height,
        generator=generator
    )

    frames = output.frames[0]
    os.makedirs(args.output_dir, exist_ok=True)
    
    frame_dir = os.path.join(args.output_dir, "frames")
    os.makedirs(frame_dir, exist_ok=True)
    for i, frame in enumerate(frames):
        frame.save(os.path.join(frame_dir, f"{i:04d}.png"))
        
    print(f">> SUCCESS. Output saved to: {args.output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--negative-prompt", type=str, default="bad quality")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--num-frames", type=int, default=16)
    parser.add_argument("--num-steps", type=int, default=25)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=384)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--guidance-scale", type=float, default=7.5)
    args = parser.parse_args()
    run(args)
