# tools/debug_pipeline.py
import sys, json
from pathlib import Path
import torch

def inspect_pipeline(base_model="SG161222/Realistic_Vision_V5.1_noVAE", motion_adapter_model="guoyww/animatediff-motion-adapter-v1-5-2"):
    print("=== DEBUG PIPELINE INSPECT ===")
    print("Torch:", torch.__version__)

    try:
        from diffusers import AnimateDiffPipeline
    except Exception as e:
        print("ERROR: diffusers AnimateDiffPipeline import failed:", e)
        return

    # Attempt to load pipeline with motion adapter passed (but do not run inference)
    try:
        print(f"Loading pipeline {base_model} with motion-adapter arg (no heavy run)...")
        # Lazy: pass motion_adapter param if MotionAdapter is importable
        try:
            from diffusers import MotionAdapter
            has_motion_adapter = True
        except Exception:
            MotionAdapter = None
            has_motion_adapter = False

        if has_motion_adapter:
            print("MotionAdapter class import OK")
            # We will not call MotionAdapter.from_pretrained here to avoid heavy download if not needed
            pipe = AnimateDiffPipeline.from_pretrained(base_model, torch_dtype=torch.float32)
            # Show top-level attributes
            print("PIPE ATTRS:", [k for k in dir(pipe) if not k.startswith("_")][:40])
            # inspect unet modules presence
            unet = getattr(pipe, "unet", None)
            if unet is None:
                print("UNET: NOT FOUND on pipeline (fatal).")
            else:
                # Count submodule names that look like attention or motion
                names = [n for n, _ in unet.named_modules()]
                print("UNET module count:", len(names))
                # Detect typical motion-adapter insertion points
                found_motion = [n for n in names if "motion" in n.lower() or "adapter" in n.lower()]
                print("UNET modules with 'motion' or 'adapter' in name (sample):", found_motion[:10])
        else:
            print("MotionAdapter class unavailable in this environment. Cannot attempt adapter-based patching.")

        # Also show scheduler type
        print("Scheduler type:", type(pipe.scheduler), getattr(pipe.scheduler, "__class__", None))
        print("Scheduler config keys sample:", list(getattr(pipe.scheduler, "config", {}).keys())[:20])
    except Exception as e:
        print("PIPELOAD ERROR:", repr(e))
        return

    # Now try a very small inference call to observe returned structure
    try:
        print("Attempting tiny inference to inspect output shape (1 step, 2 frames)...")
        g = torch.Generator(device="cpu").manual_seed(1)
        out = pipe(
            prompt="test",
            negative_prompt="",
            width=128,
            height=128,
            num_inference_steps=1,
            guidance_scale=1.0,
            generator=g,
            num_frames=2
        )
        # Print available attributes on output
        print("Output attributes:", [a for a in dir(out) if not a.startswith("_")][:50])
        if hasattr(out, "frames"):
            print("OUT.frames is present. frames len:", len(out.frames))
        if hasattr(out, "images"):
            print("OUT.images present. images len:", len(out.images) if isinstance(out.images, list) else None)
        # Print dtype/shape of first tensor-like frame if possible
        first = None
        if hasattr(out, "frames") and len(out.frames) > 0:
            first = out.frames[0]
        elif hasattr(out, "images") and len(out.images) > 0:
            first = out.images[0]
        print("Type of first frame:", type(first))
    except Exception as e:
        print("Tiny inference error (expected if pipeline heavy):", repr(e))

if __name__ == "__main__":
    inspect_pipeline()
