# tools/diagnose_vae.py
import sys
from pathlib import Path
import torch
from diffusers import AutoencoderKL
from PIL import Image

# Add project root to path
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

def main():
    """A simple, fast, CPU-only tool to verify a VAE can load and work."""
    print("--- VAE Diagnostic Tool ---")
    vae_id = "stabilityai/sd-vae-ft-mse"
    device = torch.device("cpu")
    dtype = torch.float32

    try:
        print(f"Attempting to load VAE: {vae_id} with dtype: {dtype}...")
        vae = AutoencoderKL.from_pretrained(vae_id, torch_dtype=dtype).to(device)
        print("✅ VAE loaded successfully.")

        # Create a dummy black image tensor
        dummy_tensor = torch.zeros(1, 3, 512, 512, device=device, dtype=dtype)
        print("Created a dummy 512x512 black image tensor.")

        # Test the encode/decode loop
        print("Testing encode -> decode loop...")
        with torch.no_grad():
            latent = vae.encode(dummy_tensor).latent_dist.sample()
            decoded_tensor = vae.decode(latent).sample
        print("✅ Encode/decode loop completed without errors.")

        # Convert back to an image
        decoded_tensor = (decoded_tensor / 2 + 0.5).clamp(0, 1)
        img = Image.fromarray((decoded_tensor[0].permute(1, 2, 0).cpu().numpy() * 255).astype("uint8"))
        
        output_path = project_root / "output/diagnostics/vae_test_output.png"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(output_path)
        
        print(f"✅ SUCCESS: Saved a reconstructed black image to {output_path}")
        print("--- Diagnosis: The VAE is working correctly on your system. ---")
        return 0
    except Exception as e:
        print(f"❌ FAILED: A critical error occurred: {e}")
        import traceback
        traceback.print_exc()
        print("--- Diagnosis: The VAE is broken or incompatible with your system. ---")
        return 1

if __name__ == "__main__":
    sys.exit(main())