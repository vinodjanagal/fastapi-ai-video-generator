# test_engine.py (Upgraded Version)
import subprocess
import sys
import os

# --- CONSTANTS EXTRACTED FROM YOUR ADVANCED SCRIPT (for easy testing) ---
BASE_QUALITY_PROMPT = "masterpiece, best quality, ultra-detailed, photorealistic, cinematic lighting, sharp focus, 8k"
BASE_NEGATIVE_PROMPT = "nsfw, lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry, deformed, mutation, mutilated, extra limbs, gross proportions, malformed limbs, disfigured face, ugly"
# -----------------------------------------------------------------------

# --- TUNE YOUR EXPERIMENT HERE ---

# 1. This is the CORE idea for the scene. This would come from your storyboard.
CORE_PROMPT = "A painter's hands on canvas, medium-shot, soft warm studio light, dreamy soft focus, cinematic, cluttered artist studio, easels, paintbrushes"

# 2. Assemble the final positive prompt using your advanced logic.
FINAL_POSITIVE_PROMPT = f"{CORE_PROMPT}, {BASE_QUALITY_PROMPT}"

# 3. Use the powerful negative prompt.
FINAL_NEGATIVE_PROMPT = BASE_NEGATIVE_PROMPT

# 4. Set the generation parameters.
NUM_STEPS = 4
GUIDANCE = 1.5
IP_ADAPTER_SCALE = 0.30 
OUTPUT_DIR = "test_output_v7" # Use a new folder for the new results
# -----------------------------------

# --- EXECUTION LOGIC (No changes needed below here) ---

print(f"--- Testing AnimateDiff Engine (Advanced Prompts) ---")
print(f"POSITIVE: {FINAL_POSITIVE_PROMPT}")
print(f"NEGATIVE: {FINAL_NEGATIVE_PROMPT}")
print(f"Steps: {NUM_STEPS}, Guidance: {GUIDANCE}")
print("-----------------------------------------------------")

# Ensure the output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# This calls your engine directly, just like the orchestrator does, now with the negative prompt.
command = [
    sys.executable,
    "app/video_engines/animate_diff_engine.py",
    "--prompt", FINAL_POSITIVE_PROMPT,
    "--negative-prompt", FINAL_NEGATIVE_PROMPT, # <<<--- THE NEW ARGUMENT IS HERE
    "--output-dir", OUTPUT_DIR,
    "--num-steps", str(NUM_STEPS),
    "--guidance-scale", str(GUIDANCE),
    "--ip-adapter-scale", str(IP_ADAPTER_SCALE),
    "--ip-adapter-image-path", "reference_image.jpeg"
]

# We use Popen to stream the output in real-time
try:
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace')

    while True:
        output = process.stdout.readline()
        if output == '' and process.poll() is not None:
            break
        if output:
            print(output.strip())

    rc = process.poll()
    print(f"------------------------------------")
    print(f"Process finished with return code: {rc}")
    if rc == 0:
        print(f"✅ Success! Frames saved to '{os.path.abspath(OUTPUT_DIR)}'")
    else:
        print(f"❌ Error! The engine script failed.")

except FileNotFoundError:
    print(f"❌ FATAL ERROR: Could not find 'app/video_engines/animate_diff_engine.py'. Make sure you are running this script from the project's root directory ('D:\\revision').")
except Exception as e:
    print(f"❌ An unexpected error occurred: {e}")