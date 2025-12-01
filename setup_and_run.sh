
pip install -q "protobuf==3.20.3" "moviepy==1.0.3" "imageio[ffmpeg]" 
pip install -q diffusers transformers accelerate safetensors opencv-python groq
python app/orchestrator/phoenix_director.py --prompt "$1" --mode production
