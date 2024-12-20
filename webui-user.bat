@echo off

set PYTHON=
set GIT=
set VENV_DIR=
set COMMANDLINE_ARGS=--api  --deforum-api --opt-sdp-attention --autolaunch --ckpt-dir E:/stable-diffusion/includes/checkpoints --lora-dir E:/stable-diffusion/includes/loras  --controlnet-dir E:/stable-diffusion/includes/controlnet

call webui.bat
