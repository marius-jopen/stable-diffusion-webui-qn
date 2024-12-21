import gradio as gr
import os
import sys
import importlib
from modules import rng, rng_qn_config
import json
from modules.rng_qn_config import DEFAULT_RNG_MODE  # Import the default mode

# Global variable to store pending changes
pending_mode = DEFAULT_RNG_MODE  # Initialize with default from config

def get_current_mode():
    global pending_mode
    # Return True if mode is "custom", False if "classic"
    return pending_mode == "custom"

def toggle_rng_mode(value):
    global pending_mode
    from modules.rng import ImageRNG
    
    mode = "custom" if value else "classic"
    pending_mode = mode
    
    # Update the RNG class-level mode
    ImageRNG.set_mode(mode)
    
    print(f"[QN EXTENSION] Toggling mode to: {mode}")
    return value

def save_settings(value=None):
    print(f"[QN EXTENSION] Current mode: {pending_mode}")
    return f"Current mode: {pending_mode}"

def create_ui():
    print("[QN EXTENSION] Starting UI creation")
    with gr.Blocks(analytics_enabled=False) as quantum_noise_control:
        with gr.Row():
            gr.HTML("<h1>Quantum Noise Control</h1>")
        
        initial_state = get_current_mode()
        print(f"[QN EXTENSION] Initial checkbox state: {initial_state}")
        print(f"[QN EXTENSION] Current pending_mode: {pending_mode}")
        
        with gr.Row():
            toggle = gr.Checkbox(
                label="Use Quantum Noise",
                value=initial_state,
                interactive=True
            )
            save_button = gr.Button(value="Save Settings")
            result = gr.Textbox(label="Status", interactive=False)
            
        toggle.change(
            fn=toggle_rng_mode,
            inputs=toggle,
            outputs=toggle,
            queue=False
        )
        
        save_button.click(
            fn=save_settings,
            inputs=[],
            outputs=[result]
        )
        
    return quantum_noise_control