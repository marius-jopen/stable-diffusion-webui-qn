import gradio as gr
import os
import sys
import importlib
from modules import rng, rng_qn_config
import json
from modules.rng_qn_config import DEFAULT_RNG_MODE, NOISE_SETTINGS  # Import the default mode

# Global variable to store pending changes
pending_mode = DEFAULT_RNG_MODE  # Initialize with default from config
pending_norm_strength = NOISE_SETTINGS["norm_strength"]

def get_current_mode():
    global pending_mode
    print(f"[QN EXTENSION] Getting current mode: {pending_mode}")
    return pending_mode == "custom"

def toggle_rng_mode(value):
    global pending_mode
    from modules.rng import ImageRNG
    
    mode = "custom" if value else "classic"
    pending_mode = mode
    
    # Update the RNG class-level mode
    ImageRNG.set_mode(mode)
    
    print(f"[QN EXTENSION] Toggling mode to: {mode}")
    print(f"[QN EXTENSION] New pending_mode value: {pending_mode}")
    return value

def update_norm_strength(value):
    global pending_norm_strength
    pending_norm_strength = float(value)
    print(f"[QN EXTENSION] Pending norm_strength update to: {pending_norm_strength}")
    return value

def save_settings(norm_value=None):
    global pending_mode, pending_norm_strength
    
    # Update pending values from inputs
    if norm_value is not None:
        pending_norm_strength = float(norm_value)
    
    # Actually apply the pending changes
    from modules.rng_qn_load import _current_norm_strength
    import modules.rng_qn_load as rng_qn_load
    
    # Update the norm strength
    rng_qn_load._current_norm_strength = pending_norm_strength
    
    print(f"[QN EXTENSION] Saving settings...")
    print(f"[QN EXTENSION] Current mode: {pending_mode}")
    print(f"[QN EXTENSION] Applied norm_strength: {pending_norm_strength}")
    result = f"Current mode: {pending_mode}, Norm Strength: {pending_norm_strength}"
    print(f"[QN EXTENSION] Returning status: {result}")
    return result

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
            with gr.Column(visible=initial_state) as strength_controls:
                norm_strength = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    value=NOISE_SETTINGS["norm_strength"],
                    step=0.01,
                    label="Normalization Strength",
                    interactive=True,
                    release=True
                )
            save_button = gr.Button(value="Save Settings")
            result = gr.Textbox(label="Status", interactive=False)
            
        toggle.change(
            fn=toggle_rng_mode,  # Restore the original toggle function
            inputs=toggle,
            outputs=toggle,
            queue=False
        ).then(  # Chain the visibility update after the toggle
            fn=lambda x: gr.update(visible=x),
            inputs=[toggle],
            outputs=[strength_controls],
            queue=False
        )
        
        save_button.click(
            fn=save_settings,
            inputs=[norm_strength],
            outputs=[result]
        )
        
    return quantum_noise_control