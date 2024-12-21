import gradio as gr
import os
import sys
import importlib
from modules import rng, rng_qn_config
import json
from modules.rng_qn_config import DEFAULT_RNG_MODE, NOISE_SETTINGS, BLEND_SETTINGS  # Import the default mode

# Global variable to store pending changes
pending_mode = DEFAULT_RNG_MODE  # Initialize with default from config
pending_norm_strength = NOISE_SETTINGS["norm_strength"]
pending_option = "1"  # Default option
pending_selected_file = None
pending_blend_ratio = BLEND_SETTINGS["first_step"]["blend_ratio"]
pending_blend_mode = BLEND_SETTINGS["first_step"]["blend_mode"]

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

def update_option(value):
    global pending_option
    pending_option = value
    print(f"[QN EXTENSION] Pending option update to: {pending_option}")
    return value

def update_blend_ratio(value):
    global pending_blend_ratio
    pending_blend_ratio = float(value)
    print(f"[QN EXTENSION] Pending blend_ratio update to: {pending_blend_ratio}")
    return value

def update_blend_mode(value):
    global pending_blend_mode
    pending_blend_mode = value
    print(f"[QN EXTENSION] Pending blend_mode update to: {pending_blend_mode}")
    return value

def save_settings(norm_value=None, option_value=None, blend_ratio_value=None, blend_mode_value=None):
    global pending_mode, pending_norm_strength, pending_selected_file, pending_blend_ratio, pending_blend_mode
    
    # Update pending values from inputs
    if norm_value is not None:
        pending_norm_strength = float(norm_value)
    if option_value is not None:
        pending_selected_file = option_value
    if blend_ratio_value is not None:
        pending_blend_ratio = float(blend_ratio_value)
    if blend_mode_value is not None:
        pending_blend_mode = blend_mode_value
    
    # Actually apply the pending changes
    from modules.rng_qn_load import _current_norm_strength
    import modules.rng_qn_load as rng_qn_load
    from modules.rng_qn_config import BLEND_SETTINGS
    
    # Update the norm strength
    rng_qn_load._current_norm_strength = pending_norm_strength
    
    # Update the selected file globally
    noise_file_path = os.path.join("input_quantum-noise", pending_selected_file)
    rng_qn_load.NOISE_FILE = noise_file_path
    
    # Update blend settings for both first and subsequent steps
    BLEND_SETTINGS["first_step"]["blend_ratio"] = pending_blend_ratio
    BLEND_SETTINGS["first_step"]["blend_mode"] = pending_blend_mode
    BLEND_SETTINGS["subsequent_steps"]["blend_ratio"] = pending_blend_ratio
    BLEND_SETTINGS["subsequent_steps"]["blend_mode"] = pending_blend_mode
    
    print(f"[QN EXTENSION] Saving settings...")
    print(f"[QN EXTENSION] Current mode: {pending_mode}")
    print(f"[QN EXTENSION] Applied norm_strength: {pending_norm_strength}")
    print(f"[QN EXTENSION] Applied quantum noise file path: {noise_file_path}")
    print(f"[QN EXTENSION] Applied quantum noise filename: {pending_selected_file}")
    print(f"[QN EXTENSION] Applied blend ratio: {pending_blend_ratio}")
    print(f"[QN EXTENSION] Applied blend mode: {pending_blend_mode}")
    
    result = f"Current mode: {pending_mode}\nNorm Strength: {pending_norm_strength}\nSelected File: {pending_selected_file}\nBlend Ratio: {pending_blend_ratio}\nBlend Mode: {pending_blend_mode}"
    return result

def get_quantum_noise_files():
    folder_path = "input_quantum-noise"
    try:
        # Ensure the folder exists
        if not os.path.exists(folder_path):
            print(f"[QN EXTENSION] Warning: {folder_path} directory not found")
            return ["1", "2", "3"]  # fallback to default values
        
        # Get all files in the directory
        files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
        
        if not files:
            print(f"[QN EXTENSION] Warning: No files found in {folder_path}")
            return ["1", "2", "3"]  # fallback to default values
            
        print(f"[QN EXTENSION] Found files: {files}")
        return files
        
    except Exception as e:
        print(f"[QN EXTENSION] Error reading directory: {e}")
        return ["1", "2", "3"]  # fallback to default values

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
                blend_ratio = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    value=BLEND_SETTINGS["first_step"]["blend_ratio"],
                    step=0.01,
                    label="Blend Ratio (0=pure quantum, 1=pure standard)",
                    interactive=True,
                    release=True
                )
                blend_mode = gr.Dropdown(
                    choices=["normal", "screen", "multiply", "difference"],
                    value=BLEND_SETTINGS["first_step"]["blend_mode"],
                    label="Blend Mode",
                    interactive=True
                )
                option_dropdown = gr.Dropdown(
                    choices=get_quantum_noise_files(),
                    value=get_quantum_noise_files()[0] if get_quantum_noise_files() else "1",
                    label="Select Quantum Noise File",
                    interactive=True
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
            inputs=[norm_strength, option_dropdown, blend_ratio, blend_mode],
            outputs=[result]
        )
        
    return quantum_noise_control