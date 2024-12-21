import modules.scripts as scripts
import gradio as gr
from modules import script_callbacks

class QuantumNoiseControlScript(scripts.Script):
    def title(self):
        return "Quantum Noise Control"

    def show(self, is_img2img):
        return False

    def ui(self, is_img2img):
        return []

    def run(self, p, *args):
        return p

def on_ui_tabs():
    with gr.Blocks(analytics_enabled=False) as quantum_noise_control:
        with gr.Row():
            gr.HTML("Quantum Noise Control will appear here")
            
    return [(quantum_noise_control, "Quantum Noise Control", "quantum_noise_control")]

# Register the extension
script_callbacks.on_ui_tabs(on_ui_tabs)