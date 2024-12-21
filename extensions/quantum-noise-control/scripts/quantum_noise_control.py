import modules.scripts as scripts
from modules import script_callbacks
import sys
import os

# Add the current directory to the Python path
scripts_path = os.path.dirname(os.path.abspath(__file__))
if scripts_path not in sys.path:
    sys.path.append(scripts_path)

from quantum_noise_ui import create_ui

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
    quantum_noise_control = create_ui()
    return [(quantum_noise_control, "Quantum Noise Control", "quantum_noise_control")]

# Register the extension
script_callbacks.on_ui_tabs(on_ui_tabs)