import gradio as gr

def create_ui():
    with gr.Blocks(analytics_enabled=False) as quantum_noise_control:
        with gr.Row():
            gr.HTML("<h1>Hello World</h1>")
            
    return quantum_noise_control 