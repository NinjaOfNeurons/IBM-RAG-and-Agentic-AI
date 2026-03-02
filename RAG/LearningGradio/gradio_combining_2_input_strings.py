import gradio as gr
from huggingface_hub import HfFolder



    
def combine_input(a,b):
    return a+""+b


# define the combine number
demo = gr.Interface(
    fn=combine_input, 
    inputs=[gr.Textbox(label="Input1"), gr.Textbox(label="Input2")], 
    outputs=gr.Textbox(label="Output")  # Create numerical output fields
)

# Launch the interface
demo.launch(server_name="127.0.0.1", server_port= 7860)
