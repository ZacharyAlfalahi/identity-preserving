import gradio as gr
import torch
import spaces
from PIL import Image
from diffusers import StableDiffusionInstructPix2PixPipeline

MODEL_ID = "brandonwooding/glasses_removal_lora"
PROMPT = "Remove the glasses from this person's face. Preserve all facial features, skin texture, and expression exactly."

# Load model on CPU at startup (ZeroGPU only provides GPU inside @spaces.GPU)
pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    safety_checker=None,
)

@spaces.GPU
def remove_glasses(
    input_image: Image.Image,
    guidance_scale: float,
    image_guidance_scale: float,
    num_inference_steps: int,
):
    pipe.to("cuda")
    input_image = input_image.convert("RGB").resize((512, 512))

    generator = torch.Generator("cuda").manual_seed(42)

    result = pipe(
        prompt=PROMPT,
        image=input_image,
        guidance_scale=guidance_scale,
        image_guidance_scale=image_guidance_scale,
        num_inference_steps=int(num_inference_steps),
        generator=generator,
    ).images[0]

    return result


demo = gr.Interface(
    fn=remove_glasses,
    inputs=[
        gr.Image(type="pil", label="Upload an image of a face with glasses"),
        gr.Slider(1.0, 15.0, value=7.5, step=0.5, label="Guidance Scale"),
        gr.Slider(1.0, 3.0, value=1.5, step=0.1, label="Image Guidance Scale"),
        gr.Slider(10, 100, value=50, step=5, label="Inference Steps"),
    ],
    outputs=gr.Image(type="pil", label="Resulting image with glasses removed"),
    title="Identity-Preserving Face Editing (Glasses Removal)",
    description=(
        "Upload an image of a face wearing glasses. The model, a fine-tuned "
        "InstructPix2Pix with a Stable Diffusion of 1.5, will remove the glasses "
        "while preserving the person's identity. Adjust the sliders to control "
        "generation quality."
    ),
    examples=[],

)

if __name__ == "__main__":
    demo.launch()