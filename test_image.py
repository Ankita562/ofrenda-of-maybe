import torch
from diffusers import AutoPipelineForText2Image

pipe = AutoPipelineForText2Image.from_pretrained(
    "stabilityai/sd-turbo", torch_dtype=torch.float16, variant="fp16"
)
pipe.enable_model_cpu_offload()  # keeps VRAM use low; don't call .to("cuda")

img = pipe(
    "a candlelit ofrenda with glowing marigolds, ghostly watercolor",
    num_inference_steps=2, guidance_scale=0.0, height=512, width=512,
).images[0]
img.save("test.png")
print("done")