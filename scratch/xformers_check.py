from diffusers import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1")
pipe.enable_xformers_memory_efficient_attention()
print("xformers enabled successfully!")
