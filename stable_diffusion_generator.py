import torch
import translators as ts
from diffusers import StableDiffusionPipeline


class Generator:
    def __init__(self):
        model_id = "runwayml/stable-diffusion-v1-5"
        self.pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
        self.pipe = self.pipe.to("cuda")

    def get_images(self, text):
        num_rus = 4
        rus_promts = [text + ' high quality, high resolution, 4 k'] * num_rus
        #eng_prompts = [ts.translate_text(text) + ' high quality, high resolution, 4 k'] * num_eng
        
        images = []
        #images.extend(self.pipe(eng_prompts).images)
        images.extend(self.pipe(rus_promts).images)
        return images


if __name__ == "__main__":
    g = Generator()
    # prompt = "Российская почтовая открытка 1799 года: счастья, здоровья, благополучия, люблю"
    prompt = "Российская почтовая открытка 1991 год: перестройка, америка, свобода, пейзаж, постер, плакат"
    images = g.get_images(prompt)
    for i, image in enumerate(images):
        image.save(f"res/{prompt}_{i}.png")

