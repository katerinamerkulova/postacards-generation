import ruclip
from rudalle.pipelines import generate_images, show, super_resolution, cherry_pick_by_ruclip
from rudalle import get_rudalle_model, get_tokenizer, get_vae, get_realesrgan
from rudalle.utils import seed_everything

class Generator:
    def __init__(self):
        # prepare models:
        seed_everything(42)
        device = 'cuda'
        self.dalle = get_rudalle_model('Malevich', pretrained=True, fp16=True, device=device)
        self.tokenizer = get_tokenizer()
        self.vae = get_vae(dwt=True).to(device)

        # pipeline utils:
        realesrgan = get_realesrgan('x2', device=device)
        clip, processor = ruclip.load('ruclip-vit-base-patch32-384', device=device)
        clip_predictor = ruclip.Predictor(clip, processor, device, bs=8)

    def get_images(self, text):
        top_k, top_p, images_num = 2048, 0.995, 8
        pil_images, scores = generate_images(
            text, self.tokenizer, self.dalle, self.vae, 
            top_k=top_k, images_num=images_num, bs=8, top_p=top_p
        )
        return pil_images

if __name__ == "__main__":
    g = Generator()
    pil_images = g.get_images('Открытка 1799 года: счастья, здоровья, блогополучия, люблю')
    show(pil_images, 6, save_dir='images')
