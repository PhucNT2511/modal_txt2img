import modal
import base64
import json
from io import BytesIO

app = modal.App("txt2img")
CACHE_DIR = "/cache"
cache_vol = modal.Volume.from_name("model-cache", create_if_missing=True)

# 1. Image build
image = modal.Image.debian_slim(python_version="3.10") \
    .pip_install_from_requirements("requirements_modal.txt")

# 2. Worker Class với lazy init
@app.cls(
    image=image,
    gpu="A100",
    timeout=600,
    scaledown_window=300,
    volumes={CACHE_DIR: cache_vol},
)
@modal.concurrent(max_inputs=10)
class PipelineWorker:
    @modal.method()
    def generate(
        self,
        prompt: str,
        negative_prompt: str = None,
        num_inference_steps: int = 25,
        guidance_scale: float = 7.5,
        seed: int = 42,
        width: int = 512,
        height: int = 512,
    ):
        # Lazy init model + generator
        if not hasattr(self, "_pipe"):
            import torch
            from diffusers import StableDiffusionPipeline
            # load once per container
            self._generator = torch.Generator("cuda")
            self._pipe = StableDiffusionPipeline.from_pretrained(
                "runwayml/stable-diffusion-v1-5",
                torch_dtype=torch.float16,
                cache_dir=CACHE_DIR,
            ).to("cuda")

        # seed & inference
        gen = self._generator.manual_seed(seed)
        image = self._pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=gen,
            width=width,
            height=height,
        ).images[0]

        # encode to base64
        buf = BytesIO()
        image.save(buf, format="PNG")
        img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        return {
            "image_base64": img_b64,
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "seed": seed,
            "width": width,
            "height": height,
        }

# stub để gọi remote
worker = PipelineWorker()

# 3. FastAPI endpoint
@app.function(
    image=image,
    gpu="A100",
    volumes={CACHE_DIR: cache_vol},
    timeout=300,
)
@modal.asgi_app()
def fastapi_app():
    from fastapi import FastAPI, Request
    api = FastAPI()

    @api.get("/")
    async def root():
        return {"message": "txt2img FastAPI is up and running"}
    
    # Health check endpoint
    @api.get("/health")
    async def health():
        return {"status": "ok"}

    @api.post("/generate")
    async def generate(req: Request):
        try:
            data = await req.json()
            print("[DEBUG] Input:", data)
            out = worker.generate.remote(
                prompt=data.get("prompt", ""),
                negative_prompt=data.get("negative_prompt"),
                num_inference_steps=data.get("num_inference_steps", 25),
                guidance_scale=data.get("guidance_scale", 7.5),
                seed=data.get("seed", 42),
                width=data.get("width", 512),
                height=data.get("height", 512),
            )
            return out
        except Exception as e:
            print("[ERROR]", str(e))
            return {"error": str(e)}
        
    return api


# 4. Local entrypoint
@app.local_entrypoint()
def main():
    result = worker.generate.remote(
        prompt="A majestic cyberpunk samurai standing on a neon-lit rooftop, glowing katana in hand, overlooking a futuristic Tokyo skyline at night, ultra-detailed, volumetric lighting, cinematic angle, artstation, 4K, masterpiece, trending on ArtStation",
        negative_prompt="blurry, distorted, bad quality",
        num_inference_steps=30,
        guidance_scale=8.5,
        seed=1337,
        width=512,
        height=512,
    )
    # save outputs
    with open("output.png", "wb") as f:
        f.write(base64.b64decode(result["image_base64"]))
    with open("output.json", "w") as f:
        json.dump({k: v for k, v in result.items() if k != "image_base64"}, f, indent=2)
    print("✅ Done: output.png + output.json")
