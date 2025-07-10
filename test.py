import requests
import base64

# URL của FastAPI app đã được serve trên Modal
BASE_URL = "https://phucnt-zenai--txt2img-fastapi-app.modal.run"

# 1. Kiểm tra endpoint /health
def check_health():
    print("🔍 Checking /health ...")
    res = requests.get(f"{BASE_URL}/health")
    if res.status_code == 200:
        print("✅ Health check:", res.json())
    else:
        print(f"❌ Health check failed ({res.status_code}):", res.text)

# 2. Gửi prompt tới /generate và lưu ảnh
def generate_image():
    print("🎨 Generating image ...")
    payload = {
        "prompt": "A majestic cyberpunk samurai standing on a neon-lit rooftop, glowing katana in hand, overlooking a futuristic Tokyo skyline at night, ultra-detailed, volumetric lighting, cinematic angle, artstation, 4K, masterpiece",
        "negative_prompt": "blurry, distorted, bad quality",
        "num_inference_steps": 30,
        "guidance_scale": 8.5,
        "seed": 1337,
        "width": 512,
        "height": 512,
    }

    res = requests.post(f"{BASE_URL}/generate", json=payload)
    if res.status_code == 200:
        result = res.json()
        print("🪵 Full response JSON:", result)
        with open("output_modal.png", "wb") as f:
            f.write(base64.b64decode(result["image_base64"]))
        print("✅ Saved image to output_modal.png")
    else:
        print(f"❌ Failed to generate image ({res.status_code}):", res.text)

if __name__ == "__main__":
    #check_health()
    generate_image()
