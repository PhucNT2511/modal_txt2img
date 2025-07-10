## 1. Clone và cài đặt thư viện
```bash
git clone https://github.com/PhucNT2511/modal_txt2img
cd ./modal_txt2img

python -m venv .venv
source .venv/bin/activate
```

## 2. Cài đặt thư viện và modal
```bash
pip install -r requirements.txt
modal login
```

## 3. Build & deploy trên Modal
```bash
modal deploy txt2img_modal.py
```

## 4.Chạy local test
```bash
modal run txt2img_modal.py
```
## 5. Cách dùng curl để gọi API `/generate`
```bash
curl -X POST https://phucnt-zenai--txt2img-fastapi-app.modal.run/generate \
-H "Content-Type: application/json" \
-d '{
  "prompt": "A majestic cyberpunk samurai standing on a neon-lit rooftop, glowing katana in hand, overlooking a futuristic Tokyo skyline at night, ultra-detailed, volumetric lighting, cinematic angle, artstation, 4K, masterpiece",
  "negative_prompt": "blurry, distorted, bad quality",
  "num_inference_steps": 30,
  "guidance_scale": 8.5,
  "seed": 1337,
  "width": 512,
  "height": 512
}'
```

