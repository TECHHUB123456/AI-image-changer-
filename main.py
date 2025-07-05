from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import replicate
import base64
import io
from PIL import Image
import os

app = FastAPI()

# CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# Replicate API
client = replicate.Client(api_token=os.environ["r8_E0qsUXfaFITjq67wSbPLaiWE9jOixMg44MuUU"])


@app.post("/transform/")
async def transform_image(file: UploadFile = File(...)):
    try:
        # Read image bytes
        contents = await file.read()

        # Decode and convert to base64
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        base64_img = base64.b64encode(buffer.getvalue()).decode("utf-8")
        image_input = f"data:image/png;base64,{base64_img}"

        # Call Replicate model
        output = replicate.run(
            "timothybrooks/instruct-pix2pix:7d8b4e8b32b862e96df43d2ec4f306d33d0bb48c9e4b43ae90b50e5f0f5c90f5",
            input={
                "image": image_input,
                "prompt": "Make it modern"
            }
        )

        return {"output_url": output}

    except Exception as e:
        print("Error:", e)
        return JSONResponse(status_code=500, content={"error": str(e)})
