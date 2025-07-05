from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import replicate
import base64
import io
from PIL import Image

app = FastAPI()

# Allow all origins (for testing)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# Set your Replicate API token here (will use env var)
client = replicate.Client(api_token="r8_E0qsUXfaFITjq67wSbPLaiWE9jOixMg44MuUU")  # <- We will override this on Render

@app.post("/transform/")
async def transform_image(file: UploadFile = File(...)):
    image = Image.open(io.BytesIO(await file.read()))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    base64_img = base64.b64encode(buffer.getvalue()).decode("utf-8")
    image_input = f"data:image/png;base64,{base64_img}"

    output = replicate.run(
        "timothybrooks/instruct-pix2pix:7d8b4e8b32b862e96df43d2ec4f306d33d0bb48c9e4b43ae90b50e5f0f5c90f5",
        input={"image": image_input, "prompt": "Make it modern"}
    )

    return {"output_url": output}
