import os
import uuid
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from io import BytesIO
import numpy as np
from test_wholeimage_swapsingle import run_image_single
from test_wholeimage_swapmulti import run_image_multiple
from test_video_swapsingle import run_video_single
from test_video_swapmulti import run_video_multiple
import cv2
import shutil
from pytube import YouTube
from options.inference_options import Options
opt = Options()
opt.crop_size = 224
opt.use_mask = True
opt.name = 'people'
opt.Arc_path = 'arcface_model/arcface_checkpoint.tar'
opt.temp_path = './temp_results'

app = FastAPI()

# CORS middleware to allow requests from any origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
@app.post("/swap-face-single")
async def swap_face(
    image_a: UploadFile = File(...),
    image_b: UploadFile = File(...),
    add_logo: bool = Form(False),
):
    try:
        # Read the images directly into memory
        if not image_a or not image_b:
            raise HTTPException(status_code=400, detail="Both images must be provided")
        if not image_a.content_type.startswith("image/") or not image_b.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="Uploaded files must be images")
        img_a = cv2.imdecode(np.frombuffer(await image_a.read(), np.uint8), cv2.IMREAD_COLOR)
        img_b = cv2.imdecode(np.frombuffer(await image_b.read(), np.uint8), cv2.IMREAD_COLOR)

        if add_logo:
            opt.no_simswaplogo = not add_logo
            # logging.info("Swap face single triggered, logo: enabled")
        else:
            pass
            # logging.info("Swap face single triggered, logo: disabled")

        output = run_image_single(img_a, img_b, opt)
        # Convert the output image to bytes
        _, img_bytes = cv2.imencode(".jpg", output)
        img_io = BytesIO(img_bytes.tobytes())
        # logger.info("Swap face single completed")

        # Return the result directly without saving to disk
        return StreamingResponse(content=img_io, media_type="image/jpeg", headers={'Content-Disposition': 'inline; filename=output.jpg'})

    except Exception as e:
        # logger.error("Swap face single failed: %s", str(e))
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/swap-face-multiple")
async def swap_face_multiple(
    image_a: UploadFile = File(...),
    image_b: UploadFile = File(...),
    add_logo: bool = Form(False),
):
    try:
        # Read the images directly into memory
        if not image_a or not image_b:
            raise HTTPException(status_code=400, detail="Both images must be provided")
        if not image_a.content_type.startswith("image/") or not image_b.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="Uploaded files must be images")
        img_a = cv2.imdecode(np.frombuffer(await image_a.read(), np.uint8), cv2.IMREAD_COLOR)
        img_b = cv2.imdecode(np.frombuffer(await image_b.read(), np.uint8), cv2.IMREAD_COLOR)

        if add_logo:
            opt.no_simswaplogo = not add_logo
            # logging.info("Swap face multiple triggered, logo: enabled")
        else:
            pass
            # logging.info("Swap face multiple triggered, logo: disabled")

        output = run_image_multiple(img_a, img_b, opt)

        # Convert the output image to bytes
        _, img_bytes = cv2.imencode(".jpg", output)
        img_io = BytesIO(img_bytes.tobytes())
        # logger.info("Swap face multiple completed")

        # Return the result directly without saving to disk
        return StreamingResponse(content=img_io, media_type="image/jpeg", headers={'Content-Disposition': 'inline; filename=output.jpg'})

    except Exception as e:
        # logger.error("Swap face multiple failed: %s", str(e))
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/swap-video-single")
async def swap_video_single(
    video: UploadFile = File(None),
    video_url: str = Form(None),
    image: UploadFile = File(...),
    add_logo: bool = Form(False),
):
    try:
        opt.output_path = './output/swap_video_single_temp.mp4'

        # Check if either video file or video URL is provided
        if not video and not video_url:
            raise HTTPException(status_code=400, detail="Either video file or video URL must be provided")
        
        if video or video_url:
            if not image:
                raise HTTPException(status_code=400, detail="Image must be provided")
            elif not image.content_type.startswith("image/"):
                raise HTTPException(status_code=400, detail="Uploaded file must be an image")
            
        if video and not video.content_type.startswith("video/"):
            raise HTTPException(status_code=400, detail="Uploaded file must be a video")
        
        # Handle video file upload
        if video:
            video_path = f"{uuid.uuid4()}.mp4"
            with open(video_path, "wb") as buffer:
                shutil.copyfileobj(video.file, buffer)
            opt.video_path = video_path
        # Handle YouTube video URL
        elif video_url:
            video_path = download_youtube_video(video_url)
            opt.video_path = video_path
        
        img = cv2.imdecode(np.frombuffer(await image.read(), np.uint8), cv2.IMREAD_COLOR)

        if add_logo:
            opt.no_simswaplogo = not add_logo
            # logging.info("Swap video single triggered, logo: enabled")
        else:
            pass
            # logging.info("Swap video single triggered, logo: disabled")

        output_path = run_video_single(img, opt)
        os.remove(video_path)
        # logger.info(f"Swap video single completed, result saved to {output_path}")
        return FileResponse(output_path, media_type="video/mp4", filename='output.mp4')
    
    except Exception as e:
        # logger.error("Swap video single failed: %s", str(e))
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/swap-video-multiple")
async def swap_video_multiple(
    video: UploadFile = File(None),
    video_url: str = Form(None),
    image: UploadFile = File(...),
    add_logo: bool = Form(False),
):
    try:
        opt.output_path = './output/swap_video_multiple_temp.mp4'

        # Check if either video file or video URL is provided
        if not video and not video_url:
            raise HTTPException(status_code=400, detail="Either video file or video URL must be provided")
        
        if video or video_url:
            if not image:
                raise HTTPException(status_code=400, detail="Image must be provided")
            elif not image.content_type.startswith("image/"):
                raise HTTPException(status_code=400, detail="Uploaded file must be an image")
            
        if video and not video.content_type.startswith("video/"):
            raise HTTPException(status_code=400, detail="Uploaded file must be a video")

        # Handle video file upload
        if video:
            video_path = f"{uuid.uuid4()}.mp4"
            with open(video_path, "wb") as buffer:
                shutil.copyfileobj(video.file, buffer)
            opt.video_path = video_path
        # Handle YouTube video URL
        elif video_url:
            video_path = download_youtube_video(video_url)
            opt.video_path = video_path

        img = cv2.imdecode(np.frombuffer(await image.read(), np.uint8), cv2.IMREAD_COLOR)

        if add_logo:
            # Set the global variable or pass it as an argument to run_video_single
            opt.no_simswaplogo = not add_logo
            # logging.info("Swap video multiple triggered, logo: enabled")
        else:
            pass
            # logging.info("Swap video multiple triggered, logo: disabled")

        output_path = run_video_multiple(img, opt)
        # logger.info(f"Swap video multiple completed, result saved to {output_path}")
        return FileResponse(output_path, media_type="video/mp4", filename='output.mp4')

    except Exception as e:
        # logger.error("Swap video multiple failed: %s", str(e))
        raise HTTPException(status_code=500, detail=str(e))

import os

def download_youtube_video(video_url, output_directory='youtube'):
    try:
        yt = YouTube(video_url)
        video = yt.streams.filter(file_extension='mp4', progressive=True).first()
        downloaded_video_path = output_directory
        video.download(output_path=output_directory)
        del yt
        
        # Retrieve the first file in the output_directory (assuming it's the downloaded video)
        video_files = [f for f in os.listdir(output_directory) if f.endswith('.mp4')]
        if video_files:
            return os.path.join(output_directory, video_files[0])
        else:
            raise HTTPException(status_code=400, detail="No video file found after downloading")

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to download YouTube video: {str(e)}")

    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
