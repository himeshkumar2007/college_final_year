import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import numpy as np
import cv2

from utils import (
    preprocess,
    kMeans_cluster,
    edgeDetection,
    getBoundingBox,
    drawCnt,
    cropOrig,
    overlayImage,
    calcFeetSize,
)

app = FastAPI(title="Foot Measurement API")

# Ensure output directory exists
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def process_image(oimg: np.ndarray) -> float:
    """
    Ye function pura pipeline run karega
    aur final feet size (cm) return karega.
    """

    # 1) Preprocess
    preprocessedOimg = preprocess(oimg)
    cv2.imwrite(os.path.join(OUTPUT_DIR, 'preprocessedOimg.jpg'), preprocessedOimg)

    # 2) KMeans clustering
    clusteredImg = kMeans_cluster(preprocessedOimg)
    cv2.imwrite(os.path.join(OUTPUT_DIR, 'clusteredImg.jpg'), clusteredImg)

    # 3) Edge detection
    edgedImg = edgeDetection(clusteredImg)
    cv2.imwrite(os.path.join(OUTPUT_DIR, 'edgedImg.jpg'), edgedImg)

    # 4) First bounding box + draw
    boundRect, cnts, contours_poly, img = getBoundingBox(edgedImg)
    pdraw = drawCnt(boundRect[1], cnts, contours_poly, img)
    cv2.imwrite(os.path.join(OUTPUT_DIR, 'pdraw.jpg'), pdraw)

    # 5) Crop original
    croppedImg, pcropedImg = cropOrig(boundRect[1], clusteredImg)
    cv2.imwrite(os.path.join(OUTPUT_DIR, 'croppedImg.jpg'), croppedImg)

    # 6) Overlay image
    newImg = overlayImage(croppedImg, pcropedImg)
    cv2.imwrite(os.path.join(OUTPUT_DIR, 'newImg.jpg'), newImg)

    # 7) Second edge detection + bounding box
    fedged = edgeDetection(newImg)
    fboundRect, fcnt, fcntpoly, fimg = getBoundingBox(fedged)
    fdraw = drawCnt(fboundRect[2], fcnt, fcntpoly, fimg)
    cv2.imwrite(os.path.join(OUTPUT_DIR, 'fdraw.jpg'), fdraw)

    # 8) Calculate feet size
    feet_size_mm = calcFeetSize(pcropedImg, fboundRect)
    feet_size_cm = feet_size_mm / 10.0

    return feet_size_cm


@app.post("/measure-foot")
async def measure_foot(file: UploadFile = File(...)):
    """
    Image upload karo, ye endpoint feet size (cm) dega.
    """

    # Validate file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload a valid image file.")

    # Read bytes
    contents = await file.read()

    # Decode image using OpenCV
    np_arr = np.frombuffer(contents, np.uint8)
    oimg = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if oimg is None:
        # Image decode nahi ho paayi
        raise HTTPException(status_code=400, detail="Could not read image data.")

    try:
        feet_size_cm = process_image(oimg)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

    return JSONResponse(
        {
            "feet_size_cm": feet_size_cm,
            "message": "Foot size calculated successfully.",
        }
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
