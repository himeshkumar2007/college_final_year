import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import cv2
import traceback

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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"status": "ok", "message": "Foot Measurement API running"}


def process_image(oimg: np.ndarray) -> float:
    preprocessedOimg = preprocess(oimg)
    clusteredImg = kMeans_cluster(preprocessedOimg)
    edgedImg = edgeDetection(clusteredImg)

    boundRect, cnts, contours_poly, img = getBoundingBox(edgedImg)
    if not boundRect:
        raise ValueError("No bounding box detected in first stage")

    _ = drawCnt(boundRect[0], cnts, contours_poly, img)

    croppedImg, pcropedImg = cropOrig(boundRect[0], clusteredImg)
    newImg = overlayImage(croppedImg, pcropedImg)

    fedged = edgeDetection(newImg)
    fboundRect, fcnt, fcntpoly, fimg = getBoundingBox(fedged)

    if not fboundRect:
        raise ValueError("No bounding box detected in final stage")

    idx = 2 if len(fboundRect) > 2 else len(fboundRect) - 1
    if idx < 0:
        raise ValueError("Bounding box detection failed")

    _ = drawCnt(fboundRect[idx], fcnt, fcntpoly, fimg)

    feet_size_mm = calcFeetSize(pcropedImg, fboundRect)
    return feet_size_mm / 10.0


@app.post("/measure-foot")
async def measure_foot(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload a valid image file.")

    contents = await file.read()
    np_arr = np.frombuffer(contents, np.uint8)
    oimg = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if oimg is None:
        raise HTTPException(status_code=400, detail="Could not read image data.")

    try:
        feet_size_cm = process_image(oimg)
        return {
            "feet_size_cm": feet_size_cm,
            "message": "Foot size calculated successfully.",
        }
    except ValueError as e:
        print("⚠️ VALIDATION ERROR:", str(e))
        raise HTTPException(status_code=422, detail=str(e))
    except Exception:
        print("🔥 ERROR IN process_image()")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail="Internal processing error")
