from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request

from ultralytics import YOLO
import cv2

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")

model = YOLO("best.pt")

camera = cv2.VideoCapture(0)

def generate_frames():

    while True:

        success, frame = camera.read()

        if not success:
            break

        results = model(frame)

        annotated_frame = results[0].plot()

        _, buffer = cv2.imencode(".jpg", annotated_frame)

        frame_bytes = buffer.tobytes()

        yield (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n'
            + frame_bytes +
            b'\r\n'
        )

@app.get("/")
async def home(request: Request):

    return templates.TemplateResponse(
        "index.html",
        {"request": request}
    )

@app.get("/video")
def video():

    return StreamingResponse(
        generate_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )