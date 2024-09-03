import cv2
import ffmpeg
import os

# from border_remover import remove_border
import random


def extract_key_frames(
    src_video_temp_path: str,
    output_images_temp_path: str,
    crop_width: int = 1280,
    crop_height: int = 720,
    start_x: int = 0,
    start_y: int = 0,
):
    print(
        f"Extracting key frames from {src_video_temp_path} to {output_images_temp_path}"
    )

    # Set the crop filter
    crop_filter = f"crop={crop_width}:{crop_height}:{start_x}:{start_y}"

    # Extract frames without scaling
    ffmpeg.input(src_video_temp_path, skip_frame="nokey", vsync=0).filter_(
        "crop", crop_width, crop_height, start_x, start_y
    ).output(
        f"{output_images_temp_path}/%d.png",
        qscale=2,
    ).run()

    print("Completed extracting images")


def upscale_image(image_path: str, dest_image_path: str, scale: float):
    print(f"Attempting to upscale image {image_path} by scale {scale}.")
    img = cv2.imread(image_path)
    result = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    cv2.imwrite(dest_image_path, result)
