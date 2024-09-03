import shutil
import tempfile

# from border_remover import enhance_image_quality
from dimension_detection import recover_logo
import boto3

# from celery_app import app
from typing import List

# from thumbnail_extraction import extract_key_frames
from os import listdir
from os.path import isfile, join
from thumbnail_extraction import extract_key_frames

# from border_remover import enhance_image_quality

import cv2
import random


def save_video_clips(filename, x, y, width, height):
    fileLocation = "./Input/Error5/" + filename
    outputLocation = "./Output/"

    # Open the video file
    vidcap = cv2.VideoCapture(fileLocation)

    # Get the total number of frames in the video
    total_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))

    print("Total frame ", total_frames)

    # Choose a random frame number
    random_frame_number = random.randint(0, total_frames - 1)

    # Set the frame position to the chosen random frame for the first clip (without crop)
    vidcap.set(cv2.CAP_PROP_POS_FRAMES, random_frame_number)

    # Read the selected frame for the first clip
    success, image = vidcap.read()
    outputNameNoCrop = outputLocation + "random_frame_without_crop.jpg"
    # If reading the frame was successful, save it without crop
    if success:
        cv2.imwrite(outputNameNoCrop, image)
        print("Random frame without crop saved successfully.")
    else:
        print("Error reading frame for the first clip.")

    # Reset video capture object to read the same frame for the second clip (with crop)
    vidcap.set(cv2.CAP_PROP_POS_FRAMES, random_frame_number)

    # Read the selected frame for the second clip
    success, image = vidcap.read()
    outputNameWithCrop = outputLocation + "random_frame_with_crop.jpg"
    # If reading the frame was successful, crop and save it
    if success:
        # Crop the image to desired dimensions
        cropped_image = image[y : y + height, x : x + width]

        # Save cropped frame as JPEG file
        cv2.imwrite(outputNameWithCrop, cropped_image)
        # remove_border(outputNameWithCrop, outputNameWithCrop)
        print("Random frame with crop saved successfully.")
    else:
        print("Error reading frame for the second clip.")

    # Release the video capture object
    vidcap.release()

    enhance_image_quality(outputNameWithCrop, outputNameWithCrop)


fileName = "KNZEG02HM01.mp4"
fileLocation = "./Input/HDError/" + fileName
outputLocation = "./Output/"
print("Starting the process")


INPUT_WIDTH = 1280
INPUT_HEIGHT = 720
CROP_WIDTH = 1376
CROP_HEIGHT = 774
# CROP_WIDTH = 1104
# CROP_HEIGHT = 621
CROP_X_OFFSET = 0
CROP_Y_OFFSET = -25

process_width = CROP_WIDTH
process_height = CROP_HEIGHT

start_x = ((INPUT_WIDTH - CROP_WIDTH) // 2) + CROP_X_OFFSET
start_y = ((INPUT_HEIGHT - CROP_HEIGHT) // 2) + CROP_Y_OFFSET


INPUT_WIDTH_HD = 1920
INPUT_HEIGHT_HD = 1080
CROP_WIDTH_HD = 1376  # WIDTH X 23/40
CROP_HEIGHT_HD = 774  # WIDTH X 23/40

# extract_key_frames(
#                     fileLocation,
#                     outputLocation,
#                     crop_width=process_width,
#                     crop_height=process_height,
#                     start_x=start_x,
#                     start_y=start_y,
#                     # main_width=INPUT_WIDTH,
#                     # main_height=INPUT_HEIGHT,
#                 )


num_frames = 5000
horizontal, vertical, startCoor, frame_path = recover_logo(
    fileLocation,
    limit_num_frames=5000,
    limit_skip_frames=2000,
    main_width=INPUT_WIDTH_HD,
    main_height=INPUT_HEIGHT_HD,
)

# save_video_clips(fileName, startCoor[0], startCoor[1], vertical, horizontal)

# onlyfiles = [f for f in listdir("./Input/") if isfile(join("./Input/", f))]

# x = []
# for i in onlyfiles:
#     # fileName = "KHIJM01HM01.mp4"
#     # fileName = i
#     # fileLocation = "./Input/" + fileName
#     # print(fileLocation)

#     num_frames = 5000
#     print("Starting the process")

#     # Add more logging
#     print(f"Recovering logo for {num_frames} frames")
#     type, prediction, startCoor = recover_logo(fileLocation, num_frames)
#     # break
#     x.append(recover_logo(fileLocation, num_frames))


# for j in x:
#     show(j[0], j[1], j[2], j[3], j[4], j[5])
