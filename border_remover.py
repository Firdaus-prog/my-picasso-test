import cv2
from cv2 import dnn_superres
from PIL import Image, ImageChops
import os
import logging


def remove_border(image_input, image_output):
    # Input validation
    if not os.path.exists(image_input):
        logging.error(f"Input image file '{image_input}' not found.")

    # Read the image
    img = cv2.imread(image_input)

    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply GaussianBlur to reduce noise and improve contour detection
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Adaptive thresholding to handle varying lighting conditions
    _, thresholded = cv2.threshold(
        blurred, 10, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # Find contours in the thresholded image
    contours, _ = cv2.findContours(
        thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if contours:
        # Find the bounding box of the non-black region
        x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
        # if y < bestY and ((bestY - y) < 100 or bestY == 9999):
        bestY = y
        # if x < bestX:
        bestX = x
        # if ((y + h) > bestHeight) and (((y + h) - bestHeight) < 10 or bestHeight == 0):
        bestHeight = y + h
        # if (x + w) > bestWidth:
        bestWidth = x + w
        # print("-----------------before")
        # print("bestY ", bestY)
        # print("bestHeight ", bestHeight)
        # Corner case: width
        if bestX > 300:
            bestX = 0
        if bestWidth < 980:
            bestWidth = 1280
        # Corner case: height
        if bestY > 100:
            bestY = 100
        if bestHeight < 540:
            bestHeight = 540

        # Corner case: balancing
        Xbalance = min(bestX, 1280 - bestWidth)
        bestX = Xbalance
        bestWidth = 1280 - Xbalance

        # print("-----------------")
        # print(image_input)
        # print("bestY ", bestY)
        # print("bestHeight ", bestHeight)
        # print("bestX ", bestX)
        # print("bestWidth ", bestWidth)

        # Crop the image using the bounding box
        cropped_img = img[(bestY + 2) : (bestHeight - 2), bestX:bestWidth]

        # Save the cropped image
        cv2.imwrite(image_output, cropped_img)
        print(f"Border removed, and the result is saved to {image_output}")
    else:
        print("No valid contour found, the output image will be empty")


def resize_image(input_path, output_path, width=1280, height=720):
    try:
        image = Image.open(input_path)
        original_width, original_height = image.size

        # Calculate aspect ratios
        aspect_ratio_width = width / original_width
        aspect_ratio_height = height / original_height

        # Choose the smallest aspect ratio to ensure the entire image fits within the target dimensions
        max_aspect_ratio = max(aspect_ratio_width, aspect_ratio_height)

        # Calculate the new dimensions
        new_width = int(original_width * max_aspect_ratio)
        new_height = int(original_height * max_aspect_ratio)

        # Resize the image to the new dimensions
        image = image.resize((new_width, new_height), Image.LANCZOS)

        # Calculate the positioning to center the image
        left = (width - new_width) // 2
        top = 0

        # Create a new image with the target dimensions and paste the resized image onto it
        new_image = Image.new("RGB", (width, height))
        new_image.paste(image, (left, top))

        # Save the resized image without black borders
        new_image.save(output_path)

        print(
            f"Image resized to {width}x{height} without black borders and saved to {output_path}"
        )
    except Exception as e:
        print(f"Error resizing the image: {str(e)}")


def enhance_image_quality(input_path, output_path):
    logging.basicConfig(filename="enhance_image.log", level=logging.DEBUG)
    try:
        # Load the image
        img = cv2.imread(input_path)

        # Create an SR object
        sr = dnn_superres.DnnSuperResImpl.create()

        # Read the desired model
        script_dir = os.path.dirname(os.path.realpath(__file__))
        model_path = os.path.join(script_dir, "ESPCN_x3.pb")
        logging.debug(f"Model path: {model_path}")

        if not os.path.isfile(model_path):
            logging.error(f"Model file not found at: {model_path}")
            raise Exception("Model file not found")

        sr.readModel(model_path)

        # Set the desired model and scale to get correct pre- and post-processing
        sr.setModel("espcn", 3)

        # Upscale the image
        enhanced_img = sr.upsample(img)

        # Save the enhanced image
        cv2.imwrite(output_path, enhanced_img)

        resize_image(output_path, output_path, 1280, 720)

        logging.info(f"Image enhanced and saved to {output_path}")
    except Exception as e:
        logging.error(f"Error enhancing the image: {str(e)}")
