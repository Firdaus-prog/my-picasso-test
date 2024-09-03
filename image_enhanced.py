import cv2
from dimension_detection import recover_logo, show
import numpy as np


def get_image_from_file(file_path: str) -> None:
    # Load the image from the local file
    img = cv2.imread(file_path)

    # Convert the image to LAB color space
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_channel, a, b = cv2.split(lab)

    # Applying CLAHE to L-channel
    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(10, 10))
    cl = clahe.apply(l_channel)

    # Merge the CLAHE enhanced L-channel with the a and b channel
    limg = cv2.merge((cl, a, b))

    # Convert image from LAB Color model to BGR color space
    enhanced_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    # Process the enhanced image to recover logo dimensions
    # dimensionType, result, startCoor = recover_logo(enhanced_img)

    # Display the final result using cv2.imshow
    cv2.imshow("Final Result", enhanced_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Example usage:
get_image_from_file(
    "./image/KLNAC_thumb009.jpg"
)  # Replace "image.jpg" with the path to your local image file
