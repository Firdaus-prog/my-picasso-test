import cv2
import numpy as np
import tempfile


def get_border(average_frame, main_width=1280, main_height=720):
    # average_frame = cv2.GaussianBlur(average_frame, (3, 3), 0)

    # edges = cv2.Canny(average_frame, 100, 255)
    edges = auto_canny(average_frame)
    white_pixel_coordinates = detect_white_pixel_coordinates(edges)

    center_x, center_y = main_width // 2, main_height // 2
    (
        line1_y,
        line2_y,
        first_limitY,
        second_limitY,
        first_centreLimitX,
        second_centreLimitX,
    ) = calculate_horizontal_lines(
        white_pixel_coordinates=white_pixel_coordinates,
        center_x=center_x,
        center_y=center_y,
        main_width=main_width,
        main_height=main_height,
    )
    (
        line1_x,
        line2_x,
        first_limitX,
        second_limitX,
        first_centreLimitY,
        second_centreLimitY,
    ) = calculate_vertical_lines(
        white_pixel_coordinates=white_pixel_coordinates,
        center_x=center_x,
        center_y=center_y,
        main_width=main_width,
        main_height=main_height,
    )

    return (
        line1_x,
        line2_x,
        line1_y,
        line2_y,
        average_frame,
        white_pixel_coordinates,
        first_limitX,
        second_limitX,
        first_limitY,
        second_limitY,
        first_centreLimitY,
        second_centreLimitY,
        first_centreLimitX,
        second_centreLimitX,
    )


def auto_canny(image, sigma=0.5):
    # compute the median of the single channel pixel intensities
    v = np.median(image)
    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)
    # return the edged image
    return edged


def detect_white_pixel_coordinates(canny_edges):
    # Find coordinates with white pixels in Canny edges image
    white_pixel_coordinates = np.column_stack(np.where(canny_edges > 0))
    return white_pixel_coordinates


def calculate_horizontal_lines(
    white_pixel_coordinates,
    center_x,
    center_y,
    first_limit=0,
    second_limit=999999,
    main_width=1280,
    main_height=720,
):
    topLine = [0, 0]
    bestTop = 9999
    botLine = [0, 0]
    bestBot = 9999

    # Limit for noise that is close to border top/bot/left/right (5%)
    limitY = main_height * 0.05
    first_limitY = limitY
    second_limitY = main_height - limitY

    limitX = main_width * 0.05
    first_limitX = limitX
    second_limitX = main_width - limitX

    # Limit for noise near middle of border (7.5%)
    limitZ = main_width * 0.2
    first_centreLimit = center_x - limitZ
    second_centreLimit = center_x + limitZ

    for i in white_pixel_coordinates:

        if i[1] <= second_centreLimit and i[1] >= first_centreLimit:
            continue
        if (i[0] <= first_limitY and i[0] >= 10) or (i[0] >= second_limitY and i[0] <= main_height-10):
            continue
        if i[1] <= first_limitX or i[1] >= second_limitX:
            continue
        if i[1] <= first_limit or i[1] >= second_limit:
            continue

        if i[0] < center_y:  # Top coordinate
            if i[0] > (main_height * 203 // 720):
                continue
            if (center_y - i[0]) < bestTop:
                bestTop = center_y - i[0]
                topLine = i
            else:
                continue

        if i[0] > center_y:  # Bot coordinate
            if i[0] < (main_height * 517 // 720):
                continue
            if (i[0] - center_y) < bestBot:
                bestBot = i[0] - center_y
                botLine = i
            else:
                continue

    # add +- for buffer
    if first_limit == 0 and (
        topLine[0] < (main_height * 325 // 720)
        or botLine[0] > (main_height * 375 // 720)
    ):
        topLine[0] *= 1.02
        botLine[0] *= 0.98

    # Check if bottom coordinate is found, if not, set it to max values
    if botLine[0] == 0:
        botLine[0] = main_height  # Set to max values

    # NOTE: Coordinate detected -> topLine, botLine
    # print("first_limit", first_limit)
    # print("second_limit", second_limit)
    # print("Top Line", topLine)
    # print("first_limitY", first_limitY)

    return (
        int(topLine[0]),
        int(botLine[0]),
        round(first_limitY),
        round(second_limitY),
        round(first_centreLimit),
        round(second_centreLimit),
    )


def calculate_vertical_lines(
    white_pixel_coordinates,
    center_x,
    center_y,
    first_limit=0,
    second_limit=999999,
    main_width=1280,
    main_height=720,
):
    leftLine = [0, 0]
    bestLeft = 9999
    rightLine = [0, 0]
    bestRight = 9999

    # Limit for noise that is close to border top/bot/left/right (5%)
    limitY = main_height * 0.05
    first_limitY = limitY
    second_limitY = main_height - limitY

    limitX = main_width * 0.05
    first_limitX = limitX
    second_limitX = main_width - limitX

    # Special limit for bottom. Mostly related to banner or substitle (20%)
    # special_limitY = main_height * 0.80

    # Limit for noise near middle of border (7.5%)
    limitZ = main_height * 0.2
    first_centreLimit = center_y - limitZ
    second_centreLimit = center_y + limitZ

    for i in white_pixel_coordinates:
        if i[0] <= second_centreLimit and i[0] >= first_centreLimit:
            continue
        if (i[0] <= first_limitY and i[0] >= 10) or (i[0] >= second_limitY and i[0] <= main_height-10):
            continue
        if i[1] <= first_limitX or i[1] >= second_limitX:
            continue
        if i[0] <= first_limit or i[0] >= second_limit:
            continue

        # if i[0] >= special_limitY:
        #     continue

        if i[1] < center_x:  # Left coordinate
            if i[1] > (main_width * 315 // 1280):
                continue
            if (center_x - i[1]) < bestLeft:
                bestLeft = center_x - i[1]
                leftLine = i
            else:
                continue

        if i[1] > center_x:  # Right coordinate
            if i[1] < (main_width * 965 // 1280):
                continue
            if (i[1] - center_x) < bestRight:
                bestRight = i[1] - center_x
                rightLine = i
            else:
                continue

    # add +- for buffer
    if first_limit == 0 and (
        leftLine[1] < (main_width * 610 // 1280)
        or rightLine[1] > (main_width * 665 // 1280)
    ):
        leftLine[1] *= 1.02
        rightLine[1] *= 0.98

    # Check if bottom coordinate is found, if not, set it to max values
    if rightLine[1] == 0:
        rightLine[1] = main_width  # Set to max values

    # NOTE: Coordinate detected -> leftLine, rightLine
    # print("firs_limit", first_limit)
    # print("second_limit", second_limit)
    # print("Right Line", rightLine)
    # print("second_limitY", second_limitY)

    return (
        int(leftLine[1]),
        int(rightLine[1]),
        round(first_limitX),
        round(second_limitX),
        round(first_centreLimit),
        round(second_centreLimit),
    )


def recover_logo(
    video_path,
    limit_num_frames=5000,
    limit_skip_frames=2000,
    main_width=1280,
    main_height=720,
):
    cap = cv2.VideoCapture(video_path)

    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    skip_frames = round(length / 100 * 20)
    gap_frames = round(length / 100 * 30)
    num_frames = length - skip_frames - gap_frames
    if skip_frames > limit_skip_frames:
        skip_frames = limit_skip_frames
    if gap_frames > 5000:
        gap_frames = 5000
    if num_frames > limit_num_frames:
        num_frames = limit_num_frames

    # Skip the initial 2000 frames
    for _ in range(skip_frames):
        ret, _ = cap.read()  # Read and discard frames

    # Read and process the remaining frames
    ret, first_frame = cap.read()
    first_frame = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
    accumulator = np.zeros_like(first_frame, dtype=np.float32)

    frame_counter = 0
    while frame_counter < num_frames + gap_frames:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_counter % 11 == 0:  # Process every 11th frame
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            accumulator += frame_gray.astype(np.float32)

        frame_counter += 1

    average_frame = (accumulator / num_frames).astype(np.uint8)
    blurred_frame = cv2.GaussianBlur(average_frame, (3, 3), 0)

    (
        average_line1_x,
        average_line2_x,
        average_line1_y,
        average_line2_y,
        average_frame,
        average_white_pixel_coordinates,
        first_limitX,
        second_limitX,
        first_limitY,
        second_limitY,
        first_centreLimitY,
        second_centreLimitY,
        first_centreLimitX,
        second_centreLimitX,
    ) = get_border(average_frame, main_width=main_width, main_height=main_height)

    (
        blurred_line1_x,
        blurred_line2_x,
        blurred_line1_y,
        blurred_line2_y,
        blurred_frame,
        blurred_white_pixel_coordinates,
        first_limitX,
        second_limitX,
        first_limitY,
        second_limitY,
        first_centreLimitY,
        second_centreLimitY,
        first_centreLimitX,
        second_centreLimitX,
    ) = get_border(blurred_frame, main_width=main_width, main_height=main_height)

    areaAverage = (average_line2_x - average_line1_x) * (
        average_line2_y - average_line1_y
    )
    areaBlurred = (blurred_line2_x - blurred_line1_x) * (
        blurred_line2_y - blurred_line1_y
    )

    # print(f"areaAverage: {areaAverage}, areaBlurred: {areaBlurred}")
    control = False
    if areaBlurred > ((main_width * main_height) * 0.95):
        control = True
    # print("Control value ", control)

    # Result: getting area with smallest but bigger than half original frame
    if control:
        if areaAverage < areaBlurred and ((areaAverage / areaBlurred) > 0.7):
            line1_x, line2_x, line1_y, line2_y, final_frame, white_pixel_coordinates = (
                average_line1_x,
                average_line2_x,
                average_line1_y,
                average_line2_y,
                average_frame,
                average_white_pixel_coordinates,
            )
            # print("USING AVERAGE FRAME")
        else:
            line1_x, line2_x, line1_y, line2_y, final_frame, white_pixel_coordinates = (
                blurred_line1_x,
                blurred_line2_x,
                blurred_line1_y,
                blurred_line2_y,
                blurred_frame,
                blurred_white_pixel_coordinates,
            )
            # print("USING BLURRED FRAME")
    else:
        if areaAverage > areaBlurred:
            line1_x, line2_x, line1_y, line2_y, final_frame, white_pixel_coordinates = (
                average_line1_x,
                average_line2_x,
                average_line1_y,
                average_line2_y,
                average_frame,
                average_white_pixel_coordinates,
            )
            # print("USING AVERAGE FRAME")
        else:
            line1_x, line2_x, line1_y, line2_y, final_frame, white_pixel_coordinates = (
                blurred_line1_x,
                blurred_line2_x,
                blurred_line1_y,
                blurred_line2_y,
                blurred_frame,
                blurred_white_pixel_coordinates,
            )
            # print("USING BLURRED FRAME")

    # Extra Step
    horizontal = line1_y + (main_height - line2_y)
    horizontalRatio = (main_height - horizontal) / main_height
    vertical = line1_x + (main_width - line2_x)
    verticalRatio = (main_width - vertical) / main_width

    horizontalArea = (line1_y * main_width) + ((main_height - line2_y) * main_width)
    verticalArea = (line1_x * main_height) + ((main_width - line2_x) * main_height)
    # print("horizontalArea", horizontalArea)
    # print("verticalArea", verticalArea)

    # print(f"Horizontal Area: {horizontalArea}, Vertical Area: {verticalArea}")
    center_x, center_y = main_width // 2, main_height // 2
    if horizontalArea < verticalArea and ((horizontalArea / verticalArea) < 0.8):
        # print("Recalculate LEFT RIGHT")
        (
            line1_x,
            line2_x,
            first_limitX,
            second_limitX,
            first_centreLimitY,
            second_centreLimitY,
        ) = calculate_vertical_lines(
            white_pixel_coordinates=white_pixel_coordinates,
            center_x=center_x,
            center_y=center_y,
            first_limit=line1_y,
            second_limit=line2_y,
            main_width=main_width,
            main_height=main_height,
        )
    else:
        # print("Recalculate TOP BOT")
        (
            line1_y,
            line2_y,
            first_limitY,
            second_limitY,
            first_centreLimitX,
            second_centreLimitX,
        ) = calculate_horizontal_lines(
            white_pixel_coordinates=white_pixel_coordinates,
            center_x=center_x,
            center_y=center_y,
            first_limit=line1_x,
            second_limit=line2_x,
            main_width=main_width,
            main_height=main_height,
        )

    # Recalculate
    # print(f"line1_x: {line1_x}")
    # print(f"line2_x: {line2_x}")
    horizontal = line2_y - line1_y
    vertical = line2_x - line1_x
    startCoor = [line1_x, line1_y]

    # Exception for area below 40%
    tempArea = horizontal * vertical
    tempFullArea = main_width * main_height
    if tempArea / tempFullArea < 0.4:
        horizontal = main_height * 550 // 720
        vertical = main_width * 810 // 1280
        startCoor = [(main_width * 235 // 1280), 0]

        # For visual perposes
        line1_y = startCoor[1]
        line2_y = horizontal + startCoor[1]
        line1_x = startCoor[0]
        line2_x = vertical + startCoor[0]

    # return [final_frame, white_pixel_coordinates, line1_y, line2_y, line1_x, line2_x]

    # ----------------------------------------------------------------------

    final_frame_bgr = cv2.cvtColor(final_frame, cv2.COLOR_GRAY2BGR)

    # Logo detected (Red)
    for x, y in white_pixel_coordinates:
        cv2.circle(final_frame_bgr, (y, x), 2, (0, 0, 255), -1)

    # Limit line (Orange)
    # Create an orangeOverlay image to draw the translucent rectangle
    orangeOverlay = final_frame_bgr.copy()

    # Left & Right limit
    cv2.rectangle(
        orangeOverlay, (0, 10), (first_limitX, main_height-10), (0, 165, 255), -1
    )
    cv2.rectangle(
        orangeOverlay, (second_limitX, 10), (main_width, main_height-10), (0, 165, 255), -1
    )

    # Top & Bottom limit
    cv2.rectangle(orangeOverlay, (0, 10), (main_width, first_limitY), (0, 165, 255), -1)
    cv2.rectangle(
        orangeOverlay, (0, second_limitY), (main_width, main_height-10), (0, 165, 255), -1
    )

    # Centre limit
    cv2.rectangle(
        orangeOverlay,
        (first_centreLimitX, first_centreLimitY),
        (second_centreLimitX, second_centreLimitY),
        (0, 165, 255),
        -1,
    )

    # Blend the orangeOverlay with the original image
    cv2.addWeighted(orangeOverlay, 0.5, final_frame_bgr, 1 - 0.5, 0, final_frame_bgr)

    # Accept lines (Green)
    greenOverlay = final_frame_bgr.copy()
    cv2.line(final_frame_bgr, (0, line1_y), (main_width, line1_y), (0, 255, 0), 2)
    cv2.line(final_frame_bgr, (0, line2_y), (main_width, line2_y), (0, 255, 0), 2)

    cv2.line(final_frame_bgr, (line1_x, 0), (line1_x, main_height), (0, 255, 0), 2)
    cv2.line(final_frame_bgr, (line2_x, 0), (line2_x, main_height), (0, 255, 0), 2)

    cv2.rectangle(greenOverlay, (line1_x, line1_y), (line2_x, line2_y), (0, 255, 0), -1)
    cv2.addWeighted(greenOverlay, 0.2, final_frame_bgr, 1 - 0.5, 0, final_frame_bgr)

    # Save the final frame to a temporary file
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    cv2.imwrite(temp_file.name, final_frame_bgr)

    cv2.imshow("Average Frame with Marks", final_frame_bgr)  # Update the display

    # Wait for a key press to close the windows
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print(temp_file.name)

    # ----------------------------------------------------------------------
    print(f"horizontal: {horizontal}")
    print(f"vertical: {vertical}")
    print(f"startCoor: {startCoor}")

    return horizontal, vertical, startCoor, temp_file.name
