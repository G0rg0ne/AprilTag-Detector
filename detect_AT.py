import click
import numpy as np
import apriltag
import cv2

@click.command()
@click.option('--input_path', prompt='Input image path', help='Path to the input image.')
@click.option('--output_path', prompt='Output image path', help='Path to save the output image with detected AprilTags.')
def detect_apriltags(input_path, output_path):
    # Load the image
    image = cv2.imread(input_path)
    if image is None:
        print(f"Error: Unable to load image at {input_path}")
        return
    scale_percent = 40  # percent of original size
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized_image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    # Convert the image to grayscale
    gray = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # Initialize the AprilTag detector
    options = apriltag.DetectorOptions(families="tag16h5")
    detector = apriltag.Detector(blurred)

    # Detect AprilTags
    results = detector.detect(gray)
    # Draw bounding boxes around detected AprilTags
    for r in results:
        (ptA, ptB, ptC, ptD) = r.corners
        ptA = (int(ptA[0]), int(ptA[1]))
        ptB = (int(ptB[0]), int(ptB[1]))
        ptC = (int(ptC[0]), int(ptC[1]))
        ptD = (int(ptD[0]), int(ptD[1]))

        # Draw the bounding box
        cv2.line(image, ptA, ptB, (0, 255, 0), 2)
        cv2.line(image, ptB, ptC, (0, 255, 0), 2)
        cv2.line(image, ptC, ptD, (0, 255, 0), 2)
        cv2.line(image, ptD, ptA, (0, 255, 0), 2)

        # Draw the center (optional)
        cX, cY = (int(r.center[0]), int(r.center[1]))
        cv2.circle(image, (cX, cY), 5, (0, 0, 255), -1)

    # Save the output image
    cv2.imwrite(output_path, image)
    print(f"Output image saved to {output_path}")

if __name__ == '__main__':
    detect_apriltags()