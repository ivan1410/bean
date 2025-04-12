import cv2
import numpy as np

def order_corners(pts):
    """
    Sorts 4 corner points in the order:
    top-left, top-right, bottom-right, bottom-left.
    This is needed before computing the perspective transform.
    """
    # Sort by sum of coordinates
    pts_sorted = sorted(pts, key=lambda p: p[0] + p[1])
    tl, br = pts_sorted[0], pts_sorted[3]
    tr, bl = pts_sorted[1], pts_sorted[2]
    # Ensure tr is top-right by comparing y-values
    if tr[1] > bl[1]:
        tr, bl = bl, tr
    return np.array([tl, tr, br, bl], dtype=np.float32)

def measure_beans(image_path,
                  postit_side_mm=76.0,
                  # HSV range for the Post-it color (example: yellowish)
                  postit_lower=(20, 80, 80),
                  postit_upper=(40, 255, 255),
                  # HSV range for green beans (example range, adjust as needed)
                  bean_lower=(35, 50, 50),
                  bean_upper=(90, 255, 255),
                  debug=False):
    """
    Measures the width of green beans in an image using a known-size Post-it
    for calibration. Returns an annotated image and a list of (bean_id, width_mm).
    """

    # 1. Load Image
    image = cv2.imread(image_path)
    if image is None:
        raise IOError(f"Could not load image: {image_path}")

    orig = image.copy()  # Keep an original copy for final annotation

    # 2. Convert to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 3. Detect the Post-it note for calibration
    #    We assume it's a brightly-colored square (e.g., yellow).
    mask_postit = cv2.inRange(hsv, postit_lower, postit_upper)

    # Optional: morphological clean-up if needed
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    mask_postit = cv2.morphologyEx(mask_postit, cv2.MORPH_CLOSE, kernel)

    # Find largest (or first valid) quadrilateral that should be the Post-it
    contours_postit, _ = cv2.findContours(mask_postit, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    square_contour = None
    max_area = 0
    for cnt in contours_postit:
        area = cv2.contourArea(cnt)
        if area < 100:  # skip very small
            continue
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) == 4 and area > max_area:
            max_area = area
            square_contour = approx

    if square_contour is None:
        raise Exception("Could not find a valid Post-it note contour. Check HSV range or image setup.")

    # Sort the 4 corners
    square_contour = square_contour.reshape((4, 2))
    corners = order_corners(square_contour)

    # 4. Compute perspective transform so the Post-it looks like a perfect square
    #    We'll warp it to a 'square' of size postit_side_mm x postit_side_mm in pixels.
    #    (So 1 pixel = 1 mm after warping.)
    dst_pts = np.array([
        [0, 0],
        [postit_side_mm, 0],
        [postit_side_mm, postit_side_mm],
        [0, postit_side_mm]
    ], dtype=np.float32)

    # Perspective transform matrix
    M = cv2.getPerspectiveTransform(corners, dst_pts)
    # Also compute inverse transform for drawing text back onto the original image
    M_inv = np.linalg.inv(M)

    # 5. Detect beans using a green color range in HSV
    mask_beans = cv2.inRange(hsv, bean_lower, bean_upper)
    # Remove the Post-it area from the bean mask (just in case it's partially greenish)
    mask_beans = cv2.bitwise_and(mask_beans, cv2.bitwise_not(mask_postit))

    # Morphological cleaning
    mask_beans = cv2.morphologyEx(mask_beans, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8))
    mask_beans = cv2.morphologyEx(mask_beans, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))

    # Find contours of beans
    bean_contours, _ = cv2.findContours(mask_beans, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter out very small noise
    bean_contours = [cnt for cnt in bean_contours if cv2.contourArea(cnt) > 50]

    # 6. For each bean contour, measure its width after perspective transform
    results = []
    bean_id = 0

    for cnt in bean_contours:
        bean_id += 1

        # Convert contour to float32 for perspectiveTransform
        cnt_float = cnt.astype(np.float32)
        # Warp contour to "top-down" metric space
        cnt_warped = cv2.perspectiveTransform(cnt_float, M)

        # Use minAreaRect in the warped plane
        rect = cv2.minAreaRect(cnt_warped)
        (center_warped), (w, h), angle = rect
        # The smaller dimension is the bean's width in mm (since in warped space 1 px = 1 mm)
        bean_width_mm = min(w, h)

        # We want to place text in the original image, so transform the center back
        center_warped_pt = np.array([[center_warped]], dtype=np.float32)  # shape (1,1,2)
        center_original = cv2.perspectiveTransform(center_warped_pt, M_inv)
        cx, cy = center_original[0][0]

        # Draw the contour on the original image for visualization
        cv2.drawContours(orig, [cnt], -1, (0, 255, 0), 2)

        # Annotate with bean ID and measured width
        text_label = f"Bean {bean_id}: {bean_width_mm:.1f} mm"
        # Draw text with a black outline first
        cv2.putText(orig, text_label, (int(cx), int(cy)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
        # Then draw white text on top for clarity
        cv2.putText(orig, text_label, (int(cx), int(cy)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

        results.append((bean_id, bean_width_mm))

        if debug:
            print(f"[DEBUG] Bean {bean_id}: width={bean_width_mm:.2f} mm")

    # 7. Save the result image
    out_name = "output_annotated.jpg"
    cv2.imwrite(out_name, orig)
    if debug:
        print(f"[INFO] Processed image saved as: {out_name}")
        print("[INFO] Bean widths (mm):", results)

    return orig, results


if __name__ == "__main__":
    # Example usage 
    input_image = "C:\\beans\\beans_example.jpg"# <-- Replace with your image path
    annotated_img, bean_info = measure_beans(
        image_path=input_image,
        postit_side_mm=76.0,      # Post-it note side in mm
        postit_lower = (35,  0,  180),   # H=35, S=0,   V=180
        postit_upper = (55, 80, 255),   # H=55, S=80,  V=255
        bean_lower = (20, 100, 100),  # Example HSV lower for green beans
        bean_upper = (70, 255, 255),  # Example HSV upper for green beans
        debug=True
    )
    print("Finished measuring beans!")
    print("Bean info (bean_id, width_mm):", bean_info)