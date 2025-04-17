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
    Measures the maximum internal width and length of green beans in an image using 
    a known-size Post-it for calibration. Returns an annotated image and 
    a list of (bean_id, dimension_type, measurement_mm).
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

    # 6. For each bean contour, measure its maximum internal width and length
    results = []
    bean_id = 0

    for cnt in bean_contours:
        bean_id += 1

        # Convert contour to float32 for perspectiveTransform and reshape correctly
        cnt_float = cnt.astype(np.float32)
        # Reshape to have the right format for perspectiveTransform
        cnt_reshaped = cnt_float.reshape(-1, 1, 2)
        # Warp contour to "top-down" metric space
        cnt_warped = cv2.perspectiveTransform(cnt_reshaped, M)
        # Reshape to a simpler format for further processing
        cnt_warped = cnt_warped.reshape(-1, 2)
        
        # Initialize variables for maximum width calculation
        max_width_mm = 0.0
        best_p1_warped = None
        best_p2_warped = None
        num_points = len(cnt_warped)
        
        # Handle edge case with too few points
        if num_points < 4:
            print(f"Skipping bean {bean_id} - too few contour points ({num_points})")
            continue
            
        # Iterate through all unique point pairs to find maximum width
        for i in range(num_points):
            p1_warped = cnt_warped[i]
            for j in range(i + 1, num_points):
                p2_warped = cnt_warped[j]
                
                # Calculate midpoint
                midpoint_warped = (p1_warped + p2_warped) / 2.0
                
                # Check if midpoint is inside the contour
                is_inside = cv2.pointPolygonTest(cnt_warped, tuple(midpoint_warped.flatten()), measureDist=False) >= 0
                
                if is_inside:
                    distance = np.linalg.norm(p1_warped - p2_warped)
                    if distance > max_width_mm:
                        max_width_mm = distance
                        best_p1_warped = p1_warped
                        best_p2_warped = p2_warped
        
        # Check if a valid width was found
        if best_p1_warped is not None and best_p2_warped is not None:
            # Calculate width direction vector 
            width_vector = best_p2_warped - best_p1_warped
            width_direction = width_vector / np.linalg.norm(width_vector)  # Normalize
            
            # Find perpendicular length
            max_length_mm = 0.0
            best_length_p1 = None
            best_length_p2 = None
            
            # Second pass through points to find length
            for i in range(num_points):
                p1 = cnt_warped[i]
                for j in range(i + 1, num_points):
                    p2 = cnt_warped[j]
                    
                    # Calculate candidate direction vector
                    candidate_vector = p2 - p1
                    candidate_length = np.linalg.norm(candidate_vector)
                    if candidate_length == 0:
                        continue
                        
                    candidate_direction = candidate_vector / candidate_length
                    
                    # Check if midpoint is inside
                    midpoint = (p1 + p2) / 2.0
                    is_inside = cv2.pointPolygonTest(cnt_warped, tuple(midpoint.flatten()), False) >= 0
                    
                    # Calculate alignment with width (dot product of unit vectors)
                    alignment = abs(np.dot(width_direction, candidate_direction))
                    
                    # If points are reasonably perpendicular to width (alignment close to 0)
                    # and midpoint is inside and length is greater than current max
                    if is_inside and alignment < 0.3 and candidate_length > max_length_mm:
                        max_length_mm = candidate_length
                        best_length_p1 = p1
                        best_length_p2 = p2
            
            # Transform width points back to original image space
            p1_to_transform = np.array([[best_p1_warped]], dtype=np.float32)
            p2_to_transform = np.array([[best_p2_warped]], dtype=np.float32)
            
            p1_original = cv2.perspectiveTransform(p1_to_transform, M_inv)
            p2_original = cv2.perspectiveTransform(p2_to_transform, M_inv)
            
            p1_orig = tuple(np.intp(p1_original[0][0]))
            p2_orig = tuple(np.intp(p2_original[0][0]))
            
            # Draw the max width line in green
            cv2.line(orig, p1_orig, p2_orig, (0, 255, 0), 2)  # BGR Green
            
            # Calculate text position for width
            width_text_pos = tuple(np.intp((np.array(p1_orig) + np.array(p2_orig)) / 2))
            
            # Annotate with measured width
            width_label = f"{max_width_mm:.1f} mm"
            cv2.putText(orig, width_label, width_text_pos,
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(orig, width_label, width_text_pos,
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            
            # Store width result
            results.append((bean_id, "width", max_width_mm))
            
            if debug:
                print(f"[DEBUG] Bean {bean_id}: Width={max_width_mm:.2f} mm")
            
            # Draw the length line if found
            if best_length_p1 is not None and best_length_p2 is not None:
                # Transform length points back to original image space
                len_p1_transform = np.array([[best_length_p1]], dtype=np.float32)
                len_p2_transform = np.array([[best_length_p2]], dtype=np.float32)
                
                len_p1_original = cv2.perspectiveTransform(len_p1_transform, M_inv)
                len_p2_original = cv2.perspectiveTransform(len_p2_transform, M_inv)
                
                len_p1_orig = tuple(np.intp(len_p1_original[0][0]))
                len_p2_orig = tuple(np.intp(len_p2_original[0][0]))
                
                # Draw the max length line in blue
                cv2.line(orig, len_p1_orig, len_p2_orig, (255, 0, 0), 2)  # BGR: Blue
                
                # Add length measurement text
                length_text_pos = tuple(np.intp((np.array(len_p1_orig) + np.array(len_p2_orig)) / 2))
                length_label = f"{max_length_mm:.1f} mm"
                cv2.putText(orig, length_label, length_text_pos,
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(orig, length_label, length_text_pos,
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                
                # Add length to results
                results.append((bean_id, "length", max_length_mm))
                
                if debug:
                    print(f"[DEBUG] Bean {bean_id}: Length={max_length_mm:.2f} mm")
            else:
                print(f"Warning: Could not determine valid length for bean {bean_id}")
        else:
            print(f"Warning: Could not determine valid width for bean {bean_id}")

    # 7. Save the result image
    out_name = "output_width_length.jpg"
    cv2.imwrite(out_name, orig)
    if debug:
        print(f"[INFO] Processed image saved as: {out_name}")
        print("[INFO] Bean measurements (mm):", results)

    return orig, results


if __name__ == "__main__":
    # Example usage 
    input_image = "C:\\beans\\beans_example.jpg"  # <-- Replace with your image path
    annotated_img, bean_info = measure_beans(
        image_path=input_image,
        postit_side_mm=76.0,      # Post-it note side in mm
        postit_lower=(35, 0, 180),   # H=35, S=0,   V=180
        postit_upper=(55, 80, 255),  # H=55, S=80,  V=255
        bean_lower=(20, 100, 100),   # Example HSV lower for green beans
        bean_upper=(70, 255, 255),   # Example HSV upper for green beans
        debug=True
    )
    print("Finished measuring beans!")
    print("Bean info:", bean_info) 