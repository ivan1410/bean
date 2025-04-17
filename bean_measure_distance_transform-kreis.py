import cv2
import numpy as np
import math # Needed for trig functions

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

def measure_beans_dt(image_path, # Renamed function slightly for clarity
                     postit_side_mm=76.0,
                     # HSV range for the Post-it color (example: yellowish)
                     postit_lower=(20, 80, 80),
                     postit_upper=(40, 255, 255),
                     # HSV range for green beans (example range, adjust as needed)
                     bean_lower=(35, 50, 50),
                     bean_upper=(90, 255, 255),
                     debug=False,
                     visualize=False):  # Added visualize parameter
    """
    Measures the maximum internal width of green beans in an image using
    Distance Transform and a known-size Post-it for calibration.
    Returns an annotated image and a list of (bean_id, width_mm).
    """

    # 1. Load Image
    image = cv2.imread(image_path)
    if image is None:
        raise IOError(f"Could not load image: {image_path}")

    orig = image.copy()  # Keep an original copy for final annotation

    # 2. Convert to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 3. Detect the Post-it note for calibration
    mask_postit = cv2.inRange(hsv, postit_lower, postit_upper)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    mask_postit = cv2.morphologyEx(mask_postit, cv2.MORPH_CLOSE, kernel)
    contours_postit, _ = cv2.findContours(mask_postit, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # VISUALIZATION: Show the Post-it detection
    if visualize:
        postit_vis = image.copy()
        cv2.drawContours(postit_vis, contours_postit, -1, (0, 255, 0), 2)
        cv2.imshow("Post-it Detection", postit_vis)
        cv2.imshow("Post-it Mask", mask_postit)
        cv2.waitKey(0)

    square_contour = None
    max_area = 0
    for cnt in contours_postit:
        area = cv2.contourArea(cnt)
        if area < 100:
            continue
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) == 4 and area > max_area:
            max_area = area
            square_contour = approx

    if square_contour is None:
        raise Exception("Could not find a valid Post-it note contour. Check HSV range or image setup.")

    square_contour = square_contour.reshape((4, 2))
    corners = order_corners(square_contour)

    # VISUALIZATION: Show the ordered corners
    if visualize:
        corners_vis = image.copy()
        for i, corner in enumerate(corners):
            cv2.circle(corners_vis, tuple(map(int, corner)), 5, (0, 0, 255), -1)
            cv2.putText(corners_vis, str(i), tuple(map(int, corner)), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.imshow("Post-it Corners", corners_vis)
        cv2.waitKey(0)

    # 4. Compute perspective transform
    dst_pts = np.array([
        [0, 0],
        [postit_side_mm, 0],
        [postit_side_mm, postit_side_mm],
        [0, postit_side_mm]
    ], dtype=np.float32)

    M = cv2.getPerspectiveTransform(corners, dst_pts)
    M_inv = np.linalg.inv(M)

    # 5. Detect beans
    mask_beans = cv2.inRange(hsv, bean_lower, bean_upper)
    mask_beans = cv2.bitwise_and(mask_beans, cv2.bitwise_not(mask_postit))
    mask_beans = cv2.morphologyEx(mask_beans, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8))
    mask_beans = cv2.morphologyEx(mask_beans, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))

    # VISUALIZATION: Show bean detection
    if visualize:
        cv2.imshow("Bean Mask", mask_beans)
        cv2.waitKey(0)

    bean_contours, _ = cv2.findContours(mask_beans, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bean_contours = [cnt for cnt in bean_contours if cv2.contourArea(cnt) > 50]

    # VISUALIZATION: Show detected bean contours
    if visualize:
        bean_contours_vis = image.copy()
        cv2.drawContours(bean_contours_vis, bean_contours, -1, (0, 255, 0), 2)
        cv2.imshow("Bean Contours", bean_contours_vis)
        cv2.waitKey(0)

    # 6. For each bean contour, measure its width using Distance Transform
    results = []
    bean_id = 0
    padding = 10 # Padding around bean for mask creation

    for cnt in bean_contours:
        bean_id += 1

        # Convert contour to float32 for perspectiveTransform
        cnt_float = cnt.astype(np.float32)
        # Warp contour to "top-down" metric space (1 px = 1 mm)
        cnt_warped = cv2.perspectiveTransform(cnt_float, M)

        # VISUALIZATION: Show the warped contour
        if visualize:
            # Create a blank warped space image
            warped_vis = np.zeros((int(postit_side_mm * 1.5), int(postit_side_mm * 1.5), 3), dtype=np.uint8)
            # Draw the warped contour
            cv2.drawContours(warped_vis, [cnt_warped.astype(int)], -1, (0, 255, 0), 2)
            cv2.imshow(f"Bean {bean_id} - Warped Contour", warped_vis)
            cv2.waitKey(0)

        # --- Start of Distance Transform Logic (Replaces minAreaRect) ---

        # 2. Create a Bean Mask
        try:
            # Calculate bounding box of the warped contour
            x_w, y_w, w_w, h_w = cv2.boundingRect(cnt_warped)

            # Create a black mask with padding
            mask_h = h_w + 2 * padding
            mask_w = w_w + 2 * padding
            mask = np.zeros((mask_h, mask_w), dtype=np.uint8)

            # Create offset contour relative to the mask's origin
            cnt_mask_relative = cnt_warped - np.array([x_w - padding, y_w - padding])

            # Draw the filled contour onto the mask
            cv2.drawContours(mask, [cnt_mask_relative.astype(int)], -1, (255), thickness=cv2.FILLED)

            # VISUALIZATION: Show the bean mask
            if visualize:
                cv2.imshow(f"Bean {bean_id} - Mask", mask)
                print(f"Bean {bean_id} - Mask size: {mask.shape}")
                print(f"Bean {bean_id} - Bounding box: x={x_w}, y={y_w}, w={w_w}, h={h_w}")
                mask_with_contour = cv2.cvtColor(mask.copy(), cv2.COLOR_GRAY2BGR)
                cv2.drawContours(mask_with_contour, [cnt_mask_relative.astype(int)], -1, (0, 0, 255), 1)
                cv2.imshow(f"Bean {bean_id} - Mask with Contour", mask_with_contour)
                cv2.waitKey(0)

            # 3. Apply Distance Transform
            dist_transform = cv2.distanceTransform(mask, distanceType=cv2.DIST_L2, maskSize=5)

            # VISUALIZATION: Show the distance transform
            if visualize:
                # Normalize for visualization
                norm_dist = cv2.normalize(dist_transform, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
                dist_vis = cv2.applyColorMap(norm_dist, cv2.COLORMAP_JET)
                cv2.imshow(f"Bean {bean_id} - Distance Transform", dist_vis)
                cv2.waitKey(0)

            # 4. Find Maximum Width and Location
            minVal, max_radius, minLoc, max_loc = cv2.minMaxLoc(dist_transform)
            # Max width is twice the radius of the largest inscribed circle
            bean_width_mm = 2 * max_radius

            # VISUALIZATION: Show the max distance point on the distance transform
            if visualize:
                dist_center_vis = dist_vis.copy()
                cv2.circle(dist_center_vis, max_loc, 5, (0, 0, 255), -1)
                cv2.putText(dist_center_vis, f"Max: {max_radius:.2f}", max_loc, 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.imshow(f"Bean {bean_id} - Max Distance Point", dist_center_vis)
                print(f"Bean {bean_id} - Max distance: {max_radius:.2f} at location: {max_loc}")
                cv2.waitKey(0)

            # 5. Determine Endpoints for Visualization Line (Approximation)
            # Get the center point in warped coordinates
            center_mask = max_loc # (mx, my)
            center_warped = (center_mask[0] + x_w - padding, center_mask[1] + y_w - padding)

            # VISUALIZATION: Show the center point in warped coordinates
            if visualize:
                warped_center_vis = warped_vis.copy()
                cv2.circle(warped_center_vis, (int(center_warped[0]), int(center_warped[1])), 5, (0, 0, 255), -1)
                cv2.putText(warped_center_vis, "Center", (int(center_warped[0]), int(center_warped[1]) - 10), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.imshow(f"Bean {bean_id} - Center in Warped Space", warped_center_vis)
                print(f"Bean {bean_id} - Center in mask: {center_mask}")
                print(f"Bean {bean_id} - Center in warped: {center_warped}")
                cv2.waitKey(0)

            # Estimate orientation using fitEllipse (requires > 5 points)
            if len(cnt_warped) >= 5:
                ellipse = cv2.fitEllipse(cnt_warped)
                # Ellipse angle is orientation of major axis. Width is perpendicular.
                width_angle_deg = ellipse[2] + 90.0
                
                # VISUALIZATION: Show the fitted ellipse
                if visualize:
                    ellipse_vis = warped_vis.copy()
                    cv2.ellipse(ellipse_vis, ellipse, (0, 255, 255), 2)
                    # Draw major axis
                    major_angle_rad = np.deg2rad(ellipse[2])
                    major_axis_len = max(ellipse[1][0], ellipse[1][1]) / 2
                    ex1 = int(ellipse[0][0] + major_axis_len * np.cos(major_angle_rad))
                    ey1 = int(ellipse[0][1] + major_axis_len * np.sin(major_angle_rad))
                    ex2 = int(ellipse[0][0] - major_axis_len * np.cos(major_angle_rad))
                    ey2 = int(ellipse[0][1] - major_axis_len * np.sin(major_angle_rad))
                    cv2.line(ellipse_vis, (ex1, ey1), (ex2, ey2), (255, 0, 0), 2)  # Blue line for major axis
                    
                    # Draw minor axis (perpendicular = width direction)
                    minor_angle_rad = np.deg2rad(width_angle_deg)
                    minor_axis_len = min(ellipse[1][0], ellipse[1][1]) / 2
                    wx1 = int(ellipse[0][0] + minor_axis_len * np.cos(minor_angle_rad))
                    wy1 = int(ellipse[0][1] + minor_axis_len * np.sin(minor_angle_rad))
                    wx2 = int(ellipse[0][0] - minor_axis_len * np.cos(minor_angle_rad))
                    wy2 = int(ellipse[0][1] - minor_axis_len * np.sin(minor_angle_rad))
                    cv2.line(ellipse_vis, (wx1, wy1), (wx2, wy2), (0, 255, 255), 2)  # Yellow line for minor axis
                    
                    cv2.imshow(f"Bean {bean_id} - Fitted Ellipse", ellipse_vis)
                    print(f"Bean {bean_id} - Ellipse: center={ellipse[0]}, axes={ellipse[1]}, angle={ellipse[2]}")
                    print(f"Bean {bean_id} - Width angle: {width_angle_deg}")
                    cv2.waitKey(0)
            else:
                # Fallback for very small contours: Use horizontal line
                width_angle_deg = 0.0
                print(f"Bean {bean_id} - Too few points for ellipse, using horizontal angle")

            width_angle_rad = np.deg2rad(width_angle_deg)

            # Calculate displacement vector (dx, dy) along width angle
            dx = max_radius * np.cos(width_angle_rad)
            dy = max_radius * np.sin(width_angle_rad)

            # Calculate endpoints in warped space
            p1_warped = (center_warped[0] - dx, center_warped[1] - dy)
            p2_warped = (center_warped[0] + dx, center_warped[1] + dy)

            # VISUALIZATION: Show the endpoints in warped space
            if visualize:
                endpoints_vis = warped_center_vis.copy()
                cv2.line(endpoints_vis, 
                        (int(p1_warped[0]), int(p1_warped[1])), 
                        (int(p2_warped[0]), int(p2_warped[1])), 
                        (0, 255, 255), 2)  # Yellow line
                cv2.imshow(f"Bean {bean_id} - Width Line in Warped Space", endpoints_vis)
                print(f"Bean {bean_id} - Line endpoints in warped: {p1_warped}, {p2_warped}")
                cv2.waitKey(0)

            # 6. Transform Endpoints Back to Original Image
            endpoints_warped = np.array([[p1_warped, p2_warped]], dtype=np.float32)
            endpoints_original = cv2.perspectiveTransform(endpoints_warped, M_inv)
            p1_orig = tuple(map(int, endpoints_original[0][0]))
            p2_orig = tuple(map(int, endpoints_original[0][1]))

            # VISUALIZATION: Show the endpoints in original image
            if visualize:
                orig_endpoints_vis = orig.copy()
                cv2.line(orig_endpoints_vis, p1_orig, p2_orig, (255, 255, 0), 2)  # Cyan line
                cv2.imshow(f"Bean {bean_id} - Width Line in Original Image", orig_endpoints_vis)
                print(f"Bean {bean_id} - Line endpoints in original: {p1_orig}, {p2_orig}")
                cv2.waitKey(0)

            # 7. Draw the Width Line
            cv2.line(orig, p1_orig, p2_orig, (255, 255, 0), 2) # Cyan line

            # 8. Annotate with Measurement
            # Calculate text position near the line midpoint (offset slightly)
            text_x = (p1_orig[0] + p2_orig[0]) // 2 + 5
            text_y = (p1_orig[1] + p2_orig[1]) // 2 - 5
            text_pos = (text_x, text_y)

            text_label = f"{bean_width_mm:.1f} mm"
            # Draw text with black outline
            cv2.putText(orig, text_label, text_pos,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
            # Draw white text
            cv2.putText(orig, text_label, text_pos,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

            # 9. Store Result
            results.append((bean_id, bean_width_mm))

            if debug:
                print(f"[DEBUG] Bean {bean_id}: width={bean_width_mm:.2f} mm, center_warped={center_warped}")

        except Exception as e:
            print(f"[ERROR] Failed processing bean {bean_id}: {e}")
            # Optionally skip this bean or assign default value
            results.append((bean_id, -1.0)) # Indicate error
            continue # Skip to next bean

        # --- End of Distance Transform Logic ---

    # VISUALIZATION: Show final result
    if visualize:
        cv2.imshow("Final Result", orig)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # 7. Save the result image
    out_name = "output_annotated_dt.jpg" # Different output name
    cv2.imwrite(out_name, orig)
    if debug:
        print(f"[INFO] Processed image saved as: {out_name}")
        print("[INFO] Bean widths (mm):", results)

    return orig, results


if __name__ == "__main__":
    # Example usage - Adjust paths and parameters as needed
    input_image = "C:\\beans\\beans_example.jpg" # <-- Replace with your image path
    try:
        annotated_img, bean_info = measure_beans_dt( # Call the new function
            image_path=input_image,
            postit_side_mm=76.0,      # Post-it note side in mm
            # Adjusted Post-it HSV range based on original script's main block
            postit_lower = (35,  0,  180),
            postit_upper = (55, 80, 255),
            # Adjusted Bean HSV range based on original script's main block
            bean_lower = (20, 100, 100),
            bean_upper = (70, 255, 255),
            debug=True,
            visualize=True  # Enable visualization
        )
        print("Finished measuring beans using Distance Transform!")
        print("Bean info (bean_id, width_mm):", bean_info)

    except Exception as e:
        print(f"An error occurred: {e}") 