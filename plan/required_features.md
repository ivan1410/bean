**Project Task: Implement Bean Maximum Width Measurement (Method 1)**

**Goal:**
Modify the existing Python script (`measure_beans` function) to calculate the *maximum internal width* of each detected bean contour. Instead of drawing a bounding box, you must draw a single line segment representing this maximum width *at its location* on the bean and display the measurement in mm.

**Context:**
You will be working within the provided `measure_beans` function. You should **reuse** the existing code for:
1.  Image loading.
2.  HSV conversion.
3.  Post-it note detection and perspective transform calculation (generating matrices `M` and `M_inv`). This calibration step is crucial for converting pixel measurements to millimeters.
4.  Bean color masking and contour finding (`cv2.findContours` generating `bean_contours`).

Your core task is to **replace** the section within the loop `for cnt in bean_contours:` that currently uses `cv2.minAreaRect` to calculate width and draws a bounding box.

**Chosen Method (Option 1 - Detailed): Iterate Through Contour Points with Midpoint Check**

**Why this Method:**
The goal is to find the maximum width *internal* to the bean's shape, roughly perpendicular to its length. We will approximate this by finding the longest line segment between any two points on the bean's contour whose *midpoint* also lies *inside* the contour. This acts as a heuristic to favor segments that cut across the bean's body rather than long segments along its edges or outside it.

**Detailed Implementation Steps:**

1.  **Loop Through Contours:** Start inside the existing loop: `for cnt in bean_contours:`. Increment `bean_id`.
2.  **Warp Contour:** Convert the contour `cnt` to float32 and apply the perspective transform using matrix `M` to get the contour in the calibrated space (where distances are in mm):
    ```python
    # Keep this part
    cnt_float = cnt.astype(np.float32)
    cnt_warped = cv2.perspectiveTransform(np.array([cnt_float]), M)[0] # Ensure shape is correct
    # cnt_warped is now a list of [x, y] points in mm
    ```
3.  **Initialization:** Before checking points for the current contour, initialize variables to store the best result found so far:
    ```python
    max_width_mm = 0.0
    best_p1_warped = None
    best_p2_warped = None
    num_points = len(cnt_warped)
    ```
4.  **Handle Few Points:** If `num_points` is very small (e.g., less than 4), it might not be possible to calculate a meaningful width. You can either skip this contour or handle it as an edge case.
    ```python
    if num_points < 4:
         print(f"Skipping bean {bean_id} - too few contour points ({num_points})")
         continue # Skip to the next bean contour
    ```
5.  **Iterate Through Point Pairs:** Use nested loops to iterate through all unique pairs of points on the `cnt_warped`. To avoid redundant checks (p1, p2 vs p2, p1) and checking a point against itself:
    ```python
    for i in range(num_points):
        p1_warped = cnt_warped[i]
        for j in range(i + 1, num_points): # Start j from i+1
            p2_warped = cnt_warped[j]

            # --- Inside the inner loop ---
    ```
6.  **Calculate Midpoint:** Find the midpoint of the line segment connecting `p1_warped` and `p2_warped`:
    ```python
            midpoint_warped = (p1_warped + p2_warped) / 2.0
    ```
7.  **Check if Midpoint is Inside Contour:** Use `cv2.pointPolygonTest` to verify if the `midpoint_warped` lies strictly inside or on the boundary of the `cnt_warped`. We need the contour points in the correct format (e.g., `np.array(cnt_warped, dtype=np.float32)`). `measureDist=False` is faster if you only need inside/outside/on-edge classification. A result >= 0 means inside or on the boundary.
    ```python
            # Ensure midpoint_warped is a tuple for pointPolygonTest
            is_inside = cv2.pointPolygonTest(cnt_warped, tuple(midpoint_warped.flatten()), measureDist=False) >= 0
    ```
8.  **Calculate Distance & Update Max Width:** If the midpoint is inside:
    * Calculate the Euclidean distance between `p1_warped` and `p2_warped`. This is a candidate width in mm.
    * Compare this distance with the current `max_width_mm`. If it's larger, update `max_width_mm` and store the corresponding points (`p1_warped`, `p2_warped`) as `best_p1_warped` and `best_p2_warped`.
    ```python
            if is_inside:
                distance = np.linalg.norm(p1_warped - p2_warped)
                if distance > max_width_mm:
                    max_width_mm = distance
                    best_p1_warped = p1_warped
                    best_p2_warped = p2_warped
    ```
9.  **After Point Pair Loops:** Once the loops finish for the current bean contour, `max_width_mm` will hold the maximum width found, and `best_p1_warped`, `best_p2_warped` will hold the endpoints of that width segment in the warped (mm) space.
10. **Check if Width Was Found:** It's possible (though unlikely for reasonable beans) that no pairs satisfied the midpoint condition.
    ```python
    if best_p1_warped is not None and best_p2_warped is not None:
        # Proceed to transform back and draw
        # ...
    else:
        print(f"Warning: Could not determine valid width for bean {bean_id}")
        continue # Skip drawing for this bean
    ```
11. **Transform Points Back:** If a valid width was found, transform the endpoints (`best_p1_warped`, `best_p2_warped`) back to the original image coordinate space using the inverse perspective transform `M_inv`. Remember `cv2.perspectiveTransform` expects input like `[[[x1, y1]], [[x2, y2]]]`.
    ```python
        points_to_transform = np.array([[best_p1_warped.flatten()], [best_p2_warped.flatten()]], dtype=np.float32)
        points_original = cv2.perspectiveTransform(points_to_transform, M_inv)[0]
        p1_orig = tuple(np.intp(points_original[0])) # Convert to integer tuple
        p2_orig = tuple(np.intp(points_original[1])) # Convert to integer tuple
    ```
12. **Draw the Max Width Line:** Draw **only** the line segment between `p1_orig` and `p2_orig` on the `orig` image. Do **not** draw the bounding box anymore.
    ```python
        cv2.line(orig, p1_orig, p2_orig, (0, 255, 0), 2) # Draw a green line for width
    ```
13. **Annotate the Measurement:** Calculate the midpoint of the drawn line (`p1_orig`, `p2_orig`) to place the text label. Format the `max_width_mm` value. Use the existing `cv2.putText` logic, but position the text near this new midpoint.
    ```python
        text_pos = tuple(np.intp((np.array(p1_orig) + np.array(p2_orig)) / 2))
        text_label = f"{max_width_mm:.1f} mm"
        # Use existing putText calls with the new text_pos and text_label
        cv2.putText(orig, text_label, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(orig, text_label, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    ```
14. **Store Result:** Append the bean ID and the calculated `max_width_mm` to the `results` list.
    ```python
        results.append((bean_id, max_width_mm))
        if debug:
             print(f"[DEBUG] Bean {bean_id}: Max Width={max_width_mm:.2f} mm")
    ```

**Key Considerations & What to Watch Out For:**

* **Computational Cost:** This `O(N^2)` approach (where N is the number of points in a contour) can be **very slow** for contours with many points. Test performance. Potential optimizations (if needed later):
    * Use contours found with `cv2.CHAIN_APPROX_SIMPLE` (already in the original code, likely okay).
    * Consider simplifying the contour further (`cv2.approxPolyDP`) before the O(N^2) loop, but this might reduce accuracy.
    * Consider sampling points instead of using all pairs (e.g., check point `i` against points `i + N/4`, `i + N/2`, `i + 3N/4`, reducing checks from N^2 to N*k).
* **Accuracy:** The "midpoint inside" check is a heuristic. It should work reasonably well for convex or mostly convex shapes like beans but might not find the perfect perpendicular width for highly irregular or concave shapes. It prioritizes finding the longest line segment *across* the bean.
* **Data Types:** Pay close attention to data types (`float32` for transforms and calculations, `intp` or `int` for pixel coordinates/drawing). Ensure array shapes are correct for OpenCV functions (e.g., `perspectiveTransform` needs points enclosed in extra brackets).
* **Visualization:** Ensure you **only** draw the calculated width line segment (`cv2.line`) and **remove** the old `cv2.drawContours` call that drew the red bounding box for the beans.

**Expected Output:**
The function should return the annotated `orig` image showing the beans. Each bean should have a single line segment drawn across its widest part (as determined by the implemented logic), with the corresponding width measurement in mm placed nearby. The function should also return the `results` list containing tuples of `(bean_id, max_width_mm)`.
