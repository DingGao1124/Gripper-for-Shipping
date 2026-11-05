from pathlib import Path

import cv2


if __name__ == "__main__":
    # intrinsics for camera, for a 500 x 500 image; assume no distortion.
    # K = [[fx, 0, cx], [0, fy, cy], [0, 0, 1]].
    fx, fy = 500.0, 500.0
    cx, cy = 250.0, 250.0

    # 36h11 dictionary, marker ID 0, 400 x 400 pixels.
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    tag = cv2.aruco.generateImageMarker(dictionary, 0, 400)

    # White margin separates the black marker border from its surroundings.
    image = cv2.copyMakeBorder(tag, 50, 50, 50, 50, cv2.BORDER_CONSTANT, value=255)
    output = Path(__file__).with_name("apriltag_36h11_0.png")
    if not cv2.imwrite(str(output), image):
        raise OSError(f"Could not save {output}")
    print(f"Saved: {output}")

    # Detect on the clean grayscale image; draw on a separate color copy.
    detector = cv2.aruco.ArucoDetector(dictionary)
    corners, ids, _ = detector.detectMarkers(image)
    annotated = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    # Pixel coordinates: origin at top left, +x right, +y down.
    if ids is None:
        print("No AprilTag detected.")
    else:
        cv2.aruco.drawDetectedMarkers(annotated, corners, ids)
        for marker_id, marker_corners in zip(ids.ravel(), corners):
            points = marker_corners.reshape(4, 2)
            center = points.mean(axis=0)
            print(f"AprilTag ID {marker_id} — pixel coordinates (u, v):")

            # Detector order: marker top-left, top-right, bottom-right, bottom-left.
            for index, (u, v) in enumerate(points):
                print(f"  Corner {index}: ({u:.2f}, {v:.2f})")
                pixel = (round(float(u)), round(float(v)))
                cv2.circle(annotated, pixel, 5, (0, 0, 255), -1)
                cv2.putText(annotated, str(index), (pixel[0] + 8, pixel[1] - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            print(f"  Center: ({center[0]:.2f}, {center[1]:.2f})")
            center_pixel = tuple(round(float(value)) for value in center)
            cv2.circle(annotated, center_pixel, 5, (255, 0, 0), -1)
            cv2.putText(annotated, "Center", (center_pixel[0] + 8, center_pixel[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

            # Pinhole model: u = fx * X/Z + cx, v = fy * Y/Z + cy.
            # Thus x = (u-cx)/fx = X/Z, y = (v-cy)/fy = Y/Z (dimensionless).
            # This is camera normalization, not scaling pixels into [0, 1].
            normalized_points = (points - [cx, cy]) / [fx, fy]
            print(f"Intrinsics: fx={fx}, fy={fy}, cx={cx}, cy={cy}")
            print("Normalized image coordinates (x, y):")
            for index, (x, y) in enumerate(normalized_points):
                print(f"  Corner {index}: ({x:.4f}, {y:.4f})")

    try:
        cv2.imshow("AprilTag - corners and center", annotated)
        cv2.waitKey(0)
    finally:
        cv2.destroyAllWindows()
