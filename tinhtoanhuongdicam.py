import cv2
import numpy as np

cap = cv2.VideoCapture(r"C:\Users\Vitus\Downloads\drone_video.avi")


prev = None
while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if prev is None:
        prev = gray
        continue

    # 1) detect keypoints
    pts = cv2.goodFeaturesToTrack(prev, 1000, 0.01, 10)
    if pts is None:
        prev = gray
        continue

    # 2) track
    pts2, st, err = cv2.calcOpticalFlowPyrLK(prev, gray, pts, None)
    good1 = pts[st == 1]
    good2 = pts2[st == 1]

    # 3) estimate global motion
    H, inlier = cv2.estimateAffinePartial2D(good1, good2, method=cv2.RANSAC)

    # Draw flow
    disp = frame.copy()
    for pt1, pt2, ok in zip(good1, good2, st):  # Unpack từng point array
        x1, y1 = pt1  # pt1 là [x, y], unpack thành scalars
        x2, y2 = pt2
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])  # Map scalars trực tiếp
        cv2.circle(disp, (x2, y2), 2, (0, 255, 0), -1)
        cv2.line(disp, (x1, y1), (x2, y2), (0, 255, 0), 1)

    if H is not None:
        dx, dy = H[0, 2], H[1, 2]
        angle = np.arctan2(dy, dx) * 180/np.pi
        text = f"dx={dx:.2f}, dy={dy:.2f}, angle={angle:.1f}"
        print(text)
        cv2.putText(disp, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 0, 255), 2)

    # Show
    cv2.imshow("Motion", disp)

    # quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    prev = gray

cap.release()
cv2.destroyAllWindows()
