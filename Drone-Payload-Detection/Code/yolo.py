import cv2
from inference_sdk import InferenceHTTPClient

# ---------------------------
# CONFIG
# ---------------------------
API_KEY = "TdeJ8iafXWtHsUASMq5J"
WORKSPACE_NAME = "reude-technologies"
WORKFLOW_ID = "find-containers"
VIDEO_PATH = "RC5_OR_0000.mp4"

FRAME_SKIP = 150   # 👈 process 1 frame every 10 frames (IMPORTANT)

# ---------------------------
# CLIENT
# ---------------------------
client = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key=API_KEY
)

# ---------------------------
# OPEN VIDEO
# ---------------------------
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print("❌ ERROR: Video not opened")
    exit()

print("✅ Video opened")

frame_id = 0

# ---------------------------
# LOOP
# ---------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        print("✅ End of video")
        break

    frame_id += 1

    # ---------------------------
    # FRAME SKIPPING (KEY FIX)
    # ---------------------------
    if frame_id % FRAME_SKIP != 0:
        cv2.imshow("Roboflow Container Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        continue

    # ---------------------------
    # WORKFLOW INFERENCE
    # ---------------------------
    result = client.run_workflow(
        workspace_name=WORKSPACE_NAME,
        workflow_id=WORKFLOW_ID,
        images={"image": frame}
    )

    # ---------------------------
    # SAFE PARSING
    # ---------------------------
    predictions = []
    if isinstance(result, list) and len(result) > 0:
        predictions = result[0].get("outputs", {}).get("predictions", [])

    # ---------------------------
    # DRAW BOXES
    # ---------------------------
    for p in predictions:
        if not isinstance(p, dict):
            continue

        if p.get("class") != "container":
            continue

        x = int(p["x"])
        y = int(p["y"])
        w = int(p["width"])
        h = int(p["height"])

        x1 = int(x - w / 2)
        y1 = int(y - h / 2)
        x2 = int(x + w / 2)
        y2 = int(y + h / 2)

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            frame,
            f'container {p["confidence"]:.2f}',
            (x1, y1 - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    cv2.imshow("Roboflow Container Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()