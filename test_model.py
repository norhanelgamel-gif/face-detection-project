from ultralytics import YOLO
import cv2

# Load trained YOLOv8 model
model = YOLO("best.pt")

# Path to a test image (put the image in the same folder)
img_path = "test.jpg"

# Run inference
results = model(img_path)

# Draw bounding boxes
annotated_img = results[0].plot()

# Show the image
cv2.imshow("Face Detection - Test", annotated_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Print detection details in terminal
print("Detection results:")
for i, box in enumerate(results[0].boxes):
    cls_id = int(box.cls[0])
    conf = box.conf[0].item()
    xyxy = box.xyxy[0].cpu().numpy()
    print(f"Face {i+1} | Class: {cls_id} | Confidence: {conf:.2f} | Box: {xyxy}")

print("Test completed successfully.")
