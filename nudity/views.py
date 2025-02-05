from django.http import HttpResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt
from nudenet import NudeDetector
import os
import cv2

# Define necessary directories
UPLOAD_DIR = "uploads"
FRAMES_DIR = "frames"

# Initialize NudeDetector only once
detector = NudeDetector()

def home(request):
    return HttpResponse("Hello, Django is working!")

@csrf_exempt
def predict(request):
    if request.method != 'POST':
        return JsonResponse({"error": "Only POST method is allowed"}, status=405)

    # Check if an image was uploaded
    if 'image' not in request.FILES:
        return JsonResponse({"error": "No image provided"}, status=400)

    file = request.FILES['image']

    # Ensure directories exist
    os.makedirs(UPLOAD_DIR, exist_ok=True)

    # Save the uploaded file
    image_path = os.path.join(UPLOAD_DIR, file.name)
    with open(image_path, "wb") as f:
        for chunk in file.chunks():
            f.write(chunk)

    try:
        # Run Nude Detection on the image
        classification_result = detector.detect(image_path)

        # Analyze the results
        result = is_explicit_content(classification_result)

        # Remove the saved image after processing
        os.remove(image_path)

        # Return response
        return JsonResponse({
            "details": result,
            "explicit_content": bool(result)
        })

    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)

def is_explicit_content(predictions, threshold=0.50):
    explicit_classes = {"FEMALE_BREAST_EXPOSED", "BUTTOCKS_EXPOSED", "FEMALE_GENITALIA_EXPOSED", "MALE_GENITALIA_EXPOSED"}
    result = []

    for item in predictions:
        if item["score"] > threshold and item["class"] in explicit_classes:
            result.append({
                "class": item["class"],
                "score_percentage": round(item["score"] * 100, 2),
                "explicit": True
            })

    return result

@csrf_exempt
def predict_video(request):
    if request.method != 'POST':
        return JsonResponse({"error": "Only POST method is allowed"}, status=405)

    # Check if video file is provided
    if 'video' not in request.FILES:
        return JsonResponse({"error": "No video file provided"}, status=400)

    video_file = request.FILES['video']
    video_path = os.path.join(UPLOAD_DIR, video_file.name)
    os.makedirs(UPLOAD_DIR, exist_ok=True)  # Ensure directory exists

    # Save the uploaded video file
    with open(video_path, "wb") as f:
        for chunk in video_file.chunks():
            f.write(chunk)

    # Extract frames from video
    frames = extract_frames(video_path, frame_interval=1)
    flagged_frames = []

    for timestamp, frame_path in frames:
        detections = detector.detect(frame_path)
        flagged_detections = is_explicit_content(detections, threshold=0.50)

        if flagged_detections:
            flagged_frames.append({
                "timestamp": timestamp,
                "frame_path": frame_path,
                "detections": flagged_detections
            })
        else:
            os.remove(frame_path)

    os.remove(video_path)  # Clean up uploaded video file

    responseData = []
    for frame_data in flagged_frames:
        for item in frame_data["detections"]:
            responseData.append({
                "timestamp": frame_data["timestamp"],
                "class": item["class"],
                "score_percentage": item["score_percentage"],
                "frame_path": frame_data["frame_path"]
            })

    return JsonResponse({
        "flagged_timestamps": responseData,
        "explicit_detected": len(flagged_frames) > 0
    })

def extract_frames(video_path, frame_interval=5):
    os.makedirs(FRAMES_DIR, exist_ok=True)
    frames = []
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS)) if cap.get(cv2.CAP_PROP_FPS) > 0 else 30
    frame_count = 0

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        if frame_count % (frame_interval * fps) == 0:
            timestamp = frame_count / fps
            timestamp_str = f"{timestamp:.2f}"
            frame_filename = os.path.join(FRAMES_DIR, f"frame_{timestamp_str}.jpg")
            cv2.imwrite(frame_filename, frame)
            frames.append((timestamp_str, frame_filename))

        frame_count += 1

    cap.release()
    return frames
