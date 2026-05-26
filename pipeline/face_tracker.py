from __future__ import annotations

from typing import Any, Dict, List, Optional

import cv2

try:
    import mediapipe as mp  # type: ignore
except Exception:
    mp = None


FaceSample = Dict[str, float]


def _detect_face_center_with_cv(frame) -> Optional[float]:
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(cascade_path)
    if detector.empty():
        return None

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
    if len(faces) == 0:
        return None

    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    width = frame.shape[1]
    if width <= 0:
        return None
    center_x = (x + (w / 2.0)) / float(width)
    return max(0.0, min(1.0, float(center_x)))


def _mediapipe_solutions_available() -> bool:
    return bool(mp is not None and hasattr(mp, "solutions") and hasattr(mp.solutions, "face_detection"))


def detect_face_timeline(
    video_path: str,
    start_time: float,
    end_time: float,
    sample_fps: float = 5.0,
) -> List[FaceSample]:
    """Return sampled face center positions over the clip range.

    Output times are relative to start_time.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_step = max(1, int(round(src_fps / sample_fps)))
    start_frame = int(start_time * src_fps)
    end_frame = int(end_time * src_fps)

    use_mediapipe = _mediapipe_solutions_available()
    detector = None
    if use_mediapipe:
        mp_face = mp.solutions.face_detection
        detector = mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.5)

    frame_idx = 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    out: List[FaceSample] = []

    while frame_idx <= (end_frame - start_frame):
        ok, frame = cap.read()
        if not ok:
            break

        if frame_idx % frame_step != 0:
            frame_idx += 1
            continue

        center_x: Optional[float] = None
        if detector is not None:
            try:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = detector.process(rgb)
                if result.detections:
                    box = result.detections[0].location_data.relative_bounding_box
                    center_x = float(box.xmin + (box.width / 2.0))
                    center_x = max(0.0, min(1.0, center_x))
            except Exception:
                center_x = None

        if center_x is None:
            center_x = _detect_face_center_with_cv(frame)

        if center_x is not None:
            t_rel = frame_idx / src_fps
            out.append({"time": float(t_rel), "center_x": float(center_x)})

        frame_idx += 1

    cap.release()
    if detector is not None:
        detector.close()
    return out


def suggest_safe_side(
    face_timeline: List[FaceSample],
    start_offset: float,
    end_offset: float,
    default: str = "right",
) -> str:
    """Pick overlay side opposite to dominant face position in interval."""
    scoped = [s for s in face_timeline if start_offset <= s["time"] <= end_offset]
    if not scoped:
        return default

    avg_x = sum(s["center_x"] for s in scoped) / len(scoped)
    return "left" if avg_x > 0.52 else "right"


def detect_face_center_x(
    video_path: str,
    start_time: float,
    end_time: float,
    sample_fps: float = 5.0,
) -> Optional[float]:
    """Return a normalized x-center [0,1] for the dominant face in a clip range.

    Uses EMA smoothing on sampled detections for stable framing.
    """
    timeline = detect_face_timeline(video_path, start_time, end_time, sample_fps)
    if not timeline:
        return None

    ema = None
    alpha = 0.2

    for sample in timeline:
        center_x = float(sample["center_x"])
        if ema is None:
            ema = center_x
        else:
            ema = (alpha * center_x) + ((1.0 - alpha) * ema)

    return ema
