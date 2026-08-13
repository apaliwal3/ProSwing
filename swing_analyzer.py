from __future__ import annotations

import argparse
import math
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2 as cv
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision


LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_ELBOW = 13
RIGHT_ELBOW = 14
LEFT_WRIST = 15
RIGHT_WRIST = 16
LEFT_HIP = 23
RIGHT_HIP = 24

MP_POSE_CONNECTIONS = (
    (11, 12),
    (11, 13),
    (13, 15),
    (15, 17),
    (15, 19),
    (15, 21),
    (17, 19),
    (12, 14),
    (14, 16),
    (16, 18),
    (16, 20),
    (16, 22),
    (18, 20),
    (11, 23),
    (12, 24),
    (23, 24),
    (23, 25),
    (25, 27),
    (27, 29),
    (29, 31),
    (24, 26),
    (26, 28),
    (28, 30),
    (30, 32),
    (27, 31),
    (28, 32),
)

BODY_LANDMARK_INDICES = (
    11,
    12,
    13,
    14,
    15,
    16,
    17,
    18,
    19,
    20,
    21,
    22,
    23,
    24,
    25,
    26,
    27,
    28,
    29,
    30,
    31,
    32,
)

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_full/float16/1/pose_landmarker_full.task"
)
DEFAULT_MODEL_PATH = Path(__file__).with_name("pose_landmarker_full.task")
DEFAULT_OUTPUT_PATH = Path(__file__).with_name("swing_analysed.mp4")

BRIGHT_LINE_COLOR = (0, 255, 255)
BRIGHT_POINT_COLOR = (0, 255, 0)
TEXT_COLOR = (255, 255, 255)
TEXT_BG = (0, 0, 0)
MAX_COACHING_TIPS = 3

# --- Pose validation gate -------------------------------------------------
# The gate checks for the one thing that actually characterizes a real golf swing pose:
# the hands stay together on the grip. That single "arms together" check
# catches mistracked limbs (the usual failure mode, especially on the lead
# arm) without vetoing normal swing variation.
POSE_VISIBILITY_THRESHOLD = 0.5
POSE_MIN_BODY_SPAN = 0.02
POSE_MAX_HAND_SEPARATION_MULTIPLIER = 0.9  # wrists should stay close together (hands on the grip)
POSE_MAX_ELBOW_SEPARATION_MULTIPLIER = 3.0  # loose backstop sanity check on elbow spread

TIP_CONFIRMATION_FRAMES = 3


@dataclass(frozen=True)
class SwingSides:
    lead: str
    trail: str


@dataclass
class StickyTipTracker:
    active_tips: tuple[str, ...] = ()

    def update(self, candidate_tips: Optional[list[str]]) -> list[str]:
        if not candidate_tips:
            self.active_tips = ()
            return list(self.active_tips)

        self.active_tips = tuple(unique_tips(candidate_tips)[:MAX_COACHING_TIPS])
        return list(self.active_tips)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze a golf swing video with pose overlays.")
    parser.add_argument("input_video", type=Path, help="Path to the Phone swing video")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Output annotated video path",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Path to the MediaPipe pose landmarker task file",
    )
    parser.add_argument(
        "--right-handed",
        action="store_true",
        help="Force a right-handed swing interpretation",
    )
    parser.add_argument(
        "--left-handed",
        action="store_true",
        help="Force a left-handed swing interpretation",
    )
    parser.add_argument(
        "--slowdown",
        type=float,
        default=1.0,
        help="Playback slowdown factor. 1.0 keeps normal speed, 2.0 plays at half speed.",
    )
    return parser.parse_args()


def ensure_model_file(model_path: Path) -> Path:
    if model_path.exists():
        return model_path

    model_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading pose model to {model_path}")
    with urllib.request.urlopen(MODEL_URL) as response, model_path.open("wb") as output_file:
        output_file.write(response.read())
    return model_path


def build_landmarker(model_path: Path):
    BaseOptions = mp_python.BaseOptions
    PoseLandmarker = mp_vision.PoseLandmarker
    PoseLandmarkerOptions = mp_vision.PoseLandmarkerOptions
    RunningMode = mp_vision.RunningMode

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model_path)),
        running_mode=RunningMode.IMAGE,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_segmentation_masks=False,
    )
    return PoseLandmarker.create_from_options(options)


def landmark_xy(landmarks, landmark_index: int):
    point = landmarks[landmark_index]
    return point.x, point.y, point.visibility


def point_to_pixel(point, width: int, height: int):
    x = int(round(max(0.0, min(1.0, point.x)) * (width - 1)))
    y = int(round(max(0.0, min(1.0, point.y)) * (height - 1)))
    return x, y


def angle_2d(a, b, c) -> Optional[float]:
    ab = (a[0] - b[0], a[1] - b[1])
    cb = (c[0] - b[0], c[1] - b[1])
    ab_norm = math.hypot(*ab)
    cb_norm = math.hypot(*cb)
    if ab_norm == 0 or cb_norm == 0:
        return None
    dot = ab[0] * cb[0] + ab[1] * cb[1]
    cosine = max(-1.0, min(1.0, dot / (ab_norm * cb_norm)))
    return math.degrees(math.acos(cosine))


def signed_line_angle(left_point, right_point) -> float:
    dx = right_point[0] - left_point[0]
    dy = right_point[1] - left_point[1]
    return math.degrees(math.atan2(dy, dx))


def normalize_angle(angle: float) -> float:
    while angle <= -180:
        angle += 360
    while angle > 180:
        angle -= 360
    return angle


def distance_2d(a, b) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def get_landmark_point(landmarks, landmark_index: int):
    point = landmarks[landmark_index]
    return point.x, point.y


def assess_pose_quality(landmarks) -> bool:
    """Lightweight gate: are the arms tracked in a plausible 'hands together
    on the grip' configuration for a golf swing, rather than a full
    biomechanical check on every joint.
    """
    key_indices = (
        LEFT_SHOULDER,
        RIGHT_SHOULDER,
        LEFT_ELBOW,
        RIGHT_ELBOW,
        LEFT_WRIST,
        RIGHT_WRIST,
    )
    if any(landmarks[index].visibility < POSE_VISIBILITY_THRESHOLD for index in key_indices):
        return False

    left_shoulder = get_landmark_point(landmarks, LEFT_SHOULDER)
    right_shoulder = get_landmark_point(landmarks, RIGHT_SHOULDER)
    left_elbow = get_landmark_point(landmarks, LEFT_ELBOW)
    right_elbow = get_landmark_point(landmarks, RIGHT_ELBOW)
    left_wrist = get_landmark_point(landmarks, LEFT_WRIST)
    right_wrist = get_landmark_point(landmarks, RIGHT_WRIST)

    shoulder_span = distance_2d(left_shoulder, right_shoulder)
    if shoulder_span < POSE_MIN_BODY_SPAN:
        return False

    # Core check: the hands stay together on the grip through almost the
    # entire swing, so the two wrists should sit close to each other
    # relative to shoulder width. A large separation almost always means
    # one wrist got mistracked (this is what was rejecting good lead-arm
    # frames under the old, stricter gate).
    wrist_span = distance_2d(left_wrist, right_wrist)
    if wrist_span > shoulder_span * POSE_MAX_HAND_SEPARATION_MULTIPLIER:
        return False

    # Loose backstop so a wildly flailing elbow doesn't sneak through even
    # when the wrists happen to line up.
    elbow_span = distance_2d(left_elbow, right_elbow)
    if elbow_span > shoulder_span * POSE_MAX_ELBOW_SEPARATION_MULTIPLIER:
        return False

    return True


def pick_swing_sides(right_handed: Optional[bool], landmarks) -> SwingSides:
    if right_handed is True:
        return SwingSides(lead="left", trail="right")
    if right_handed is False:
        return SwingSides(lead="right", trail="left")

    left_shoulder = landmarks[LEFT_SHOULDER]
    right_shoulder = landmarks[RIGHT_SHOULDER]
    left_wrist = landmarks[LEFT_WRIST]
    right_wrist = landmarks[RIGHT_WRIST]

    left_activity = left_shoulder.visibility + left_wrist.visibility
    right_activity = right_shoulder.visibility + right_wrist.visibility
    if left_activity >= right_activity:
        return SwingSides(lead="left", trail="right")
    return SwingSides(lead="right", trail="left")


def build_pose_canvas(frame, results, swing_sides: SwingSides):
    annotated = frame.copy()
    height, width = annotated.shape[:2]

    if not results.pose_landmarks:
        cv.putText(
            annotated,
            "Pose not found",
            (30, 60),
            cv.FONT_HERSHEY_SIMPLEX,
            1.0,
            TEXT_COLOR,
            3,
            cv.LINE_AA,
        )
        return annotated

    landmarks = results.pose_landmarks[0]
    pixel_points = [point_to_pixel(point, width, height) for point in landmarks]

    for connection in MP_POSE_CONNECTIONS:
        start = pixel_points[connection[0]]
        end = pixel_points[connection[1]]
        cv.line(annotated, start, end, BRIGHT_LINE_COLOR, 8, cv.LINE_AA)

    for index in BODY_LANDMARK_INDICES:
        x, y = pixel_points[index]
        cv.circle(annotated, (x, y), 7, (0, 0, 0), -1, cv.LINE_AA)
        cv.circle(annotated, (x, y), 5, BRIGHT_POINT_COLOR, -1, cv.LINE_AA)

    return annotated


def calculate_metrics(landmarks, swing_sides: SwingSides) -> dict[str, Optional[float]]:
    def get_point(index: int):
        point = landmarks[index]
        return (point.x, point.y)

    left_shoulder = get_point(LEFT_SHOULDER)
    right_shoulder = get_point(RIGHT_SHOULDER)
    left_elbow = get_point(LEFT_ELBOW)
    right_elbow = get_point(RIGHT_ELBOW)
    left_wrist = get_point(LEFT_WRIST)
    right_wrist = get_point(RIGHT_WRIST)
    left_hip = get_point(LEFT_HIP)
    right_hip = get_point(RIGHT_HIP)

    lead_elbow = left_elbow if swing_sides.lead == "left" else right_elbow
    lead_shoulder = left_shoulder if swing_sides.lead == "left" else right_shoulder
    lead_wrist = left_wrist if swing_sides.lead == "left" else right_wrist
    trail_elbow = right_elbow if swing_sides.lead == "left" else left_elbow
    trail_shoulder = right_shoulder if swing_sides.lead == "left" else left_shoulder
    trail_wrist = right_wrist if swing_sides.lead == "left" else left_wrist

    lead_elbow_angle = angle_2d(lead_shoulder, lead_elbow, lead_wrist)
    trail_elbow_angle = angle_2d(trail_shoulder, trail_elbow, trail_wrist)

    shoulder_line = signed_line_angle(left_shoulder, right_shoulder)
    hip_line = signed_line_angle(left_hip, right_hip)
    shoulder_rotation = normalize_angle(shoulder_line)
    hip_rotation = normalize_angle(hip_line)

    return {
        "lead_elbow": lead_elbow_angle,
        "trail_elbow": trail_elbow_angle,
        "shoulder_rotation": shoulder_rotation,
        "hip_rotation": hip_rotation,
    }


def derive_coaching_tips(metrics: dict[str, Optional[float]], swing_sides: SwingSides) -> list[str]:
    tips: list[str] = []

    lead_elbow = metrics["lead_elbow"]
    trail_elbow = metrics["trail_elbow"]

    if lead_elbow is not None:
        if lead_elbow < 155:
            tips.append(f"Keep your {swing_sides.lead} arm a little longer.")
        elif lead_elbow > 175:
            tips.append(f"Relax your {swing_sides.lead} arm a touch.")

    if trail_elbow is not None:
        if trail_elbow < 90:
            tips.append(f"Let your {swing_sides.trail} arm extend a bit more.")
        elif trail_elbow > 145:
            tips.append(f"Let your {swing_sides.trail} arm fold naturally.")

    return unique_tips(tips)[:MAX_COACHING_TIPS]


def unique_tips(tips: list[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for tip in tips:
        if tip in seen:
            continue
        seen.add(tip)
        unique.append(tip)
    return unique


def draw_tip_banner(image, coaching_tips: list[str]) -> int:
    if not coaching_tips:
        return 18

    height, width = image.shape[:2]
    x0 = 18
    y0 = 18
    x1 = width - 18
    header_height = 34
    line_height = 28
    padding = 16
    banner_height = padding * 2 + header_height + len(coaching_tips) * line_height
    y1 = min(height - 18, y0 + banner_height)

    overlay = image.copy()
    cv.rectangle(overlay, (x0, y0), (x1, y1), (0, 0, 0), -1)
    cv.addWeighted(overlay, 0.48, image, 0.52, 0, image)

    text_x = x0 + 16
    draw_text_box(image, "Tips", (text_x, y0 + 30))
    for offset, tip in enumerate(coaching_tips[:MAX_COACHING_TIPS]):
        draw_text_box(image, f"- {tip}", (text_x, y0 + 30 + (offset + 1) * line_height))

    return y1


def draw_text_box(image, text: str, origin: tuple[int, int]):
    x, y = origin
    cv.putText(image, text, (x, y), cv.FONT_HERSHEY_SIMPLEX, 0.7, TEXT_BG, 6, cv.LINE_AA)
    cv.putText(image, text, (x, y), cv.FONT_HERSHEY_SIMPLEX, 0.7, TEXT_COLOR, 2, cv.LINE_AA)


def draw_metrics_panel(
    image,
    metrics: dict[str, Optional[float]],
    swing_sides: SwingSides,
    top_offset: int = 18,
):
    height, width = image.shape[:2]
    panel_width = 460
    panel_height = 220
    x1 = width - 18
    y1 = top_offset
    x0 = max(18, x1 - panel_width)
    y2 = min(height - 18, y1 + panel_height)

    overlay = image.copy()
    cv.rectangle(overlay, (x0, y1), (x1, y2), (0, 0, 0), -1)
    cv.addWeighted(overlay, 0.48, image, 0.52, 0, image)

    text_x = x0 + 16
    draw_text_box(image, f"Lead side: {swing_sides.lead}", (text_x, 34))
    draw_text_box(image, f"Trail side: {swing_sides.trail}", (text_x, 64))
    draw_text_box(image, f"Lead elbow: {format_angle(metrics['lead_elbow'])}", (text_x, 100))
    draw_text_box(image, f"Trail elbow: {format_angle(metrics['trail_elbow'])}", (text_x, 130))
    draw_text_box(image, f"Shoulder rotation: {format_angle(metrics['shoulder_rotation'])}", (text_x, 160))
    draw_text_box(image, f"Hip rotation: {format_angle(metrics['hip_rotation'])}", (text_x, 190))


def format_angle(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    return f"{value:.1f}\xb0"


def open_writer(output_path: Path, fps: float, size: tuple[int, int]):
    fourcc_candidates = ["mp4v", "avc1"]
    fourcc_factory = getattr(cv, "VideoWriter_fourcc")
    for codec in fourcc_candidates:
        writer = cv.VideoWriter(str(output_path), fourcc_factory(*codec), fps, size)
        if writer.isOpened():
            return writer
    raise RuntimeError(f"Could not open a video writer for {output_path}")


def process_video(
    input_path: Path,
    output_path: Path,
    model_path: Path,
    right_handed: Optional[bool],
    slowdown: float,
):
    capture = cv.VideoCapture(str(input_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open input video: {input_path}")

    source_fps = capture.get(cv.CAP_PROP_FPS) or 30.0
    slowdown = max(1.0, slowdown)
    fps = max(1.0, source_fps / slowdown)
    capture.set(cv.CAP_PROP_ORIENTATION_AUTO, 0) if hasattr(cv, "CAP_PROP_ORIENTATION_AUTO") else None

    with build_landmarker(model_path) as landmarker:
        first_ok, first_frame = capture.read()
        if not first_ok:
            raise RuntimeError("Input video contains no frames")

        output_size = (first_frame.shape[0], first_frame.shape[1])
        writer = open_writer(output_path, fps, output_size)

        try:
            frame_index = 0
            capture.set(cv.CAP_PROP_POS_FRAMES, 0)
            swing_sides: Optional[SwingSides] = None
            tip_tracker = StickyTipTracker()

            while True:
                success, frame = capture.read()
                if not success:
                    break

                rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                result = landmarker.detect(mp_image)

                if result.pose_landmarks and swing_sides is None:
                    swing_sides = pick_swing_sides(right_handed, result.pose_landmarks[0])

                if swing_sides is None:
                    swing_sides = SwingSides(lead="left", trail="right")

                pose_is_valid = bool(result.pose_landmarks) and assess_pose_quality(result.pose_landmarks[0])
                candidate_tips: Optional[list[str]] = None
                metrics = None
                if pose_is_valid:
                    metrics = calculate_metrics(result.pose_landmarks[0], swing_sides)
                    candidate_tips = derive_coaching_tips(metrics, swing_sides)

                sticky_tips = tip_tracker.update(candidate_tips)

                annotated = build_pose_canvas(frame, result, swing_sides)
                annotated = cv.rotate(annotated, cv.ROTATE_90_CLOCKWISE)
                tip_bottom = draw_tip_banner(annotated, sticky_tips)
                if metrics is not None:
                    draw_metrics_panel(annotated, metrics, swing_sides, top_offset=tip_bottom + 12)
                writer.write(annotated)
                frame_index += 1
        finally:
            writer.release()

    capture.release()


def main() -> int:
    args = parse_args()
    if args.right_handed and args.left_handed:
        raise SystemExit("Choose only one of --right-handed or --left-handed")

    right_handed: Optional[bool]
    if args.right_handed:
        right_handed = True
    elif args.left_handed:
        right_handed = False
    else:
        right_handed = None

    if not args.input_video.exists():
        raise SystemExit(f"Input video not found: {args.input_video}")

    model_path = ensure_model_file(args.model)
    process_video(args.input_video, args.output, model_path, right_handed, args.slowdown)
    print(f"Wrote annotated video to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())