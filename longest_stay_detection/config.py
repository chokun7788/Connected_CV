from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

# รวมค่า default หลักไว้ที่ไฟล์นี้ เพื่อให้สคริปต์รันงานอ่านง่าย
# และจูน threshold ต่าง ๆ ได้จากจุดเดียว
DEFAULT_SOURCE = Path(r"C:\Users\Chokhun\Downloads\entrance.mov")
LOCAL_MODEL = ROOT / "yolo26s.pt"
DOWNLOADS_MODEL = Path(r"C:\Users\Chokhun\Downloads\Standing-ID-Detection\yolo26s.pt")
DEFAULT_OUTPUT_DIR = ROOT / "standing_detail_output_live"

DETECTION_CONF = 0.35
DETECTION_IOU = 0.45
IMAGE_SIZE = 640
FRAME_SKIP = 1
DEVICE = "0"

WINDOW_SIZE_FRAMES = 30
STATIONARY_STD_PX = 12.0
MIN_STATIONARY_SEC = 2.0
MAX_LOST_FRAMES = 30
DUPLICATE_IOU_THRESHOLD = 0.85


def default_model_path() -> str:
    """Prefer a project-local model, then reuse the existing downloaded model."""
    if LOCAL_MODEL.exists():
        return str(LOCAL_MODEL)
    if DOWNLOADS_MODEL.exists():
        return str(DOWNLOADS_MODEL)
    return "yolo26s.pt"
