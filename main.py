from __future__ import annotations

import sys

from longest_stay_detection import config
from yolo_standing_detail_video import main as run_standing_detail


if __name__ == "__main__":
    if len(sys.argv) == 1:
        # โหมดรันค่าเริ่มต้น: ใช้วิดีโอ entrance.mov และเปิดหน้าต่าง preview
        # ถ้าผู้ใช้ส่ง argument มาเอง จะไม่เติมค่า default เพื่อให้ควบคุม
        # source, output, model และ preview ได้เต็มที่จาก command line
        sys.argv.extend(
            [
                "--source",
                str(config.DEFAULT_SOURCE),
                "--out",
                str(config.DEFAULT_OUTPUT_DIR),
                "--model",
                config.default_model_path(),
                "--conf",
                str(config.DETECTION_CONF),
                "--imgsz",
                str(config.IMAGE_SIZE),
                "--frame-skip",
                str(config.FRAME_SKIP),
                "--device",
                config.DEVICE,
                "--show",
                "--show-scale",
                "0.60",
            ]
        )
    run_standing_detail()
