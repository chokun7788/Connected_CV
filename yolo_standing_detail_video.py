from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2

from longest_stay_detection import config
from longest_stay_detection.tracker import LongestStayTracker, TrackInfo


def parse_args() -> argparse.Namespace:
    # สิ่งที่อาจเปลี่ยนระหว่างวิดีโอถูกเปิดเป็น command-line argument
    # เพื่อให้โปรเจกต์ไม่ต้องพึ่งไฟล์ .bat
    parser = argparse.ArgumentParser(
        description="Render YOLO person tracking with Person IDs and cumulative standing timers."
    )
    parser.add_argument("--source", default=str(config.DEFAULT_SOURCE), help="Input video path.")
    parser.add_argument("--out", default=str(config.DEFAULT_OUTPUT_DIR), help="Output folder.")
    parser.add_argument("--model", default=config.default_model_path(), help="YOLO model path.")
    parser.add_argument("--conf", type=float, default=config.DETECTION_CONF, help="Person detection confidence.")
    parser.add_argument("--iou", type=float, default=config.DETECTION_IOU, help="Detection/tracking IoU threshold.")
    parser.add_argument("--imgsz", type=int, default=config.IMAGE_SIZE, help="YOLO image size.")
    parser.add_argument("--frame-skip", type=int, default=config.FRAME_SKIP, help="Process every Nth frame.")
    parser.add_argument("--device", default=config.DEVICE, help="Ultralytics device such as 0 or cpu.")
    parser.add_argument("--window-size", type=int, default=config.WINDOW_SIZE_FRAMES)
    parser.add_argument("--stationary-std-px", type=float, default=config.STATIONARY_STD_PX)
    parser.add_argument("--min-stationary-sec", type=float, default=config.MIN_STATIONARY_SEC)
    parser.add_argument("--max-lost-frames", type=int, default=config.MAX_LOST_FRAMES)
    parser.add_argument(
        "--duplicate-iou-threshold",
        type=float,
        default=config.DUPLICATE_IOU_THRESHOLD,
        help="Hide duplicate active IDs when two person boxes overlap above this IoU.",
    )
    parser.add_argument("--limit-frames", type=int, default=0, help="Stop after N processed frames; 0 means all.")
    parser.add_argument("--show", action="store_true", help="Show live preview while processing.")
    parser.add_argument("--show-scale", type=float, default=0.60, help="Preview scale when --show is enabled.")
    return parser.parse_args()


def fmt_time(seconds: float) -> str:
    # รูปแบบเวลาแบบสั้นสำหรับ overlay และ summary ใน CSV/JSON
    minutes = int(seconds // 60)
    sec = seconds - minutes * 60
    if minutes:
        return f"{minutes}:{sec:04.1f}"
    return f"{sec:.1f}s"


def color_for_track(track: TrackInfo, winner_id: int | None) -> tuple[int, int, int]:
    # สีเหลืองใช้เน้น ID ที่ชนะด้วยเวลาสะสม, สีเขียวคือ standing,
    # สีเทาคือ moving/checking
    if winner_id is not None and track.track_id == winner_id:
        return (0, 215, 255)
    if track.is_stationary:
        return (0, 200, 100)
    return (120, 120, 120)


def draw_label_block(frame, lines: list[str], x: int, y: int, color: tuple[int, int, int]) -> None:
    # วาดพื้นหลังทึบหลัง label เพื่อให้อ่านตัวหนังสือบนวิดีโอได้ชัด
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.47
    thickness = 1
    sizes = [cv2.getTextSize(line, font, scale, thickness)[0] for line in lines]
    width = max(size[0] for size in sizes) + 10
    line_h = 18
    height = line_h * len(lines) + 6
    top = max(0, y - height)
    right = min(frame.shape[1] - 1, x + width)

    cv2.rectangle(frame, (x, top), (right, top + height), color, -1)
    for idx, line in enumerate(lines):
        cv2.putText(
            frame,
            line,
            (x + 5, top + 16 + idx * line_h),
            font,
            scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )


def draw_box(frame, track: TrackInfo, winner_id: int | None) -> None:
    # overlay ต่อคนตาม requirement: confidence, Person ID, สถานะปัจจุบัน,
    # เวลายืนรอบนี้, เวลายืนสะสม และช่วงยืนต่อเนื่องนานสุด
    color = color_for_track(track, winner_id)
    x1, y1, x2, y2 = track.bbox
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    status = "STANDING" if track.is_stationary else "moving/checking"
    current = track.current_stationary_sec if track.is_stationary else 0.0
    labels = [
        f"YOLO person {track.confidence:.2f}",
        f"Person ID {track.track_id} | {status}",
        f"standing now {fmt_time(current)}",
        f"standing total {fmt_time(track.total_stationary_sec)}",
        f"longest one time {fmt_time(track.max_stationary_sec)}",
    ]
    draw_label_block(frame, labels, x1, y1, color)


def draw_panel(
    frame,
    active_tracks: dict[int, TrackInfo],
    all_tracks: dict[int, TrackInfo],
    winner: TrackInfo | None,
    frame_idx: int,
    fps: float,
) -> None:
    h = frame.shape[0]
    panel_w = 430
    overlay = frame.copy()
    # panel แบบโปร่งแสงช่วยให้ยังเห็นวิดีโอด้านหลัง
    # แต่ตัวหนังสือฝั่ง ranking ยังอ่านง่าย
    cv2.rectangle(overlay, (0, 0), (panel_w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.58, frame, 0.42, 0, frame)

    elapsed = frame_idx / fps if fps > 0 else 0.0
    standing_count = sum(1 for track in active_tracks.values() if track.is_stationary)
    visible_tracks = [track for track in all_tracks.values() if not track.is_duplicate_suppressed]
    winner_text = "collecting..."
    if winner and winner.total_stationary_sec > 0:
        winner_text = f"ID {winner.track_id} | total {fmt_time(winner.total_stationary_sec)}"

    lines = [
        "YOLO PERSON ID + STANDING TIMER",
        f"video time: {fmt_time(elapsed)} | frame: {frame_idx}",
        f"active people: {len(active_tracks)} | known IDs: {len(visible_tracks)} | standing: {standing_count}",
        f"top cumulative standing: {winner_text}",
        "",
        "Ranking by total standing time",
    ]

    # ranking เรียงด้วยเวลายืนนิ่งสะสมรวมจากทุกช่วง
    ranked = sorted(visible_tracks, key=lambda track: track.total_stationary_sec, reverse=True)[:8]
    if not ranked:
        lines.append("no person detected yet")
    for rank, track in enumerate(ranked, 1):
        marker = "*" if winner and track.track_id == winner.track_id else " "
        state = "standing" if track.is_stationary else ("active" if track.track_id in active_tracks else "last seen")
        lines.append(
            f"{marker}{rank}. ID {track.track_id:<4} total {fmt_time(track.total_stationary_sec):>6} "
            f"now {fmt_time(track.current_stationary_sec):>6} {state}"
        )

    font = cv2.FONT_HERSHEY_SIMPLEX
    y = 28
    for idx, line in enumerate(lines):
        scale = 0.62 if idx == 0 else 0.52
        color = (0, 215, 255) if idx == 0 or line.startswith("*") else (255, 255, 255)
        cv2.putText(frame, line, (12, y), font, scale, color, 1, cv2.LINE_AA)
        y += 26 if idx == 0 else 22


def save_id_crop(frame, track: TrackInfo, crop_dir: Path, best_crop_scores: dict[int, float]) -> None:
    # เก็บ crop ที่ดีที่สุดของแต่ละ ID โดยใช้ bbox area * confidence
    # เป็นคะแนนแบบง่าย ๆ เพื่อให้รูปย้อนหลังดีกว่าการเซฟ crop แรกที่เจอ
    x1, y1, x2, y2 = track.bbox
    h, w = frame.shape[:2]
    x1 = max(0, min(w - 1, x1))
    y1 = max(0, min(h - 1, y1))
    x2 = max(x1 + 1, min(w, x2))
    y2 = max(y1 + 1, min(h, y2))
    score = float((x2 - x1) * (y2 - y1)) * max(0.01, float(track.confidence))

    if score <= best_crop_scores.get(track.track_id, 0.0):
        return
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return

    best_crop_scores[track.track_id] = score
    cv2.imwrite(str(crop_dir / f"person_id_{track.track_id:04d}.jpg"), crop)


def show_preview(frame, scale: float) -> bool:
    # กด q หรือ Esc ในหน้าต่าง preview เพื่อหยุดการแสดง preview
    # โดยยังคงปล่อยให้ flow การรันจาก command line ชัดเจน
    preview = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale != 1.0 else frame
    cv2.imshow("YOLO Person ID + Standing Timer", preview)
    key = cv2.waitKey(1) & 0xFF
    return key not in (27, ord("q"), ord("Q"))


def write_outputs(
    out_dir: Path,
    source: Path,
    video_out: Path,
    rows: list[list[object]],
    tracks: dict[int, TrackInfo],
    crop_dir: Path,
    args: argparse.Namespace,
) -> None:
    # เซฟผลลัพธ์ทั้งแบบ machine-readable และ CSV ที่เปิดใน spreadsheet ได้ง่าย
    visible_tracks = [track for track in tracks.values() if not track.is_duplicate_suppressed]
    ranked_tracks = sorted(visible_tracks, key=lambda track: track.total_stationary_sec, reverse=True)
    cumulative_winner = ranked_tracks[0] if ranked_tracks else None
    official_winner = max(visible_tracks, key=lambda track: track.max_stationary_sec) if visible_tracks else None

    summary = {
        "video": str(source),
        "output_video": str(video_out),
        "official_result_longest_stationary_period": {
            "person_id": official_winner.track_id if official_winner else None,
            "duration_sec": round(official_winner.max_stationary_sec, 3) if official_winner else 0.0,
            "duration_str": fmt_time(official_winner.max_stationary_sec) if official_winner else "0.0s",
            "start_frame": official_winner.max_stationary_start_frame if official_winner else None,
            "end_frame": official_winner.max_stationary_end_frame if official_winner else None,
            "crop": str(crop_dir / f"person_id_{official_winner.track_id:04d}.jpg") if official_winner else None,
        },
        "additional_insight_cumulative_standing": {
            "person_id": cumulative_winner.track_id if cumulative_winner else None,
            "total_standing_sec": round(cumulative_winner.total_stationary_sec, 3) if cumulative_winner else 0.0,
            "total_standing_str": fmt_time(cumulative_winner.total_stationary_sec) if cumulative_winner else "0.0s",
            "max_standing_sec": round(cumulative_winner.max_stationary_sec, 3) if cumulative_winner else 0.0,
            "max_standing_str": fmt_time(cumulative_winner.max_stationary_sec) if cumulative_winner else "0.0s",
        },
        "people": [
            {
                "person_id": track.track_id,
                "total_standing_sec": round(track.total_stationary_sec, 3),
                "total_standing_str": fmt_time(track.total_stationary_sec),
                "max_standing_sec": round(track.max_stationary_sec, 3),
                "max_standing_str": fmt_time(track.max_stationary_sec),
                "first_frame": track.first_frame,
                "last_frame": track.last_frame,
                "standing_start_frame": track.max_stationary_start_frame,
                "standing_end_frame": track.max_stationary_end_frame,
                "crop": str(crop_dir / f"person_id_{track.track_id:04d}.jpg"),
            }
            for track in ranked_tracks
        ],
        "parameters": vars(args),
    }

    summary_path = out_dir / f"{source.stem}_standing_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    result_md_path = out_dir / f"{source.stem}_result_summary.md"
    result_md_path.write_text(build_result_markdown(summary, args), encoding="utf-8")

    tracks_csv = out_dir / f"{source.stem}_standing_tracks.csv"
    with tracks_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "source_frame",
                "output_frame",
                "track_id",
                "x1",
                "y1",
                "x2",
                "y2",
                "confidence",
                "is_standing",
                "current_standing_sec",
                "total_standing_sec",
                "max_standing_sec",
            ]
        )
        writer.writerows(rows)

    people_csv = out_dir / f"{source.stem}_people_id_summary.csv"
    with people_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "person_id",
                "total_standing_sec",
                "total_standing_str",
                "max_standing_sec",
                "max_standing_str",
                "first_frame",
                "last_frame",
                "standing_start_frame",
                "standing_end_frame",
                "crop_file",
            ]
        )
        for track in ranked_tracks:
            writer.writerow(
                [
                    track.track_id,
                    round(track.total_stationary_sec, 3),
                    fmt_time(track.total_stationary_sec),
                    round(track.max_stationary_sec, 3),
                    fmt_time(track.max_stationary_sec),
                    track.first_frame,
                    track.last_frame,
                    track.max_stationary_start_frame,
                    track.max_stationary_end_frame,
                    crop_dir / f"person_id_{track.track_id:04d}.jpg",
                ]
            )

    print(f"video: {video_out}")
    print(f"summary: {summary_path}")
    print(f"result markdown: {result_md_path}")
    print(f"tracks csv: {tracks_csv}")
    print(f"people csv: {people_csv}")
    print(f"crops: {crop_dir}")
    print(json.dumps(summary["official_result_longest_stationary_period"], indent=2))


def build_result_markdown(summary: dict[str, object], args: argparse.Namespace) -> str:
    official = summary["official_result_longest_stationary_period"]
    cumulative = summary["additional_insight_cumulative_standing"]
    assert isinstance(official, dict)
    assert isinstance(cumulative, dict)
    return f"""# Result Summary

## Official Result

โจทย์ต้องการหา `person who remained stationary for the longest duration`

ในรายงานนี้ตีความ `stationary duration` เป็นช่วงเวลาที่ยืนนิ่งต่อเนื่องนานที่สุดครั้งเดียว (`longest one time`)

```text
Person ID: {official["person_id"]}
Duration: {official["duration_str"]} ({official["duration_sec"]} seconds)
Start frame: {official["start_frame"]}
End frame: {official["end_frame"]}
Crop: {official["crop"]}
```

## Additional Insight

อีกมุมหนึ่งที่ระบบคำนวณให้ด้วยคือเวลายืนนิ่งสะสมรวม (`standing total`) ของแต่ละ ID

```text
Top cumulative Person ID: {cumulative["person_id"]}
Total standing duration: {cumulative["total_standing_str"]} ({cumulative["total_standing_sec"]} seconds)
Longest single period of that ID: {cumulative["max_standing_str"]} ({cumulative["max_standing_sec"]} seconds)
```

## Method Used

1. ใช้ YOLO ตรวจจับเฉพาะ class `person`
2. ใช้ ByteTrack เพื่อให้แต่ละคนมี `Person ID`
3. แปลง bounding box เป็น centroid
4. เก็บ centroid history ใน sliding window
5. คำนวณ `spread = sqrt(var(x) + var(y))`
6. ถ้า spread ต่ำกว่า threshold และนิ่งครบเวลาขั้นต่ำ จะถือว่า `STANDING`
7. บันทึกทั้ง `longest one time` และ `standing total`

## Parameters

```text
model: {args.model}
conf: {args.conf}
iou: {args.iou}
imgsz: {args.imgsz}
frame_skip: {args.frame_skip}
device: {args.device}
window_size: {args.window_size}
stationary_std_px: {args.stationary_std_px}
min_stationary_sec: {args.min_stationary_sec}
duplicate_iou_threshold: {args.duplicate_iou_threshold}
```

## Limitations

- `Person ID` มาจาก tracker ไม่ใช่ face recognition
- ถ้าคนถูกบังหรือหายจากเฟรม ID อาจเปลี่ยนได้
- วิธีนี้เหมาะกับกล้องนิ่งมากกว่ากล้องเคลื่อนที่
- duplicate suppression ช่วยลด ID ซ้ำในเฟรมเดียวกัน แต่ไม่ใช่ re-identification ระยะยาว
"""


def main() -> None:
    args = parse_args()
    source = Path(args.source)
    out_dir = Path(args.out)
    # สร้างโฟลเดอร์ output ให้อัตโนมัติ ถ้ามีไฟล์ผลลัพธ์ชื่อเดิมอยู่แล้ว
    # writer ของ OpenCV/CSV/JSON จะเขียนทับตามปกติ
    out_dir.mkdir(parents=True, exist_ok=True)
    crop_dir = out_dir / "person_id_crops"
    crop_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {source}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    output_fps = max(1.0, fps / max(1, args.frame_skip))

    # วิดีโอ output ใช้ resolution เท่าต้นฉบับ ถ้าใช้ frame skip จะลด FPS output
    # เพื่อให้ความเร็ว playback ใกล้เคียงกับเวลาจริงของวิดีโอเดิม
    video_out = out_dir / f"{source.stem}_standing_detail.mp4"
    writer = cv2.VideoWriter(str(video_out), cv2.VideoWriter_fourcc(*"mp4v"), output_fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Cannot create output video: {video_out}")

    tracker = LongestStayTracker(
        model_path=args.model,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        device=args.device,
        window_size=args.window_size,
        stationary_std_px=args.stationary_std_px,
        min_stationary_sec=args.min_stationary_sec,
        max_lost_frames=args.max_lost_frames,
        duplicate_iou_threshold=args.duplicate_iou_threshold,
    )

    rows: list[list[object]] = []
    best_crop_scores: dict[int, float] = {}
    frame_idx = -1
    processed = 0
    keep_showing = True

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame_idx += 1
        if frame_idx % max(1, args.frame_skip) != 0:
            continue
        if args.limit_frames and processed >= args.limit_frames:
            break

        # track คนที่ active ในเฟรมนี้ อัปเดต timer แล้วค่อยวาด overlay
        active_tracks = tracker.update(frame, frame_idx=frame_idx, fps=fps)
        cumulative_winner = tracker.cumulative_winner()
        winner_id = cumulative_winner.track_id if cumulative_winner and cumulative_winner.total_stationary_sec > 0 else None

        for track in active_tracks.values():
            save_id_crop(frame, track, crop_dir, best_crop_scores)
            draw_box(frame, track, winner_id)
            rows.append(
                [
                    frame_idx,
                    processed,
                    track.track_id,
                    *track.bbox,
                    track.confidence,
                    track.is_stationary,
                    round(track.current_stationary_sec, 4),
                    round(track.total_stationary_sec, 4),
                    round(track.max_stationary_sec, 4),
                ]
            )

        draw_panel(frame, active_tracks, tracker.tracks, cumulative_winner, frame_idx, fps)
        writer.write(frame)

        if args.show and keep_showing:
            # ปิด preview ด้วย q/Esc จะไม่ยกเลิกการสร้าง output video
            keep_showing = show_preview(frame, args.show_scale)

        processed += 1
        if processed % 100 == 0:
            print(f"{source.name}: processed {processed} frames from {frame_idx + 1}/{total_frames}")

    cap.release()
    writer.release()
    if args.show:
        cv2.destroyAllWindows()

    write_outputs(out_dir, source, video_out, rows, tracker.tracks, crop_dir, args)


if __name__ == "__main__":
    main()
