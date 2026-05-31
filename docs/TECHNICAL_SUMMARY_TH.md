# Technical Summary - No `.bat`

## สรุปสิ่งที่ทำไปแล้ว

สร้างโปรเจกต์ใหม่ชื่อ `Standing-ID-Detection-NoBat` โดยอิงจาก requirement เดิม แต่ปรับให้ใช้งานผ่าน command เท่านั้น และไม่มีไฟล์ `.bat`

สิ่งที่เพิ่ม/ปรับในเวอร์ชันนี้:

- สร้างโครงโปรเจกต์ใหม่สะอาด
- แยก config ไว้ใน `longest_stay_detection/config.py`
- แยก logic tracking และ standing timer ไว้ใน `longest_stay_detection/tracker.py`
- เพิ่ม `main.py` เป็น command entry point แบบ default
- เพิ่ม `yolo_standing_detail_video.py` เป็นตัวประมวลผลหลัก
- เพิ่ม fallback การหาโมเดล `yolo26s.pt`
- เพิ่ม duplicate track suppression ด้วย IoU เพื่อลด ID ซ้ำของคนเดียวกันในเฟรมเดียวกัน
- เพิ่ม official result แยกจาก insight โดยใช้ `max_stationary_sec` เป็นคำตอบหลักของโจทย์
- เพิ่มการ generate `*_result_summary.md` หลังรัน
- เพิ่มเอกสาร `README_TH.md`
- เพิ่มเอกสาร technical summary นี้
- ไม่สร้าง `run_show.bat`
- ไม่สร้าง `run_generate_video.bat`

## Pipeline

```text
video input
  -> OpenCV VideoCapture
  -> YOLO detect class person
  -> ByteTrack assign Person ID
  -> centroid history per ID
  -> stationary/moving decision
  -> cumulative standing timer
  -> draw bounding boxes and summary panel
  -> write MP4
  -> write JSON/CSV/crops
```

## Stationary logic

สำหรับแต่ละ `Person ID` ระบบเก็บ centroid ของ bounding box ใน sliding window

```text
cx = (x1 + x2) / 2
cy = (y1 + y2) / 2
spread = sqrt(var(x) + var(y))
```

ถ้า `spread < stationary_std_px` และมี history มากพอ จะถือว่า track นั้นนิ่ง

ค่า default:

- `window_size`: `30` เฟรม
- `stationary_std_px`: `12.0`
- `min_stationary_sec`: `2.0`

## Timer logic

ใน `TrackInfo` มีค่าหลัก:

- `current_stationary_sec`: เวลายืนนิ่งของรอบปัจจุบัน
- `total_stationary_sec`: เวลายืนนิ่งสะสมของ ID นั้น
- `max_stationary_sec`: เวลายืนนิ่งต่อเนื่องนานสุดครั้งเดียว

เมื่อนิ่งครบ `min_stationary_sec` ระบบจะเริ่มนับเวลายืน และบวกเข้า `total_stationary_sec`

เมื่อ ID เดินหรือหลุดจากเงื่อนไขนิ่ง:

- reset `current_stationary_sec`
- เก็บ `total_stationary_sec` ไว้
- ถ้ากลับมานิ่งอีก จะนับรอบใหม่และบวกสะสมต่อ

## Ranking

Ranking ใน panel ฝั่งซ้ายและ CSV summary เรียงด้วย:

```text
total_stationary_sec
```

ดังนั้นอันดับ 1 คือ ID ที่ยืนนิ่งสะสมรวมมากที่สุด ไม่ใช่ ID ที่ยืนต่อเนื่องรอบเดียวนานที่สุด

## Command arguments สำคัญ

- `--source`: path วิดีโอ input
- `--out`: output folder
- `--model`: YOLO model path
- `--conf`: confidence threshold
- `--iou`: IoU threshold
- `--imgsz`: image size
- `--frame-skip`: ข้ามเฟรมเพื่อเร่งประมวลผล
- `--device`: `0` สำหรับ GPU หรือ `cpu`
- `--show`: เปิด preview
- `--show-scale`: ขนาด preview
- `--limit-frames`: จำกัดจำนวนเฟรมสำหรับทดสอบเร็ว
- `--window-size`: ขนาด sliding window
- `--stationary-std-px`: threshold ความนิ่ง
- `--min-stationary-sec`: เวลาขั้นต่ำก่อนถือว่า standing
- `--max-lost-frames`: จำนวนเฟรมที่ยอมให้ track หายก่อนถือว่า stale
- `--duplicate-iou-threshold`: threshold สำหรับซ่อน ID ซ้ำเมื่อ box ทับกันมากในเฟรมเดียวกัน

## Dependencies

ดูใน `requirements.txt`

ถ้าใช้ environment เดิมจาก `Car-Counting-1\.yolo_env` และมี ultralytics/opencv/numpy/torch แล้ว ไม่จำเป็นต้องติดตั้งใหม่
