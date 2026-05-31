## สิ่งที่ระบบทำ

1. อ่านวิดีโอด้วย OpenCV ทีละเฟรม
2. ตรวจจับเฉพาะ class `person` ด้วย Ultralytics YOLO
3. ติดตามคนแต่ละคนด้วย ByteTrack เพื่อให้ได้ `Person ID`
4. วิเคราะห์ centroid history ของแต่ละ ID ว่ายืนนิ่งหรือเคลื่อนที่
5. นับเวลายืนนิ่งแบบสะสมต่อ ID
6. วาด overlay ลงวิดีโอ
7. แสดง preview ระหว่างรันเมื่อใช้ `--show`
8. เซฟวิดีโอ, JSON summary, CSV รายเฟรม, CSV สรุปต่อ ID, และรูป crop ของแต่ละ ID

## ไฟล์หลัก

```text
Standing-ID-Detection-NoBat/
  main.py
  yolo_standing_detail_video.py
  longest_stay_detection/
    __init__.py
    config.py
    tracker.py
  docs/
    INTERVIEW_EXPLANATION_DEEP_DIVE_TH.md
    METHODS_AND_TECHNIQUES_TH.md
    PROJECT_BRIEF_TH.md
    TECHNICAL_SUMMARY_TH.md
  README_TH.md
  requirements.txt
```

## Input เริ่มต้น

ค่า default ใช้วิดีโอ:

```text
C:\Users\Chokhun\Downloads\entrance.mov
```

## วิธีรันแบบ default

เปิด PowerShell ในโฟลเดอร์โปรเจกต์ แล้วรัน:

```powershell
& "C:\Users\Chokhun\Downloads\Car-Counting-1\.yolo_env\Scripts\python.exe" .\main.py
```

คำสั่งนี้จะ:

- ใช้ `entrance.mov`
- เปิด preview
- เซฟ output ไปที่ `standing_detail_output_live`
- ใช้ confidence `0.35`
- ใช้ image size `640`
- ใช้ GPU device `0`
- ซ่อน duplicate ID ที่ box ทับกันมากด้วย `duplicate-iou-threshold=0.85`

## วิธีรันโดยระบุวิดีโอเอง

```powershell
& "C:\Users\Chokhun\Downloads\Car-Counting-1\.yolo_env\Scripts\python.exe" .\yolo_standing_detail_video.py --source "PATH_TO_VIDEO.mov" --out standing_detail_output_live --show
```

## วิธีรันแบบไม่เปิด preview

ตัด `--show` ออก:

```powershell
& "C:\Users\Chokhun\Downloads\Car-Counting-1\.yolo_env\Scripts\python.exe" .\yolo_standing_detail_video.py --source "C:\Users\Chokhun\Downloads\entrance.mov" --out standing_detail_output_live
```

## วิธีทดสอบเร็ว

ใช้ `--limit-frames` เพื่อประมวลผลแค่บางส่วน:

```powershell
& "C:\Users\Chokhun\Downloads\Car-Counting-1\.yolo_env\Scripts\python.exe" .\yolo_standing_detail_video.py --source "C:\Users\Chokhun\Downloads\entrance.mov" --out standing_detail_output_test --limit-frames 120
```

## Output

หลังรันเสร็จจะได้:

```text
standing_detail_output_live/
  entrance_standing_detail.mp4
  entrance_result_summary.md
  entrance_standing_summary.json
  entrance_standing_tracks.csv
  entrance_people_id_summary.csv
  person_id_crops/
```

ความหมาย:

- `entrance_standing_detail.mp4` วิดีโอที่มีกรอบคน, ID, confidence, สถานะ, และเวลายืน
- `entrance_result_summary.md` สรุปคำตอบหลักสำหรับโจทย์และ insight เสริม
- `entrance_standing_summary.json` summary รวมและ parameter ที่ใช้
- `entrance_standing_tracks.csv` ข้อมูลรายเฟรมของแต่ละ track
- `entrance_people_id_summary.csv` สรุปต่อ Person ID เรียงตามเวลายืนนิ่งสะสม
- `person_id_crops/` รูป crop ที่ดีที่สุดของแต่ละ Person ID

## Overlay บนวิดีโอ

บนกรอบของแต่ละคนจะแสดง:

- `YOLO person confidence`
- `Person ID`
- `STANDING` หรือ `moving/checking`
- `standing now`
- `standing total`
- `longest one time`

ฝั่งซ้ายของวิดีโอมี panel สรุป:

- เวลาของวิดีโอ ณ เฟรมนั้น
- จำนวนคน active
- จำนวน ID ทั้งหมดที่เคยเห็น
- จำนวนคนที่กำลังยืนนิ่ง
- ranking ของ ID ที่ยืนนิ่งสะสมมากที่สุด

## วิธีอ่านผลสำหรับส่งงาน

โจทย์ใช้คำว่า `stationary period` ดังนั้นคำตอบหลักของโปรเจกต์นี้จะใช้:

```text
longest one time
```

หรือช่วงยืนนิ่งต่อเนื่องครั้งเดียวที่นานที่สุด

ส่วน:

```text
standing total
```

เป็น insight เสริมที่บอกว่า ID นั้นยืนนิ่งสะสมรวมทุกช่วงกี่วินาที

หลังรันเสร็จให้ดูไฟล์:

```text
entrance_result_summary.md
```

ไฟล์นี้จะสรุป official result, duration, additional insight, method used และ limitations

## หมายเหตุเรื่องโมเดล

ระบบจะหาโมเดลตามลำดับนี้:

1. `yolo26s.pt` ในโฟลเดอร์โปรเจกต์นี้
2. `C:\Users\Chokhun\Downloads\Standing-ID-Detection\yolo26s.pt`
3. ชื่อ `yolo26s.pt` ให้ Ultralytics จัดการเอง

ถ้าต้องการให้โฟลเดอร์ใหม่นี้ standalone ให้ copy `yolo26s.pt` มาไว้ในโฟลเดอร์โปรเจกต์นี้

## ข้อจำกัด

- `Person ID` มาจาก ByteTrack ไม่ใช่การจำหน้าคนจริง
- ถ้าคนถูกบัง หายจากเฟรม หรือเดินผ่านกันใกล้มาก ID อาจเปลี่ยนได้
- ระบบไม่ได้ทำ face recognition หรือ re-identification ระยะยาว
- preview อาจไม่คมเท่าวิดีโอที่เซฟถ้าใช้ `--show-scale 0.60`

## เอกสารเพิ่มเติม

- `docs/INTERVIEW_EXPLANATION_DEEP_DIVE_TH.md` อธิบายแนวคิดแบบละเอียดสำหรับใช้เตรียม interview ตั้งแต่ pipeline, เทคนิค, insight, การ debug, ข้อจำกัด และคำตอบที่ควรพูด
- `docs/PROJECT_BRIEF_TH.md` สรุปโจทย์, objective, rules, output ที่ต้องการ, สิ่งที่ต้อง submit และเกณฑ์ประเมินเป็นภาษาไทย
- `docs/METHODS_AND_TECHNIQUES_TH.md` อธิบายโมเดล YOLO, ByteTrack, centroid history, spread, การล็อคตำแหน่ง, การนับเวลา, output และข้อจำกัดแบบละเอียด
- `docs/TECHNICAL_SUMMARY_TH.md` สรุปสิ่งที่สร้างไว้ในเวอร์ชันนี้
