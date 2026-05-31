# โจทย์โปรเจกต์: Stationary Person Detection

## Objective

โจทย์นี้ใช้เพื่อประเมินความสามารถด้าน:

- Python
- Computer Vision
- การออกแบบระบบ AI
- การแก้ปัญหา
- คุณภาพของโค้ด
- การอธิบายแนวคิดและข้อจำกัดของวิธีที่ใช้

## Dataset

ใช้ชุดข้อมูลวิดีโอ:

```text
VDO Dataset
```

ในโปรเจกต์นี้ใช้วิดีโอเริ่มต้น:

```text
C:\Users\Chokhun\Downloads\entrance.mov
```

## Rules

สามารถใช้เครื่องมือใดก็ได้ เช่น:

- Python
- OpenCV
- YOLO
- Ultralytics
- LLM / AI tools
- เครื่องมืออื่น ๆ ที่ช่วยพัฒนา solution

แต่ผู้เข้าสอบ/ผู้พัฒนาต้องสามารถอธิบายแนวทาง วิธีคิด และข้อจำกัดของระบบได้ด้วยตัวเองในรอบ interview ถัดไป

## Task ที่เลือก

เลือกทำข้อ:

```text
A) Develop a system to identify:
   1. The person who remained stationary for the longest duration
   2. The duration of the stationary period
```

แปลเป็นไทย:

```text
พัฒนาระบบเพื่อระบุว่า:
1. คนใดอยู่นิ่ง/ยืนนิ่งนานที่สุด
2. ระยะเวลาที่ยืนนิ่งนานเท่าไร
```

## Desired Output

ระบบควรให้ผลลัพธ์หลักดังนี้:

1. ระบุคนที่อยู่นิ่งนานที่สุด
2. ระบุระยะเวลาของช่วงที่อยู่นิ่ง
3. อธิบายวิธีที่ใช้วัดระยะเวลานิ่ง
4. มีวิดีโอผลลัพธ์ที่แสดงการตรวจจับและ tracking
5. มีไฟล์สรุปผล เช่น CSV หรือ JSON

ในโปรเจกต์นี้ คำตอบหลักสำหรับโจทย์ใช้ค่า:

```text
longest one time
```

เพราะโจทย์พูดถึง `stationary period` ซึ่งหมายถึงช่วงเวลานิ่งต่อเนื่องหนึ่งช่วง

ส่วน `standing total` เป็น insight เสริมสำหรับดูพฤติกรรมสะสมของแต่ละ ID

## สิ่งที่ต้อง Submit

ต้องส่ง:

1. Source code
2. README
3. `requirements.txt`
4. VDO result
5. Short idea description

สำหรับโปรเจกต์นี้ ไฟล์ที่เกี่ยวข้องคือ:

```text
Standing-ID-Detection-NoBat/
  main.py
  yolo_standing_detail_video.py
  longest_stay_detection/
    config.py
    tracker.py
  README_TH.md
  requirements.txt
  docs/
    PROJECT_BRIEF_TH.md
    METHODS_AND_TECHNIQUES_TH.md
    TECHNICAL_SUMMARY_TH.md
```

หลังรันระบบจะได้ output เช่น:

```text
standing_detail_output_live/
  entrance_standing_detail.mp4
  entrance_result_summary.md
  entrance_standing_summary.json
  entrance_standing_tracks.csv
  entrance_people_id_summary.csv
  person_id_crops/
```

## Evaluation Criteria

เกณฑ์ประเมิน:

- Idea: แนวคิดเหมาะสมกับโจทย์หรือไม่
- Code quality: โค้ดอ่านง่าย แยกส่วนดี ปรับค่าได้ และ maintain ได้หรือไม่
- Solution design: ออกแบบ pipeline และ output ได้ครบถ้วนหรือไม่
- Reasoning: อธิบายเหตุผลของวิธีที่เลือกได้หรือไม่
- Creativity: มีการเสริมแนวคิดหรือ output ที่ช่วยให้ตรวจสอบผลได้ดีขึ้นหรือไม่

## แนวทาง Solution ของโปรเจกต์นี้

แนวทางที่ใช้คือ:

```text
YOLO person detection
  + ByteTrack person tracking
  + centroid movement analysis
  + cumulative stationary timer
```

อธิบายสั้น ๆ:

1. ใช้ YOLO ตรวจจับคนในแต่ละเฟรม
2. ใช้ ByteTrack ให้ `Person ID` กับคนแต่ละคน
3. เก็บตำแหน่ง centroid ของ bounding box ย้อนหลัง
4. วัดว่าคนขยับมากน้อยแค่ไหนด้วยค่า spread
5. ถ้า centroid แทบไม่เคลื่อนที่ต่อเนื่องเกินเวลาที่กำหนด ถือว่า `STANDING`
6. นับเวลายืนนิ่งของแต่ละ ID
7. สรุปว่า ID ใดยืนนิ่งสะสมมากที่สุด และยืนต่อเนื่องนานที่สุดเท่าไร

## Method Used to Measure Stationary Duration

ระบบวัดความนิ่งจาก centroid ของ bounding box

สำหรับแต่ละคน:

```text
cx = (x1 + x2) / 2
cy = (y1 + y2) / 2
```

จากนั้นเก็บค่า `(cx, cy)` ใน sliding window แล้วคำนวณ:

```text
spread = sqrt(var(x) + var(y))
```

ถ้า:

```text
spread < stationary_std_px
```

แปลว่าคนนั้นเคลื่อนที่น้อยมาก

และถ้านิ่งต่อเนื่องนานกว่า:

```text
min_stationary_sec
```

ระบบจะถือว่าอยู่ในสถานะ:

```text
STANDING
```

เวลาที่นับมี 3 ค่า:

- `standing now`: เวลานิ่งของรอบปัจจุบัน
- `standing total`: เวลานิ่งสะสมทั้งหมดของ Person ID นั้น
- `longest one time`: เวลานิ่งต่อเนื่องนานสุดครั้งเดียว

คำตอบหลักที่ใช้ส่งคือ `longest one time`

## จุดที่เสริมเพื่อให้ตรวจสอบง่ายขึ้น

โปรเจกต์นี้เสริมจากโจทย์หลักด้วย:

- แสดง overlay บนวิดีโอ
- แสดง panel สรุปฝั่งซ้าย
- แสดง ranking ตามเวลานิ่งสะสม
- เซฟ crop ของแต่ละ Person ID
- เซฟ CSV รายเฟรม
- เซฟ CSV สรุปต่อ ID
- เซฟ JSON summary
- เพิ่ม duplicate ID suppression เพื่อลดกรณีคนเดียวกันถูกสร้างเป็นหลาย ID ในเฟรมเดียวกัน
- ไม่มีไฟล์ `.bat` ใช้ command line ตรง ๆ

## ข้อจำกัดที่ควรอธิบายได้ใน Interview

1. `Person ID` มาจาก ByteTrack ไม่ใช่ face recognition
2. ถ้าคนถูกบัง หายจากเฟรม หรือเดินผ่านกันใกล้มาก ID อาจเปลี่ยนได้
3. ถ้ากล้องเคลื่อนที่ ระบบอาจเข้าใจว่าคนขยับ เพราะ centroid เปลี่ยนตามภาพ
4. Threshold เช่น `stationary_std_px` และ `min_stationary_sec` ต้องจูนตามวิดีโอจริง
5. Duplicate ID suppression ช่วยกรณี box ทับกันมากในเฟรมเดียวกัน แต่ไม่ได้แก้ re-identification ระยะยาว
