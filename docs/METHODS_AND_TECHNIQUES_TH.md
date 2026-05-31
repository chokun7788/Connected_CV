# Methods and Techniques - Standing ID Detection

เอกสารนี้อธิบายหลักการและเทคนิคของระบบตั้งแต่การใช้โมเดล YOLO, การ tracking ด้วย ByteTrack, การวิเคราะห์ตำแหน่ง, การตัดสินว่า "ยืนนิ่ง", และการนับเวลายืนสะสมต่อ `Person ID`

## 1. เป้าหมายของระบบ

ระบบนี้ไม่ได้พยายามจำตัวบุคคลด้วยใบหน้า แต่พยายามตอบคำถามว่า:

```text
ในวิดีโอนี้ แต่ละ tracking ID ยืนนิ่ง/ค้างอยู่ตำแหน่งเดิมนานเท่าไร
```

ดังนั้นคำว่า `Person ID` ในระบบนี้หมายถึง ID ที่ตัว tracker ให้มา ไม่ใช่ identity จริงของคนแบบ face recognition

## 2. ภาพรวม Pipeline

```text
วิดีโอ input
  -> อ่านเฟรมด้วย OpenCV
  -> YOLO ตรวจจับเฉพาะ class person
  -> ByteTrack เชื่อม detections ระหว่างเฟรมเป็น Person ID
  -> เก็บ centroid history ของแต่ละ ID
  -> วัดความนิ่งจากการกระจายของ centroid
  -> ตัดสินสถานะ STANDING หรือ moving/checking
  -> นับ standing now, standing total, longest one time
  -> วาด overlay และ panel
  -> เซฟวิดีโอ, CSV, JSON, crop ต่อ ID
```

## 3. Model ที่ใช้

### 3.1 YOLO

ระบบใช้ Ultralytics YOLO ผ่าน package `ultralytics`

โมเดล default:

```text
yolo26s.pt
```

หน้าที่ของ YOLO คือหา bounding box ของคนในแต่ละเฟรม โดยระบบกำหนดให้ตรวจเฉพาะ class `person`

ในโค้ดใช้:

```python
classes=[0]
```

สำหรับ COCO class index:

```text
0 = person
```

ผลลัพธ์สำคัญจาก YOLO:

- bounding box: `x1, y1, x2, y2`
- confidence: ความมั่นใจว่า box นั้นเป็นคน
- class: ใช้เฉพาะ person

### 3.2 เหตุผลที่ใช้ YOLO

YOLO เหมาะกับงานนี้เพราะ:

- ตรวจจับวัตถุแบบ real-time ได้ดี
- มี implementation พร้อมใช้ใน Ultralytics
- ใช้ร่วมกับ ByteTrack ได้ง่ายผ่าน `model.track(...)`
- เหมาะกับงานที่ต้อง overlay กรอบคนลงวิดีโอ

### 3.3 ค่า YOLO default

```text
confidence threshold = 0.35
IoU threshold        = 0.45
image size           = 640
device               = 0
```

ความหมาย:

- `conf=0.35`: box ที่มั่นใจต่ำกว่า 0.35 จะถูกตัดออก
- `iou=0.45`: ใช้ควบคุมการซ้อนทับของ box และ tracking
- `imgsz=640`: resize ภาพก่อนเข้าโมเดลเพื่อบาลานซ์ความเร็ว/ความแม่น
- `device=0`: ใช้ GPU ตัวแรก ถ้าไม่มี GPU ให้ใช้ `--device cpu`

## 4. Tracking ด้วย ByteTrack

### 4.1 ByteTrack คืออะไร

ByteTrack เป็น object tracking algorithm ที่เชื่อม detection ของวัตถุในแต่ละเฟรมเข้าด้วยกัน เพื่อให้วัตถุเดิมมี ID เดิมต่อเนื่อง

ในระบบนี้ ByteTrack ทำหน้าที่ให้:

```text
คนในเฟรมที่ 1 -> Person ID 3
คนเดิมในเฟรมที่ 2 -> Person ID 3
คนเดิมในเฟรมที่ 3 -> Person ID 3
```

ถ้าไม่มี tracking ระบบจะรู้แค่ว่าแต่ละเฟรมมีคนกี่คน แต่จะไม่รู้ว่า box ไหนคือคนเดิมจากเฟรมก่อน

### 4.2 วิธีเรียกใช้ในโค้ด

ระบบเรียกผ่าน Ultralytics:

```python
model.track(
    source=frame,
    persist=True,
    tracker="bytetrack.yaml",
    classes=[0],
)
```

ความหมาย:

- `source=frame`: ส่งภาพทีละเฟรมเข้าโมเดล
- `persist=True`: ให้ tracker จำ state ต่อเนื่องระหว่างเฟรม
- `tracker="bytetrack.yaml"`: ใช้ ByteTrack config ของ Ultralytics
- `classes=[0]`: track เฉพาะคน

### 4.3 ข้อจำกัดของ Person ID

`Person ID` ไม่ใช่การจำหน้าหรือจำคนจริง 100%

ID อาจเปลี่ยนได้ถ้า:

- คนถูกบังนาน
- คนหายออกจากเฟรมแล้วกลับมา
- คนเดินผ่านกันใกล้มาก
- detection ขาดหายหลายเฟรม
- มุมกล้องหรือแสงทำให้ YOLO detect ไม่สม่ำเสมอ

ดังนั้นคำว่า "ล็อคคนเดิม" ในระบบนี้หมายถึง:

```text
ByteTrack ยังสามารถเชื่อม track เดิมได้อย่างต่อเนื่อง
```

ไม่ใช่การ re-identification แบบจำเสื้อผ้า/ใบหน้าข้ามช่วงยาว

## 5. เทคนิคการล็อคตำแหน่ง

ระบบไม่ได้ล็อคตำแหน่งด้วยการกำหนด ROI ตายตัว แต่ใช้การ "ติดตามตำแหน่งของ Person ID" จาก bounding box แล้วดูว่าจุดกลางของ box เคลื่อนที่มากน้อยแค่ไหน

### 5.1 Bounding Box

YOLO ให้กรอบคนในรูปแบบ:

```text
x1, y1 = มุมซ้ายบน
x2, y2 = มุมขวาล่าง
```

ตัวอย่าง:

```text
(x1, y1, x2, y2) = (420, 130, 510, 430)
```

### 5.2 Centroid

ระบบแปลง bounding box เป็นจุดกลาง หรือ centroid:

```text
cx = (x1 + x2) / 2
cy = (y1 + y2) / 2
```

เช่น:

```text
x1 = 420, x2 = 510 -> cx = 465
y1 = 130, y2 = 430 -> cy = 280
```

จุด `cx, cy` นี้คือ "ตำแหน่งตัวแทน" ของคนคนนั้นในเฟรม

### 5.3 Centroid History

สำหรับแต่ละ `Person ID` ระบบเก็บ centroid ย้อนหลังไว้ใน sliding window

ค่า default:

```text
window_size = 30 เฟรม
```

ถ้าวิดีโอ 30 FPS ค่า 30 เฟรมคือประมาณ 1 วินาทีล่าสุด

ตัวอย่าง history:

```text
ID 12:
  frame 100 -> (465, 280)
  frame 101 -> (466, 281)
  frame 102 -> (464, 279)
  frame 103 -> (465, 280)
```

ถ้าค่า centroid กระจุกอยู่ใกล้กัน แปลว่าคนน่าจะยืนนิ่ง

ถ้าค่า centroid เปลี่ยนไปไกลเรื่อย ๆ แปลว่าคนน่าจะเดินหรือเคลื่อนที่

### 5.4 Spread

ระบบวัดความกระจายของ centroid ด้วยสูตร:

```text
spread = sqrt(var(x) + var(y))
```

โดย:

- `var(x)` คือ variance ของค่า x ใน history
- `var(y)` คือ variance ของค่า y ใน history
- `spread` ต่ำ = จุด centroid เกาะกลุ่มกัน = แทบไม่ขยับ
- `spread` สูง = centroid กระจาย = เคลื่อนที่

### 5.5 Threshold ความนิ่ง

ค่า default:

```text
stationary_std_px = 12.0
```

ระบบตัดสินว่า "นิ่ง" เมื่อ:

```text
spread < stationary_std_px
```

แปลว่า centroid ของคนคนนั้นกระจายตัวน้อยกว่า 12 pixels ในช่วง sliding window

### 5.6 ทำไมใช้ centroid แทนการดู box ทั้งกล่อง

เพราะ bounding box อาจสั่นเล็กน้อยจาก YOLO แม้คนไม่ขยับ เช่น:

- ขอบ box กว้าง/แคบขึ้นเล็กน้อย
- หัวหรือขาขยับนิดหน่อย
- confidence เปลี่ยน
- pose เปลี่ยนนิดเดียว

การใช้ centroid ทำให้ระบบสนใจการเคลื่อนที่ของตำแหน่งโดยรวมมากกว่า noise ของขอบ box

## 5.7 การกรอง ID ซ้ำในเฟรมเดียวกัน

ในบางเฟรม YOLO + ByteTrack อาจสร้าง ID ซ้ำของคนเดียวกันได้ เช่น คนจริงคือ `ID 46` แต่มี `ID 164` หรือ `ID 177` โผล่มาทับในตำแหน่งเดียวกัน 1-5 เฟรม

สาเหตุคือ detector อาจปล่อย box ซ้ำหรือใกล้เคียงกันมาก และ tracker เข้าใจว่าเป็น track ใหม่

เวอร์ชันนี้เพิ่มขั้นตอนหลัง ByteTrack:

```text
ถ้าในเฟรมเดียวกันมี 2 ID ที่ bounding box ทับกันมาก
  คำนวณ IoU ของ box ทั้งสอง
  ถ้า IoU >= duplicate_iou_threshold
    เก็บ ID ที่ดูเป็น track หลัก
    ซ่อน ID ที่น่าจะเป็น duplicate
```

ค่า default:

```text
duplicate_iou_threshold = 0.85
```

### IoU คืออะไร

IoU ย่อมาจาก Intersection over Union ใช้วัดว่า box สองอันทับกันมากแค่ไหน

```text
IoU = พื้นที่ซ้อนทับ / พื้นที่รวมของ box ทั้งสอง
```

ถ้า IoU ใกล้ 1.0 แปลว่า box แทบจะทับตำแหน่งเดียวกัน

ในเคสคนเดียวกันถูกแยกเป็นหลาย ID มักเห็น box ทับกันสูงมาก เช่น `0.85+`

### เลือกเก็บ ID ไหน

ถ้ามี 2 ID ทับกันมาก ระบบจะให้คะแนนแบบง่าย ๆ แล้วเก็บ track ที่น่าจะเป็น ID จริงกว่า โดยดูจาก:

1. track อยู่มานานกว่า
2. มี `total_stationary_sec` มากกว่า
3. มี `max_stationary_sec` มากกว่า
4. confidence สูงกว่า
5. ถ้ายังเท่ากัน ให้ ID เลขน้อยกว่าได้เปรียบ

เหตุผลคือ ID จริงมักอยู่ต่อเนื่องมาหลายเฟรม ส่วน duplicate มักเพิ่งโผล่แค่ 1-5 เฟรม

### ผลของการกรอง duplicate

ID ที่ถูกมองว่าเป็น duplicate จะไม่ถูกนำไป:

- วาด overlay
- บันทึกลง CSV รายเฟรม
- เซฟ crop
- แสดงใน ranking
- แสดงใน summary หลัก

แต่ไม่ได้ลบข้อมูล track ทิ้งทันที เพราะถ้าในอนาคต ID นั้นไม่ทับกับ track อื่นอีก ระบบยังมีโอกาสใช้ต่อได้

## 6. วิธีตัดสิน STANDING

ระบบไม่ได้ขึ้นสถานะ `STANDING` ทันทีที่ centroid นิ่ง แต่ต้องนิ่งต่อเนื่องเกินเวลาขั้นต่ำก่อน

ค่า default:

```text
min_stationary_sec = 2.0
```

Logic:

```text
ถ้า spread ต่ำกว่า threshold:
  เริ่มจับเวลาช่วงนิ่ง

ถ้านิ่งต่อเนื่องครบ 2 วินาที:
  สถานะ = STANDING
  เริ่มแสดง standing now
  เริ่มบวกเข้า standing total

ถ้าขยับ:
  สถานะ = moving/checking
  reset standing now
  total standing ยังเก็บไว้
```

เหตุผลที่ต้องมี `min_stationary_sec`:

- ลด false positive จากคนเดินช้า
- ลดการนับช่วงหยุดสั้นมาก ๆ
- ให้สถานะ STANDING มีความหมายว่า "ค้างอยู่จริง"

## 7. การนับเวลา

ระบบมีเวลาหลัก 3 ค่า

### 7.1 standing now

```text
current_stationary_sec
```

คือเวลายืนนิ่งในรอบปัจจุบัน

ถ้าคนขยับ ค่านี้ reset เป็น 0

### 7.2 standing total

```text
total_stationary_sec
```

คือเวลายืนนิ่งสะสมทั้งหมดของ ID นั้น

ตัวอย่าง:

```text
ID 12 ยืนนิ่ง 5 วินาที
ID 12 เดิน
ID 12 กลับมายืนนิ่งอีก 3 วินาที
standing total = 8 วินาที
```

### 7.3 longest one time

```text
max_stationary_sec
```

คือช่วงยืนนิ่งต่อเนื่องที่นานที่สุดครั้งเดียว

จากตัวอย่างด้านบน:

```text
standing total    = 8 วินาที
longest one time  = 5 วินาที
```

### 7.4 ค่าไหนใช้เป็นคำตอบหลักของโจทย์

โจทย์ใช้คำว่า:

```text
stationary period
```

ดังนั้นคำตอบหลักของระบบนี้เลือกใช้:

```text
longest one time
```

เพราะเป็น "ช่วงเวลานิ่งต่อเนื่อง" ที่นานที่สุดครั้งเดียว

ส่วน `standing total` ถูกเก็บเป็น additional insight เพราะช่วยตอบอีกมุมว่า ID ไหนมีเวลายืนนิ่งสะสมรวมมากที่สุด

## 8. การจัดอันดับใน panel

Panel ฝั่งซ้ายเรียงอันดับด้วย:

```text
total_stationary_sec
```

ดังนั้นอันดับ 1 คือ ID ที่ยืนนิ่งสะสมรวมมากที่สุด

ไม่ใช่:

- ID ที่ยืนต่อเนื่องรอบเดียวนานที่สุด
- ID ที่อยู่ในเฟรมนานที่สุด
- ID ที่ confidence สูงที่สุด

หมายเหตุ: ranking ใน panel เป็น insight ระหว่างดูวิดีโอ ส่วนคำตอบหลักหลังรันให้ดูใน `*_result_summary.md` ซึ่งเลือกจาก `longest one time`

## 9. การทำ Crop ต่อ Person ID

ระบบเซฟรูป crop ของแต่ละ `Person ID` ลงใน:

```text
person_id_crops/
```

ชื่อไฟล์:

```text
person_id_0001.jpg
person_id_0002.jpg
...
```

### 9.1 เลือก crop ที่ดีที่สุดอย่างไร

ระบบไม่ได้เซฟ crop แรกที่เจอเสมอ แต่ให้คะแนน crop ด้วย:

```text
score = bbox_area * confidence
```

โดย:

- box ใหญ่กว่า มักเห็นคนชัดกว่า
- confidence สูงกว่า มักน่าเชื่อถือกว่า

ถ้าเจอ crop ใหม่ของ ID เดิมที่ score ดีกว่า จะเขียนทับรูปเดิม

## 10. Output ที่ได้

```text
standing_detail_output_live/
  entrance_standing_detail.mp4
  entrance_result_summary.md
  entrance_standing_summary.json
  entrance_standing_tracks.csv
  entrance_people_id_summary.csv
  person_id_crops/
```

### 10.1 MP4

`entrance_standing_detail.mp4`

วิดีโอ output พร้อม overlay:

- bounding box
- Person ID
- confidence
- status
- standing now
- standing total
- longest one time
- summary panel ฝั่งซ้าย

### 10.2 Result Summary Markdown

`entrance_result_summary.md`

ไฟล์สรุปสำหรับส่งงานหรืออ่านเร็ว มี:

- Official result จาก `longest one time`
- ระยะเวลา stationary duration
- frame เริ่ม/จบของช่วงนิ่ง
- crop ของ Person ID ที่ชนะ
- additional insight จาก `standing total`
- method used
- limitations

### 10.3 JSON Summary

`entrance_standing_summary.json`

เหมาะสำหรับอ่านด้วยโปรแกรมต่อ เช่น Python หรือ dashboard

มี:

- path input
- path output video
- winner
- people list
- parameters ที่ใช้ตอนรัน

### 10.4 Tracks CSV

`entrance_standing_tracks.csv`

ข้อมูลรายเฟรมต่อ ID:

- frame
- track_id
- bbox
- confidence
- is_standing
- current_standing_sec
- total_standing_sec
- max_standing_sec

### 10.5 People Summary CSV

`entrance_people_id_summary.csv`

ข้อมูลสรุปต่อ ID เรียงตามเวลายืนสะสม:

- person_id
- total_standing_sec
- total_standing_str
- max_standing_sec
- max_standing_str
- first_frame
- last_frame
- standing_start_frame
- standing_end_frame
- crop_file

## 11. พารามิเตอร์ที่ควรจูน

### 11.1 `--conf`

ค่า default:

```text
0.35
```

ถ้าเจอคนหลุด detection บ่อย:

```text
ลด conf เช่น 0.25-0.30
```

ถ้าเจอ false detection:

```text
เพิ่ม conf เช่น 0.45-0.60
```

### 11.2 `--stationary-std-px`

ค่า default:

```text
12.0
```

ถ้าระบบบอกว่า moving ทั้งที่คนยืนนิ่ง:

```text
เพิ่มเป็น 15-20
```

ถ้าระบบบอกว่า standing ทั้งที่คนเดินช้า:

```text
ลดเป็น 6-10
```

### 11.3 `--min-stationary-sec`

ค่า default:

```text
2.0
```

ถ้าอยากนับเร็วขึ้น:

```text
ลดเป็น 1.0
```

ถ้าอยากให้นับเฉพาะคนที่ค้างนานจริง:

```text
เพิ่มเป็น 3.0-5.0
```

### 11.4 `--window-size`

ค่า default:

```text
30
```

ถ้าวิดีโอ 30 FPS จะเท่ากับประมาณ 1 วินาที

ถ้าอยากให้ระบบนิ่งขึ้นและลด noise:

```text
เพิ่ม window-size
```

ถ้าอยากให้ระบบตอบสนองไวขึ้น:

```text
ลด window-size
```

### 11.5 `--frame-skip`

ค่า default:

```text
1
```

ถ้าต้องการประมวลผลเร็วขึ้น:

```text
--frame-skip 2
```

ข้อควรระวัง: frame skip มากเกินไปอาจทำให้ tracking และเวลายืนละเอียดน้อยลง

### 11.6 `--duplicate-iou-threshold`

ค่า default:

```text
0.85
```

ใช้ซ่อน ID ซ้ำเมื่อ box ของคน 2 ID ทับกันมากในเฟรมเดียวกัน

ถ้ายังเห็น ID ซ้ำของคนเดียวกัน:

```text
ลดเป็น 0.75-0.80
```

ถ้ากลัวว่าคนสองคนที่ยืนใกล้กันมากจะถูก merge ผิด:

```text
เพิ่มเป็น 0.90-0.95
```

ข้อควรระวัง: วิธีนี้แก้ duplicate ที่เกิดพร้อมกันในเฟรมเดียวกันได้ดี แต่ไม่ได้แก้ re-identification ระยะยาว เช่น คนหายไปนานแล้วกลับมาเป็น ID ใหม่

## 12. ข้อจำกัดเชิงเทคนิค

### 12.1 ID Switch

ByteTrack อาจเปลี่ยน ID เมื่อ:

- คนเดินสวนกัน
- คนซ้อนทับกัน
- detection หายหลายเฟรม
- คนออกจากเฟรมแล้วกลับมา

ผลคือเวลาสะสมอาจถูกแยกเป็นคนละ ID

### 12.2 Camera Motion

ถ้ากล้องขยับ ระบบอาจเข้าใจว่าคนขยับ เพราะ centroid เปลี่ยนตามภาพ

ระบบนี้เหมาะกับ:

```text
กล้องนิ่ง / CCTV / fixed camera
```

ถ้ากล้องเคลื่อนที่ ควรเพิ่มขั้นตอน video stabilization หรือ background compensation

### 12.3 Perspective

คนที่อยู่ไกลจะมี box เล็กกว่า และ centroid noise อาจมีผลมากกว่า

ค่า `stationary_std_px` อาจต้องจูนตาม:

- ความละเอียดวิดีโอ
- ระยะกล้อง
- มุมกล้อง
- ขนาดคนในเฟรม

### 12.4 ไม่ใช่ Face Recognition

ระบบไม่สามารถบอกได้ว่า:

```text
ID 5 ในช่วงต้นวิดีโอ กับ ID 18 ตอนท้ายวิดีโอ คือคนเดียวกันหรือไม่
```

ถ้าต้องการแบบนั้น ต้องเพิ่ม person re-identification หรือ face recognition แยกอีกชั้น

## 13. เหตุผลของแนวทางนี้

แนวทาง YOLO + ByteTrack + centroid spread เหมาะกับโจทย์นี้เพราะ:

- ทำได้เร็วและเข้าใจง่าย
- ไม่ต้อง train model ใหม่
- ไม่ต้องใช้ข้อมูลใบหน้า
- เหมาะกับ fixed camera
- อธิบายผลย้อนหลังได้จาก CSV/JSON
- จูน threshold ได้ตามสถานที่จริง

ข้อแลกเปลี่ยนคือ identity ยังขึ้นกับ tracker และความนิ่งขึ้นกับ pixel movement จึงต้องเข้าใจข้อจำกัดเรื่อง ID switch และ camera motion

## 14. แนวทางที่อาจเพิ่มต่อได้

ถ้าต้องการพัฒนาให้แม่นขึ้น สามารถเพิ่ม:

- ROI mask เพื่อวิเคราะห์เฉพาะพื้นที่ทางเข้า/จุดสนใจ
- perspective-aware threshold ให้คนไกล/ใกล้ใช้ threshold ต่างกัน
- re-identification ด้วย embedding เสื้อผ้า/ลักษณะคน
- smoothing เพิ่มเติมเพื่อลด box jitter
- export Excel report พร้อมรูป crop
- dashboard สำหรับดู ranking และคลิกดู crop ของแต่ละ ID
- config file แบบ YAML เพื่อไม่ต้องพิมพ์ argument ยาว
