# Interview Explanation Deep Dive

เอกสารนี้เขียนเพื่อช่วยอธิบายแนวคิดของโปรเจกต์ในรอบ interview แบบละเอียด โดยเน้นให้เข้าใจจริง ไม่ใช่แค่ท่องว่าใช้ YOLO + ByteTrack

เป้าหมายคือให้สามารถตอบได้ว่า:

- ทำไมเลือกวิธีนี้
- ระบบทำงานอย่างไรตั้งแต่ต้นจนจบ
- วัดความนิ่งยังไง
- ทำไมบางคนถูกแยกเป็นหลาย ID
- แก้ duplicate ID ยังไง
- อ่านผลลัพธ์ยังไง
- มี insight อะไร
- ข้อจำกัดคืออะไร
- ถ้าจะปรับปรุงต่อควรทำอะไร

## 1. เข้าใจโจทย์ก่อน

โจทย์ต้องการ:

```text
The person who remained stationary for the longest duration
and the duration of the stationary period.
```

แปลแบบตรงตัว:

```text
หาว่าคนไหนอยู่นิ่งนานที่สุด
และช่วงเวลาที่อยู่นิ่งนั้นนานเท่าไร
```

จุดสำคัญคือคำว่า:

```text
stationary period
```

คำนี้หมายถึง "ช่วงเวลานิ่งต่อเนื่องหนึ่งช่วง" มากกว่า "เวลานิ่งสะสมรวมทุกช่วง"

ดังนั้นโปรเจกต์นี้จึงแยกผลลัพธ์เป็น 2 แบบ:

1. **Official result**

   ใช้ `longest one time`

   คือช่วงเวลายืนนิ่งต่อเนื่องที่นานที่สุดครั้งเดียว ใช้เป็นคำตอบหลักของโจทย์

2. **Additional insight**

   ใช้ `standing total`

   คือเวลายืนนิ่งสะสมรวมของแต่ละ ID แม้จะยืนหลายรอบแล้วเดินออก/กลับมายืนใหม่

ตัวอย่าง:

```text
ID 46 ยืนนิ่งรอบแรก 9.1 วินาที
ID 46 เดิน/ขยับ
ID 46 ยืนนิ่งรอบสอง 40.8 วินาที

longest one time = 40.8 วินาที
standing total   = 49.9 วินาที
```

คำตอบหลักของโจทย์คือ `40.8 วินาที`

แต่ insight เสริมคือ ID นี้มีเวลายืนนิ่งสะสมรวม `49.9 วินาที`

## 2. แนวคิดหลักของ Solution

แนวคิดหลักคือ:

```text
Detect -> Track -> Measure movement -> Count stationary time -> Summarize
```

แยกเป็นขั้นตอน:

1. ใช้ YOLO ตรวจจับคนในแต่ละเฟรม
2. ใช้ ByteTrack เชื่อมคนระหว่างเฟรมให้เป็น `Person ID`
3. สำหรับแต่ละ ID เก็บตำแหน่งจุดกลางของ bounding box
4. ดูว่าจุดกลางนั้นขยับมากหรือน้อยในช่วงเวลาล่าสุด
5. ถ้าขยับน้อยต่อเนื่องเกินเวลาขั้นต่ำ ถือว่า stationary/standing
6. นับระยะเวลานิ่ง
7. สรุปว่า ID ไหนมีช่วงนิ่งต่อเนื่องยาวที่สุด
8. สร้างวิดีโอ overlay, CSV, JSON, markdown summary และ crop

Pipeline:

```text
Video
  -> OpenCV อ่านเฟรม
  -> YOLO ตรวจจับคน
  -> ByteTrack ให้ Person ID
  -> คำนวณ centroid ของแต่ละ box
  -> เก็บ centroid history
  -> คำนวณ spread
  -> ตัดสิน STANDING / moving
  -> นับเวลา
  -> สร้าง output
```

## 3. ทำไมใช้ YOLO

YOLO เป็น object detection model ที่เหมาะกับงานวิดีโอ เพราะ:

- ตรวจจับวัตถุได้เร็ว
- ใช้กับคนได้ดีผ่าน COCO class `person`
- ใช้งานง่ายผ่าน Ultralytics
- ต่อเข้ากับ tracker ได้ด้วย `model.track(...)`
- ไม่ต้อง train model ใหม่สำหรับโจทย์นี้

ในระบบนี้ YOLO ไม่ได้ทำหน้าที่จำว่าใครเป็นใคร

YOLO ทำหน้าที่แค่ตอบว่า:

```text
ในเฟรมนี้ มีคนอยู่ตรงไหนบ้าง
```

ผลลัพธ์ที่ได้จาก YOLO:

```text
bounding box = x1, y1, x2, y2
confidence   = ความมั่นใจว่าเป็นคน
class        = person
```

ตัวอย่าง:

```text
box = (1202, 271, 1376, 772)
confidence = 0.69
class = person
```

ค่าที่ใช้:

```text
model: yolo26s.pt
conf: 0.35
iou: 0.45
imgsz: 640
device: 0
```

อธิบาย:

- `conf=0.35`: ถ้าความมั่นใจต่ำกว่า 0.35 จะไม่เอา box นั้น
- `iou=0.45`: ช่วยจัดการ box ที่ซ้อนกันใน detection/tracking
- `imgsz=640`: resize ภาพให้โมเดลประมวลผล
- `device=0`: ใช้ GPU ตัวแรก

ถ้าถามว่า “ทำไมไม่ใช้ face recognition”

ตอบได้ว่า:

> โจทย์ต้องการวัด stationary duration จากวิดีโอ ไม่ได้ต้องการระบุตัวตนจริงของบุคคล การใช้ YOLO + tracker เพียงพอและเหมาะกว่า เพราะไม่ต้องใช้ข้อมูลใบหน้า และทำงานได้แม้ใบหน้าไม่ชัด

## 4. ทำไมต้องใช้ Tracker

ถ้าใช้ YOLO อย่างเดียว จะรู้แค่ว่าแต่ละเฟรมมีคนตรงไหน แต่จะไม่รู้ว่า:

```text
คนในเฟรมที่ 100
กับคนในเฟรมที่ 101
คือคนเดียวกันหรือไม่
```

ดังนั้นต้องมี tracker เพื่อให้ ID ต่อเนื่อง

ระบบนี้ใช้ ByteTrack

หน้าที่ของ ByteTrack:

```text
เชื่อม bounding box ระหว่างเฟรม
แล้วให้ track ID กับคนแต่ละคน
```

ตัวอย่าง:

```text
frame 100 -> person box A -> ID 46
frame 101 -> person box A moved a little -> ID 46
frame 102 -> person box A moved a little -> ID 46
```

ถ้าไม่มี ByteTrack เราจะนับเวลาต่อคนไม่ได้ เพราะไม่รู้ว่า box ไหนคือคนเดิม

ในโค้ดใช้:

```python
model.track(
    source=frame,
    persist=True,
    tracker="bytetrack.yaml",
    classes=[0],
)
```

ความหมาย:

- `source=frame`: ส่งภาพทีละเฟรม
- `persist=True`: ให้ tracker จำ state ต่อเนื่องระหว่างเฟรม
- `tracker="bytetrack.yaml"`: ใช้ ByteTrack
- `classes=[0]`: track เฉพาะคน

## 5. Person ID คืออะไร

`Person ID` ในโปรเจกต์นี้คือ tracking ID จาก ByteTrack

ไม่ใช่:

- ชื่อคน
- ใบหน้าคน
- identity จริง
- re-identification ข้ามช่วงยาว

ควรอธิบายใน interview ว่า:

> Person ID หมายถึง track ID ที่ ByteTrack สามารถติดตามต่อเนื่องได้ในวิดีโอ ถ้าคนถูกบังหรือออกจากเฟรมนาน ID อาจเปลี่ยนได้

ข้อจำกัดนี้สำคัญมาก เพราะถ้ากรรมการถามว่า “รู้ได้ไงว่าเป็นคนเดียวกันจริง ๆ” ควรตอบตรง ๆ ว่า:

> ระบบนี้ไม่ได้ยืนยันตัวตนจริง แต่ใช้ tracking continuity จากตำแหน่งและการเคลื่อนที่

## 6. การวัดตำแหน่งของคน

เมื่อ YOLO ตรวจจับคน จะได้ bounding box:

```text
x1, y1 = มุมซ้ายบน
x2, y2 = มุมขวาล่าง
```

แต่การดูว่าคนขยับไหม ถ้าเอาทั้ง box มาเทียบตรง ๆ จะมี noise เยอะ เพราะ YOLO box อาจสั่นเล็กน้อย

ระบบจึงแปลง box เป็นจุดกลาง หรือ centroid:

```text
cx = (x1 + x2) / 2
cy = (y1 + y2) / 2
```

ตัวอย่าง:

```text
x1 = 1202
y1 = 271
x2 = 1376
y2 = 772

cx = (1202 + 1376) / 2 = 1289
cy = (271 + 772) / 2 = 521.5
```

จุดนี้ใช้แทนตำแหน่งของคนในเฟรมนั้น

เหตุผลที่ใช้ centroid:

- เข้าใจง่าย
- คำนวณเร็ว
- ลดผลจาก box jitter บางส่วน
- เหมาะกับกล้องนิ่ง
- เพียงพอสำหรับวัดว่าอยู่ตำแหน่งเดิมหรือไม่

## 7. Sliding Window คืออะไร

ระบบไม่ได้ดูแค่ centroid เฟรมเดียว เพราะเฟรมเดียวบอกไม่ได้ว่านิ่งหรือขยับ

จึงเก็บ centroid ย้อนหลังของแต่ละ ID ในช่วงเวลาสั้น ๆ

เรียกว่า sliding window

ค่า default:

```text
window_size = 30 เฟรม
```

ถ้าวิดีโอประมาณ 30 FPS แปลว่าเก็บประมาณ 1 วินาทีล่าสุด

ตัวอย่าง:

```text
ID 46:
frame 100 -> (1289, 522)
frame 101 -> (1290, 521)
frame 102 -> (1288, 522)
frame 103 -> (1289, 523)
...
```

ถ้าค่าพวกนี้อยู่ใกล้กัน แปลว่าคนแทบไม่ขยับ

ถ้าค่ากระจายไปไกล แปลว่าคนกำลังเคลื่อนที่

## 8. Spread คืออะไร

ระบบวัดความกระจายของ centroid ด้วยสูตร:

```text
spread = sqrt(var(x) + var(y))
```

อธิบายแบบง่าย:

- เอาค่า x ของ centroid หลายเฟรมมาดูว่ากระจายมากไหม
- เอาค่า y ของ centroid หลายเฟรมมาดูว่ากระจายมากไหม
- รวมความกระจายทั้ง x และ y เป็นค่าเดียว

ถ้า:

```text
spread ต่ำ
```

แปลว่า centroid แทบไม่เปลี่ยน

ถ้า:

```text
spread สูง
```

แปลว่า centroid เคลื่อนที่มาก

ค่า default:

```text
stationary_std_px = 12.0
```

ระบบถือว่านิ่งเมื่อ:

```text
spread < 12.0 pixels
```

## 9. จาก “นิ่ง” ไปเป็น “STANDING”

ระบบไม่ได้บอกว่า `STANDING` ทันทีเมื่อ spread ต่ำ

เพราะอาจมีคนเดินช้า ๆ หรือหยุดแป๊บเดียว

จึงมีค่า:

```text
min_stationary_sec = 2.0
```

Logic:

```text
ถ้า spread ต่ำ:
  เริ่มจับเวลาช่วงนิ่ง

ถ้านิ่งต่อเนื่องครบ 2 วินาที:
  สถานะ = STANDING
  เริ่มนับเวลายืน

ถ้าขยับ:
  สถานะ = moving/checking
  reset standing now
```

เหตุผล:

- ลด false positive
- ไม่อยากนับการหยุดแค่เสี้ยววินาที
- ทำให้คำว่า stationary มีความหมายมากขึ้น

## 10. เวลาที่ระบบนับ

ระบบนับ 3 ค่า:

### 10.1 standing now

เวลายืนนิ่งในรอบปัจจุบัน

ถ้าขยับ จะ reset เป็น 0

ใช้ดู live overlay

### 10.2 standing total

เวลายืนนิ่งสะสมรวมของ ID นั้น

เช่น:

```text
ยืนนิ่ง 5 วิ
เดิน
กลับมายืนนิ่ง 3 วิ

standing total = 8 วิ
```

ค่านี้เป็น behavioral insight

### 10.3 longest one time

ช่วงยืนนิ่งต่อเนื่องนานที่สุดครั้งเดียว

เช่น:

```text
ยืนนิ่ง 5 วิ
เดิน
กลับมายืนนิ่ง 3 วิ

longest one time = 5 วิ
```

ค่านี้ใช้เป็น official result สำหรับโจทย์

## 11. ทำไมต้องแยก longest one time กับ standing total

เพราะสองค่านี้ตอบคนละคำถาม

```text
longest one time
```

ตอบว่า:

```text
ช่วง stationary period ที่ยาวที่สุดคือเท่าไร
```

```text
standing total
```

ตอบว่า:

```text
คนนี้มีพฤติกรรมหยุดนิ่งรวมทั้งหมดนานเท่าไร
```

ในผลล่าสุด:

```text
Official Result:
Person ID 46
Longest stationary period = 40.8s

Additional Insight:
Person ID 46
Total standing duration = 49.9s
```

Insight:

```text
49.9 - 40.8 = 9.1s
```

แปลว่า ID 46 มีช่วงนิ่งอื่น ๆ เพิ่มอีกราว 9.1 วินาที นอกจากช่วงนิ่งยาวสุด

อันนี้เป็น insight ที่ทำให้งานดูดีกว่าแค่ส่ง ID กับ duration เฉย ๆ

## 12. Duplicate ID เกิดจากอะไร

จาก output จริงเจอว่า:

```text
person_id_0164.jpg
person_id_0177.jpg
```

เป็นคนเดียวกันกับ ID หลัก แต่ถูกแยกเป็น ID ใหม่สั้น ๆ

หลังตรวจ CSV พบว่า:

```text
ID 164 อยู่แค่ 1 เฟรม
ID 177 อยู่แค่ 5 เฟรม
ID 46 อยู่ต่อเนื่องยาวกว่า
```

และในเฟรมเดียวกัน box ของ ID เหล่านี้ทับกับ ID 46 มาก

แปลว่า:

```text
YOLO/ByteTrack สร้าง duplicate track ของคนเดียวกัน
```

สาเหตุที่เกิดได้:

- YOLO detect คนเดียวกันออกมาเป็น box ซ้ำ
- คนอยู่ใน pose ที่เปลี่ยน เช่น ยกแก้ว ดื่มน้ำ
- box jitter
- คนอยู่ใกล้ขอบภาพหรือ background ซับซ้อน
- tracker แยก box ใกล้เคียงเป็น ID ใหม่

## 13. วิธีแก้ Duplicate ID

เพิ่ม post-processing หลัง ByteTrack:

```text
ถ้าในเฟรมเดียวกันมี 2 ID ที่ bbox ทับกันมาก
  คำนวณ IoU
  ถ้า IoU >= 0.85
    ถือว่าอาจเป็น duplicate
    เก็บ ID หลัก
    ซ่อน ID ซ้ำ
```

IoU คือ:

```text
IoU = พื้นที่ซ้อนทับ / พื้นที่รวมของ box ทั้งสอง
```

ถ้า IoU สูง เช่น 0.85-0.95 แปลว่า box แทบจะทับกัน

การเลือก ID หลักดูจาก:

1. track ที่อยู่มานานกว่า
2. มี total standing มากกว่า
3. มี longest standing มากกว่า
4. confidence สูงกว่า
5. ID เลขน้อยกว่า ถ้ายังเสมอ

เหตุผล:

> ID ที่อยู่มานานมักเป็น track จริง ส่วน ID ที่เพิ่งโผล่ 1-5 เฟรมมักเป็น ghost/duplicate

ผลของการแก้:

- duplicate ID ไม่ถูกวาด overlay
- ไม่ถูกเซฟ crop
- ไม่เข้า CSV rows
- ไม่เข้า summary/ranking
- ไม่รบกวนผลลัพธ์หลัก

ข้อจำกัด:

> วิธีนี้แก้ duplicate ที่เกิดพร้อมกันในเฟรมเดียวกันได้ดี แต่ไม่ใช่ re-identification ถ้าคนหายไปนานแล้วกลับมาเป็น ID ใหม่ วิธีนี้ยังรวมกลับไม่ได้

## 14. Insight ที่ระบบให้ได้

ระบบไม่ได้ให้แค่ “ใครชนะ” แต่ให้ insight หลายชั้น

### 14.1 Official Answer

```text
ID ไหนมี stationary period ต่อเนื่องนานที่สุด
```

ใช้ตอบโจทย์โดยตรง

### 14.2 Cumulative Behavior

```text
ID ไหนมีเวลายืนนิ่งสะสมรวมมากที่สุด
```

ใช้ดูพฤติกรรมรวม เช่น คนนี้ไม่ได้ยืนยาวครั้งเดียว แต่หยุดนิ่งหลายช่วง

### 14.3 Track Quality

ดูได้จาก:

- ID อยู่กี่เฟรม
- มี duplicate ID หรือไม่
- crop ตรงกับคนจริงหรือเปล่า
- confidence ต่ำหรือสูง

### 14.4 Debug Insight

ถ้าเห็น ID แปลก ๆ:

- ดู crop
- ดู CSV ว่า ID นั้นอยู่กี่เฟรม
- ดู bbox ว่าทับกับ ID อื่นไหม
- ดู confidence
- ดูช่วง frame ที่เกิด

ตัวอย่าง:

```text
ID 164 อยู่ 1 เฟรม
ID 177 อยู่ 5 เฟรม
box ทับกับ ID 46
```

สรุปได้ว่าเป็น duplicate ไม่ใช่คนใหม่จริง

## 15. Output แต่ละไฟล์ใช้ทำอะไร

### 15.1 entrance_standing_detail.mp4

วิดีโอพร้อม overlay

ใช้ดูด้วยตา:

- กรอบคน
- Person ID
- confidence
- STANDING/moving
- standing now
- standing total
- longest one time
- ranking panel

### 15.2 entrance_result_summary.md

ไฟล์สำคัญสำหรับส่งงาน/อ่านผลเร็ว

มี:

- Official result
- Duration
- Start/end frame
- Crop
- Additional insight
- Method used
- Parameters
- Limitations

### 15.3 entrance_standing_summary.json

ไฟล์ machine-readable

เหมาะสำหรับ:

- เอาไปทำ dashboard
- โหลดด้วย Python
- ใช้วิเคราะห์ต่อ

### 15.4 entrance_standing_tracks.csv

ข้อมูลรายเฟรมต่อ ID

ใช้ debug:

- ID นี้โผล่ frame ไหน
- bbox อยู่ตรงไหน
- confidence เท่าไร
- is_standing หรือไม่
- total เพิ่มเมื่อไร

### 15.5 entrance_people_id_summary.csv

สรุปต่อ ID

ใช้ดู ranking ง่าย ๆ ใน Excel

### 15.6 person_id_crops

รูป crop ของแต่ละ ID

ใช้ตรวจว่า:

- ID ที่ชนะคือคนไหน
- ID ซ้ำหรือเปล่า
- crop ชัดพอไหม

## 16. วิธีอ่านผลจริงของโปรเจกต์นี้

ผลล่าสุด:

```text
Official Result
Person ID: 46
Duration: 40.8s
Start frame: 1006
End frame: 2226
```

แปลว่า:

```text
ID 46 มีช่วงยืนนิ่งต่อเนื่องนานที่สุด ประมาณ 40.8 วินาที
```

Insight เสริม:

```text
Top cumulative Person ID: 46
Total standing duration: 49.9s
Longest single period: 40.8s
```

แปลว่า:

```text
ID 46 เป็นคนที่ยืนนิ่งสะสมรวมมากที่สุดด้วย
และมีช่วงนิ่งอื่น ๆ อีกประมาณ 9.1 วินาที
```

## 17. การจูนค่า

### 17.1 conf

```text
--conf 0.35
```

ถ้า detect คนไม่ครบ:

```text
ลดเป็น 0.25-0.30
```

ถ้า detect มั่ว:

```text
เพิ่มเป็น 0.45-0.60
```

### 17.2 stationary_std_px

```text
--stationary-std-px 12.0
```

ถ้าคนยืนนิ่งแต่ระบบบอก moving:

```text
เพิ่มเป็น 15-20
```

ถ้าคนเดินช้าแต่ระบบบอก standing:

```text
ลดเป็น 6-10
```

### 17.3 min_stationary_sec

```text
--min-stationary-sec 2.0
```

ถ้าอยากให้นับไวขึ้น:

```text
ลดเป็น 1.0
```

ถ้าอยากนับเฉพาะการค้างนานจริง:

```text
เพิ่มเป็น 3.0-5.0
```

### 17.4 duplicate_iou_threshold

```text
--duplicate-iou-threshold 0.85
```

ถ้ายังมี duplicate ID:

```text
ลดเป็น 0.80
```

ถ้าคนสองคนยืนชิดกันแล้วกลัวโดนรวมผิด:

```text
เพิ่มเป็น 0.90-0.95
```

### 17.5 frame_skip

```text
--frame-skip 1
```

ละเอียดที่สุด แต่ช้ากว่า

ถ้าอยากเร็วขึ้น:

```text
--frame-skip 2
```

ข้อเสียคือเวลาและ tracking อาจละเอียดน้อยลง

## 18. ทำไมรันช้า

เพราะระบบประมวลผลทุกเฟรม:

1. อ่านภาพ
2. YOLO inference
3. ByteTrack update
4. centroid/spread calculation
5. overlay drawing
6. video writing
7. preview display ถ้าเปิด `--show`

การรันช้าไม่ได้แปลว่าระบบจำหน้าละเอียดขึ้น

แต่แปลว่า:

```text
ระบบกำลังวิเคราะห์ทุกเฟรมอย่างละเอียด
```

ถ้าใช้ `--frame-skip 1` tracking จะต่อเนื่องกว่า และเวลายืนแม่นกว่า

ถ้าใช้ `--frame-skip 2` จะเร็วขึ้น แต่มีโอกาสหลุดมากขึ้น

## 19. ข้อจำกัดสำคัญ

### 19.1 ไม่ใช่ face recognition

ระบบไม่รู้ชื่อคนหรือ identity จริง

รู้แค่ว่า:

```text
track นี้น่าจะเป็นคนเดิมจากเฟรมก่อน
```

### 19.2 ID switch

ID อาจเปลี่ยนเมื่อ:

- คนถูกบัง
- คนออกจากเฟรม
- คนเดินสวนกันใกล้มาก
- detection หลุดหลายเฟรม

### 19.3 กล้องควรนิ่ง

ถ้ากล้องเคลื่อนที่ centroid จะเปลี่ยนทั้งภาพ ทำให้ระบบเข้าใจว่าคนขยับ

เหมาะกับ:

```text
CCTV / fixed camera
```

### 19.4 threshold ต้องจูนตามวิดีโอ

ค่าที่ใช้เหมาะกับวิดีโอนี้ แต่อาจต้องปรับกับวิดีโออื่น

เช่น:

- ความละเอียดต่างกัน
- คนอยู่ไกล/ใกล้ต่างกัน
- กล้องสั่น
- FPS ต่างกัน

### 19.5 duplicate suppression ไม่ใช่ re-ID

แก้ได้เฉพาะ ID ซ้ำที่ box ทับกันในเวลาเดียวกัน

ถ้าคนหายไปนานแล้วกลับมาเป็น ID ใหม่ ต้องใช้ person re-identification เพิ่ม

## 20. สิ่งที่ทำให้ Solution นี้น่าสนใจ

จุดที่ควรเล่า:

1. ไม่ได้ใช้ YOLO วาดกรอบอย่างเดียว แต่เพิ่ม temporal reasoning จากหลายเฟรม
2. ใช้ tracker เพื่อแยกคนแต่ละคนและนับเวลาต่อ ID
3. ใช้ centroid spread เป็น metric ที่อธิบายง่ายและ debug ได้
4. แยก official result กับ cumulative insight
5. เพิ่ม crop เพื่อ verify ว่า ID คือใคร
6. เพิ่ม CSV/JSON เพื่อวิเคราะห์ย้อนหลังได้
7. ตรวจเจอปัญหา duplicate ID จาก output จริง แล้วแก้ด้วย IoU suppression
8. มีเอกสารอธิบายข้อจำกัดชัดเจน

## 21. คำอธิบายสั้นสำหรับ Interview

ถ้าต้องตอบแบบสั้น:

> ผมใช้ YOLO ตรวจจับคนในแต่ละเฟรม แล้วใช้ ByteTrack เชื่อมคนระหว่างเฟรมให้เป็น Person ID จากนั้นแปลง bounding box ของแต่ละ ID เป็น centroid และเก็บ centroid history ใน sliding window เพื่อดูว่าตำแหน่งกระจายมากแค่ไหน ถ้า spread ต่ำกว่า threshold และนิ่งต่อเนื่องเกินเวลาขั้นต่ำ จะถือว่าเป็น stationary period ระบบนับทั้ง longest one time สำหรับคำตอบหลักของโจทย์ และ standing total เป็น insight เสริม นอกจากนี้ผมเพิ่ม duplicate suppression ด้วย IoU เพื่อลดกรณี tracker สร้าง ID ซ้ำของคนเดียวกัน

ใช้้ Yolo ในการ Detect คนในแต่ละเฟรม จากนั้นใช้ ByteTrack ให้รู้ว่าเฟรมนั้นเป็น ID ของคนที่ตรวจจับได้ และทำเป็น bbox แต่ละ ID เพื่อเก็บ

## 22. คำอธิบายแบบละเอียดขึ้น

ถ้ามีเวลาพูด 1-2 นาที:

> Pipeline ของผมเริ่มจากใช้ OpenCV อ่านวิดีโอทีละเฟรม จากนั้นใช้ Ultralytics YOLO ตรวจจับเฉพาะ class person แล้วส่งผลเข้า ByteTrack เพื่อสร้าง Person ID ต่อเนื่องในวิดีโอ หลังจากได้ box ของแต่ละ ID ผมคำนวณ centroid ของ box และเก็บค่า centroid ย้อนหลังใน sliding window ประมาณ 30 เฟรม จากนั้นคำนวณ spread ด้วยสูตร sqrt(var(x)+var(y)) ถ้า spread ต่ำกว่า 12 pixels และนิ่งต่อเนื่องเกิน 2 วินาที จะถือว่า ID นั้นอยู่ในสถานะ STANDING
>
> ผมนับเวลา 2 แบบ คือ longest one time ซึ่งเป็นช่วงนิ่งต่อเนื่องนานที่สุด ใช้เป็น official answer ของโจทย์ และ standing total ซึ่งเป็นเวลานิ่งสะสมรวมทุกช่วง ใช้เป็น behavioral insight เพิ่มเติม ผลลัพธ์ถูก export เป็นวิดีโอ overlay, CSV, JSON, markdown summary และ crop ของแต่ละ ID เพื่อให้ตรวจสอบย้อนหลังได้
>
> ระหว่างตรวจ output ผมเจอเคส duplicate ID ที่คนเดียวกันถูก tracker สร้างเป็น ID ใหม่ 1-5 เฟรม จึงเพิ่ม IoU-based duplicate suppression ถ้า box ของ 2 ID ทับกันมากกว่า 0.85 จะเก็บ ID ที่อยู่มานานกว่าและซ่อน ID ที่เป็น duplicate วิธีนี้ช่วยให้ summary สะอาดขึ้น แต่ยังยอมรับข้อจำกัดว่าไม่ใช่ face recognition หรือ re-identification ระยะยาว

## 23. ถ้ากรรมการถามว่า “ทำไมถึงรู้ว่าคนนั้นนิ่ง”

ตอบ:

> ผมไม่ได้ดูจากเฟรมเดียว แต่ดูตำแหน่ง centroid ย้อนหลังหลายเฟรม ถ้า centroid กระจายน้อย แปลว่าตำแหน่งโดยรวมของคนนั้นแทบไม่เปลี่ยน และต้องนิ่งต่อเนื่องเกิน 2 วินาทีก่อนจึงนับเป็น stationary เพื่อลด false positive จากการหยุดสั้น ๆ

## 24. ถ้ากรรมการถามว่า “ถ้า ID เปลี่ยนล่ะ”

ตอบ:

> เป็นข้อจำกัดของ tracking-based system ครับ Person ID มาจาก ByteTrack ไม่ใช่ identity จริง ถ้าถูกบังหรือหายจากเฟรม ID อาจเปลี่ยนได้ ผมลดบางกรณีด้วย duplicate suppression สำหรับ ID ซ้ำในเฟรมเดียวกัน แต่ถ้าต้องการรวมคนเดิมหลังหายไปนาน ต้องเพิ่ม person re-identification หรือ face recognition

## 25. ถ้ากรรมการถามว่า “ทำไมไม่ใช้ optical flow”

ตอบ:

> Optical flow วัดการเคลื่อนที่ของ pixel ได้ละเอียด แต่สำหรับโจทย์นี้เราต้องนับเวลาแยกเป็นรายบุคคล จึงต้องมี detection และ tracking ก่อน การใช้ centroid ของ tracked person ทำให้ผูก movement metric กับ Person ID ได้ง่ายกว่า และอธิบายผลใน CSV/summary ได้ชัดเจนกว่า

## 26. ถ้ากรรมการถามว่า “ทำไมไม่ใช้ pose estimation”

ตอบ:

> Pose estimation อาจช่วยแยกการยืน/นั่ง/เดินได้ละเอียดขึ้น แต่โจทย์หลักคือ stationary duration หรือการค้างตำแหน่งเดิม ไม่จำเป็นต้องรู้ skeleton ทุกจุด วิธี centroid spread จึงเรียบง่าย เร็ว และพอเพียงสำหรับ fixed camera แต่ถ้าต้องแยกท่าทางจริง ๆ เช่น ยืนกับนั่ง อาจเพิ่ม pose model ในเวอร์ชันถัดไป

## 27. ถ้ากรรมการถามว่า “จะปรับปรุงต่อยังไง”

ตอบได้หลายทาง:

1. เพิ่ม person re-identification เพื่อรวม ID ที่หลุดแล้วกลับมา
2. เพิ่ม ROI mask เพื่อวัดเฉพาะพื้นที่สนใจ
3. เพิ่ม perspective-aware threshold เพราะคนไกล/ใกล้มี pixel movement ต่างกัน
4. เพิ่ม smoothing ของ bounding box ลด jitter
5. เพิ่ม dashboard แสดง top stationary IDs และ crop
6. เพิ่ม Excel/PDF report
7. เพิ่ม video stabilization ถ้ากล้องไม่นิ่ง
8. ใช้ pose estimation ถ้าต้องแยกท่าทางยืน/นั่งจริง

## 28. One-liner ที่ควรจำ

```text
ระบบนี้ไม่ได้จำหน้าคน แต่ติดตาม track ของคน แล้ววัดความนิ่งจากการกระจายของ centroid ในช่วงเวลาต่อเนื่อง
```

อีกประโยค:

```text
YOLO บอกว่าคนอยู่ตรงไหน, ByteTrack บอกว่าคนเดิมคือ ID ไหน, centroid spread บอกว่า ID นั้นนิ่งหรือขยับ
```

และอีกประโยค:

```text
longest one time ใช้ตอบโจทย์ ส่วน standing total ใช้เป็น insight เสริม
```

