from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from itertools import combinations
from math import sqrt

import numpy as np
from ultralytics import YOLO


@dataclass
class TrackInfo:
    # ข้อมูลของคน 1 track จาก ByteTrack โดย object นี้เก็บทั้งสถานะปัจจุบัน
    # และสถิติเวลายืนนิ่งสะสมของ Person ID เดียวกัน
    track_id: int
    bbox: tuple[int, int, int, int]
    confidence: float
    centroid: tuple[float, float]
    first_frame: int
    last_frame: int
    centroid_history: deque[tuple[float, float]] = field(default_factory=deque)
    is_stationary: bool = False
    stationary_start_frame: int | None = None
    stationary_counted_until_frame: int | None = None
    current_stationary_sec: float = 0.0
    total_stationary_sec: float = 0.0
    max_stationary_sec: float = 0.0
    max_stationary_start_frame: int | None = None
    max_stationary_end_frame: int | None = None
    is_duplicate_suppressed: bool = False

    def update(
        self,
        bbox: tuple[int, int, int, int],
        confidence: float,
        frame_idx: int,
        fps: float,
        window_size: int,
        stationary_std_px: float,
        min_stationary_sec: float,
    ) -> None:
        self.bbox = bbox
        self.confidence = confidence
        # ใช้การเคลื่อนที่ของ centroid เป็นสัญญาณหลักในการตัดสินว่า ID นี้
        # ยืนนิ่งหรือไม่ วิธีนี้ตั้งใจให้เรียบง่ายกว่า pose/face analysis
        self.centroid = ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)
        self.last_frame = frame_idx

        if self.centroid_history.maxlen != window_size:
            self.centroid_history = deque(self.centroid_history, maxlen=window_size)
        self.centroid_history.append(self.centroid)

        spread = self._centroid_spread()
        enough_history = len(self.centroid_history) >= max(5, min(window_size, 10))
        stationary_now = enough_history and spread < stationary_std_px

        if stationary_now:
            self._update_stationary_segment(frame_idx, fps, min_stationary_sec)
        else:
            self._reset_current_segment()

    def _update_stationary_segment(self, frame_idx: int, fps: float, min_stationary_sec: float) -> None:
        # ช่วงนิ่งเริ่มเมื่อ centroid กระจายน้อยพอ แต่จะเริ่มนับ/แสดงผล
        # เป็น STANDING ก็ต่อเมื่อนิ่งครบ min_stationary_sec แล้ว
        if self.stationary_start_frame is None:
            self.stationary_start_frame = frame_idx
            self.stationary_counted_until_frame = None

        elapsed_sec = max(0.0, (frame_idx - self.stationary_start_frame) / fps)
        self.is_stationary = elapsed_sec >= min_stationary_sec
        self.current_stationary_sec = elapsed_sec if self.is_stationary else 0.0

        if self.is_stationary:
            # บวกเฉพาะเวลาที่เพิ่มจากเฟรมที่นับล่าสุด เพื่อให้ total standing
            # เป็นเวลาสะสมข้ามหลายช่วงการยืน ไม่ได้นับซ้ำทั้ง segment
            if self.stationary_counted_until_frame is None:
                self.total_stationary_sec += elapsed_sec
            else:
                self.total_stationary_sec += max(0.0, (frame_idx - self.stationary_counted_until_frame) / fps)
            self.stationary_counted_until_frame = frame_idx

        if elapsed_sec > self.max_stationary_sec:
            self.max_stationary_sec = elapsed_sec
            self.max_stationary_start_frame = self.stationary_start_frame
            self.max_stationary_end_frame = frame_idx

    def _reset_current_segment(self) -> None:
        # เมื่อขยับจะจบรอบยืนปัจจุบัน แต่ยังเก็บ total และ longest ไว้
        # สำหรับสรุปผลท้ายวิดีโอ
        self.is_stationary = False
        self.current_stationary_sec = 0.0
        self.stationary_start_frame = None
        self.stationary_counted_until_frame = None

    def _centroid_spread(self) -> float:
        if len(self.centroid_history) < 2:
            return float("inf")
        # spread = sqrt(var(x) + var(y)); ค่ายิ่งต่ำแปลว่ายิ่งขยับน้อย
        pts = np.asarray(self.centroid_history, dtype=np.float32)
        return float(sqrt(np.var(pts[:, 0]) + np.var(pts[:, 1])))


class LongestStayTracker:
    def __init__(
        self,
        model_path: str,
        conf: float,
        iou: float,
        imgsz: int,
        device: str,
        window_size: int,
        stationary_std_px: float,
        min_stationary_sec: float,
        max_lost_frames: int,
        duplicate_iou_threshold: float,
    ) -> None:
        self.model = YOLO(model_path)
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        self.device = device
        self.window_size = window_size
        self.stationary_std_px = stationary_std_px
        self.min_stationary_sec = min_stationary_sec
        self.max_lost_frames = max_lost_frames
        self.duplicate_iou_threshold = duplicate_iou_threshold
        self.tracks: dict[int, TrackInfo] = {}

    def update(self, frame, frame_idx: int, fps: float) -> dict[int, TrackInfo]:
        # Ultralytics จะเรียก ByteTrack ภายใน model.track
        # classes=[0] คือเลือกเฉพาะ COCO class person เพื่อไม่ให้รถ/วัตถุอื่นได้ ID
        result = self.model.track(
            source=frame,
            persist=True,
            tracker="bytetrack.yaml",
            imgsz=self.imgsz,
            conf=self.conf,
            iou=self.iou,
            classes=[0],
            device=self.device,
            verbose=False,
        )[0]

        seen_ids: set[int] = set()
        if result.boxes is not None and result.boxes.id is not None:
            boxes = result.boxes.xyxy.detach().cpu().numpy()
            ids = result.boxes.id.detach().cpu().numpy().astype(int)
            confs = result.boxes.conf.detach().cpu().numpy()

            for box, track_id, conf in zip(boxes, ids, confs):
                x1, y1, x2, y2 = [int(round(v)) for v in box]
                bbox = (x1, y1, x2, y2)
                track_id = int(track_id)
                seen_ids.add(track_id)

                if track_id not in self.tracks:
                    # ครั้งแรกที่ ByteTrack ID นี้ปรากฏในวิดีโอ
                    self.tracks[track_id] = TrackInfo(
                        track_id=track_id,
                        bbox=bbox,
                        confidence=float(conf),
                        centroid=((x1 + x2) / 2.0, (y1 + y2) / 2.0),
                        first_frame=frame_idx,
                        last_frame=frame_idx,
                        centroid_history=deque(maxlen=self.window_size),
                    )

                self.tracks[track_id].is_duplicate_suppressed = False
                self.tracks[track_id].update(
                    bbox=bbox,
                    confidence=float(conf),
                    frame_idx=frame_idx,
                    fps=fps,
                    window_size=self.window_size,
                    stationary_std_px=self.stationary_std_px,
                    min_stationary_sec=self.min_stationary_sec,
                )

        active_ids = self._suppress_duplicate_tracks(seen_ids)

        for track_id, track in self.tracks.items():
            if track_id not in seen_ids and frame_idx - track.last_frame > self.max_lost_frames:
                # ไม่ลบ ID เก่า เพราะ summary ท้ายงานยังต้องใช้ยอดสะสมของ ID นั้น
                track.is_stationary = False
                track.current_stationary_sec = 0.0

        return {track_id: self.tracks[track_id] for track_id in active_ids if track_id in self.tracks}

    def _suppress_duplicate_tracks(self, seen_ids: set[int]) -> set[int]:
        # ByteTrack/YOLO บางเฟรมอาจปล่อย box ซ้ำของคนเดียวกันออกมาเป็น 2 ID
        # ถ้า box ทับกันสูงมาก ให้เก็บ track ที่น่าเชื่อถือกว่าและซ่อน ID ซ้ำ
        active_ids = set(seen_ids)
        for id_a, id_b in combinations(sorted(seen_ids), 2):
            if id_a not in active_ids or id_b not in active_ids:
                continue
            track_a = self.tracks.get(id_a)
            track_b = self.tracks.get(id_b)
            if track_a is None or track_b is None:
                continue
            if bbox_iou(track_a.bbox, track_b.bbox) < self.duplicate_iou_threshold:
                continue

            _, drop_id = choose_primary_track(track_a, track_b)
            active_ids.discard(drop_id)
            dropped = self.tracks[drop_id]
            dropped.is_stationary = False
            dropped.current_stationary_sec = 0.0
            dropped.is_duplicate_suppressed = True

        return active_ids

    def cumulative_winner(self) -> TrackInfo | None:
        visible_tracks = [track for track in self.tracks.values() if not track.is_duplicate_suppressed]
        if not visible_tracks:
            return None
        # winner คือ ID ที่มีเวลายืนนิ่งสะสมสูงสุด ไม่ใช่คนที่ยืนต่อเนื่อง
        # รอบเดียวได้นานที่สุด
        return max(visible_tracks, key=lambda track: track.total_stationary_sec)

    def longest_period_winner(self) -> TrackInfo | None:
        visible_tracks = [track for track in self.tracks.values() if not track.is_duplicate_suppressed]
        if not visible_tracks:
            return None
        # official winner สำหรับโจทย์ "stationary period" คือช่วงยืนนิ่ง
        # ต่อเนื่องครั้งเดียวที่นานที่สุด
        return max(visible_tracks, key=lambda track: track.max_stationary_sec)

    def top_stationary(self, limit: int = 8) -> list[TrackInfo]:
        visible_tracks = [track for track in self.tracks.values() if not track.is_duplicate_suppressed]
        ranked = sorted(visible_tracks, key=lambda track: track.total_stationary_sec, reverse=True)
        return [track for track in ranked if track.total_stationary_sec > 0][:limit]


def bbox_iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    if inter_area == 0:
        return 0.0

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter_area
    return inter_area / union if union > 0 else 0.0


def choose_primary_track(track_a: TrackInfo, track_b: TrackInfo) -> tuple[int, int]:
    # track ที่อยู่มานานกว่าและมีประวัติสะสมมากกว่ามักเป็น ID จริง
    # ส่วน ID ที่เพิ่งโผล่ 1-5 เฟรมมักเป็น duplicate/ghost track
    score_a = (
        track_a.last_frame - track_a.first_frame,
        track_a.total_stationary_sec,
        track_a.max_stationary_sec,
        track_a.confidence,
        -track_a.track_id,
    )
    score_b = (
        track_b.last_frame - track_b.first_frame,
        track_b.total_stationary_sec,
        track_b.max_stationary_sec,
        track_b.confidence,
        -track_b.track_id,
    )
    if score_a >= score_b:
        return track_a.track_id, track_b.track_id
    return track_b.track_id, track_a.track_id
