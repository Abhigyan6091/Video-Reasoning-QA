"""
Keyframe extraction – pulls representative frames from each scene.
"""
from __future__ import annotations

from pathlib import Path

import cv2

from backend.config import settings


def extract_keyframes(video_path: str, video_id: str,
                      scenes: list[dict],
                      interval_sec: float | None = None,
                      max_per_scene: int | None = None) -> dict[str, list[str]]:
    """
    Extract keyframes for each scene.

    Returns: {scene_id: [frame_path, ...]}
    """
    interval = interval_sec or settings.KEYFRAME_INTERVAL_SEC
    cap_max = max_per_scene or settings.MAX_KEYFRAMES_PER_SCENE

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    result: dict[str, list[str]] = {}

    for scene in scenes:
        scene_id = scene["id"]
        start_sec = scene["start_time"]
        end_sec = scene["end_time"]

        out_dir = Path(settings.FRAMES_DIR) / video_id / scene_id
        out_dir.mkdir(parents=True, exist_ok=True)

        # Determine which timestamps to sample
        timestamps: list[float] = [start_sec]  # always grab first frame of scene
        t = start_sec + interval
        while t < end_sec and len(timestamps) < cap_max:
            timestamps.append(t)
            t += interval

        paths: list[str] = []
        for i, ts in enumerate(timestamps):
            frame_no = int(ts * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
            ret, frame = cap.read()
            if not ret:
                continue
            fname = f"frame_{i:04d}_{ts:.2f}s.jpg"
            fpath = out_dir / fname
            cv2.imwrite(str(fpath), frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            paths.append(str(fpath))

        result[scene_id] = paths

    cap.release()
    return result
