"""
Scene segmentation using PySceneDetect.
Detects scene boundaries and returns temporal chunks.
"""
from __future__ import annotations

from scenedetect import open_video, SceneManager
from scenedetect.detectors import ContentDetector

from backend import database as db


async def detect_scenes(video_id: str, video_path: str,
                        threshold: float = 27.0,
                        min_scene_len_sec: float = 1.0) -> list[dict]:
    """
    Detect scene boundaries in a video file.

    Returns list of scene dicts with scene_idx, start_time, end_time.
    Also persists scenes to the database.
    """
    video = open_video(video_path)
    scene_manager = SceneManager()
    scene_manager.add_detector(
        ContentDetector(threshold=threshold, min_scene_len=round(min_scene_len_sec * video.frame_rate))
    )

    scene_manager.detect_scenes(video)
    scene_list_raw = scene_manager.get_scene_list()

    if not scene_list_raw:
        # Treat entire video as one scene
        import cv2
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        duration = frame_count / fps if fps > 0 else 0.0
        cap.release()
        scene_list_raw = [(0, duration)]
        scenes = [{"scene_idx": 0, "start_time": 0.0, "end_time": duration}]
    else:
        scenes = []
        for idx, (start, end) in enumerate(scene_list_raw):
            scenes.append({
                "scene_idx": idx,
                "start_time": start.get_seconds(),
                "end_time": end.get_seconds(),
            })

    # Persist to DB
    scene_ids = await db.create_scenes(video_id, scenes)

    # Attach IDs to scene dicts
    for scene, sid in zip(scenes, scene_ids):
        scene["id"] = sid
        scene["video_id"] = video_id

    return scenes
