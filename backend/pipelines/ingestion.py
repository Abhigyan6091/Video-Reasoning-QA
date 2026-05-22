"""
Video ingestion – accept upload, save to disk, extract metadata via OpenCV.
"""
from pathlib import Path

import cv2

from backend.config import settings
from backend import database as db


async def ingest_video(filename: str, file_bytes: bytes, language: str = "english") -> dict:
    """
    Save uploaded video and extract basic metadata.
    Returns the video record dict with id, metadata, etc.
    """
    # Save to uploads/
    video_id_record = await db.create_video(
        filename=filename,
        filepath="",  # will update after saving
        language=language,
    )
    video_id = video_id_record["id"]

    video_dir = settings.UPLOAD_DIR / video_id
    video_dir.mkdir(parents=True, exist_ok=True)
    video_path = video_dir / filename
    video_path.write_bytes(file_bytes)

    # Update filepath in DB
    async with __import__("aiosqlite").connect(str(settings.DB_PATH)) as conn:
        await conn.execute(
            "UPDATE videos SET filepath = ?, updated_at = datetime('now') WHERE id = ?",
            (str(video_path), video_id),
        )
        await conn.commit()

    # Extract metadata via OpenCV
    cap = cv2.VideoCapture(str(video_path))
    try:
        if cap.isOpened():
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration = frame_count / fps if fps > 0 else 0.0
            await db.update_video_metadata(video_id, duration, width, height, fps)
        else:
            duration, width, height, fps = 0.0, 0, 0, 0.0
    finally:
        cap.release()

    return {
        "id": video_id,
        "filename": filename,
        "filepath": str(video_path),
        "duration": duration,
        "width": width,
        "height": height,
        "fps": fps,
        "status": "uploaded",
    }
