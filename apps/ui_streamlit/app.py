from __future__ import annotations

import logging
import sys
from pathlib import Path

import cv2
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from animaluniversity.core.video import extract_frames, get_video_metadata
from animaluniversity.utils.paths import get_data_dir


def _list_videos(raw_dir: Path) -> list[Path]:
    exts = {".mp4", ".avi", ".mkv"}
    return sorted([p for p in raw_dir.glob("*") if p.suffix.lower() in exts])


def _preview_frames(frame_dir: Path, max_images: int = 12) -> None:
    frames = sorted(frame_dir.glob("frame_*.jpg"))[:max_images]
    if not frames:
        st.info("No frames to preview.")
        return
    cols = st.columns(4)
    for i, frame_path in enumerate(frames):
        img = cv2.imread(str(frame_path))
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cols[i % 4].image(img, caption=frame_path.name, use_container_width=True)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    st.set_page_config(page_title="AnimalUniversity", layout="wide")

    st.title("AnimalUniversity - Local Pipeline")
    st.caption("Phase 0–2: Video ingest and frame extraction")

    data_dir = get_data_dir()
    raw_dir = data_dir / "raw_videos"
    frames_dir = data_dir / "frames"
    raw_dir.mkdir(parents=True, exist_ok=True)
    frames_dir.mkdir(parents=True, exist_ok=True)

    with st.sidebar:
        st.header("Video Source")
        upload = st.file_uploader("Upload a video", type=["mp4", "avi", "mkv"])
        existing = _list_videos(raw_dir)
        selected = st.selectbox("Or select existing", [""] + [p.name for p in existing])
        fps = st.number_input("FPS", min_value=1, max_value=60, value=1, step=1)
        start_sec = st.number_input("Start (sec)", min_value=0.0, value=0.0, step=1.0)
        end_sec = st.number_input("End (sec, optional)", min_value=0.0, value=0.0, step=1.0)
        end_sec = None if end_sec == 0.0 else float(end_sec)
        run_extract = st.button("Extract Frames")

    video_path: Path | None = None
    if upload is not None:
        video_path = raw_dir / upload.name
        if not video_path.exists():
            video_path.write_bytes(upload.read())
        st.success(f"Uploaded: {video_path}")
    elif selected:
        video_path = raw_dir / selected

    if video_path and video_path.exists():
        metadata = get_video_metadata(video_path)
        st.subheader("Metadata")
        st.json(metadata)

    if run_extract:
        if not video_path or not video_path.exists():
            st.error("Please upload or select a video first.")
        else:
            out_dir = frames_dir / video_path.stem
            manifest = extract_frames(
                video_path=video_path,
                out_dir=out_dir,
                fps=int(fps),
                start_sec=float(start_sec),
                end_sec=end_sec,
                overwrite=True,
            )
            st.success(f"Extracted {len(manifest)} frames to {out_dir}")
            _preview_frames(out_dir)


if __name__ == "__main__":
    main()
