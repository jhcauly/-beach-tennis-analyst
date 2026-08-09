from __future__ import annotations

import json
from html import escape
from pathlib import Path

import cv2
import numpy as np

VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".webm"}


def _discover_athlete_videos(video_path: str | Path) -> list[Path]:
    source = Path(video_path).resolve()
    folder = source.parent
    videos = sorted(
        [path.resolve() for path in folder.iterdir() if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS],
        key=lambda path: path.name.lower(),
    )
    if source not in videos and source.exists():
        videos.insert(0, source)
    return videos


def _format_duration(seconds: float | None) -> str:
    if seconds is None or seconds < 0:
        return "--:--"
    total = int(round(seconds))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def _video_duration(path: Path) -> float | None:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return None
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        frames = float(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
        if fps <= 0 or frames <= 0:
            return None
        return frames / fps
    finally:
        cap.release()


def _frame_is_useful(frame: np.ndarray) -> bool:
    if frame.size == 0:
        return False
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(gray.mean()) > 14.0 and float(gray.std()) > 7.0


def _create_thumbnail(video: Path, destination: Path) -> Path | None:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        return None
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        candidates = [0.12, 0.25, 0.45, 0.65]
        chosen: np.ndarray | None = None
        for ratio in candidates:
            if total > 0:
                cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, min(total - 1, int(total * ratio))))
            ok, frame = cap.read()
            if ok and frame is not None and _frame_is_useful(frame):
                chosen = frame
                break
        if chosen is None:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = cap.read()
            if not ok or frame is None:
                return None
            chosen = frame

        height, width = chosen.shape[:2]
        target_width = 240
        target_height = 135
        scale = max(target_width / max(1, width), target_height / max(1, height))
        resized = cv2.resize(
            chosen,
            (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
            interpolation=cv2.INTER_AREA,
        )
        rh, rw = resized.shape[:2]
        left = max(0, (rw - target_width) // 2)
        top = max(0, (rh - target_height) // 2)
        crop = resized[top : top + target_height, left : left + target_width]
        destination.parent.mkdir(parents=True, exist_ok=True)
        if cv2.imwrite(str(destination), crop):
            return destination
        return None
    finally:
        cap.release()


def render_final_dashboard(
    *,
    output_path: str | Path,
    video_path: str | Path,
    annotated_video: str | Path,
    movement_2d_video: str | Path,
    performance_report: str | Path,
) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    annotated = Path(annotated_video).name
    movement_2d = Path(movement_2d_video).name
    report = Path(performance_report).name
    athlete_videos = _discover_athlete_videos(video_path)

    assets_dir = output.parent / "dashboard_assets"
    assets_dir.mkdir(parents=True, exist_ok=True)

    video_rows: list[str] = []
    playlist: list[dict[str, object]] = []
    for index, path in enumerate(athlete_videos):
        thumbnail_path = assets_dir / f"athlete_video_{index + 1:03d}.jpg"
        created_thumbnail = _create_thumbnail(path, thumbnail_path)
        thumbnail_uri = created_thumbnail.as_uri() if created_thumbnail else None
        duration = _video_duration(path)
        uri = path.as_uri()
        playlist.append(
            {
                "name": path.name,
                "uri": uri,
                "duration": duration,
                "duration_text": _format_duration(duration),
                "thumbnail": thumbnail_uri,
            }
        )

        thumb_html = (
            f'<img class="thumb" src="{escape(thumbnail_uri)}" alt="Prévia de {escape(path.name)}">'
            if thumbnail_uri
            else '<div class="thumb placeholder">▶</div>'
        )
        video_rows.append(
            f'<button class="video-item" type="button" data-index="{index}">'
            f'{thumb_html}<span class="video-copy"><strong>{escape(path.name)}</strong>'
            f'<small>{escape(_format_duration(duration))}</small></span>'
            f'<span class="open-icon">▶</span></button>'
        )

    playlist_json = json.dumps(playlist, ensure_ascii=False).replace("</", "<\\/")
    rows_html = "".join(video_rows) or '<div class="empty">Nenhum vídeo encontrado na pasta do atleta.</div>'
    video_count = len(athlete_videos)

    html = f'''<!doctype html>
<html lang="pt-BR">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Painel Final - Beach Tennis Analyst</title>
<style>
:root{{--navy:#12325b;--teal:#0b8b94;--bg:#eef3f7;--card:#ffffff;--line:#d5dfe7;--muted:#63778b;--ink:#17324f}}
*{{box-sizing:border-box}}
html,body{{margin:0;min-height:100%;background:var(--bg);font-family:Arial,Helvetica,sans-serif;color:var(--ink)}}
body{{padding:12px}}
.header{{display:flex;justify-content:space-between;align-items:center;gap:16px;margin:0 0 10px;padding:0 4px}}
.header h1{{margin:0;font-size:22px;color:var(--navy)}}
.header-meta{{display:flex;gap:8px;align-items:center;flex-wrap:wrap;justify-content:flex-end}}
.chip{{font-size:12px;color:var(--navy);background:white;border:1px solid var(--line);border-radius:8px;padding:7px 10px}}
.dashboard{{display:grid;grid-template-columns:1fr 1fr;grid-template-rows:minmax(330px,47vh) minmax(360px,47vh);gap:10px;min-height:calc(100vh - 58px)}}
.panel{{position:relative;overflow:hidden;background:var(--card);border:1px solid var(--line);border-radius:12px}}
.panel-title{{position:absolute;z-index:3;top:8px;left:10px;background:rgba(18,50,91,.9);color:white;border-radius:7px;padding:5px 9px;font-size:12px;font-weight:700;pointer-events:none}}
.sync-badge{{position:absolute;z-index:3;top:8px;right:10px;background:rgba(11,139,148,.9);color:white;border-radius:7px;padding:5px 9px;font-size:11px;font-weight:700;pointer-events:none}}
video{{width:100%;height:100%;display:block;background:#0d1722;object-fit:contain}}
iframe{{width:100%;height:100%;border:0;background:white}}
.library{{display:grid;grid-template-columns:minmax(230px,34%) 1fr;height:100%;padding-top:34px}}
.video-list{{overflow:auto;border-right:1px solid var(--line);padding:8px;background:#f9fbfc}}
.video-item{{width:100%;display:grid;grid-template-columns:82px 1fr 24px;align-items:center;gap:9px;text-align:left;border:1px solid transparent;background:white;padding:7px;border-radius:10px;color:var(--ink);cursor:pointer;margin-bottom:7px}}
.video-item:hover,.video-item.active{{background:#e9f4f5;border-color:#b8dadd}}
.thumb{{width:82px;height:48px;object-fit:cover;border-radius:7px;background:#0d1722}}
.placeholder{{display:flex;align-items:center;justify-content:center;color:white}}
.video-copy{{display:flex;flex-direction:column;min-width:0}}
.video-copy strong{{font-size:12px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}}
.video-copy small{{font-size:11px;color:var(--muted);margin-top:4px}}
.open-icon{{color:var(--teal);font-size:12px}}
.library-player{{min-width:0;display:grid;grid-template-rows:1fr auto;background:#0d1722}}
.library-info{{background:white;border-top:1px solid var(--line);padding:9px 12px;display:flex;justify-content:space-between;align-items:center;gap:10px}}
.library-info strong{{font-size:12px;color:var(--navy)}}
.library-info span{{font-size:11px;color:var(--muted)}}
.empty{{padding:12px;color:var(--muted);font-size:13px}}
@media(max-width:900px){{body{{padding:6px}}.header{{align-items:flex-start;flex-direction:column}}.header-meta{{justify-content:flex-start}}.dashboard{{grid-template-columns:1fr;grid-template-rows:repeat(4,minmax(320px,70vh))}}.library{{grid-template-columns:190px 1fr}}}}
</style>
</head>
<body>
<div class="header">
  <div><h1>Painel Final de Análise - Beach Tennis</h1></div>
  <div class="header-meta">
    <span class="chip">{escape(Path(video_path).name)}</span>
    <span class="chip">meia quadra</span>
    <span class="chip">2 atletas</span>
    <span class="chip">{video_count} vídeo{'s' if video_count != 1 else ''} do atleta</span>
  </div>
</div>
<main class="dashboard">
<section class="panel">
<div class="panel-title">VÍDEO ANALISADO + MARCAÇÕES</div>
<div class="sync-badge">MASTER</div>
<video id="annotated-video" controls playsinline src="{escape(annotated)}"></video>
</section>
<section class="panel">
<div class="panel-title">MOVIMENTO 2D DA DUPLA</div>
<div class="sync-badge">SINCRONIZADO</div>
<video id="movement-video" controls muted playsinline src="{escape(movement_2d)}"></video>
</section>
<section class="panel">
<div class="panel-title">ANÁLISE GRÁFICA</div>
<iframe src="{escape(report)}" title="Relatório gráfico de desempenho"></iframe>
</section>
<section class="panel">
<div class="panel-title">VÍDEOS DO ATLETA</div>
<div class="library">
<div class="video-list">{rows_html}</div>
<div class="library-player">
  <video id="athlete-player" controls playsinline muted></video>
  <div class="library-info"><strong id="library-name">Selecione um vídeo</strong><span id="library-duration">--:--</span></div>
</div>
</div>
</section>
</main>
<script>
const playlist = {playlist_json};
const libraryPlayer = document.getElementById('athlete-player');
const libraryName = document.getElementById('library-name');
const libraryDuration = document.getElementById('library-duration');
const buttons = [...document.querySelectorAll('.video-item')];
const master = document.getElementById('annotated-video');
const follower = document.getElementById('movement-video');
let syncing = false;
const MAX_DRIFT_SECONDS = 0.08;

function openVideo(index, fromUser = false) {{
  const item = playlist[index];
  if (!item) return;
  libraryPlayer.src = item.uri;
  libraryPlayer.muted = !fromUser;
  libraryPlayer.load();
  libraryName.textContent = item.name;
  libraryDuration.textContent = item.duration_text || '--:--';
  libraryPlayer.play().catch(() => {{}});
  buttons.forEach((button, i) => button.classList.toggle('active', i === index));
}}
buttons.forEach(button => button.addEventListener('click', () => openVideo(Number(button.dataset.index), true)));
if (playlist.length) openVideo(0, false);

function alignFollower(force = false) {{
  if (!Number.isFinite(master.currentTime) || follower.readyState < 1) return;
  const drift = Math.abs(follower.currentTime - master.currentTime);
  if (force || drift > MAX_DRIFT_SECONDS) {{
    syncing = true;
    follower.currentTime = master.currentTime;
    syncing = false;
  }}
  if (Math.abs(follower.playbackRate - master.playbackRate) > 0.001) follower.playbackRate = master.playbackRate;
}}

master.addEventListener('loadedmetadata', () => alignFollower(true));
follower.addEventListener('loadedmetadata', () => alignFollower(true));
master.addEventListener('play', () => {{ alignFollower(true); follower.playbackRate = master.playbackRate; follower.play().catch(() => {{}}); }});
master.addEventListener('pause', () => follower.pause());
master.addEventListener('seeking', () => alignFollower(true));
master.addEventListener('seeked', () => alignFollower(true));
master.addEventListener('ratechange', () => {{ follower.playbackRate = master.playbackRate; }});
master.addEventListener('timeupdate', () => alignFollower(false));
master.addEventListener('ended', () => {{ follower.pause(); alignFollower(true); }});

follower.addEventListener('play', () => {{ if (!syncing && master.paused) master.play().catch(() => {{}}); }});
follower.addEventListener('pause', () => {{ if (!syncing && !master.paused && !master.ended) master.pause(); }});
follower.addEventListener('seeking', () => {{
  if (syncing || !Number.isFinite(follower.currentTime)) return;
  syncing = true;
  master.currentTime = follower.currentTime;
  syncing = false;
}});
follower.addEventListener('ratechange', () => {{
  if (syncing) return;
  syncing = true;
  master.playbackRate = follower.playbackRate;
  syncing = false;
}});
</script>
</body>
</html>'''
    output.write_text(html, encoding="utf-8")
