from __future__ import annotations

import json
from html import escape
from pathlib import Path

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

    video_rows = []
    playlist = []
    for index, path in enumerate(athlete_videos):
        uri = path.as_uri()
        playlist.append({"name": path.name, "uri": uri})
        video_rows.append(
            f'<button class="video-item" type="button" data-index="{index}">'
            f'<span class="play">▶</span><span>{escape(path.name)}</span></button>'
        )

    playlist_json = json.dumps(playlist, ensure_ascii=False).replace("</", "<\\/")
    rows_html = "".join(video_rows) or '<div class="empty">Nenhum vídeo encontrado na pasta do atleta.</div>'

    html = f'''<!doctype html>
<html lang="pt-BR">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Painel Final - Beach Tennis Analyst</title>
<style>
:root{{--navy:#12325b;--teal:#0b8b94;--bg:#eef3f7;--card:#ffffff;--line:#d5dfe7;--muted:#63778b}}
*{{box-sizing:border-box}}
html,body{{margin:0;min-height:100%;background:var(--bg);font-family:Arial,Helvetica,sans-serif;color:#17324f}}
body{{padding:12px}}
.header{{display:flex;justify-content:space-between;align-items:center;gap:16px;margin:0 0 10px;padding:0 4px}}
.header h1{{margin:0;font-size:22px;color:var(--navy)}}
.header span{{font-size:12px;color:var(--muted)}}
.dashboard{{display:grid;grid-template-columns:1fr 1fr;grid-template-rows:minmax(320px,47vh) minmax(320px,47vh);gap:10px;min-height:calc(100vh - 58px)}}
.panel{{position:relative;overflow:hidden;background:var(--card);border:1px solid var(--line);border-radius:12px}}
.panel-title{{position:absolute;z-index:3;top:8px;left:10px;background:rgba(18,50,91,.88);color:white;border-radius:7px;padding:5px 9px;font-size:12px;font-weight:700;pointer-events:none}}
video{{width:100%;height:100%;display:block;background:#0d1722;object-fit:contain}}
iframe{{width:100%;height:100%;border:0;background:white}}
.library{{display:grid;grid-template-columns:220px 1fr;height:100%;padding-top:34px}}
.video-list{{overflow:auto;border-right:1px solid var(--line);padding:8px}}
.video-item{{width:100%;display:flex;align-items:center;gap:8px;text-align:left;border:1px solid transparent;background:transparent;padding:10px 8px;border-radius:8px;color:#17324f;cursor:pointer}}
.video-item:hover,.video-item.active{{background:#e9f4f5;border-color:#b8dadd}}
.play{{color:var(--teal)}}
.library-player{{min-width:0;display:flex;align-items:stretch;background:#0d1722}}
.empty{{padding:12px;color:var(--muted);font-size:13px}}
@media(max-width:900px){{body{{padding:6px}}.dashboard{{grid-template-columns:1fr;grid-template-rows:repeat(4,minmax(300px,70vh))}}.library{{grid-template-columns:180px 1fr}}}}
</style>
</head>
<body>
<div class="header"><h1>Painel Final de Análise - Beach Tennis</h1><span>{escape(Path(video_path).name)} | meia quadra | 2 atletas</span></div>
<main class="dashboard">
<section class="panel">
<div class="panel-title">VÍDEO ANALISADO + MARCAÇÕES</div>
<video controls autoplay muted playsinline src="{escape(annotated)}"></video>
</section>
<section class="panel">
<div class="panel-title">MOVIMENTO 2D DA DUPLA</div>
<video controls autoplay muted loop playsinline src="{escape(movement_2d)}"></video>
</section>
<section class="panel">
<div class="panel-title">ANÁLISE GRÁFICA</div>
<iframe src="{escape(report)}" title="Relatório gráfico de desempenho"></iframe>
</section>
<section class="panel">
<div class="panel-title">VÍDEOS DO ATLETA</div>
<div class="library">
<div class="video-list">{rows_html}</div>
<div class="library-player"><video id="athlete-player" controls playsinline></video></div>
</div>
</section>
</main>
<script>
const playlist = {playlist_json};
const player = document.getElementById('athlete-player');
const buttons = [...document.querySelectorAll('.video-item')];
function openVideo(index) {{
  const item = playlist[index];
  if (!item) return;
  player.src = item.uri;
  player.load();
  player.play().catch(() => {{}});
  buttons.forEach((button, i) => button.classList.toggle('active', i === index));
}}
buttons.forEach(button => button.addEventListener('click', () => openVideo(Number(button.dataset.index))));
if (playlist.length) openVideo(0);
</script>
</body>
</html>'''
    output.write_text(html, encoding="utf-8")
