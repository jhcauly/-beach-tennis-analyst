from __future__ import annotations

from html import escape
from pathlib import Path

from beach_tennis_analyst.analytics.motion import MotionSummary
from beach_tennis_analyst.analytics.pair import PairFrame, PairSummary
from beach_tennis_analyst.domain.models import ObservationStatus, PlayerFrame

NEAR_IDS = ("near_left", "near_right")


def _fmt(value: float, digits: int = 2) -> str:
    return f"{value:.{digits}f}".replace(".", ",")


def _duration(seconds: float) -> str:
    total = max(0, round(seconds))
    minutes, secs = divmod(total, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {secs:02d}s"
    return f"{minutes}m {secs:02d}s"


def _valid_near_frames(trajectories: dict[str, list[PlayerFrame]]) -> list[PlayerFrame]:
    result: list[PlayerFrame] = []
    for athlete_id in NEAR_IDS:
        for frame in trajectories.get(athlete_id, []):
            if frame.observation_status not in {ObservationStatus.REJECTED, ObservationStatus.SUSPECT}:
                result.append(frame)
    return result


def _occupancy_matrix(trajectories: dict[str, list[PlayerFrame]]) -> list[list[float]]:
    counts = [[0 for _ in range(3)] for _ in range(3)]
    total = 0
    for frame in _valid_near_frames(trajectories):
        x = min(max(frame.x_m, 0.0), 7.999999)
        y = min(max(frame.y_m, 0.0), 7.999999)
        col = min(2, int(x / (8.0 / 3.0)))
        depth = min(2, int(y / (8.0 / 3.0)))
        row = 2 - depth
        counts[row][col] += 1
        total += 1
    if not total:
        return [[0.0 for _ in range(3)] for _ in range(3)]
    return [[100.0 * value / total for value in row] for row in counts]


def _heatmap_svg(trajectories: dict[str, list[PlayerFrame]]) -> str:
    width, height = 430, 340
    left, top, court_w, court_h = 35, 25, 360, 285
    parts = [
        f'<svg class="heatmap" viewBox="0 0 {width} {height}" role="img" aria-label="Heatmap de ocupação da meia quadra">',
        '<rect width="100%" height="100%" rx="14" fill="#e8d2ad"/>',
        f'<rect x="{left}" y="{top}" width="{court_w}" height="{court_h}" fill="none" stroke="#ffffff" stroke-width="3"/>',
        f'<line x1="{left + court_w/2}" y1="{top}" x2="{left + court_w/2}" y2="{top + court_h}" stroke="#ffffff" stroke-width="2"/>',
        f'<line x1="{left}" y1="{top + court_h/3}" x2="{left + court_w}" y2="{top + court_h/3}" stroke="#ffffff" stroke-width="1" opacity="0.6"/>',
        f'<line x1="{left}" y1="{top + 2*court_h/3}" x2="{left + court_w}" y2="{top + 2*court_h/3}" stroke="#ffffff" stroke-width="1" opacity="0.6"/>',
        f'<line x1="{left}" y1="{top}" x2="{left + court_w}" y2="{top}" stroke="#16385f" stroke-width="5"/>',
    ]
    colors = {"near_left": "#1556b8", "near_right": "#0d8f93"}
    for athlete_id in NEAR_IDS:
        frames = trajectories.get(athlete_id, [])
        if not frames:
            continue
        step = max(1, len(frames) // 180)
        for frame in frames[::step]:
            if frame.observation_status in {ObservationStatus.REJECTED, ObservationStatus.SUSPECT}:
                continue
            x = left + min(max(frame.x_m, 0.0), 8.0) / 8.0 * court_w
            y = top + (1.0 - min(max(frame.y_m, 0.0), 8.0) / 8.0) * court_h
            parts.append(
                f'<circle cx="{x:.1f}" cy="{y:.1f}" r="15" fill="{colors[athlete_id]}" opacity="0.065"/>'
            )
    parts.append("</svg>")
    return "".join(parts)


def _matrix_html(matrix: list[list[float]]) -> str:
    max_value = max((value for row in matrix for value in row), default=0.0) or 1.0
    rows = []
    labels = ("REDE", "MEIO", "FUNDO")
    for label, values in zip(labels, matrix, strict=True):
        cells = []
        for value in values:
            alpha = 0.08 + 0.72 * value / max_value
            cells.append(
                f'<td style="background:rgba(11,139,148,{alpha:.3f})"><strong>{value:.0f}%</strong></td>'
            )
        rows.append(f'<tr><th>{label}</th>{"".join(cells)}</tr>')
    return (
        '<table class="matrix"><thead><tr><th></th><th>ESQUERDA</th><th>CENTRO</th><th>DIREITA</th></tr></thead>'
        f'<tbody>{"".join(rows)}</tbody></table>'
    )


def _insights(matrix: list[list[float]], pair: PairSummary) -> list[str]:
    zone_names = [
        ["rede/esquerda", "rede/centro", "rede/direita"],
        ["meio/esquerda", "meio/centro", "meio/direita"],
        ["fundo/esquerda", "fundo/centro", "fundo/direita"],
    ]
    best = max(((matrix[r][c], r, c) for r in range(3) for c in range(3)), default=(0.0, 0, 0))
    return [
        f"Maior ocupação conjunta na zona {zone_names[best[1]][best[2]]}: {best[0]:.0f}% das amostras válidas.",
        f"Distância média entre parceiros: {_fmt(pair.average_partner_distance_m)} m.",
        f"Abertura lateral acima do limite em {pair.too_open_ratio * 100:.0f}% das amostras sincronizadas.",
        f"Desalinhamento em profundidade acima do limite em {pair.depth_misaligned_ratio * 100:.0f}% das amostras sincronizadas.",
    ]


def render_performance_report(
    *,
    output_path: str | Path,
    video_name: str,
    trajectories: dict[str, list[PlayerFrame]],
    athlete_summaries: dict[str, MotionSummary],
    pair_frames: list[PairFrame],
    pair_summary: PairSummary,
) -> None:
    summaries = [athlete_summaries[athlete_id] for athlete_id in NEAR_IDS if athlete_id in athlete_summaries]
    total_distance = sum(item.total_distance_m for item in summaries)
    avg_speed = sum(item.average_speed_mps for item in summaries) / max(1, len(summaries))
    max_speed = max((item.maximum_speed_mps for item in summaries), default=0.0)
    max_acc = max((item.maximum_acceleration_mps2 for item in summaries), default=0.0)
    duration = max((item.duration_s for item in summaries), default=0.0)
    matrix = _occupancy_matrix(trajectories)
    insights = _insights(matrix, pair_summary)

    athlete_rows = "".join(
        f"<tr><td>{escape(item.athlete_id)}</td><td>{_fmt(item.total_distance_m)} m</td>"
        f"<td>{_fmt(item.average_speed_mps)} m/s</td><td>{_fmt(item.maximum_speed_mps)} m/s</td>"
        f"<td>{_fmt(item.maximum_acceleration_mps2)} m/s²</td></tr>"
        for item in summaries
    )
    insight_html = "".join(f"<li>{escape(text)}</li>" for text in insights)

    html = f'''<!doctype html>
<html lang="pt-BR"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Relatório Beach Tennis</title>
<style>
:root{{--navy:#12325b;--teal:#0b8b94;--sand:#f5efe4;--line:#d9e1e8;--text:#17324f}}
*{{box-sizing:border-box}} body{{margin:0;background:#f7f9fb;color:var(--text);font-family:Arial,Helvetica,sans-serif}}
.page{{max-width:1380px;margin:0 auto;padding:26px}} h1{{margin:0;color:var(--navy);font-size:34px}} .sub{{margin:8px 0 22px;color:#52677c}}
.cards{{display:grid;grid-template-columns:repeat(5,1fr);gap:14px}} .card,.panel{{background:white;border:1px solid var(--line);border-radius:16px;padding:18px}}
.card span{{font-size:12px;font-weight:700}} .card strong{{display:block;font-size:30px;color:var(--navy);margin-top:7px}}
.grid{{display:grid;grid-template-columns:1.05fr .95fr .95fr;gap:16px;margin-top:16px}} .panel h2{{font-size:16px;color:var(--navy);margin:0 0 14px}}
.heatmap{{display:block;width:100%;height:auto}} .matrix{{width:100%;border-collapse:collapse;text-align:center}} .matrix th,.matrix td{{border:1px solid #dbe4ea;padding:18px 8px}} .matrix th{{font-size:11px}}
.pair strong{{font-size:27px;color:var(--navy)}} .pair div{{padding:13px 0;border-bottom:1px solid var(--line)}} .pair div:last-child{{border-bottom:0}}
.bottom{{display:grid;grid-template-columns:1.2fr .8fr;gap:16px;margin-top:16px}} table.stats{{width:100%;border-collapse:collapse}} .stats th,.stats td{{padding:12px;border-bottom:1px solid var(--line);text-align:left}} .stats th{{font-size:12px}}
li{{margin:10px 0;line-height:1.4}} .note{{margin-top:16px;font-size:12px;color:#687b8d}}
body.embedded{{overflow:hidden;background:white}} body.embedded .page{{max-width:none;padding:8px}} body.embedded h1,body.embedded .sub,body.embedded .bottom{{display:none}}
body.embedded .cards{{gap:6px}} body.embedded .card{{padding:7px 8px;border-radius:9px}} body.embedded .card span{{font-size:9px}} body.embedded .card strong{{font-size:18px;margin-top:3px}}
body.embedded .grid{{grid-template-columns:1.05fr .92fr .85fr;gap:6px;margin-top:6px}} body.embedded .panel{{padding:7px;border-radius:9px}} body.embedded .panel h2{{font-size:10px;margin-bottom:6px}}
body.embedded .heatmap{{max-height:215px}} body.embedded .matrix th,body.embedded .matrix td{{padding:7px 3px;font-size:10px}} body.embedded .matrix th{{font-size:8px}}
body.embedded .pair div{{padding:5px 0;font-size:9px}} body.embedded .pair strong{{font-size:16px}}
@media(max-width:900px){{.cards{{grid-template-columns:repeat(2,1fr)}}.grid,.bottom{{grid-template-columns:1fr}}}}
</style></head><body><main class="page">
<h1>Relatório de Análise de Desempenho – Beach Tennis</h1>
<div class="sub">Vídeo: {escape(video_name)} &nbsp;|&nbsp; Escopo: meia quadra do lado da câmera &nbsp;|&nbsp; Atletas: near_left e near_right</div>
<section class="cards">
<div class="card"><span>DISTÂNCIA TOTAL DA DUPLA</span><strong>{_fmt(total_distance)} m</strong></div>
<div class="card"><span>VELOCIDADE MÉDIA</span><strong>{_fmt(avg_speed)} m/s</strong></div>
<div class="card"><span>VELOCIDADE MÁXIMA</span><strong>{_fmt(max_speed)} m/s</strong></div>
<div class="card"><span>ACELERAÇÃO MÁXIMA</span><strong>{_fmt(max_acc)} m/s²</strong></div>
<div class="card"><span>TEMPO ANALISADO</span><strong>{_duration(duration)}</strong></div>
</section>
<section class="grid">
<div class="panel"><h2>HEATMAP DE OCUPAÇÃO – MEIA QUADRA</h2>{_heatmap_svg(trajectories)}</div>
<div class="panel"><h2>OCUPAÇÃO PERCENTUAL DA QUADRA (3×3)</h2>{_matrix_html(matrix)}</div>
<div class="panel pair"><h2>ORGANIZAÇÃO DA DUPLA</h2>
<div>Distância média entre parceiros<br><strong>{_fmt(pair_summary.average_partner_distance_m)} m</strong></div>
<div>Abertura lateral média<br><strong>{_fmt(pair_summary.average_lateral_opening_m)} m</strong></div>
<div>Desalinhamento médio em profundidade<br><strong>{_fmt(pair_summary.average_depth_misalignment_m)} m</strong></div>
<div>Amostras sincronizadas<br><strong>{pair_summary.valid_samples}</strong></div>
</div></section>
<section class="bottom">
<div class="panel"><h2>DESEMPENHO POR ATLETA</h2><table class="stats"><thead><tr><th>ATLETA</th><th>DISTÂNCIA</th><th>VEL. MÉDIA</th><th>VEL. MÁX.</th><th>ACEL. MÁX.</th></tr></thead><tbody>{athlete_rows}</tbody></table>
<div class="note">A identidade é estabilizada por posição, ID do detector e assinatura visual da roupa no tronco.</div></div>
<div class="panel"><h2>PRINCIPAIS INSIGHTS</h2><ul>{insight_html}</ul><div class="note">Os insights usam apenas métricas disponíveis no tracking da dupla. Métricas de rally/bola só aparecem quando a análise de bola estiver habilitada.</div></div>
</section>
</main>
<script>if(new URLSearchParams(window.location.search).get('embedded')==='1')document.body.classList.add('embedded');</script>
</body></html>'''
    Path(output_path).write_text(html, encoding="utf-8")
