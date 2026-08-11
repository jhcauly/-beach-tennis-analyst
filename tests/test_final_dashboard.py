from pathlib import Path

from beach_tennis_analyst.render.final_dashboard import render_final_dashboard


def test_final_dashboard_contains_four_quadrants_and_synchronization(tmp_path: Path) -> None:
    videos = tmp_path / "videos"
    videos.mkdir()
    for index in range(1, 6):
        (videos / f"clip_{index:03d}.mp4").write_bytes(b"not-a-real-video")

    output_dir = tmp_path / "output"
    output_dir.mkdir()
    annotated = output_dir / "annotated_match.mp4"
    movement = output_dir / "movement_2d.mp4"
    report = output_dir / "performance_report.html"
    annotated.write_bytes(b"")
    movement.write_bytes(b"")
    report.write_text("<html><body>report</body></html>", encoding="utf-8")

    dashboard = output_dir / "final_dashboard.html"
    render_final_dashboard(
        output_path=dashboard,
        video_path=videos / "clip_001.mp4",
        annotated_video=annotated,
        movement_2d_video=movement,
        performance_report=report,
    )

    html = dashboard.read_text(encoding="utf-8")
    assert "VÍDEO ANALISADO + MARCAÇÕES" in html
    assert "MOVIMENTO 2D DA DUPLA" in html
    assert "ANÁLISE GRÁFICA" in html
    assert "VÍDEOS DO ATLETA" in html
    assert 'id="annotated-video"' in html
    assert 'id="movement-video"' in html
    assert "MAX_DRIFT_SECONDS = 0.08" in html
    assert "5 vídeos do atleta" in html
    for index in range(1, 6):
        assert f"clip_{index:03d}.mp4" in html


def test_final_dashboard_autostarts_library_muted(tmp_path: Path) -> None:
    video = tmp_path / "clip_001.mp4"
    video.write_bytes(b"not-a-real-video")
    output = tmp_path / "result"
    output.mkdir()
    annotated = output / "annotated_match.mp4"
    movement = output / "movement_2d.mp4"
    report = output / "performance_report.html"
    annotated.write_bytes(b"")
    movement.write_bytes(b"")
    report.write_text("report", encoding="utf-8")

    dashboard = output / "final_dashboard.html"
    render_final_dashboard(
        output_path=dashboard,
        video_path=video,
        annotated_video=annotated,
        movement_2d_video=movement,
        performance_report=report,
    )

    html = dashboard.read_text(encoding="utf-8")
    assert 'id="athlete-player" controls playsinline muted' in html
    assert "if (playlist.length) openVideo(0, false);" in html
    assert "libraryPlayer.muted = !fromUser;" in html
