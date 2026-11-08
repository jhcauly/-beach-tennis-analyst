from __future__ import annotations

import queue
import subprocess
import threading
from dataclasses import dataclass
from pathlib import Path
import tkinter as tk
from tkinter import messagebox, ttk


VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".m4v"}


@dataclass(frozen=True, slots=True)
class VideoItem:
    path: Path
    calibration_path: Path
    output_dir: Path

    @property
    def name(self) -> str:
        return self.path.name

    @property
    def calibrated(self) -> bool:
        return self.calibration_path.exists()

    @property
    def analyzed(self) -> bool:
        return (self.output_dir / "summary.json").exists()


class BeachTennisLauncher:
    def __init__(self, root: tk.Tk, project_root: Path) -> None:
        self.root = root
        self.project_root = project_root
        self.videos_dir = project_root / "videos"
        self.output_root = project_root / "output"
        self.items: list[VideoItem] = []
        self.item_by_iid: dict[str, VideoItem] = {}
        self.events: queue.Queue[tuple[str, str]] = queue.Queue()
        self.running = False

        root.title("Beach Tennis Analyst")
        root.geometry("760x480")
        root.minsize(680, 420)

        container = ttk.Frame(root, padding=14)
        container.pack(fill=tk.BOTH, expand=True)

        ttk.Label(
            container,
            text="Selecione os vídeos que deseja analisar",
            font=("Segoe UI", 13, "bold"),
        ).pack(anchor=tk.W, pady=(0, 10))

        columns = ("video", "calibration", "analysis")
        self.tree = ttk.Treeview(
            container,
            columns=columns,
            show="headings",
            selectmode="extended",
            height=13,
        )
        self.tree.heading("video", text="Vídeo")
        self.tree.heading("calibration", text="Calibração")
        self.tree.heading("analysis", text="Análise")
        self.tree.column("video", width=400, anchor=tk.W)
        self.tree.column("calibration", width=130, anchor=tk.CENTER)
        self.tree.column("analysis", width=130, anchor=tk.CENTER)
        self.tree.pack(fill=tk.BOTH, expand=True)

        buttons = ttk.Frame(container)
        buttons.pack(fill=tk.X, pady=(12, 8))
        self.refresh_button = ttk.Button(buttons, text="Atualizar lista", command=self.refresh)
        self.refresh_button.pack(side=tk.LEFT)
        self.select_all_button = ttk.Button(
            buttons, text="Selecionar todos", command=self.select_all
        )
        self.select_all_button.pack(side=tk.LEFT, padx=(8, 0))
        self.analyze_button = ttk.Button(
            buttons, text="ANALISAR SELECIONADOS", command=self.analyze_selected
        )
        self.analyze_button.pack(side=tk.RIGHT)

        self.status = tk.StringVar(value="Pronto.")
        ttk.Label(container, textvariable=self.status).pack(anchor=tk.W)

        self.refresh()
        self.root.after(150, self._poll_events)

    def refresh(self) -> None:
        self.videos_dir.mkdir(parents=True, exist_ok=True)
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.items = []
        self.item_by_iid.clear()
        for iid in self.tree.get_children():
            self.tree.delete(iid)

        for path in sorted(self.videos_dir.iterdir(), key=lambda p: p.name.lower()):
            if not path.is_file() or path.suffix.lower() not in VIDEO_EXTENSIONS:
                continue
            stem = path.stem
            item = VideoItem(
                path=path,
                calibration_path=self.project_root / f"calibration_{stem}.json",
                output_dir=self.output_root / stem,
            )
            self.items.append(item)
            iid = self.tree.insert(
                "",
                tk.END,
                values=(
                    item.name,
                    "Calibrado" if item.calibrated else "Sem calibração",
                    "Analisado" if item.analyzed else "Não analisado",
                ),
            )
            self.item_by_iid[iid] = item

        self.status.set(f"{len(self.items)} vídeo(s) encontrado(s) em {self.videos_dir}")

    def select_all(self) -> None:
        children = self.tree.get_children()
        if children:
            self.tree.selection_set(children)

    def analyze_selected(self) -> None:
        if self.running:
            return
        selection = self.tree.selection()
        selected = [self.item_by_iid[iid] for iid in selection if iid in self.item_by_iid]
        if not selected:
            messagebox.showinfo("Beach Tennis Analyst", "Selecione pelo menos um vídeo.")
            return

        self.running = True
        self._set_controls_enabled(False)
        threading.Thread(target=self._worker, args=(selected,), daemon=True).start()

    def _worker(self, selected: list[VideoItem]) -> None:
        try:
            total = len(selected)
            for index, item in enumerate(selected, start=1):
                self.events.put(("status", f"[{index}/{total}] Preparando {item.name}..."))
                if not item.calibration_path.exists():
                    self.events.put(
                        (
                            "status",
                            f"[{index}/{total}] Calibração necessária para {item.name}. "
                            "Marque os 4 cantos e confirme.",
                        )
                    )
                    preview = self.project_root / f"calibration_{item.path.stem}_preview.jpg"
                    top_down = self.project_root / f"calibration_{item.path.stem}_topdown.jpg"
                    subprocess.run(
                        [
                            "beach-tennis-calibrate",
                            str(item.path),
                            str(item.calibration_path),
                            "--preview",
                            str(preview),
                            "--top-down",
                            str(top_down),
                        ],
                        cwd=self.project_root,
                        check=True,
                    )

                self.events.put(("status", f"[{index}/{total}] Analisando {item.name}..."))
                item.output_dir.mkdir(parents=True, exist_ok=True)
                subprocess.run(
                    [
                        "beach-tennis-analyze",
                        str(item.path),
                        str(item.calibration_path),
                        str(item.output_dir),
                    ],
                    cwd=self.project_root,
                    check=True,
                )
                self.events.put(("refresh", ""))

            self.events.put(("done", f"Análise concluída para {total} vídeo(s)."))
        except subprocess.CalledProcessError as exc:
            self.events.put(("error", f"O processamento parou com código {exc.returncode}."))
        except Exception as exc:  # pragma: no cover - GUI safety net
            self.events.put(("error", str(exc)))

    def _poll_events(self) -> None:
        try:
            while True:
                event, payload = self.events.get_nowait()
                if event == "status":
                    self.status.set(payload)
                elif event == "refresh":
                    self.refresh()
                elif event == "done":
                    self.running = False
                    self._set_controls_enabled(True)
                    self.refresh()
                    self.status.set(payload)
                    messagebox.showinfo("Beach Tennis Analyst", payload)
                elif event == "error":
                    self.running = False
                    self._set_controls_enabled(True)
                    self.status.set("Processamento interrompido.")
                    messagebox.showerror("Beach Tennis Analyst", payload)
        except queue.Empty:
            pass
        self.root.after(150, self._poll_events)

    def _set_controls_enabled(self, enabled: bool) -> None:
        state = tk.NORMAL if enabled else tk.DISABLED
        self.refresh_button.configure(state=state)
        self.select_all_button.configure(state=state)
        self.analyze_button.configure(state=state)


def main() -> int:
    project_root = Path.cwd()
    root = tk.Tk()
    BeachTennisLauncher(root, project_root)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
