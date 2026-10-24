"""Control panel: pick your video folder, train, export, and play videos with the upscaling player.

    ~/vsr-venv/bin/python panel.py
"""
import json, queue, re, signal, subprocess, sys, threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, ttk

ROOT = Path(__file__).resolve().parent
SETTINGS = ROOT / "panel.json"
PLAYER = ROOT / "build" / "open-player"
STEP_LINE = re.compile(r"step (\d+) .*?\+?(-?[\d.]+) dB")  # train.py's progress lines


class Panel:
    def __init__(self, root):
        self.root, self.q, self.job = root, queue.Queue(), None
        try:
            s = json.loads(SETTINGS.read_text())
        except (OSError, ValueError):
            s = {}
        self.videos = tk.StringVar(value=s.get("videos", str(ROOT / "videos")))
        self.steps = tk.StringVar(value=s.get("steps", "100000"))
        self.size = tk.StringVar(value=s.get("size", "standard"))
        self.video = tk.StringVar(value=s.get("video", ""))
        self.status = tk.StringVar(value="idle")

        self.stats, self.check_out, self.player_info = [], [], None
        root.title("Open Player")
        root.columnconfigure(0, weight=1)
        root.rowconfigure(4, weight=1)
        pad = dict(padx=8, pady=4)

        tr = ttk.LabelFrame(root, text="Training")
        tr.grid(row=0, column=0, sticky="ew", **pad)
        tr.columnconfigure(1, weight=1)
        ttk.Label(tr, text="Video folder").grid(row=0, column=0, sticky="w", **pad)
        ttk.Entry(tr, textvariable=self.videos).grid(row=0, column=1, sticky="ew", **pad)
        ttk.Button(tr, text="Browse…", command=self.pick_folder).grid(row=0, column=2, **pad)
        ttk.Label(tr, text="Total steps").grid(row=1, column=0, sticky="w", **pad)
        opts = ttk.Frame(tr)
        opts.grid(row=1, column=1, sticky="w", **pad)
        ttk.Entry(opts, textvariable=self.steps, width=10).pack(side="left")
        ttk.Label(opts, text="Model size").pack(side="left", padx=(16, 4))
        ttk.Combobox(opts, textvariable=self.size, values=["standard", "fast"], state="readonly", width=9).pack(side="left")
        ttk.Label(opts, text="standard: best for up to 30 fps · fast: 60 fps at 1080p → 4K").pack(side="left", padx=8)
        btns = ttk.Frame(tr)
        btns.grid(row=1, column=2, **pad)
        self.train_btn = ttk.Button(btns, text="Train", command=self.train)
        self.train_btn.pack(side="left")
        self.stop_btn = ttk.Button(btns, text="Stop", command=self.stop, state="disabled")
        self.stop_btn.pack(side="left", padx=(4, 0))
        self.progress = ttk.Progressbar(tr, maximum=1)
        self.progress.grid(row=2, column=0, columnspan=3, sticky="ew", **pad)

        mo = ttk.LabelFrame(root, text="Model")
        mo.grid(row=1, column=0, sticky="ew", **pad)
        self.export_btn = ttk.Button(mo, text="Export for player + self-test", command=self.export)
        self.export_btn.pack(side="left", **pad)
        ttk.Label(mo, text="vsr.pt  →  vsr.onnx").pack(side="left", **pad)

        pl = ttk.LabelFrame(root, text="Play")
        pl.grid(row=2, column=0, sticky="ew", **pad)
        pl.columnconfigure(0, weight=1)
        ttk.Entry(pl, textvariable=self.video).grid(row=0, column=0, sticky="ew", **pad)
        ttk.Button(pl, text="Browse…", command=self.pick_video).grid(row=0, column=1, **pad)
        ttk.Button(pl, text="Play", command=self.play).grid(row=0, column=2, **pad)

        dg = ttk.LabelFrame(root, text="Diagnostics")
        dg.grid(row=3, column=0, sticky="ew", **pad)
        dg.columnconfigure(0, weight=1)
        self.info = tk.StringVar(value="Play a video to see live numbers here.")
        self.live, self.verdict = tk.StringVar(), tk.StringVar()
        ttk.Label(dg, textvariable=self.info).grid(row=0, column=0, sticky="w", **pad)
        ttk.Label(dg, textvariable=self.live, font="TkFixedFont").grid(row=1, column=0, sticky="w", **pad)
        self.verdict_label = ttk.Label(dg, textvariable=self.verdict)
        self.verdict_label.grid(row=2, column=0, sticky="w", **pad)
        side = ttk.Frame(dg)
        side.grid(row=0, column=1, rowspan=3, sticky="ne", **pad)
        ttk.Button(side, text="System check", command=self.system_check).pack(fill="x")
        ttk.Button(side, text="Copy report", command=self.copy_report).pack(fill="x", pady=(4, 0))
        self.graph = tk.Canvas(dg, height=56, background="#151515", highlightthickness=0)
        self.graph.grid(row=3, column=0, columnspan=2, sticky="ew", **pad)

        logf = ttk.Frame(root)
        logf.grid(row=4, column=0, sticky="nsew", **pad)
        logf.rowconfigure(0, weight=1)
        logf.columnconfigure(0, weight=1)
        self.log = tk.Text(logf, height=14, wrap="none", state="disabled")
        self.log.grid(row=0, column=0, sticky="nsew")
        sb = ttk.Scrollbar(logf, command=self.log.yview)
        sb.grid(row=0, column=1, sticky="ns")
        self.log["yscrollcommand"] = sb.set
        ttk.Label(root, textvariable=self.status).grid(row=5, column=0, sticky="w", **pad)

        root.protocol("WM_DELETE_WINDOW", self.close)
        root.after(100, self.poll)

    def save_settings(self):
        SETTINGS.write_text(json.dumps({"videos": self.videos.get(), "steps": self.steps.get(), "size": self.size.get(),
                                        "video": self.video.get()}))

    def pick_folder(self):
        if d := filedialog.askdirectory(initialdir=self.videos.get() or ROOT):
            self.videos.set(d)
            self.save_settings()

    def pick_video(self):
        f = filedialog.askopenfilename(
            initialdir=Path(self.video.get()).parent if self.video.get() else self.videos.get(),
            filetypes=[("Video", "*.mp4 *.mkv *.webm *.mov *.avi *.m4v *.ts"), ("All files", "*")])
        if f:
            self.video.set(f)
            self.save_settings()

    def write(self, text):
        self.log["state"] = "normal"
        self.log.insert("end", text)
        self.log.see("end")
        self.log["state"] = "disabled"

    def start(self, args, what, then=None):
        """Run one job (train / export / self-test) at a time, streaming its output into the log."""
        if self.job:
            return
        self.save_settings()
        self.write(f"\n$ {' '.join(map(str, args))}\n")
        self.job = subprocess.Popen(args, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        self.status.set(f"{what}…")
        self.busy(True)
        threading.Thread(target=self.pump, args=(self.job, what, then), daemon=True).start()

    def pump(self, p, what, then):  # background thread: Tk is only touched from poll()
        for line in p.stdout:
            self.q.put(("line", line))
        self.q.put(("exit", (p, what, p.wait(), then)))

    def poll(self):
        while not self.q.empty():
            kind, val = self.q.get()
            if kind == "line":
                tag, _, payload = val.partition(" ")
                if tag in ("info", "stats") and payload.startswith("{"):  # the player's --stats lines
                    try:
                        self.diagnostics(tag, json.loads(payload))
                        continue
                    except ValueError:
                        pass
                if val[:6] in ("PASS  ", "WARN  ", "FAIL  "):  # --check results, kept for the report
                    self.check_out.append(val.rstrip())
                self.write(val)
                if m := STEP_LINE.search(val):
                    self.progress["value"] = int(m[1])
                    self.status.set(f"training: step {m[1]}, {float(m[2]):+.2f} dB vs bilinear")
            else:
                p, what, code, then = val
                if p is self.job:  # the player runs on its own and doesn't hold the job slot
                    self.job = None
                    self.busy(False)
                self.status.set(f"{what} finished" if code == 0 else f"{what} failed (exit code {code}), see log")
                if code == 0 and then:
                    then()
        self.root.after(100, self.poll)

    def busy(self, on):
        for b in (self.train_btn, self.export_btn):
            b["state"] = "disabled" if on else "normal"
        self.stop_btn["state"] = "normal" if on else "disabled"

    def train(self):
        if not self.steps.get().isdigit() or int(self.steps.get()) < 1:
            return self.status.set("steps must be a whole number")
        self.progress["maximum"], self.progress["value"] = int(self.steps.get()), 0
        args = [sys.executable, "-u", "train.py", self.videos.get(), "--out", "vsr.pt", "--steps", self.steps.get(),
                "--size", self.size.get()]
        if (ROOT / "vsr.pt").exists():
            args += ["--resume", "vsr.pt"]  # keep improving the same model
        self.start(args, "training")

    def stop(self):
        if self.job:
            self.job.send_signal(signal.SIGINT)  # train.py saves on SIGINT before exiting
            self.status.set("stopping after this step…")

    def export(self):
        if not (ROOT / "vsr.pt").exists():
            return self.status.set("no vsr.pt yet: train first")
        selftest = lambda: PLAYER.exists() and self.start([PLAYER, "--selftest", "vsr.onnx"], "self-test")
        self.start([sys.executable, "-u", "export.py", "vsr.pt"], "export", then=selftest)

    def play(self):
        if not PLAYER.exists():
            return self.status.set(f"player not built yet: {PLAYER}")
        if not Path(self.video.get()).is_file():
            return self.status.set("pick a video to play")
        self.save_settings()
        args = [PLAYER, self.video.get(), "--stats"]
        args += ["--model", "vsr.onnx"] if (ROOT / "vsr.onnx").exists() else ["--no-vsr"]
        self.stats = []
        self.write(f"\n$ {' '.join(map(str, args))}\n")
        p = subprocess.Popen(args, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        threading.Thread(target=self.pump, args=(p, "player", None), daemon=True).start()

    def diagnostics(self, tag, d):
        if tag == "info":
            self.player_info = d
            self.info.set(f"{d['gpu']}  |  decode: {d['decode']}  |  display: {d['display']}  |  "
                          f"{d['video']} on a {d.get('screen', '?')} screen  |  VSR {d['vsr']}  |  TensorRT {d['tensorrt']}")
            return
        self.stats = (self.stats + [d])[-60:]
        self.live.set(f"{d['fps']:>3}/{d['src_fps']:.0f} fps   dropped {d['dropped']:<3}  "
                      f"VSR {d['vsr_ms']:5.1f} ms (max {d['vsr_max_ms']:.1f})   "
                      f"display {d['copy_ms'] + d['map_ms'] + d['present_ms']:5.1f} ms   late {d['late_ms']:4.0f} ms   "
                      f"queue {d['queue']:2.0f}   audio {d['audio_s']:.1f} s   GPU {d['gpu_mem_mb']:.0f} MB")
        ok = d["verdict"] == "ok"
        self.verdict.set("Keeping up." if ok else f"Dropping frames. Bottleneck: {d['verdict']}")
        self.verdict_label["foreground"] = "#2a8a2a" if ok else "#cc3333"
        self.draw_graph()

    def draw_graph(self):
        """Last minute: bar height = fps reached / source fps, red where frames were dropped."""
        c = self.graph
        c.delete("all")
        W, H, n = max(c.winfo_width(), 1), int(c["height"]), 60
        for i, d in enumerate(self.stats):
            h = min(d["fps"] / max(d["src_fps"], 1), 1) * (H - 14)
            x = i * W / n
            c.create_rectangle(x, H - h, x + W / n - 1, H, width=0, fill="#cc3333" if d["dropped"] else "#2a8a2a")
        c.create_text(4, 2, anchor="nw", fill="#999", text="fps vs source, last minute (red: frames dropped)")

    def system_check(self):
        if not PLAYER.exists():
            return self.status.set(f"player not built yet: {PLAYER}")
        self.check_out = []
        self.start([PLAYER, "--check", "vsr.onnx"], "system check")

    def copy_report(self):
        report = {"info": self.player_info, "check": self.check_out, "stats": self.stats[-10:]}
        self.root.clipboard_clear()
        self.root.clipboard_append(json.dumps(report, indent=1))
        self.status.set("diagnostics report copied to the clipboard")

    def close(self):
        if self.job:  # let training save instead of dying mid-step
            self.job.send_signal(signal.SIGINT)
            try:
                self.job.wait(timeout=120)
            except subprocess.TimeoutExpired:
                self.job.kill()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    root.geometry("760x560")
    Panel(root)
    root.mainloop()
