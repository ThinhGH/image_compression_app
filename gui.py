import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk
import cv2
import numpy as np

from jpeg_codec import compress_rgb, decompress_rgb
from metrics import psnr, ssim

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure


class JPEGApp:
    def __init__(self, root):
        self.root = root
        root.title("JPEG-like Image Compression")
        root.configure(bg="#0f172a")
        root.minsize(980, 640)

        # Modern-ish ttk styling
        self.style = ttk.Style()
        self.style.theme_use("clam")
        self.style.configure("TFrame", background="#0f172a")
        self.style.configure("TLabel", background="#0f172a", foreground="#e2e8f0", font=("Segoe UI", 11))
        self.style.configure("Header.TLabel", font=("Segoe UI Semibold", 16))
        self.style.configure("Accent.TButton", background="#10b981", foreground="#0b1021")
        self.style.map(
            "Accent.TButton",
            background=[("active", "#34d399")],
            relief=[("pressed", "groove"), ("!pressed", "raised")],
        )
        self.style.configure("Card.TFrame", background="#111827", relief="ridge", borderwidth=1)
        self.style.configure("Card.TLabel", background="#111827", foreground="#e2e8f0")

        self.img = None
        self.recon = None
        self.canvas = None

        # === Layout containers ===
        header = ttk.Label(root, text="JPEG-like Image Compression", style="Header.TLabel")
        header.pack(pady=(14, 6))

        main = ttk.Frame(root)
        main.pack(fill="both", expand=True, padx=14, pady=8)

        left = ttk.Frame(main, style="Card.TFrame")
        left.pack(side="left", fill="y", padx=(0, 10), pady=4)

        right = ttk.Frame(main, style="TFrame")
        right.pack(side="left", fill="both", expand=True, pady=4)

        # === Controls panel ===
        ttk.Label(left, text="Controls", style="Header.TLabel").pack(anchor="w", padx=14, pady=(12, 6))

        ttk.Button(left, text="Open Image", command=self.load_image, style="Accent.TButton").pack(padx=14, pady=6, fill="x")

        self.quality_var = tk.DoubleVar(value=50)
        quality_label_frame = ttk.Frame(left, style="Card.TFrame")
        quality_label_frame.pack(anchor="w", padx=14, pady=(12, 2), fill="x")
        ttk.Label(quality_label_frame, text="Quality", style="Card.TLabel").pack(side="left")
        self.quality_value_label = ttk.Label(quality_label_frame, text="50", style="Card.TLabel")
        self.quality_value_label.pack(side="right")
        
        self.scale = ttk.Scale(left, from_=1, to=100, orient=tk.HORIZONTAL, variable=self.quality_var, command=self.update_quality_label)
        self.scale.pack(padx=14, fill="x")

        ttk.Button(left, text="Compress", command=self.compress, style="Accent.TButton").pack(padx=14, pady=(12, 6), fill="x")
        ttk.Button(left, text="Quality – PSNR Curve", command=self.show_psnr_curve).pack(padx=14, pady=(0, 12), fill="x")

        self.info = ttk.Label(left, text="Load an image to begin", style="TLabel")
        self.info.pack(anchor="w", padx=14, pady=(0, 14))

        self.status = ttk.Label(left, text="", style="TLabel")
        self.status.pack(anchor="w", padx=14, pady=(0, 12))

        # === Image panels ===
        panels = ttk.Frame(right, style="TFrame")
        panels.pack(fill="both", expand=True)

        orig_card = ttk.Frame(panels, style="Card.TFrame")
        orig_card.pack(side="left", fill="both", expand=True, padx=(0, 6), pady=4)

        recon_card = ttk.Frame(panels, style="Card.TFrame")
        recon_card.pack(side="left", fill="both", expand=True, padx=(6, 0), pady=4)

        ttk.Label(orig_card, text="Original", style="Card.TLabel").pack(anchor="w", padx=12, pady=(10, 6))
        self.panel_orig = tk.Label(orig_card, bg="#0b1224")
        self.panel_orig.pack(fill="both", expand=True, padx=12, pady=(0, 12))

        ttk.Label(recon_card, text="Reconstructed", style="Card.TLabel").pack(anchor="w", padx=12, pady=(10, 6))
        self.panel_rec = tk.Label(recon_card, bg="#0b1224")
        self.panel_rec.pack(fill="both", expand=True, padx=12, pady=(0, 12))

        # === Chart area ===
        chart_card = ttk.Frame(right, style="Card.TFrame")
        chart_card.pack(fill="x", padx=2, pady=(6, 2))
        ttk.Label(chart_card, text="Quality vs PSNR", style="Card.TLabel").pack(anchor="w", padx=12, pady=(10, 0))
        self.canvas_frame = ttk.Frame(chart_card, style="Card.TFrame")
        self.canvas_frame.pack(fill="both", expand=True, padx=8, pady=8)

    def update_quality_label(self, value):
        """Update the quality value label as slider moves"""
        q_val = int(round(float(value)))
        self.quality_value_label.config(text=str(q_val))

    def load_image(self):
        path = filedialog.askopenfilename()
        if not path:
            return

        self.img = cv2.imread(path)
        self.show_image(self.img, self.panel_orig)
        self.info.config(text="Image loaded. Choose quality and Compress.")
        self.status.config(text=f"Size: {self.img.shape[1]}x{self.img.shape[0]}")

    def compress(self):
        if self.img is None:
            return

        q = int(round(self.quality_var.get()))
        Yq, Cbq, Crq, QY, QC = compress_rgb(self.img, q)
        self.recon = decompress_rgb(Yq, Cbq, Crq, QY, QC, self.img.shape[:2])

        self.show_image(self.recon, self.panel_rec)

        p = psnr(self.img, self.recon)
        s = ssim(self.img, self.recon)
        self.info.config(text=f"PSNR: {p:.2f} dB   SSIM: {s:.4f}")
        self.status.config(text=f"Quality {q} processed")

    def show_psnr_curve(self):
        if self.img is None:
            return

        qualities = list(range(5, 101, 5))
        psnr_vals = []

        h, w = self.img.shape[:2]

        for q in qualities:
            Yq, Cbq, Crq, QY, QC = compress_rgb(self.img, q)
            recon = decompress_rgb(Yq, Cbq, Crq, QY, QC, (h, w))
            psnr_vals.append(psnr(self.img, recon))

        if self.canvas is not None:
            self.canvas.get_tk_widget().destroy()

        fig = Figure(figsize=(5, 4), dpi=100)
        ax = fig.add_subplot(111)
        ax.plot(qualities, psnr_vals, marker="o")
        ax.set_title("Quality vs PSNR")
        ax.set_xlabel("Quality")
        ax.set_ylabel("PSNR (dB)")
        ax.grid(True)

        self.canvas = FigureCanvasTkAgg(fig, master=self.canvas_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

    def show_image(self, img, panel):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        img_pil.thumbnail((420, 420))
        img_tk = ImageTk.PhotoImage(img_pil)
        panel.config(image=img_tk)
        panel.image = img_tk


if __name__ == "__main__":
    root = tk.Tk()
    app = JPEGApp(root)
    root.mainloop()
