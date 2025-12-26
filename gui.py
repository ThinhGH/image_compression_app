import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk
import cv2
import numpy as np

from jpeg_codec import compress_rgb, decompress_rgb
from metrics import psnr, ssim

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


class JPEGApp:
    def __init__(self, root):
        self.root = root
        root.title("JPEG-like Image Compression")
        self.use_entropy = tk.BooleanVar(value=False)
        self.img = None
        self.recon = None

        # ===== Notebook =====
        notebook = ttk.Notebook(root)
        notebook.pack(fill="both", expand=True)

        self.tab_compress = ttk.Frame(notebook)
        self.tab_experiment = ttk.Frame(notebook)

        notebook.add(self.tab_compress, text="Compression")
        notebook.add(self.tab_experiment, text="Experiment")

        self.build_compression_tab()
        self.build_experiment_tab()

    # =======================
    # Compression TAB
    # =======================
    def build_compression_tab(self):
        frame = self.tab_compress

        control = ttk.Frame(frame)
        control.pack(pady=10)

        ttk.Button(control, text="Open Image", command=self.load_image).grid(row=0, column=0, padx=5)

        self.scale = tk.Scale(
            control, from_=1, to=100, orient=tk.HORIZONTAL,
            label="Quality", length=300
        )
        self.scale.set(50)
        self.scale.grid(row=0, column=1, padx=10)

        self.entropy_var = tk.BooleanVar()
        tk.Checkbutton(
                root,
                text="Enable Entropy Coding (Zigzag + RLE + Huffman)",
                variable=self.use_entropy
            ).pack()
        ttk.Button(control, text="Compress", command=self.compress).grid(row=0, column=3, padx=5)

        self.info = ttk.Label(frame, text="")
        self.info.pack(pady=5)

        image_frame = ttk.Frame(frame)
        image_frame.pack(pady=10)

        self.panel_orig = ttk.Label(image_frame)
        self.panel_orig.pack(side="left", padx=10)

        self.panel_rec = ttk.Label(image_frame)
        self.panel_rec.pack(side="right", padx=10)

    # =======================
    # Experiment TAB
    # =======================
    def build_experiment_tab(self):
        frame = self.tab_experiment

        ttk.Label(
            frame,
            text="Quality – PSNR Experiment",
            font=("Arial", 12, "bold")
        ).pack(pady=10)

        ttk.Button(
            frame,
            text="Run PSNR Curve (New Window)",
            command=self.show_psnr_curve
        ).pack(pady=5)

    # =======================
    # Core functions
    # =======================
    def load_image(self):
        path = filedialog.askopenfilename()
        if not path:
            return

        self.img = cv2.imread(path)
        self.show_image(self.img, self.panel_orig)

    def compress(self):
        if self.img is None:
            return

        q = self.scale.get()
        enable_entropy = self.use_entropy.get()

        Yq, Cbq, Crq, QY, QC, info = compress_rgb(self.img, q, enable_entropy)
        self.recon = decompress_rgb(Yq, Cbq, Crq, QY, QC, self.img.shape[:2])

        self.show_image(self.recon, self.panel_rec)

        p = psnr(self.img, self.recon)
        s = ssim(self.img, self.recon)
        self.info.config(
    text=f"PSNR: {p:.2f} dB | SSIM: {s:.4f} | "
         f"bpp: {info['bpp']:.3f} | "
         f"Entropy: {'ON' if enable_entropy else 'OFF'}"
)


        enable_entropy = self.use_entropy.get()

        print("Entropy:", enable_entropy)
        print("Compressed size (bits):", info["bits"])
        print("bpp:", info["bpp"])

    # =======================
    # PSNR Curve (separate window)
    # =======================
    def show_psnr_curve(self):
        if self.img is None:
            return

        win = tk.Toplevel(self.root)
        win.title("Quality vs PSNR")

        qualities = list(range(5, 101, 5))
        psnr_no_entropy = []
        psnr_entropy = []

        h, w = self.img.shape[:2]

        for q in qualities:
            # No entropy
            Yq, Cbq, Crq, QY, QC, info = compress_rgb(self.img, q, False)
            bpp = info["bpp"]
            rec = decompress_rgb(Yq, Cbq, Crq, QY, QC, (h, w))
            psnr_no_entropy.append(psnr(self.img, rec))

            # With entropy
            Yq, Cbq, Crq, QY, QC, info = compress_rgb(self.img, q, True)
            bpp = info["bpp"]
            rec = decompress_rgb(Yq, Cbq, Crq, QY, QC, (h, w))
            psnr_entropy.append(psnr(self.img, rec))

        fig = Figure(figsize=(6, 4), dpi=100)
        ax = fig.add_subplot(111)

        ax.plot(qualities, psnr_entropy, marker="s", label="With Entropy")

        ax.set_title("Quality vs PSNR")
        ax.set_xlabel("Quality")
        ax.set_ylabel("PSNR (dB)")
        ax.grid(True)
        ax.legend()

        canvas = FigureCanvasTkAgg(fig, master=win)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

    # =======================
    # Image display helper
    # =======================
    def show_image(self, img, panel):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        img_pil.thumbnail((300, 300))
        img_tk = ImageTk.PhotoImage(img_pil)
        panel.config(image=img_tk)
        panel.image = img_tk


if __name__ == "__main__":
    root = tk.Tk()
    app = JPEGApp(root)
    root.mainloop()
