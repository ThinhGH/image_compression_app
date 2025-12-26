import tkinter as tk
from tkinter import filedialog
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

        self.img = None
        self.recon = None
        self.canvas = None   

        # ===== Controls =====
        tk.Button(root, text="Open Image", command=self.load_image).pack()

        self.scale = tk.Scale(
            root, from_=1, to=100,
            orient=tk.HORIZONTAL,
            label="Quality", length=300
        )
        self.scale.set(50)
        self.scale.pack()

        # 🔹 Enable Entropy checkbox
        self.use_entropy = tk.BooleanVar()
        self.entropy_check = tk.Checkbutton(
            root,
            text="Enable Entropy (Zig-zag + RLE + Huffman)",
            variable=self.use_entropy
        )
        self.entropy_check.pack()

        tk.Button(root, text="Compress", command=self.compress).pack()
        tk.Button(
            root,
            text="Quality–PSNR Curve",
            command=self.show_psnr_curve
        ).pack(pady=5)

        self.info = tk.Label(root, text="")
        self.info.pack()

        # ===== Image panels =====
        self.panel_orig = tk.Label(root)
        self.panel_orig.pack(side="left", padx=10)

        self.panel_rec = tk.Label(root)
        self.panel_rec.pack(side="right", padx=10)

    # =========================
    def load_image(self):
        path = filedialog.askopenfilename()
        if not path:
            return

        self.img = cv2.imread(path)
        self.show_image(self.img, self.panel_orig)

    # =========================
    def compress(self):
        if self.img is None:
            return

        q = self.scale.get()

        # 🔹 Hiện tại entropy chỉ là MODE FLAG
        # (bước sau sẽ gắn thuật toán thật)
        entropy_on = self.use_entropy.get()

        Yq, Cbq, Crq, QY, QC = compress_rgb(self.img, q)
        self.recon = decompress_rgb(
            Yq, Cbq, Crq, QY, QC,
            self.img.shape[:2]
        )

        self.show_image(self.recon, self.panel_rec)

        p = psnr(self.img, self.recon)
        s = ssim(self.img, self.recon)

        mode = "Entropy ON" if entropy_on else "Entropy OFF"
        self.info.config(
            text=f"{mode} | PSNR: {p:.2f} dB   SSIM: {s:.4f}"
        )

    # =========================
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

        self.canvas = FigureCanvasTkAgg(fig, master=self.root)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(pady=10)

    # =========================
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
