"""Tkinter GUI for Hydride Segmentation Tool."""
from __future__ import annotations

import tkinter as tk
from tkinter import filedialog, messagebox
from pathlib import Path
from typing import Optional
from PIL import Image, ImageTk

from ..core.image_io import load_image, save_image
from ..core.segmentation import segment_hydrides
from ..core.metrics import area_fraction


class HydrideApp(tk.Tk):
    """Main application window."""

    def __init__(self) -> None:
        super().__init__()
        self.title("Hydride Segmentation Tool")

        self.original_label = tk.Label(self)
        self.original_label.grid(row=0, column=0, padx=5, pady=5)

        self.segmented_label = tk.Label(self)
        self.segmented_label.grid(row=0, column=1, padx=5, pady=5)

        btn_frame = tk.Frame(self)
        btn_frame.grid(row=1, column=0, columnspan=2, pady=5)

        tk.Button(btn_frame, text="Load Image", command=self.load_image).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="Run Segmentation", command=self.run_segmentation).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="Save Output", command=self.save_output).pack(side=tk.LEFT, padx=5)

        self.status = tk.StringVar(value="Ready")
        tk.Label(self, textvariable=self.status).grid(row=2, column=0, columnspan=2, sticky="we")

        self.image: Optional[Image.Image] = None
        self.mask: Optional[Image.Image] = None

    def load_image(self) -> None:
        path = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.tif;*.bmp")])
        if not path:
            return
        self.image = load_image(path)
        tk_img = ImageTk.PhotoImage(self.image)
        self.original_label.configure(image=tk_img)
        self.original_label.image = tk_img
        self.status.set("Image loaded")

    def run_segmentation(self) -> None:
        if self.image is None:
            messagebox.showwarning("No image", "Please load an image first")
            return
        self.mask = segment_hydrides(self.image)
        tk_img = ImageTk.PhotoImage(self.mask)
        self.segmented_label.configure(image=tk_img)
        self.segmented_label.image = tk_img
        fraction = area_fraction(self.mask)
        self.status.set(f"Area fraction: {fraction:.2%}")

    def save_output(self) -> None:
        if self.mask is None:
            messagebox.showwarning("No result", "Run segmentation before saving")
            return
        path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG", "*.png")])
        if not path:
            return
        save_image(self.mask, path)
        self.status.set(f"Saved: {Path(path).name}")


def main() -> None:
    app = HydrideApp()
    app.mainloop()


if __name__ == "__main__":
    main()
