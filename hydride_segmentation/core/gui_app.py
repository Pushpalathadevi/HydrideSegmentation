"""Tkinter GUI application for hydride segmentation."""

import os
import sys

# Dynamically add path to semanticsegmentation
SEMANTIC_PATH_PRIMARY = r"C:\Users\Admin\PycharmProjects\semanticsegmentation"
ALTERNATIVE_SEMANTIC_PATH = r"C:\Users\ManiKrishna\PycharmProjects\semanticsegmentation"

SEMANTIC_PATH = (
    SEMANTIC_PATH_PRIMARY
    if os.path.isdir(SEMANTIC_PATH_PRIMARY)
    else ALTERNATIVE_SEMANTIC_PATH
)

print(f"[DEBUG] Adding to sys.path: {SEMANTIC_PATH}")
if SEMANTIC_PATH not in sys.path:
    sys.path.append(SEMANTIC_PATH)

# Now you can import inference

import tkinter as tk
from tkinter import filedialog, messagebox, Menu, Scrollbar, Canvas, PanedWindow, ttk, Text
#sys.path.append(r"C:\Users\Admin\anaconda3\Lib\site-packages")
from tkinterdnd2 import DND_FILES, TkinterDnD
from PIL import Image, ImageTk, ImageGrab
import os
import threading
import numpy as np
import importlib
import logging
from .analysis import orientation_analysis, combined_figure


MAX_HISTORY = 10

# Model backend mapping
MODEL_BACKENDS = {
    "Conventional Model": "segmentationMaskCreation",
    "ML Model": "inference"
}

#test
class HydrideSegmentationGUI:
    """Tkinter front-end for running segmentation and viewing results."""
    def __init__(self, master):
        self.master = master
        master.title("Hydride Segmentation GUI")
        master.geometry("1500x900")
        self.create_menu()

        self.file_entry = tk.Entry(master)
        self.file_entry.place_forget()  # hides it from view
        self.file_entry.bind("<Return>", self.run_segmentation_event)

        master.drop_target_register(DND_FILES)
        master.dnd_bind('<<Drop>>', self.drop_file)

        model_frame = tk.Frame(master)
        model_frame.pack(pady=5)
        tk.Label(model_frame, text="Select Model:").pack(side=tk.LEFT)


        self.model_var = tk.StringVar(value="ML Model")
        self.model_menu = ttk.Combobox(
            model_frame,
            textvariable=self.model_var,
            values=list(MODEL_BACKENDS.keys()),
            state="readonly",
        )
        self.model_menu.pack(side=tk.LEFT, padx=5)
        # NEW → toggle callback
        self.model_menu.bind("<<ComboboxSelected>>", self.on_model_change)



        self.param_frame = tk.LabelFrame(master, text="Segmentation Parameters")
        # self.param_frame.pack(fill="both", padx=10, pady=5)  # default visible

        self.entries = {}
        params = [
            ("CLAHE Clip Limit", "2.0"),
            ("CLAHE Tile Grid Size", "8,8"),
            ("Adaptive Block Size", "13"),
            ("Adaptive C", "40"),
            ("Morph Kernel Size", "5,5"),
            ("Morph Iterations", "0"),
            ("Area Threshold", "95"),
            ("Crop Percent", "10"),
        ]
        for idx, (label, default) in enumerate(params):
            tk.Label(self.param_frame, text=f"{label}:").grid(
                row=idx, column=0, sticky="e"
            )
            entry = tk.Entry(self.param_frame)
            entry.insert(0, default)
            entry.grid(row=idx, column=1)
            entry.bind("<Return>", self.run_segmentation_event)
            self.entries[label] = entry

        self.crop_var = tk.IntVar()
        crop_check = tk.Checkbutton(
            self.param_frame, text="Crop Image", variable=self.crop_var
        )
        crop_check.grid(row=len(params), column=0, columnspan=2, sticky="w")

        self.output_frame = tk.LabelFrame(master, text="Results")
        self.output_frame.pack(fill="both", expand=True, padx=10, pady=5)

        self.paned = PanedWindow(self.output_frame, orient=tk.HORIZONTAL)
        self.paned.pack(fill="both", expand=True)

        self.image_panels = {}
        for label in ["Input Image", "Mask", "Overlay"]:
            frame = tk.Frame(self.paned)
            self.paned.add(frame, stretch="always")
            tk.Label(frame, text=label).pack()

            canvas = Canvas(frame)
            v_scroll = Scrollbar(frame, orient="vertical", command=canvas.yview)
            h_scroll = Scrollbar(frame, orient="horizontal", command=canvas.xview)
            canvas.configure(yscrollcommand=v_scroll.set, xscrollcommand=h_scroll.set)

            v_scroll.pack(side="right", fill="y")
            h_scroll.pack(side="bottom", fill="x")
            canvas.pack(side="left", fill="both", expand=True)

            canvas.image_label = label  # Track which image it is
            canvas.bind("<Button-3>", self.show_context_menu)  # Right-click binding

            self.image_panels[label] = canvas

            self.image_panels[label] = canvas

        button_frame = tk.Frame(master)
        button_frame.pack(pady=10)

        tk.Button(button_frame, text="Run Segmentation", command=self.run_segmentation).pack(side=tk.LEFT, padx=5)
        tk.Button(button_frame, text="Reset Zoom", command=self.reset_zoom).pack(side=tk.LEFT, padx=5)
        tk.Button(button_frame, text="Clear", command=self.clear_all).pack(side=tk.LEFT, padx=5)

        self.log_frame = tk.LabelFrame(master, text="Logger")
        self.log_frame.pack(fill="both", expand=False, padx=10, pady=5)

        log_scroll = Scrollbar(self.log_frame, orient="vertical")
        self.log_text = Text(
            self.log_frame,
            height=10,
            state='disabled',
            wrap='word',
            bg='white',
            fg='black',
            yscrollcommand=log_scroll.set
        )
        log_scroll.config(command=self.log_text.yview)

        # Layout
        self.log_text.pack(side=tk.LEFT, fill="both", expand=True)
        log_scroll.pack(side=tk.RIGHT, fill="y")


        self.setup_logger()

        self.last_mask = None
        self.last_mask_filename = None

        self.master.bind('<Return>', self.run_segmentation_event)
        self.master.bind('<Control-s>', lambda event: self.save_results())

        self.context_menu = Menu(master, tearoff=0)
        self.context_menu.add_command(label="Save Image", command=self.save_context_image)

        self.show_placeholder()

        self.undo_stack: list[tuple[Image.Image, Image.Image, Image.Image]] = []
        self.redo_stack: list[tuple[Image.Image, Image.Image, Image.Image]] = []

    def on_model_change(self, event=None):
        if self.model_var.get() == "Conventional Model":
            if not self.param_frame.winfo_ismapped():
                # pack *before* the logger frame
                self.param_frame.pack(
                    fill="both",
                    padx=10,
                    pady=5,
                    before=self.log_frame
                )
        else:
            self.param_frame.pack_forget()

    def setup_logger(self):
        class TextHandler(logging.Handler):
            def __init__(self, text_widget):
                super().__init__()
                self.text_widget = text_widget

            def emit(self, record):
                msg = self.format(record)
                self.text_widget.configure(state='normal')
                self.text_widget.insert(tk.END, msg + '\n')
                self.text_widget.configure(state='disabled')
                self.text_widget.yview(tk.END)

        self.logger = logging.getLogger("HydrideGUI")
        self.logger.setLevel(logging.INFO)

        handler = TextHandler(self.log_text)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)

        self.logger.addHandler(handler)

    def create_menu(self):
        menu_bar = Menu(self.master)
        self.master.config(menu=menu_bar)

        file_menu = Menu(menu_bar, tearoff=0)
        menu_bar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Open", command=self.browse_file)
        file_menu.add_command(label="Save Results", command=self.save_results)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.master.quit)

        edit_menu = Menu(menu_bar, tearoff=0)
        menu_bar.add_cascade(label="Edit", menu=edit_menu)
        edit_menu.add_command(label="Undo", command=self.undo, accelerator="Ctrl+Z")
        edit_menu.add_command(label="Redo", command=self.redo, accelerator="Ctrl+Y")
        self.master.bind('<Control-z>', lambda event: self.undo())
        self.master.bind('<Control-y>', lambda event: self.redo())

        image_menu = Menu(menu_bar, tearoff=0)
        menu_bar.add_cascade(label="Image", menu=image_menu)
        image_menu.add_command(label="Brightness and Contrast", command=self.adjust_brightness_contrast)

        analyze_menu = Menu(menu_bar, tearoff=0)
        menu_bar.add_cascade(label="Analyze", menu=analyze_menu)
        analyze_menu.add_command(label="Area Fraction", command=self.calculate_area_fraction)
        analyze_menu.add_command(label="Hydride Orientation", command=self.analyze_orientation)
        # analyze_menu.add_command(label="Keyboard Shortcuts…", command=self.show_shortcuts)

        help_menu = Menu(menu_bar, tearoff=0)
        menu_bar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="About", command=self.show_about)
        help_menu.add_command(label="Keyboard Shortcuts…", command=self.show_shortcuts)

    def show_about(self):
        messagebox.showinfo(
            "About",
            "Hydride Segmentation GUI\n"
            "Version 1.0\n"
            "Powered by Conventional & ML backends"
        )

    def show_shortcuts(self):
        shortcuts = (
            "Ctrl+S: Save results\n"
            "Ctrl+Z / Ctrl+Y: Undo/Redo\n"
            "Crtl+Mouse wheel: Zoom\n"
            "Shift+Mouse wheel: Horizontal scroll\n"
            "Mouse wheel: Vertical scroll\n"
            "Left click and drag: Pan\n"
        )
        messagebox.showinfo("Keyboard Shortcuts", shortcuts)

    def browse_file(self):
        path = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.tif")])
        if path:
            self.file_entry.delete(0, tk.END)
            self.file_entry.insert(0, path)
            self.display_input_image(path)
            self.logger.info(f"Loaded image: {path}")

    def drop_file(self, event):
        files = self.master.tk.splitlist(event.data)
        if files:
            self.file_entry.delete(0, tk.END)
            self.file_entry.insert(0, files[0])
            self.display_input_image(files[0])
            self.logger.info(f"Dropped image: {files[0]}")

    def clear_all(self):
        self.file_entry.delete(0, tk.END)
        for canvas in self.image_panels.values():
            canvas.delete("all")
        self.last_mask = None
        self.show_placeholder()
        self.logger.info("Cleared all panels")

    def show_placeholder(self):
        canvas = self.image_panels["Input Image"]
        canvas.create_text(200, 200, text="Drag image here", font=("Arial", 20), fill="gray")

    def collect_parameters(self):
        image_path = self.file_entry.get()
        if not os.path.exists(image_path):
            messagebox.showerror("Error", "Select valid image file")
            self.logger.error("Invalid image path")
            return None

        if self.model_var.get() != "Conventional Model":
            return {"image_path": image_path}


        try:
            settings = {
                'image_path': image_path,
                'clahe': {'clip_limit': float(self.entries['CLAHE Clip Limit'].get()),
                          'tile_grid_size': [int(x) for x in self.entries['CLAHE Tile Grid Size'].get().split(',')]},
                'adaptive': {'block_size': int(self.entries['Adaptive Block Size'].get()),
                             'C': int(self.entries['Adaptive C'].get())},
                'morph': {'kernel_size': [int(x) for x in self.entries['Morph Kernel Size'].get().split(',')],
                          'iterations': int(self.entries['Morph Iterations'].get())},
                'area_threshold': int(self.entries['Area Threshold'].get()),
                'crop': bool(self.crop_var.get()),
                'crop_percent': int(self.entries['Crop Percent'].get())
            }
            return settings
        except Exception as e:
            messagebox.showerror("Parameter Error", str(e))
            self.logger.error(f"Parameter parsing error: {e}")
            return None

    def adjust_brightness_contrast(self):
        """Opens a dialog with sliders + spinboxes for brightness & contrast."""
        # 1) requires only that an input image is loaded
        if not hasattr(self, "loaded_input") or self.loaded_input is None:
            messagebox.showwarning("No Image", "Load an input image first.")
            return

        # 2) grab a NumPy copy of the current input
        orig = np.array(self.loaded_input)

        # 3) build the window
        win = tk.Toplevel(self.master)
        win.title("Brightness & Contrast")
        win.resizable(False, False)
        win.transient(self.master)
        win.grab_set()

        # variables
        brightness_var = tk.IntVar(value=0)
        contrast_var = tk.IntVar(value=100)

        # 4) update function
        def update(*_):
            b = brightness_var.get()
            c = contrast_var.get() / 100.0
            adj = np.clip((orig.astype(np.float32) - 128) * c + 128 + b, 0, 255).astype(np.uint8)
            pil_adj = Image.fromarray(adj)
            self.adjusted_input_pil = pil_adj
            # redraw both panels
            self.display_image(self.image_panels["Input Image"], pil_adj)
            # self.display_image(self.image_panels["Overlay"], pil_adj)  # or recompute overlay if you like

        # trace var changes
        brightness_var.trace_add("write", update)
        contrast_var.trace_add("write", update)

        # 5) Brightness controls
        frm_b = tk.Frame(win)
        frm_b.pack(fill="x", padx=10, pady=(10, 0))
        tk.Label(frm_b, text="Brightness").pack(anchor="w")
        sb_b = tk.Spinbox(frm_b, from_=-100, to=100, textvariable=brightness_var, width=5)
        sb_b.pack(side="right")
        scale_b = tk.Scale(frm_b, from_=-100, to=100, orient="horizontal", variable=brightness_var)
        scale_b.pack(fill="x", expand=True)

        # 6) Contrast controls
        frm_c = tk.Frame(win)
        frm_c.pack(fill="x", padx=10, pady=(10, 0))
        tk.Label(frm_c, text="Contrast (%)").pack(anchor="w")
        sb_c = tk.Spinbox(frm_c, from_=0, to=300, textvariable=contrast_var, width=5)
        sb_c.pack(side="right")
        scale_c = tk.Scale(frm_c, from_=0, to=300, orient="horizontal", variable=contrast_var)
        scale_c.pack(fill="x", expand=True)

        # 7) Buttons: Reset and Close
        btn_frame = tk.Frame(win)
        btn_frame.pack(pady=10)

        def on_reset():
            brightness_var.set(0)
            contrast_var.set(100)
            update()

        tk.Button(btn_frame, text="Reset", command=on_reset).pack(side="left", padx=5)
        tk.Button(btn_frame, text="Close", command=win.destroy).pack(side="right", padx=5)

        # initialize display
        update()


    def calculate_area_fraction(self):
        """Compute hydride area fraction from the last mask & show it."""
        if self.last_mask is None:
            messagebox.showwarning("Area Fraction", "No segmentation available.")
            return

        # last_mask is a 2D array of 0 and 255
        hydride_pixels = np.count_nonzero(self.last_mask)
        total_pixels = self.last_mask.size
        fraction = hydride_pixels / total_pixels * 100

        #log
        self.logger.info(
            f"Hydride area fraction: {fraction:.2f}% "
            f"({hydride_pixels}/{total_pixels} pixels)"
        )

        messagebox.showinfo(
            "Area Fraction",
            f"Hydrides cover {fraction:.2f}% of the image "
            f"({hydride_pixels}/{total_pixels} pixels)."
        )

    def analyze_orientation(self):
        """Generate orientation analysis plots and show them."""
        if self.last_mask is None:
            messagebox.showwarning("Orientation", "No segmentation available.")
            return
        orient, size_plot, angle_plot = orientation_analysis(self.last_mask)
        fig = combined_figure(self.current_input, self.current_mask, self.current_overlay,
                              orient, size_plot, angle_plot)
        fig.show()


    def run_segmentation_event(self, event=None):
        self.run_segmentation()

    def run_segmentation(self):
        params = self.collect_parameters()
        if not params:
            return
        model_name = self.model_var.get()
        backend_module_name = MODEL_BACKENDS[model_name]

        try:
            backend = importlib.import_module(backend_module_name)
        except ModuleNotFoundError as e:
            messagebox.showerror("Module Error", f"Could not import model: {e}")
            self.logger.error(f"ModuleNotFoundError: {e}")
            return
        except Exception as e:
            messagebox.showerror("Import Error", str(e))
            self.logger.error(f"Unexpected import error: {e}")
            return

        def process():
            try:
                self.logger.info("Running segmentation...")
                image, mask = backend.run_model(params['image_path'], params)
            except Exception as e:
                messagebox.showerror("Backend Error", str(e))
                self.logger.error(f"Segmentation failed: {e}")
                return

            self.last_mask = mask
            #mask_arr = np.array(self.last_mask)
            self.last_mask_filename = params['image_path']

            # 1) Make sure `rgb_image` is always H×W×3
            if image.ndim == 2:
                # grayscale → replicate into 3 channels
                rgb_image = np.stack([image] * 3, axis=-1)
            else:
                rgb_image = image

            # 2) Convert to PIL only once
            input_img = Image.fromarray(rgb_image)
            mask_img = Image.fromarray(mask)

            # 3) Build overlay in RGB
            overlay_np = rgb_image.copy()
            overlay_np[mask > 0] = [255, 0, 0]  # paint red wherever mask is positive
            overlay_img = Image.fromarray(overlay_np)

            overlay_img = Image.fromarray(overlay_np)


            # at segmentation time
            self.current_input, self.current_mask, self.current_overlay = (
                input_img, mask_img, overlay_img
            )
            self.undo_stack.append((input_img, mask_img, overlay_img))
            # CAP the undo history
            if len(self.undo_stack) > MAX_HISTORY:
                self.undo_stack.pop(0)

            self.redo_stack.clear()

            self.master.after(
                0,
                self.update_panels,
                input_img,
                mask_img,
                overlay_img,
            )

            self.master.after(0, self.update_panels, input_img, mask_img, overlay_img)
            self.logger.info("Segmentation completed")

        threading.Thread(target=process).start()

    def update_panels(self, input_img, mask_img, overlay_img):
        for label, img in zip(["Input Image", "Mask", "Overlay"], [input_img, mask_img, overlay_img]):
            self.display_image(self.image_panels[label], img)

    #
    def display_image(self, canvas, pil_img):
        canvas.delete("all")

        canvas.original_img = pil_img
        canvas.zoom = 1.0

        canvas.update_idletasks()
        cw, ch = canvas.winfo_width(), canvas.winfo_height()
        iw, ih = pil_img.width, pil_img.height
        ratio = min(cw / iw, ch / ih)
        resized = pil_img.resize((int(iw * ratio), int(ih * ratio)), Image.LANCZOS)

        tk_img = ImageTk.PhotoImage(resized)
        canvas.tk_img = tk_img
        canvas.image_id = canvas.create_image(0, 0, anchor="nw", image=tk_img)
        canvas.config(scrollregion=(0, 0, int(iw * ratio), int(ih * ratio)))

        self.attach_zoom_and_sync(canvas)
        canvas.x_scroll_callback = canvas.xview
        canvas.y_scroll_callback = canvas.yview

    def attach_zoom_and_sync(self, canvas):
        def zoom(event):
            if not hasattr(canvas, 'original_img'):
                return

            if event.delta > 0:
                canvas.zoom *= 1.1
            else:
                canvas.zoom /= 1.1

            new_w = int(canvas.original_img.width * canvas.zoom)
            new_h = int(canvas.original_img.height * canvas.zoom)
            resized = canvas.original_img.resize((new_w, new_h), Image.LANCZOS)
            tk_img = ImageTk.PhotoImage(resized)

            canvas.tk_img = tk_img
            canvas.itemconfig(canvas.image_id, image=tk_img)
            canvas.config(scrollregion=(0, 0, new_w, new_h))

            for lbl, other_canvas in self.image_panels.items():
                if other_canvas != canvas and hasattr(other_canvas, 'original_img'):
                    other_canvas.zoom = canvas.zoom
                    new_w = int(other_canvas.original_img.width * canvas.zoom)
                    new_h = int(other_canvas.original_img.height * canvas.zoom)
                    resized = other_canvas.original_img.resize((new_w, new_h), Image.LANCZOS)
                    tk_other_img = ImageTk.PhotoImage(resized)
                    other_canvas.tk_img = tk_other_img
                    other_canvas.itemconfig(other_canvas.image_id, image=tk_other_img)
                    other_canvas.config(scrollregion=(0, 0, new_w, new_h))

        def scroll_y(*args):
            for other_canvas in self.image_panels.values():
                other_canvas.yview_moveto(args[0])

        def scroll_x(*args):
            for other_canvas in self.image_panels.values():
                other_canvas.xview_moveto(args[0])

        def mouse_scroll_y(event):
            delta = -1 * (event.delta / 120)
            for other_canvas in self.image_panels.values():
                other_canvas.yview_scroll(int(delta), "units")

        def mouse_scroll_x(event):
            delta = -1 * (event.delta / 120)
            for other_canvas in self.image_panels.values():
                other_canvas.xview_scroll(int(delta), "units")

        def start_drag(event):
            canvas.scan_mark(event.x, event.y)

        def on_drag(event):
            canvas.scan_dragto(event.x, event.y, gain=1)
            for other_canvas in self.image_panels.values():
                if other_canvas != canvas:
                    other_canvas.xview_moveto(canvas.xview()[0])
                    other_canvas.yview_moveto(canvas.yview()[0])

        canvas.bind("<Control-MouseWheel>", zoom)
        canvas.bind("<MouseWheel>", mouse_scroll_y)
        canvas.bind("<Shift-MouseWheel>", mouse_scroll_x)
        canvas.bind("<ButtonPress-1>", start_drag)
        canvas.bind("<B1-Motion>", on_drag)

        canvas.configure(yscrollcommand=scroll_y, xscrollcommand=scroll_x)

    def reset_zoom(self):
        for label, canvas in self.image_panels.items():
            if hasattr(canvas, 'original_img'):
                canvas.zoom = 1.0
                canvas.update_idletasks()
                cw = canvas.winfo_width()
                ch = canvas.winfo_height()
                iw, ih = canvas.original_img.width, canvas.original_img.height
                ratio = min(cw / iw, ch / ih)
                new_w = int(iw * ratio)
                new_h = int(ih * ratio)
                resized = canvas.original_img.resize((new_w, new_h), Image.LANCZOS)
                tk_img = ImageTk.PhotoImage(resized)
                canvas.tk_img = tk_img
                canvas.itemconfig(canvas.image_id, image=tk_img)
                canvas.config(scrollregion=(0, 0, new_w, new_h))

    def display_input_image(self, image_path):
        try:
            img = Image.open(image_path)
            self.loaded_input = img.copy()  # ← store for later adjustment
            self.display_image(self.image_panels["Input Image"], img)
        except Exception as e:
            messagebox.showerror("Error", f"Unable to load image: {e}")

    def save_results(self):
        if self.last_mask is None or self.last_mask_filename is None:
            messagebox.showwarning("No Results", "Please run segmentation first.")
            self.logger.warning("Save attempted without results")
            return

        save_dir = filedialog.askdirectory(title="Select directory to save results")
        if not save_dir:
            return

        base_name = os.path.splitext(os.path.basename(self.last_mask_filename))[0]

        # Save the composite panel as PNG
        composite_path = os.path.join(save_dir, f"{base_name}_results.png")
        self.master.update()
        x = self.output_frame.winfo_rootx()
        y = self.output_frame.winfo_rooty()
        w = self.output_frame.winfo_width()
        h = self.output_frame.winfo_height()
        img = ImageGrab.grab(bbox=(x, y, x + w, y + h))
        img.save(composite_path)
        self.logger.info(f"Saved composite image to {composite_path}")

        # Save input, mask, and overlay images
        try:
            self.current_input.save(os.path.join(save_dir, f"{base_name}_input.png"))
            self.current_mask.save(os.path.join(save_dir, f"{base_name}_prediction.png"))
            self.current_overlay.save(os.path.join(save_dir, f"{base_name}_overlay.png"))
            self.logger.info("Saved individual images (input, prediction, overlay)")
        except Exception as e:
            self.logger.error(f"Failed to save individual images: {e}")
            messagebox.showerror("Save Error", f"Could not save individual images:\n{e}")
            return

        messagebox.showinfo("Saved", f"Results saved in {save_dir}")

    def undo(self, event=None):
        if not self.undo_stack:
            self.logger.info("Nothing to undo")
            return
        # pop last action
        last_input, last_mask, last_overlay = self.undo_stack.pop()
        # push current state onto redo
        current = (
            getattr(self, 'current_input', last_input),
            getattr(self, 'current_mask', last_mask),
            getattr(self, 'current_overlay', last_overlay),
        )
        self.redo_stack.append(current)

        # restore
        self.current_input, self.current_mask, self.current_overlay = (
            last_input, last_mask, last_overlay
        )
        self.update_panels(last_input, last_mask, last_overlay)
        self.logger.info("Undid last action")

    def redo(self, event=None):
        if not self.redo_stack:
            self.logger.info("Nothing to redo")
            return
        # pop from redo
        inp, msk, ovl = self.redo_stack.pop()
        # push current onto undo
        self.undo_stack.append((inp, msk, ovl))
        # restore
        self.current_input, self.current_mask, self.current_overlay = (inp, msk, ovl)
        self.update_panels(inp, msk, ovl)
        self.logger.info("Redid action")

    def show_context_menu(self, event):
        widget = event.widget
        if hasattr(widget, 'image_label'):
            self.context_menu_canvas = widget
            try:
                self.context_menu.tk_popup(event.x_root, event.y_root)
            finally:
                self.context_menu.grab_release()

    def save_context_image(self):
        canvas = self.context_menu_canvas
        label = getattr(canvas, 'image_label', None)
        if label and hasattr(canvas, 'original_img'):
            file_path = filedialog.asksaveasfilename(defaultextension=".png",
                                                     filetypes=[("PNG files", "*.png")],
                                                     title=f"Save {label} Image")
            if file_path:
                canvas.original_img.save(file_path)
                self.logger.info(f"{label} image saved to {file_path}")

