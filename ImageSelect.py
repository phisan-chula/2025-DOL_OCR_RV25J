import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk, ImageDraw
from pathlib import Path  # Use pathlib for modern path handling
import json
import time
import os # Keep os for compatibility with Tkinter widgets if needed

class ImageSelect(ttk.Frame):
    """
    Custom Tkinter component for displaying images, handling zoom, and managing 
    a mouse-drawn Region of Interest (ROI) rectangle.
    Uses pathlib for robust file path handling.
    """
    def __init__(self, master=None, log_callback=None, **kwargs):
        super().__init__(master, **kwargs)
        self.log_callback = log_callback if log_callback else print

        self.original_image = None
        self.tk_image = None
        self.displayed_image = None
        self.image_path: Path = None  # Stores the Path object of the image
        self.scale_factor = 1.0

        # ROI tracking
        self.start_x = None
        self.start_y = None
        self.rect_id = None
        self.original_rect_coords = None 

        self.create_canvas()
        self.bind_events()

    def create_canvas(self):
        """Creates the canvas and necessary scrollbars."""
        
        # 1. Scrollbars (placed inside this frame)
        self.v_scrollbar = ttk.Scrollbar(self, orient=tk.VERTICAL)
        self.h_scrollbar = ttk.Scrollbar(self, orient=tk.HORIZONTAL)
        
        # 2. Canvas
        self.canvas = tk.Canvas(self, bg='#333333', cursor='crosshair', 
                                yscrollcommand=self.v_scrollbar.set,
                                xscrollcommand=self.h_scrollbar.set)

        # 3. Configure Scrollbars
        self.v_scrollbar.config(command=self.canvas.yview)
        self.h_scrollbar.config(command=self.canvas.xview)
        
        # 4. Layout
        self.canvas.grid(row=0, column=0, sticky='nsew')
        self.v_scrollbar.grid(row=0, column=1, sticky='ns')
        self.h_scrollbar.grid(row=1, column=0, sticky='ew')
        
        # 5. Make the canvas cell expand
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)
        
        # 6. Bind resize event
        self.canvas.bind('<Configure>', self.on_canvas_resize)

        # Initially hide scrollbars
        self._toggle_scrollbars(0, 0, initial_setup=True)

    def on_canvas_resize(self, event):
        """Redraws the image and selection when the canvas size changes (window resize)."""
        if self.original_image:
            self.set_scale(self.scale_factor) 

    def bind_events(self):
        """Binds mouse events for ROI selection."""
        self.canvas.bind("<Button-1>", self.on_mouse_down)
        self.canvas.bind("<B1-Motion>", self.on_mouse_move)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_up)

    def log(self, message):
        """Wrapper for the log callback."""
        self.log_callback(message)

    def _get_base_path(self) -> Path | None:
        """
        Derives the base file stem (the part before _RV25J.jpg) as a string.
        E.g., C:/data/scan_001_RV25J.jpg -> "scan_001"
        """
        if not self.image_path:
            return None
        
        # Check if the filename ends with the marker (case-insensitive)
        name_lower = self.image_path.name.lower()
        marker = "_rv25j.jpg"
        
        if name_lower.endswith(marker):
            # Extract the part before the marker and extension
            base_stem = self.image_path.stem[:-6] # stem is 'scan_001_RV25J', remove '_RV25J' (6 chars)
        else:
            # Fallback to the original stem if the marker is not found
            base_stem = self.image_path.stem 
            
        return base_stem

    def _get_output_dir(self) -> Path | None:
        """
        Calculates the output directory path based on the image prefix.
        Creates the directory if it doesn't exist.
        Returns: Path to the new directory (e.g., C:/data/scan_001/)
        """
        base_stem = self._get_base_path()
        if not self.image_path or not base_stem:
            self.log("ERROR: Image path or prefix is missing.")
            return None

        # Create the new directory path (sibling to the original image)
        # e.g., C:/data/scan_001_RV25J.jpg -> C:/data/scan_001/
        output_dir = self.image_path.parent / base_stem
        
        try:
            # Create directory if it does not exist (mkdir -p)
            output_dir.mkdir(parents=True, exist_ok=True)
            self.log(f"INFO: Created/Ensured output directory: {output_dir.name}")
            return output_dir
        except Exception as e:
            self.log(f"ERROR: Failed to create output directory {output_dir.name}: {e}")
            return None

    def load_image(self, path):
        """Loads a new image from the given file path."""
        # Convert incoming path string to Path object
        self.image_path = Path(path) 
        
        try:
            if not self.image_path.exists():
                raise FileNotFoundError(f"File not found: {self.image_path}")

            self.original_image = Image.open(self.image_path)
            self.log(f"Image loaded: {self.image_path.name} ({self.original_image.width}x{self.original_image.height})")
            
            self.clear_selection()
            
            # CRITICAL FIX 1: Load existing selection BEFORE calling set_scale
            self._load_existing_selection() 

            self.scale_factor = 1.0 
            self.set_scale(self.scale_factor) 
            
            # CRITICAL FIX 2: Explicitly guarantee the selection is drawn AFTER set_scale 
            # and any potential Configure/resize events triggered by it have settled.
            if self.original_rect_coords:
                self.draw_selection_from_original()
            
        except FileNotFoundError:
            self.log(f"ERROR: Image file not found at {self.image_path}")
            self.clear_image()
        except Exception as e:
            self.log(f"ERROR loading image: {e}")
            self.clear_image()

    def clear_image(self):
        """Clears the canvas and image data."""
        self.canvas.delete(tk.ALL)
        self.original_image = None
        self.tk_image = None 
        self.displayed_image = None
        self.image_path = None
        self.clear_selection()
        self._toggle_scrollbars(0, 0)

    def _toggle_scrollbars(self, img_w, img_h, initial_setup=False):
        """Shows or hides scrollbars if the image dimensions exceed the canvas size."""
        canvas_w = self.canvas.winfo_width()
        canvas_h = self.canvas.winfo_height()
        
        need_v = img_h > canvas_h
        need_h = img_w > canvas_w

        if need_v and not initial_setup:
            self.v_scrollbar.grid(row=0, column=1, sticky='ns')
        else:
            self.v_scrollbar.grid_forget()

        if need_h and not initial_setup:
            self.h_scrollbar.grid(row=1, column=0, sticky='ew')
        else:
            self.h_scrollbar.grid_forget()

    def set_scale(self, scale):
        """Rescales the image and redraws it on the canvas, updating the scroll region."""
        if self.original_image is None:
            self.canvas.delete(tk.ALL)
            return

        self.scale_factor = scale
        self.canvas.delete(tk.ALL)
        
        new_width = int(self.original_image.width * scale)
        new_height = int(self.original_image.height * scale)
        
        self.displayed_image = self.original_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        self.tk_image = ImageTk.PhotoImage(self.displayed_image)
        
        self.canvas.config(scrollregion=(0, 0, new_width, new_height))
        self.canvas.create_image(0, 0, image=self.tk_image, anchor=tk.NW)
        
        self._toggle_scrollbars(new_width, new_height)
        
        self.log(f"Image scaled to {scale:.2f}x ({new_width}x{new_height}).")
        
        # NOTE: draw_selection_from_original is called from load_image now (for guarantee)
        # We leave this check here in case set_scale is called manually (e.g., from resize or zoom button)
        if self.original_rect_coords:
            self.draw_selection_from_original()

    def _load_existing_selection(self):
        """Reads *_rect.json from the new output directory and draws the existing ROI if found."""
        output_dir = self._get_output_dir()
        base_stem = self._get_base_path()
        
        if not output_dir or not base_stem:
            return

        # The JSON file will now be inside the new directory: [output_dir]/[base_stem]_rect.json
        json_filepath = output_dir / f"{base_stem}_rect.json"
        if json_filepath.exists():
            try:
                with open(json_filepath, 'r') as f:
                    data = json.load(f)
                
                ox1 = data['original_x_min']
                oy1 = data['original_y_min']
                ox2 = data['original_x_max']
                oy2 = data['original_y_max']
                
                self.original_rect_coords = [ox1, oy1, ox2, oy2]
                self.log(f"SUCCESS: Loaded existing ROI from {json_filepath.name}.")

            except Exception as e:
                self.log(f"ERROR: Could not read or parse existing ROI file {json_filepath.name}: {e}")
                self.original_rect_coords = None


    def draw_selection_from_original(self):
        """
        Translates the stored original_rect_coords to the current scaled canvas 
        coordinates and draws the rectangle.
        """
        if not self.original_rect_coords:
            return

        ox1, oy1, ox2, oy2 = self.original_rect_coords
        scale = self.scale_factor
        
        cx1 = ox1 * scale
        cy1 = oy1 * scale
        cx2 = ox2 * scale
        cy2 = oy2 * scale
        
        if self.rect_id:
            self.canvas.delete(self.rect_id)
        
        # Redraw the rectangle using the canvas coordinates
        self.rect_id = self.canvas.create_rectangle(
            cx1, cy1, cx2, cy2, 
            outline='red', width=2
        )
        
    def clear_selection(self):
        """Removes the current ROI rectangle and resets original coordinates."""
        if self.rect_id:
            self.canvas.delete(self.rect_id)
            self.rect_id = None
        self.start_x = None
        self.start_y = None
        self.original_rect_coords = None
        self.log("Selection cleared.")

    def on_mouse_down(self, event):
        """Starts the ROI selection process."""
        if self.original_image is None:
            self.log("ERROR: Cannot select area, no image is currently loaded.")
            return

        self.clear_selection()
        self.start_x = self.canvas.canvasx(event.x)
        self.start_y = self.canvas.canvasy(event.y)
        
        self.rect_id = self.canvas.create_rectangle(self.start_x, self.start_y, self.start_x, self.start_y, 
                                                    outline='red', width=2)

    def on_mouse_move(self, event):
        """Draws the dynamic ROI rectangle."""
        if self.original_image is None:
            return

        cur_x = self.canvas.canvasx(event.x)
        cur_y = self.canvas.canvasy(event.y)
        
        self.canvas.coords(self.rect_id, self.start_x, self.start_y, cur_x, cur_y)
        
    def on_mouse_up(self, event):
        """
        Finishes the ROI selection, converts to original image coordinates,
        and immediately saves the selection.
        """
        if self.original_image is None: 
            return
            
        end_x = self.canvas.canvasx(event.x)
        end_y = self.canvas.canvasy(event.y)
        
        # Finalize canvas coordinates (c for canvas)
        cx1 = min(self.start_x, end_x)
        cy1 = min(self.start_y, end_y)
        cx2 = max(self.start_x, end_x)
        cy2 = max(self.start_y, end_y)
        
        self.canvas.coords(self.rect_id, cx1, cy1, cx2, cy2)
        
        # Convert to UNCALED image coordinates (o for original)
        ox1 = int(cx1 / self.scale_factor)
        oy1 = int(cy1 / self.scale_factor)
        ox2 = int(cx2 / self.scale_factor)
        oy2 = int(cy2 / self.scale_factor)
        
        # Clamp to image bounds and store the final original coordinates
        width = self.original_image.width
        height = self.original_image.height
        ox1 = max(0, min(width, ox1))
        oy1 = max(0, min(height, oy1))
        ox2 = max(0, min(width, ox2))
        oy2 = max(0, min(height, oy2))

        self.original_rect_coords = [ox1, oy1, ox2, oy2]

        # Automatic Save
        self.save_selection()
        self.log(f"Selection saved automatically upon release.")

    def _convert_to_original_coords(self, *args):
        """Returns the stored original, unscaled coordinates."""
        if self.original_rect_coords is None:
            return None
            
        x_min = min(self.original_rect_coords[0], self.original_rect_coords[2])
        y_min = min(self.original_rect_coords[1], self.original_rect_coords[3])
        x_max = max(self.original_rect_coords[0], self.original_rect_coords[2])
        y_max = max(self.original_rect_coords[1], self.original_rect_coords[3])

        return [x_min, y_min, x_max, y_max]

    def save_rect_to_json(self):
        """
        Saves the ROI coordinates to a JSON file inside the new output directory.
        """
        if self.original_rect_coords is None:
            self.log("ERROR: No valid selection to save.")
            return None
            
        output_dir = self._get_output_dir()
        base_stem = self._get_base_path()
        if not output_dir or not base_stem:
            self.log("ERROR: Output directory could not be determined/created.")
            return None

        original_coords = self._convert_to_original_coords()
        
        # Define output path: [output_dir]/[base_stem]_rect.json
        json_filepath = output_dir / f"{base_stem}_rect.json"
        data = {
            "source_image": str(self.image_path), # Store path as string
            "scale_factor_at_selection": self.scale_factor,
            "original_x_min": original_coords[0],
            "original_y_min": original_coords[1],
            "original_x_max": original_coords[2],
            "original_y_max": original_coords[3],
            "width": original_coords[2] - original_coords[0],
            "height": original_coords[3] - original_coords[1]
        }

        try:
            with open(json_filepath, 'w') as f:
                json.dump(data, f, indent=4)
            self.log(f"INFO: ROI coordinates overwritten to {json_filepath.name} inside {output_dir.name}")
            return original_coords
        except Exception as e:
            self.log(f"ERROR saving JSON file: {e}")
            return None

    def clip_and_save_image(self, original_coords):
        """
        Clips the original image based on coordinates and saves it as *_table.jpg 
        inside the new output directory.
        """
        if self.original_image is None or not original_coords:
            self.log("ERROR: Cannot clip image, missing original or coordinates.")
            return

        crop_area = tuple(original_coords)
        
        output_dir = self._get_output_dir()
        base_stem = self._get_base_path()
        if not output_dir or not base_stem:
            self.log("ERROR: Output directory could not be determined/created.")
            return
            
        # Define output path: [output_dir]/[base_stem]_table.jpg
        jpg_filepath = output_dir / f"{base_stem}_table.jpg"
        
        try:
            clipped_image = self.original_image.crop(crop_area)
            
            # Save format is JPEG, file extension is jpg
            clipped_image.save(jpg_filepath, format="JPEG", quality=95)
            self.log(f"INFO: Clipped image overwritten and saved to {jpg_filepath.name} inside {output_dir.name}")
        except Exception as e:
            self.log(f"ERROR clipping or saving image: {e}")

    def save_selection(self):
        """Master function to save both JSON coordinates and the clipped image."""
        if self.original_rect_coords is None:
            self.log("Action Failed: Selection data is missing.")
            return
            
        self.log("Action: Saving selection and clipping image...")
        original_coords = self.save_rect_to_json()
        if original_coords:
            self.clip_and_save_image(original_coords)
            self.log("SUCCESS: Selection saved and image clipped successfully.")