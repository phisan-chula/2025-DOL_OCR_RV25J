import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import os
from pathlib import Path
import time
import subprocess
import sys
import argparse 

# Import the configuration manager from the other file
# Using the correct module name based on file name CONFIG_AppRV25J.py
from CONFIG_AppRV25J import Config_AppRV25J 

# --- Custom Component Imports (Assumed to be in the same directory) ---
from ImageSelect import ImageSelect 
from Toml_Verify_Edit import OCRTomlEditor 

# Default fallback if config load fails or key is missing
DEFAULT_IMAGE_DIR = r".\RV25J_L1L2"

class RV25J_OCR_Center(tk.Tk):
    """
    Main application window for the RV25J OCR Center.
    Integrates File List, Image Viewer, and TOML Editor.
    """
    def __init__(self, initial_dir_arg: str): # initial_dir_arg is now guaranteed to be set
        super().__init__()
        
        # 4. Initialize Core Variables (Ensure activity_log is initialized for early logging)
        self.activity_log: tk.Text | None = None
        self.file_list = []
        self.image_selector: ImageSelect = None
        self.ocr_editor_panel: OCRTomlEditor = None
        self.current_image_path: Path | None = None # Attribute to track current image
        
        # 1. Load Configuration using the centralized manager (Loads Global only)
        self.config_manager = self._load_central_config()
        
        # Access the initial (GLOBAL) configuration
        current_config = self.config_manager.CONFIG 

        # 2. Setup Defaults based on Config 
        
        # --- Load and assign UI parameters from configuration ---
        
        # Load Window Settings with Fallbacks
        self.window_title = current_config.get('WINDOW_TITLE', "RV25J OCR Center (Fallback)")
        self.default_width = current_config.get('DEFAULT_WIDTH', 1300)
        self.default_height = current_config.get('DEFAULT_HEIGHT', 850)

        # Load Font Settings with Fallbacks and conversion to Tkinter tuple format
        # TOML stores ['FontName', Size] list, which needs to be a tuple for Tkinter font parameter.
        default_font_list = current_config.get('DEFAULT_FONT', ['Tahoma', 16])
        self.default_font = tuple(default_font_list)
        
        log_font_list = current_config.get('LOG_FONT', ['Courier New', 16])
        self.log_font = tuple(log_font_list)

        # Load other application defaults
        self.current_scale = current_config.get('VIEW_SCALE', 0.5) 
        self.default_dir_str = current_config.get('DEFAULT_IMAGE_DIR', initial_dir_arg)
        
        self.zoom_buttons = {}

        # 3. Setup Window (Using loaded variables)
        self.title(self.window_title)
        self.geometry(f"{self.default_width}x{self.default_height}")
        
        # 5. Build UI Layout
        self.create_widgets()
        
        # 6. Load Initial Data (and trigger local config load/merge)
        self.load_initial_dir()
        
        self.protocol("WM_DELETE_WINDOW", self.on_quit)


    def _load_central_config(self):
        """
        Instantiates the Config_AppRV25J and loads GLOBAL configuration, capturing logs.
        """
        app_file_dir = Path(sys.argv[0]).parent
        
        try:
            # 1. Instantiate the Manager
            manager = Config_AppRV25J(app_file_dir=app_file_dir)
            
            # 2. Call the global load method explicitly to run the setup and capture logs
            global_log_messages = manager.load_global_config_and_log()
            
            # Log the successful messages
            for msg in global_log_messages:
                 self.log_activity(msg)
            
            return manager
            
        except RuntimeError as e:
            # Catch the specific exception raised by the config manager on failure
            self.log_activity(f"FATAL GLOBAL CONFIG ERROR: {e}. Application cannot start.")
            sys.exit(1)
        except Exception as e:
            # Catch any unexpected errors during instantiation or logging
            self.log_activity(f"CRITICAL ERROR during config startup: {e}. Application cannot start.")
            sys.exit(1)

    def create_widgets(self):
        """Builds the main UI structure."""
        self.grid_rowconfigure(0, weight=1)
        
        # --- LAYOUT FIX: Adjust column weights for larger center panel (1:4:1) ---
        self.grid_columnconfigure(0, weight=1)  # Left Panel (File List)
        self.grid_columnconfigure(1, weight=8)  # Center Panel (Image/Plot Viewer) - BIGGER
        self.grid_columnconfigure(2, weight=1)  # Right Panel (Editor/Log) - SMALLER
        
        # Initialize ttk style (needed for button appearance change)
        self.style = ttk.Style()
        
        # Restoring original styles (NO PADDING/SMALLER FONT)
        self.style.configure('Toggled.TButton', background='lightgray', foreground='black')
        self.style.configure('Default.TButton', background='SystemButtonFace', foreground='black')
        
        # Setting style for Red Button (original font size)
        self.style.configure('Danger.TButton', background='red', foreground='white', font=('Tahoma', 12, 'bold'))
        self.style.map('Danger.TButton',
                       background=[('active', '#CC0000')], # Darker red on hover
                       foreground=[('active', 'white')])


        # --- Left Panel: File List ---
        left_panel = ttk.Frame(self, padding="5")
        left_panel.grid(row=0, column=0, sticky='nsew')
        left_panel.grid_rowconfigure(1, weight=1)
        left_panel.grid_columnconfigure(0, weight=1)

        ttk.Button(left_panel, text="Change Directory", 
           command=self.select_directory).grid(row=0, column=0, sticky='ew', pady=(0, 5))
        
        # Use self.default_font
        self.file_listbox = tk.Listbox(left_panel, font=self.default_font, bg='#2E2E2E', fg='white')
        self.file_listbox.grid(row=1, column=0, sticky='nsew')
        self.file_listbox.bind('<<ListboxSelect>>', self.on_file_select)

        # --- Center Panel: Image Viewer ---
        center_panel = ttk.Frame(self, padding="5")
        center_panel.grid(row=0, column=1, sticky='nsew')
        center_panel.grid_rowconfigure(1, weight=1)
        center_panel.grid_columnconfigure(0, weight=1)

        # Top Buttons (Zoom, Quit)
        control_frame = ttk.Frame(center_panel)
        control_frame.grid(row=0, column=0, sticky='ew', pady=(0, 5))

        ttk.Label(control_frame, text="Zoom:").pack(side=tk.LEFT, padx=(0, 5))
        self._add_zoom_buttons(control_frame) # Contains 25%, 50%, 100%

        # --- New Vertical Separator and Remove Button ---
        ttk.Separator(control_frame, orient=tk.VERTICAL).pack(side=tk.LEFT, padx=10, fill='y')

        # Uses the Danger.TButton style for red background
        ttk.Button(control_frame, text="Remove OCR", style='Danger.TButton', 
                   command=self._remove_ocr_files).pack(side=tk.LEFT, padx=5)
        # --- End New Elements ---

        ttk.Button(control_frame, text="Quit", command=self.on_quit).pack(side=tk.RIGHT)
        
        # Image Selector Widget
        self.image_selector = ImageSelect(center_panel, log_callback=self.log_activity)
        self.image_selector.grid(row=1, column=0, sticky='nsew')

        # --- Right Panel: Editor and Log ---
        right_panel = ttk.Frame(self, padding="5")
        right_panel.grid(row=0, column=2, sticky='nsew')
        
        # CONFIGURE EXPANDING ROWS: 50% for Editor, 50% for Log Text
        right_panel.grid_rowconfigure(0, weight=1) # Row 0: TOML Editor 
        right_panel.grid_rowconfigure(2, weight=1) # Row 2: Activity Log Text 
        right_panel.grid_columnconfigure(0, weight=1)

        # OCR TOML Editor (Row 0)
        self.ocr_editor_panel = OCRTomlEditor(right_panel, log_callback=self.log_activity)
        self.ocr_editor_panel.grid(row=0, column=0, sticky='nsew', pady=(0, 5))

        # Activity Log Label (Row 1 - fixed height)
        ttk.Label(right_panel, text="Activity Log:").grid(row=1, column=0, sticky='sw')
        
        # Activity Log (Widget creation) (Row 2 - expanding height)
        self.activity_log = tk.Text(right_panel, height=10, font=self.log_font, bg='#1E1E1E', fg='#E0E0E0')
        self.activity_log.grid(row=2, column=0, sticky='nsew')
        self.activity_log.config(state=tk.DISABLED)
        
        self.log_activity("UI components initialized.")


    def _add_zoom_buttons(self, master):
        """Adds zoom buttons to the control frame."""
        zoom_levels = {0.25: "25%", 0.5: "50%", 1.0: "100%"}
        
        for val, text in zoom_levels.items():
            btn = ttk.Button(master, text=text, style='Default.TButton', 
                             command=lambda v=val: self._set_scale(v))
            btn.pack(side=tk.LEFT, padx=2)
            self.zoom_buttons[val] = btn
        
        self._set_scale(self.current_scale)

    def load_initial_dir(self):
        """Loads files from the directory passed via CLI argument."""
        # The initial_dir_arg passed to __init__ is already stored in self.default_dir_str
        default_path = Path(self.default_dir_str)
        if default_path.is_dir():
            # Automatically load the directory provided by the CLI
            self.select_directory(initial_dir=str(default_path), skip_dialog=True)
        else:
            self.log_activity(f"FATAL ERROR: CLI directory not found: {default_path}. Showing directory dialog.")
            self.select_directory(initial_dir=str(Path.cwd()), skip_dialog=False) # Fallback to dialog


    def load_file_list(self, directory):
        """Scans the directory for image files and populates the listbox."""
        self.file_listbox.delete(0, tk.END)
        self.file_list = []
        search_path = Path(directory)
        
        for path in search_path.rglob('*_RV25J.jpg'):
            self.file_list.append(path) 
            # Use relative path starting from the directory name for display
            display_name = path.relative_to(search_path.parent)
            self.file_listbox.insert(tk.END, str(display_name))

        self.log_activity(f"Found {len(self.file_list)} images in {search_path.name}.")
        
        if self.file_list:
            self.file_listbox.selection_set(0)
            self.on_file_select(None)

    def on_file_select(self, event):
        """Handles selection change in the listbox."""
        try:
            selection = self.file_listbox.curselection()
            if selection:
                index = selection[0]
                self.load_selected_image(self.file_list[index])
        except IndexError:
            pass
        except Exception as e:
            self.log_activity(f"ERROR during file selection: {e}")

    def load_selected_image(self, image_path: Path):
        """Loads the image into the viewer and initiates editor loading."""
        
        self.current_image_path = image_path # Store the path
        self.image_selector.load_image(str(image_path))
        
        # Ensure scale is consistent
        view_scale_config = self.config_manager.CONFIG.get('VIEW_SCALE', 0.5) 
        if self.current_scale != view_scale_config:
             self._set_scale(view_scale_config)
        else:
             self._set_scale(self.current_scale)

        # Derive base name and output directory (Original logic restored)
        base_name_str = image_path.stem
        if base_name_str.lower().endswith('_rv25j'):
            base_name_str = base_name_str[:-6] 

        output_dir = image_path.parent / base_name_str 
        
        # This call handles loading the TOML content and immediately plotting if content exists
        # Passing the CONFIG Series as originally intended
        self.ocr_editor_panel.load_files(output_dir, base_name_str, self.config_manager.CONFIG)
        
        self.log_activity(f"Loaded file: {image_path.name}")
        
    def select_directory(self, initial_dir: str = None, skip_dialog: bool = False):
        """
        Opens a file dialog to select a new base directory OR loads the initial dir.
        Loads and merges local config based on the selected directory.
        """
        if initial_dir is None:
            initial_dir = self.default_dir_str
        
        if not Path(initial_dir).exists():
            initial_dir = str(Path.cwd())
            
        directory = None

        if skip_dialog:
            directory = initial_dir
        else:
            directory = filedialog.askdirectory(
                title="Select Directory with _RV25J.jpg Files",
                initialdir=str(initial_dir)
            )
        
        if directory:
            self.default_dir_str = directory
            
            # 1. Load Local Config and Merge using the selected directory
            selected_path = Path(directory)
            try:
                # Capture and log local load and merge messages
                merge_log_messages = self.config_manager.update_with_local(working_folder=selected_path)
                for msg in merge_log_messages:
                    self.log_activity(msg)
                
                # Update UI elements that rely on config (e.g., scale) immediately
                self.current_scale = self.config_manager.CONFIG.get('VIEW_SCALE', 0.5)
                self._set_scale(self.current_scale) 
                
            except Exception as e:
                self.log_activity(f"WARN: Failed to load local config from {selected_path}: {e}")
                
            # 2. Load File List
            self.load_file_list(directory)


    def _set_scale(self, scale):
        """
        Updates zoom level and button visuals.
        """
        self.current_scale = scale
        if self.image_selector:
            self.image_selector.set_scale(scale)
            
        for val, btn in self.zoom_buttons.items():
            # Use style configuration instead of relief for modern Tkinter themes
            if val == scale:
                btn.configure(style='Toggled.TButton')
            else:
                btn.configure(style='Default.TButton')

    def _handle_save_or_edit_click(self):
        """Delegates save/edit to the editor panel."""
        if self.ocr_editor_panel:
            self.ocr_editor_panel.on_save_or_edit_click()

    # --- Method for new "Remove OCR" functionality (Logic derived safely) ---
    def _remove_ocr_files(self):
        """
        Displays a confirmation dialog and, if confirmed, deletes 
        the *_OCR.toml, *_OCRedit.toml, AND *_plot.png files associated with the current image.
        """
        # 0. Initial check for loaded image
        if not self.current_image_path: 
            self.log_activity("WARN: No image is currently loaded. Cannot remove files.")
            messagebox.showwarning("Warning", "No image is currently loaded to determine files.")
            return

        current_image_path = self.current_image_path

        # 1. Derive all related paths from the current image path (Safe path derivation)
        
        # Derive base name (same logic as load_selected_image)
        base_name_str = current_image_path.stem
        if base_name_str.lower().endswith('_rv25j'):
            base_name_str = base_name_str[:-6] 

        # Derive output directory (e.g., C0002 folder)
        output_dir = current_image_path.parent / base_name_str 
        
        # Calculate full paths for all target files (Hardcoded suffixes restored)
        toml_path = output_dir / f"{base_name_str}_OCR.toml"
        edit_path = output_dir / f"{base_name_str}_OCRedit.toml"
        plot_path = output_dir / f"{base_name_str}_plot.png" 
        
        files_to_check = [toml_path, edit_path, plot_path]
        
        # 2. Pop-up Confirmation Dialog
        confirm = messagebox.askyesno(
            "Confirm OCR File Removal",
            (
                f"You are going to remove ALL OCR-related files for:\n\n"
                f"Image Base: **{base_name_str}**\n\n"
                f"This will attempt to remove:\n"
                f"1. {toml_path.name}\n"
                f"2. {edit_path.name}\n"
                f"3. {plot_path.name}\n\n"
                f"Do you want to proceed?"
            ),
            icon=messagebox.WARNING
        )
        
        if confirm:
            files_removed = []
            
            # 3. Delete files
            for file_path in files_to_check:
                try:
                    if file_path.exists():
                        os.remove(file_path)
                        files_removed.append(file_path.name)
                        self.log_activity(f"SUCCESS: Removed {file_path.name}")
                    else:
                        self.log_activity(f"INFO: {file_path.name} not found, skipped deletion.")
                except Exception as e:
                    self.log_activity(f"ERROR: Failed to remove {file_path.name}: {e}")
            
            # 4. Clear editor state and reload image to reflect changes
            if files_removed:
                # NOTE: The explicit clear_editor_and_plot() call is still removed
                
                # Re-trigger the load to update the editor/plot view for the current image
                self.load_selected_image(current_image_path)
                messagebox.showinfo("Success", f"Successfully removed: {', '.join(files_removed)}")
            else:
                 self.log_activity("INFO: No files were found or removed.")
        else:
            self.log_activity("INFO: OCR file removal cancelled by user.")
    # --- End "Remove OCR" functionality ---


    def on_quit(self):
        """
        Handles graceful exit by stopping the Tkinter event loop.
        """
        self.log_activity("Application quit requested. Initiating graceful shutdown...")
        
        self.quit()
        self.destroy()

    def log_activity(self, message):
        """Appends text to the log window."""
        if self.activity_log is None:
            # Only print during early startup if activity_log widget isn't ready
            print(f"EARLY LOG: {message}")
            return
            
        # Ensure log entries from config_manager are clean
        if message.startswith("EARLY LOG: "):
             message = message[11:]
             
        ts = time.strftime("[%H:%M:%S]")
        self.activity_log.config(state=tk.NORMAL)
        self.activity_log.insert(tk.END, f"{ts} {message}\n")
        self.activity_log.config(state=tk.DISABLED)
        self.activity_log.see(tk.END)

    # Argument Parsing Method (Original logic restored)
    @staticmethod
    def _parse_args():
        """Parses command line arguments for the application."""
        parser = argparse.ArgumentParser(
            description="RV25J OCR Center Application.",
            epilog="The application requires a root folder and will recursively scan it for files ending in '_rv25j.jpg'.",
            formatter_class=argparse.RawTextHelpFormatter
        )
        
        # Positional Argument 'folder' (required)
        parser.add_argument(
            'folder', # Positional argument name
            type=str, 
            nargs='?', # Make positional argument optional for cleaner testing
            default=DEFAULT_IMAGE_DIR, # Restoring original constant here
            help=(
                "The **root directory** containing parcel documents.\\n"
                "The application will **recursively search** this directory and its subfolders\\n"
                "for all image files matching the pattern `*_rv25j.jpg` to populate the file list."
            )
        )
        
        args = parser.parse_args()
        return args

if __name__ == '__main__':
    # Parse arguments first
    cli_args = RV25J_OCR_Center._parse_args()
    
    try:
        # Pass the parsed folder path to the application initializer
        app = RV25J_OCR_Center(initial_dir_arg=cli_args.folder)
        app.mainloop()
    except Exception as e:
        print(f"An unexpected error occurred in mainloop: {e}", file=sys.stderr)