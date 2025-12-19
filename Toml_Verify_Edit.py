import tkinter as tk
from tkinter import ttk, messagebox
from pathlib import Path
import os
import shutil
import json
from PIL import Image, ImageTk
from pandas import Series # Ensure Series is imported for type hinting
from typing import Dict, Any # Added Dict and Any import

# For local plotting functionality
try:
    import matplotlib.pyplot as plt 
    import numpy as np              
    import pandas as pd 
    # --- THAI FONT CONFIGURATION FOR MATPLOTLIB ---
    # Set global font for Matplotlib to a Thai-compatible font like Tahoma
    plt.rcParams['font.family'] = 'Tahoma' 
    plt.rcParams['axes.unicode_minus'] = False # Fix minus sign display issue
except ImportError:
    plt = np = pd = None 
    
try:
    import tomllib 
except ImportError:
    try:
        import tomli as tomllib
    except ImportError:
        tomllib = None

# --- Configuration Constants ---
DEFAULT_FONT = ('Tahoma', 14) 
TEXT_WIDTH = 35  
TEXT_HEIGHT = 12 
# --- Global Configuration Option ---
USE_SIMULATED_TOML = False 

# --- Helper Functions for MRK_SEQ Renumbering and TOML Serialization ---

def _to_excel_style_label(n: int) -> str:
    """Converts a 0-based index (n) to an Excel-style column label (A, B, ..., AA, AB, ...)."""
    label = ''
    while n >= 0:
        label = chr(n % 26 + ord('A')) + label
        n = n // 26 - 1
    return label

def _dict_to_simple_toml(data: Dict[str, Any]) -> str:
    """
    Simplistic function to serialize a nested dict back to a TOML string, 
    focused primarily on preserving list/array formatting, as required by the environment.
    """
    toml_str = ""
    for section_name, section_data in data.items():
        if isinstance(section_data, dict):
            toml_str += f"\n[{section_name}]\n"
            for key, value in section_data.items():
                if key == 'marker' and isinstance(value, list):
                    # Special handling for marker array of arrays
                    toml_str += f"marker = [\n"
                    for row in value:
                        # Convert Python list to string representation for TOML array
                        toml_str += f"  {str(row)},\n"
                    toml_str = toml_str.rstrip(',\n') + "\n]\n"
                elif isinstance(value, str):
                    toml_str += f"{key} = \"{value}\"\n"
                else:
                    toml_str += f"{key} = {value}\n"
        elif isinstance(section_data, (str, int, float, bool)):
            # Top-level keys (META usually handles this, but included for robustness)
            if isinstance(section_data, str):
                 toml_str += f"{section_name} = \"{section_data}\"\n"
            else:
                 toml_str += f"{section_name} = {section_data}\n"
    return toml_str.strip()


# --- SIMULATED OCR DATA ---
SIMULATED_OCR_TOML_CONTENT = """[META]
DOL_Office = "Narathivas"

[Deed]
Survey_Type = "MAP-L1"
EPSG = 24047
unit = "meter"
polygon_closed = false
marker = [
  [1, "A", "s41", 711042.723, 810293.807],
  [2, "B", "520", 711275.096, 810520.089],
  [3, "C", "s21", 711343.246, 810520.089]
]
"""

class TextEditor(tk.Text):
    """Simple wrapper for a Tkinter Text widget."""
    def __init__(self, master=None, **kwargs):
        # NOTE: Using DEFAULT_FONT defined globally (Tahoma, 10)
        super().__init__(master, **kwargs)
        self.config(
            wrap=tk.NONE, 
            undo=True, 
            font=DEFAULT_FONT,
            bg='#2E2E2E', 
            fg='white', 
            insertbackground='white'
        )

    def set_content(self, content):
        self.config(state=tk.NORMAL)
        self.delete(1.0, tk.END)
        self.insert(tk.END, content)
        self.config(state=tk.DISABLED)

    def get_content(self):
        return self.get(1.0, tk.END).strip()

    def enable_editing(self):
        """Activates editing mode and sets background to yellow."""
        self.config(state=tk.NORMAL, bg='yellow', fg='black')

    def disable_editing(self):
        """Deactivates editing mode and sets background to dark gray."""
        self.config(state=tk.DISABLED, bg='#2E2E2E', fg='white')


class OCRTomlEditor(ttk.Frame):
    """Panel for verifying, editing, and saving OCR output in TOML format."""
    def __init__(self, master=None, log_callback=None, **kwargs):
        self.log = log_callback if log_callback else print
        self.current_image_path = None
        self.base_name = None
        self.output_dir: Path = None 
        self.config: Series = Series(dtype=object) # Store CONFIG here (Reverted to Series)

        # Defensive move: Filter out configuration keys that might be accidentally passed down
        if 'column_spec' in kwargs:
             kwargs.pop('column_spec')
        if 'COLSPEC_RV25J' in kwargs:
             kwargs.pop('COLSPEC_RV25J')
        
        super().__init__(master, **kwargs)

        self.create_widgets()

    def create_widgets(self):
        # Configuration for grid layout
        self.grid_rowconfigure(2, weight=1)
        self.grid_columnconfigure(0, weight=1)

        # --- Plot Preview ---
        preview_frame = ttk.Frame(self, padding="5 5 5 5", relief=tk.RIDGE)
        preview_frame.grid(row=0, column=0, sticky='ew', pady=(0, 5))
        
        # Repurposed label for Plot preview (Restoring original initialization)
        self.image_plot_label = ttk.Label(preview_frame, text="Plot Preview (Pending Load)", 
                                           background='#333', foreground='white', 
                                           anchor='center', width=TEXT_WIDTH) # Restored width=TEXT_WIDTH
        self.image_plot_label.pack(side=tk.TOP, fill=tk.X, expand=True) # Restored fill=tk.X
        self.cropped_img_tk = None 

        # --- Editor Pane Caption ---
        caption_frame = ttk.Frame(self)
        caption_frame.grid(row=1, column=0, sticky='ew')
        self.toml_caption_label = ttk.Label(caption_frame, text="TOML Editor: (No File Loaded)", font=DEFAULT_FONT)
        self.toml_caption_label.pack(side=tk.LEFT)
        
        # Editor frame remains at row=2
        editor_frame = ttk.Frame(self)
        editor_frame.grid(row=2, column=0, sticky='nsew', pady=(5, 0))
        
        # Text Editor
        self.edit_ocr_editor = TextEditor(editor_frame, width=TEXT_WIDTH, height=TEXT_HEIGHT)
        self.edit_ocr_editor.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Scrollbar
        v_scroll = ttk.Scrollbar(editor_frame, orient=tk.VERTICAL, command=self.edit_ocr_editor.yview)
        v_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.edit_ocr_editor['yscrollcommand'] = v_scroll.set

        # --- Control Buttons ---
        button_frame = ttk.Frame(self)
        button_frame.grid(row=3, column=0, sticky='ew', pady=(5, 0))
        
        # Edit/Save button
        self.save_edit_button = ttk.Button(button_frame, text="Edit/Save TOML", command=self.on_save_or_edit_click)
        self.save_edit_button.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # Rerun MRK_SEQ button with confirmation dialog
        ttk.Button(button_frame, text="Rerun MRK_SEQ", command=self.confirm_RerunMRK_SEQ).pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # Initial state
        self.edit_ocr_editor.disable_editing()
        self.is_editing = False

    def _display_plot_image(self, plot_path: Path):
        """Loads the plot image (*_plot.png) and displays it in the preview label. (Original logic restored)"""
        self.cropped_img_tk = None 

        if not plot_path.exists():
            error_text = f"INFO: Plot not found.\nExpected Path: {plot_path.name}"
            self.image_plot_label.config(text=error_text, image="")
            self.log(f"INFO: Plot file missing at {plot_path.resolve()}")
            return

        try:
            img = Image.open(plot_path)
            
            # Resize for preview (Original resize logic restored)
            max_width = 600
            if img.width > max_width:
                ratio = max_width / img.width
                new_size = (max_width, int(img.height * ratio))
                img = img.resize(new_size, Image.Resampling.LANCZOS)

            self.cropped_img_tk = ImageTk.PhotoImage(img)
            self.image_plot_label.config(image=self.cropped_img_tk, text="")
            self.log(f"INFO: Plot preview loaded: {plot_path.name}")
        except Exception as e:
            self.log(f"ERROR loading plot image for preview: {e}")
            self.image_plot_label.config(text=f"ERROR loading plot: {plot_path.name}\n{e}", image="")
            self.cropped_img_tk = None

    def load_files(self, output_dir: Path, base_name: str, config: Series): # Reverted type hint to Series
        """
        Loads the relevant files (OCR TOML, Plot) for the selected image.
        """
        self.output_dir = output_dir
        self.base_name = base_name
        self.config = config # Store the configuration Series
        
        # 2. Load TOML (Prefer edit, fallback to original OCR, then mock)
        edit_toml_path = self.output_dir / f"{self.base_name}_OCRedit.toml"
        ocr_toml_path = self.output_dir / f"{self.base_name}_OCR.toml"
        
        toml_content = ""
        toml_path = None
        toml_filename = "(No File Loaded)" # Default caption

        if edit_toml_path.exists():
            toml_path = edit_toml_path
            toml_filename = edit_toml_path.name
        elif ocr_toml_path.exists():
            toml_path = ocr_toml_path
            toml_filename = ocr_toml_path.name
        elif USE_SIMULATED_TOML:
            toml_content = SIMULATED_OCR_TOML_CONTENT
            toml_filename = "Simulated Content"
            self.log("INFO: Using simulated TOML data.")
        
        if toml_path:
            try:
                toml_content = toml_path.read_text(encoding='utf-8')
                self.log(f"INFO: Loaded TOML from: {toml_path.name}")
            except Exception as e:
                self.log(f"ERROR reading TOML file {toml_path.name}: {e}")
                toml_content = f"# ERROR loading {toml_path.name}: {e}"
        
        # 1. Update TOML Caption Label
        self.toml_caption_label.config(text=f"TOML Editor: {toml_filename}")
        
        # 4. Update Editor state
        self.edit_ocr_editor.set_content(toml_content)
        self.edit_ocr_editor.disable_editing()
        self.save_edit_button.config(text="Edit/Save TOML")
        self.is_editing = False

        # 3. Load Plot Image for Preview (Trigger plot generation if TOML exists)
        # FIX: Always call on_plot_click if we have TOML content to ensure plot is generated and displayed immediately
        if toml_content and toml_content.strip():
             self.on_plot_click()
        else:
             plot_path = self.output_dir / f"{self.base_name}_plot.png"
             self._display_plot_image(plot_path) # Show existing plot or 'not found' message


    def confirm_RerunMRK_SEQ(self):
        """
        Shows a confirmation dialog before running RerunMRK_SEQ.
        """
        # Ensure we have a file loaded before showing the dialog
        if not self.output_dir or not self.base_name:
            self.log("ERROR: No file loaded. Cannot re-sequence.")
            messagebox.showwarning("Action Required", "Please load an image file first.")
            return

        warning_message = (
            "WARNING: This action will overwrite the 'SEQ_NUM' (Marker Number) and 'MRK_SEQ' (Marker Label) columns "
            f"in {self.base_name}_OCRedit.toml.\n\n"
            "SEQ_NUM will be reset to 1, 2, 3, ...\n"
            "MRK_SEQ will be reset to A, B, C, ..., AA, AB, ... (Excel-style).\n\n"
            "Do you wish to proceed and overwrite the file?"
        )
        
        if messagebox.askyesno("Confirm Marker Re-sequence", warning_message):
            self.RerunMRK_SEQ()
        else:
            self.log("INFO: Marker re-sequencing cancelled by user.")


    def RerunMRK_SEQ(self):
        """
        Reads *_OCRedit.toml, re-sequences both SEQ_NUM and MRK_SEQ columns, 
        and overwrites the TOML content.
        """
        if not self.output_dir or not self.base_name or not tomllib or not pd:
            self.log("ERROR: Initialization incomplete or libraries missing (Pandas/TOML).")
            return
            
        edit_toml_path = self.output_dir / f"{self.base_name}_OCRedit.toml"
        
        if not edit_toml_path.exists():
            self.log(f"ERROR: TOML file not found for editing: {edit_toml_path.name}")
            return
            
        try:
            # 1. Read and Parse TOML
            toml_content = edit_toml_path.read_text(encoding='utf-8')
            toml_data = tomllib.loads(toml_content)
            
            marker_data = toml_data.get('Deed', {}).get('marker')
            if not marker_data:
                self.log("WARN: Marker array is empty or missing in [Deed]. Cannot re-sequence.")
                return

            # 2. Get Column Specification and indices
            expected_columns = self.config.get('COLSPEC_TOML', None)
            if not expected_columns or 'MRK_SEQ' not in expected_columns or 'SEQ_NUM' not in expected_columns:
                self.log("ERROR: 'COLSPEC_TOML' missing or does not contain 'MRK_SEQ' and 'SEQ_NUM'.")
                return
            
            # 3. Create DataFrame and Re-sequence
            df = pd.DataFrame(marker_data, columns=expected_columns)
            
            # Generate new sequence labels for MRK_SEQ (A, B, ...)
            new_mrk_seq = [_to_excel_style_label(i) for i in range(len(df))]
            # Generate new sequence numbers for SEQ_NUM (1, 2, 3, ...)
            new_seq_num = list(range(1, len(df) + 1))
            
            # Update the DataFrame columns
            df['MRK_SEQ'] = new_mrk_seq
            df['SEQ_NUM'] = new_seq_num
            
            # 4. Convert back to list of lists and update TOML dictionary
            updated_marker_data = df.values.tolist()
            toml_data['Deed']['marker'] = updated_marker_data
            
            # 5. Serialize dictionary back to TOML string
            new_toml_content = _dict_to_simple_toml(toml_data)

            # 6. Overwrite file and update editor view
            edit_toml_path.write_text(new_toml_content, encoding='utf-8')
            self.edit_ocr_editor.set_content(new_toml_content)
            self.log(f"SUCCESS: SEQ_NUM and MRK_SEQ columns re-sequenced and saved to {edit_toml_path.name}.")
            
            # 7. Replot the data to visualize changes immediately
            self.on_plot_click() 

            # Ensure the editor state is correctly set after direct manipulation
            self.edit_ocr_editor.disable_editing()
            self.save_edit_button.config(text="Edit/Save TOML")
            self.is_editing = False

        except Exception as e:
            self.log(f"FATAL ERROR during MRK_SEQ re-sequencing: {e}")
            messagebox.showerror("Re-sequence Error", f"Failed to re-sequence markers.\nDetails: {e}")


    def on_save_or_edit_click(self):
        """Toggles between Edit mode and Save mode."""
        if self.is_editing:
            # Currently in Edit mode, switch to Save
            if self.save_edited_toml():
                self.edit_ocr_editor.disable_editing()
                self.save_edit_button.config(text="Edit/Save TOML")
                self.is_editing = False
        else:
            # Currently in View mode, switch to Edit
            self.edit_ocr_editor.enable_editing()
            self.save_edit_button.config(text="Save TOML")
            self.is_editing = True
            self.log("INFO: TOML Editor unlocked for manual changes (Yellow Background).")
            
    def save_edited_toml(self):
        """Saves content to *_OCRedit.toml."""
        if not self.output_dir or not self.base_name:
            self.log("ERROR: Output path not set.")
            return False
            
        edit_toml_path = self.output_dir / f"{self.base_name}_OCRedit.toml"
        
        content = self.edit_ocr_editor.get_content()

        try:
            # Basic validation: try to parse TOML before saving
            if tomllib:
                tomllib.loads(content) 
            
            edit_toml_path.write_text(content, encoding='utf-8')
            
            # Regenerate and display plot immediately after successful save
            self.on_plot_click() 
            
            self.log(f"SUCCESS: Edited TOML saved to {edit_toml_path.name} inside {self.output_dir.name}")
            return True
        except Exception as e:
            self.log(f"ERROR: Invalid TOML format or save error: {e}")
            messagebox.showerror("TOML Save Error", f"Cannot save: Invalid TOML format.\nDetails: {e}")
            return False
        
    def on_plot_click(self):
        """Plots the coordinates from the edited TOML data. Always closes the polygon."""
        if not plt or not tomllib or not pd:
            self.log("ERROR: Cannot plot. Matplotlib/TOML/Pandas library not available.")
            return
        
        # 1. Retrieve the required column specification from the merged config
        expected_columns = self.config.get('COLSPEC_TOML', None)
        
        if not expected_columns or len(expected_columns) < 5:
             self.log("ERROR: Configuration missing critical spec 'COLSPEC_TOML' or it's incomplete for plotting (requires 5 columns).")
             return

        # Simulating data retrieval and plotting
        try:
            # Get content from the editor (which holds the latest data, saved or unsaved)
            content = self.edit_ocr_editor.get_content()
            data = tomllib.loads(content)
            
            # Simplified data extraction assuming a single [Deed] section structure for marker
            marker_data = data.get('Deed', {}).get('marker')
            
            if marker_data:
                # Use the retrieved list directly as column names
                df = pd.DataFrame(marker_data, columns=expected_columns)
                
                # Assume EASTING and NORTHING are the 5th and 4th columns (index 4 and 3)
                x_col = expected_columns[4]
                y_col = expected_columns[3]
                marker_col = expected_columns[2]

                # --- Plotting Logic ---
                fig, ax = plt.subplots(figsize=(6, 6))
                
                # Ensure the columns are numeric before plotting
                df[x_col] = pd.to_numeric(df[x_col], errors='coerce')
                df[y_col] = pd.to_numeric(df[y_col], errors='coerce')
                df.dropna(subset=[x_col, y_col], inplace=True)

                if df.empty:
                    self.log("WARN: No valid numeric coordinates found after cleaning. Plot not generated.")
                    plt.close(fig)
                    # Clear any existing plot image if plot failed
                    self._display_plot_image(Path("non_existent_path"))
                    return
                
                # --- FIX: Close the polygon (append first row to the end) ---
                closed_df = pd.concat([df, df.iloc[[0]]], ignore_index=True)
                
                ax.plot(closed_df[x_col], closed_df[y_col], 'o-', label='Survey Points')
                
                # Annotate markers using the original (non-closed) DataFrame
                for i, row in df.iterrows():
                    # Annotate using the marker name (e.g., MRK_DOL)
                    # Use a smaller font size for annotations so they don't overlap too much
                    ax.annotate(str(row[marker_col]), (row[x_col], row[y_col]), 
                                textcoords="offset points", xytext=(5,5), ha='center', fontsize=8)
                
                # Set font property for title and labels explicitly for Thai
                font_properties = {'family': 'Tahoma', 'size': 12}
                ax.set_title(f"Survey Plot: {self.base_name}", fontdict=font_properties)
                ax.set_xlabel("Easting (X)", fontdict=font_properties)
                ax.set_ylabel("Northing (Y)", fontdict=font_properties)
                
                ax.grid(True)
                ax.axis('equal') # Important for spatial data

                # Save plot to the output directory
                plot_path = self.output_dir / f"{self.base_name}_plot.png"
                fig.savefig(plot_path)
                
                plt.close(fig) # Close figure to free memory
                self.log(f"SUCCESS: Plot saved to {plot_path.name}")
                
                # Update the plot preview widget immediately
                self._display_plot_image(plot_path)

            else:
                self.log("ERROR: TOML data structure is missing the 'marker' array under [Deed].")

        except Exception as e:
            self.log(f"ERROR during plotting: {e}")
            messagebox.showerror("Plot Error", f"An error occurred during plotting:\n{e}")