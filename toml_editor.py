import tkinter as tk
from tkinter import ttk, messagebox
import os
import re

# --- GLOBAL CONFIGURATION CONSTANTS (User Configurable) ---
TEXT_WIDTH = 80             # Width in characters
TEXT_HEIGHT = 25            # Height in lines
TEXT_FONT_FAMILY = 'Tahoma'  # Font family (e.g., 'TlwgTypewriter', 'Arial', 'Tahoma')
TEXT_FONT_SIZE = 14         # Font size in points

class TOMLTextEditor(tk.Text):
    """
    A custom Tkinter Text widget that provides basic TOML syntax highlighting.
    """
    def __init__(self, master=None, **kwargs):
        
        self.font_family = kwargs.pop('font_family', TEXT_FONT_FAMILY)
        self.font_size = kwargs.pop('font_size', TEXT_FONT_SIZE)
        
        default_kwargs = {
            'width': kwargs.pop('width', TEXT_WIDTH),
            'height': kwargs.pop('height', TEXT_HEIGHT),
            'wrap': 'none',
            'font': (self.font_family, self.font_size), 
            'undo': True
        }
        final_kwargs = {**default_kwargs, **kwargs}
        
        super().__init__(master, **final_kwargs)
        
        self._highlight_id = None
        self._configure_tags()
        self.bind('<KeyRelease>', self._on_text_change)
        
    def _configure_tags(self):
        """Defines the color and style tags for TOML syntax highlighting."""
        base_font = (self.font_family, self.font_size)
        
        self.tag_config('comment', foreground='gray', font=(self.font_family, self.font_size, 'italic'))
        self.tag_config('table', foreground='#800080', font=(self.font_family, self.font_size, 'bold')) 
        self.tag_config('key', foreground='#00008B', font=(self.font_family, self.font_size, 'bold')) 
        self.tag_config('string', foreground='#8B0000', font=base_font) 
        self.tag_config('number_bool', foreground='#006400', font=base_font) 
        self.tag_config('datetime', foreground='#FF8C00', font=base_font)

    def _highlight_syntax(self, *args):
        """Applies syntax highlighting to the current content."""
        
        for tag in ['comment', 'table', 'key', 'string', 'number_bool', 'datetime']:
            self.tag_remove(tag, '1.0', tk.END)

        content = self.get('1.0', tk.END)
        
        patterns = {
            'comment': r'#.*$',                          
            'table': r'^(\s*\[\[?.*?\]\]?\s*)$',         
            'string': r'([\'\"])((?:\\.|[^"\\])*)\1',    
            'number_bool': r'\b(true|false)\b|\b\d+(\.\d*)?([eE][+-]?\d+)?\b', 
            'key': r'^(\s*[\w\-_"]+)\s*=',               
            'datetime': r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}([+-]\d{2}:\d{2}|Z)?' 
        }

        for i, line in enumerate(content.splitlines(), start=1):
            line_start = f'{i}.0'
            line_end = f'{i}.{len(line)}'
            
            # 1. Table Highlighting 
            table_match = re.match(patterns['table'], line.strip())
            if table_match and not line.strip().startswith('#'):
                self.tag_add('table', line_start, line_end)
                continue 

            # 2. Key Highlighting 
            key_match = re.match(patterns['key'], line)
            if key_match and not line.strip().startswith('#'):
                key_end_index = key_match.end(1)
                self.tag_add('key', line_start, f'{i}.{key_end_index}')

            # 3. Inline Highlighting 
            for tag, pattern in patterns.items():
                if tag in ['comment', 'table', 'key']:
                    continue
                
                for match in re.finditer(pattern, line):
                    start_char = match.start(0)
                    end_char = match.end(0)
                    self.tag_add(tag, f'{i}.{start_char}', f'{i}.{end_char}')

            # 4. Comment Highlighting 
            comment_match = re.search(patterns['comment'], line)
            if comment_match:
                start_index = comment_match.start(0)
                self.tag_add('comment', f'{i}.{start_index}', line_end)
    
    def _on_text_change(self, event):
        """Handler for text changes, uses a small delay (250ms debounce) before highlighting."""
        if self._highlight_id:
            self.after_cancel(self._highlight_id)
        self._highlight_id = self.after(250, self._highlight_syntax)

    # --- Public Helper Methods ---

    def set_content(self, text):
        """Sets the content of the editor and triggers highlighting."""
        self.delete('1.0', tk.END)
        self.insert('1.0', text)
        self._highlight_syntax()

    def get_content(self):
        """Gets the content of the editor, stripping trailing whitespace."""
        return self.get('1.0', tk.END).strip()
    
    def highlight(self):
        """Manually triggers a syntax highlight."""
        self._highlight_syntax()


class TOMLApp(tk.Tk):
    """
    Main application window hosting the TOMLTextEditor widget.
    """
    def __init__(self, toml_filepath):
        super().__init__()
        
        self.toml_filepath = toml_filepath
        self.title(f"TOML File Editor: {os.path.basename(self.toml_filepath)}")
        self.geometry("800x600") 

        self.is_editing = False

        self._create_widgets()
        self._load_toml_content()

    def _create_widgets(self):
        """Sets up the GUI components."""

        control_frame = ttk.Frame(self, padding="10")
        control_frame.pack(fill='x', expand=False)

        self.edit_button = ttk.Button(
            control_frame,
            text="Edit TOML",
            command=self.edit_toml_content
        )
        self.edit_button.pack(side='left', padx=5)

        self.save_button = ttk.Button(
            control_frame,
            text="Save to File",
            command=self.save_toml_content,
            state=tk.DISABLED
        )
        self.save_button.pack(side='left', padx=5)
        
        self.clear_button = ttk.Button(
            control_frame,
            text="Clear Text",
            command=self.clear_text_content,
            state=tk.DISABLED
        )
        self.clear_button.pack(side='left', padx=5)

        self.status_label = ttk.Label(
            control_frame,
            text="Status: Read-Only (Click 'Edit TOML' to change)",
            foreground="blue"
        )
        self.status_label.pack(side='right', padx=5)

        text_frame = ttk.Frame(self, padding="10")
        text_frame.pack(fill='both', expand=True)

        scrollbar = ttk.Scrollbar(text_frame)
        scrollbar.pack(side='right', fill='y')

        self.text_editor = TOMLTextEditor(
            text_frame,
            yscrollcommand=scrollbar.set,
        )
        self.text_editor.pack(fill='both', expand=True)
        scrollbar.config(command=self.text_editor.yview)

        self.text_editor.config(state=tk.DISABLED)

    def _load_toml_content(self):
        """Loads the content of the TOML file into the text editor."""
        try:
            with open(self.toml_filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            self.text_editor.config(state=tk.NORMAL)
            self.text_editor.set_content(content)
            self.text_editor.config(state=tk.DISABLED)
            
        except IOError:
            error_msg = f"Could not read the file: {self.toml_filepath}. Using empty content."
            messagebox.showerror("Load Error", error_msg)
            self.status_label.config(text="Status: Error loading file.", foreground="red")
            self.text_editor.config(state=tk.NORMAL)
            self.text_editor.set_content("# No content loaded")
            self.text_editor.config(state=tk.DISABLED)

    def clear_text_content(self):
        """Clears all content from the text editor if in editing mode."""
        if not self.is_editing:
            messagebox.showwarning("Warning", "Cannot clear: The file is not currently in editing mode.")
            return

        response = messagebox.askyesno(
            "Confirm Clear",
            "Are you sure you want to delete ALL content in the editor?"
        )
        
        if response:
            self.text_editor.set_content("")
            self.status_label.config(text="Status: Content cleared. Save or undo changes.", foreground="orange")


    def edit_toml_content(self):
        """Enables editing of the text editor."""
        if not self.is_editing:
            self.text_editor.config(state=tk.NORMAL)
            self.save_button.config(state=tk.NORMAL)
            self.clear_button.config(state=tk.NORMAL)
            self.edit_button.config(text="Editing...", state=tk.DISABLED)
            self.status_label.config(text="Status: ACTIVE EDITING", foreground="red")
            self.is_editing = True
            self.text_editor.focus_set()

    def save_toml_content(self):
        """Saves the current text editor content to the fixed file."""
        if not self.is_editing:
            messagebox.showwarning("Warning", "The file is not currently in editing mode.")
            return

        content = self.text_editor.get_content()
        try:
            with open(self.toml_filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            
            # Revert to read-only state
            self.text_editor.config(state=tk.DISABLED)
            self.save_button.config(state=tk.DISABLED)
            self.clear_button.config(state=tk.DISABLED)
            self.edit_button.config(text="Edit TOML", state=tk.NORMAL)
            self.status_label.config(text="Status: Saved successfully! Read-Only.", foreground="green")
            self.is_editing = False
            
            # Ensure the content is highlighted correctly after save/read-only switch
            self.text_editor.highlight()
            
        except IOError as e:
            messagebox.showerror("Save Error", f"Could not write to file: {e}")
            self.status_label.config(text="Status: ERROR saving file.", foreground="red")

if __name__ == "__main__":
    # --- Main Application Initialization ---
    INITIAL_FILE = "S10_MAPL1.toml"
    
    # Check if the required file exists before starting the app
    if not os.path.exists(INITIAL_FILE):
        print(f"File not found: {INITIAL_FILE}. Creating initial file.")
        try:
            # Create the file with placeholder content
            with open(INITIAL_FILE, 'w', encoding='utf-8') as f:
                f.write("# Initial TOML Configuration\n")
                f.write("filename = \"S10_MAPL1.toml\"\n")
        except IOError as e:
            print(f"Error creating file: {e}")

    app = TOMLApp(INITIAL_FILE)
    app.mainloop()