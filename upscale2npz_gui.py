import os
import cv2
import numpy as np
from PIL import Image
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from cv2 import dnn_superres
from threading import Thread

# --- FIXED SETTINGS ---
BATCH_SIZE = 10  # Number of TIFF files to process in each batch
SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "EDSR_x4.pb")  # Neural network model for 4x upscaling

class TiffProcessorApp:
    """
    GUI application for batch processing TIFF depth images.
    Converts and upscales depth images, then saves them as NPZ files in batches.
    """
    def __init__(self, root):
        """Initialize the application with the main window"""
        self.root = root
        self.root.title("TIFF Depth Batch Processor")
        self.root.geometry("690x590")
        self.root.minsize(690, 590)  # Set minimum window size
        self.root.resizable(True, True)
        
        # Initialize directory paths and state variables
        self.input_dir = ""
        self.output_dir = ""
        self.processing = False
        self.output_manually_set = False  # Track if user manually set output directory
        
        self.create_widgets()
        
    def create_widgets(self):
        """Create and layout all GUI elements"""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Title
        title_label = ttk.Label(main_frame, text="TIFF Depth Batch → NPZ Converter", font=("Arial", 16, "bold"))
        title_label.pack(pady=(0, 20))
        
        # Input directory selection
        input_frame = ttk.Frame(main_frame)
        input_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(input_frame, text="Input Directory:").pack(side=tk.LEFT, padx=(0, 10))
        self.input_path_var = tk.StringVar()
        ttk.Entry(input_frame, textvariable=self.input_path_var, width=40).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(input_frame, text="Browse", command=self.select_input_dir).pack(side=tk.LEFT, padx=(10, 0))
        
        # Output directory selection
        output_frame = ttk.Frame(main_frame)
        output_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(output_frame, text="Output Directory:").pack(side=tk.LEFT, padx=(0, 10))
        self.output_path_var = tk.StringVar()
        ttk.Entry(output_frame, textvariable=self.output_path_var, width=40).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(output_frame, text="Browse", command=self.select_output_dir).pack(side=tk.LEFT, padx=(10, 0))
        
        # Upscaling method selection
        method_frame = ttk.Frame(main_frame)
        method_frame.pack(fill=tk.X, pady=10)
        
        ttk.Label(method_frame, text="Upscaling Method:").pack(side=tk.LEFT, padx=(0, 10))
        self.method_var = tk.StringVar(value="cv2")
        method_combo = ttk.Combobox(method_frame, textvariable=self.method_var, values=["cv2", "edsr"], state="readonly", width=10)
        method_combo.pack(side=tk.LEFT)
        # Track method changes to update output directory accordingly
        self.method_var.trace_add("write", self.on_method_change)
        
        # Info frame for processing information and progress
        info_frame = ttk.LabelFrame(main_frame, text="Processing Information", padding=10)
        info_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # File count display
        file_count_frame = ttk.Frame(info_frame)
        file_count_frame.pack(fill=tk.X, pady=5)
        ttk.Label(file_count_frame, text="Files Found:").pack(side=tk.LEFT)
        self.file_count_var = tk.StringVar(value="0")
        ttk.Label(file_count_frame, textvariable=self.file_count_var).pack(side=tk.LEFT, padx=(5, 0))
        
        # Batch size display
        batch_count_frame = ttk.Frame(info_frame)
        batch_count_frame.pack(fill=tk.X, pady=5)
        ttk.Label(batch_count_frame, text="Batch Size:").pack(side=tk.LEFT)
        ttk.Label(batch_count_frame, text=str(BATCH_SIZE)).pack(side=tk.LEFT, padx=(5, 0))
        
        # Overall Progress bar
        ttk.Label(info_frame, text="Overall Progress:").pack(anchor=tk.W, pady=(10, 5))
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(info_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.pack(fill=tk.X, pady=(0, 10))
        
        # Batch Progress bar
        ttk.Label(info_frame, text="Current Batch Progress:").pack(anchor=tk.W, pady=(5, 5))
        self.batch_progress_var = tk.DoubleVar()
        self.batch_progress_bar = ttk.Progressbar(info_frame, variable=self.batch_progress_var, maximum=100)
        self.batch_progress_bar.pack(fill=tk.X, pady=(0, 10))
        
        # Batch info display
        self.batch_info_var = tk.StringVar(value="")
        self.batch_info_label = ttk.Label(info_frame, textvariable=self.batch_info_var, foreground="red")
        self.batch_info_label.pack(fill=tk.X, pady=(0, 10))
        
        # Status message display
        ttk.Label(info_frame, text="Status:").pack(anchor=tk.W)
        self.status_var = tk.StringVar(value="Ready")
        self.status_label = ttk.Label(info_frame, textvariable=self.status_var, wraplength=500)
        self.status_label.pack(fill=tk.X)
        
        # Control buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=(20, 0))
        
        self.start_button = ttk.Button(button_frame, text="Start Processing", command=self.start_processing)
        self.start_button.pack(side=tk.RIGHT, padx=5)
        
        self.cancel_button = ttk.Button(button_frame, text="Cancel", command=self.cancel_processing, state=tk.DISABLED)
        self.cancel_button.pack(side=tk.RIGHT, padx=5)
    
    def select_input_dir(self):
        """Open directory dialog to select input directory with TIFF files"""
        directory = filedialog.askdirectory(title="Select Folder with TIFF Files")
        if directory:
            self.input_dir = directory
            self.input_path_var.set(directory)
            self.update_file_count()
            
            # Reset the manual flag when selecting a new input directory
            self.output_manually_set = False
            
            # Auto-set output directory
            self.update_output_dir()
    
    def select_output_dir(self):
        """Open directory dialog to select output directory for NPZ files"""
        directory = filedialog.askdirectory(title="Select Output Directory")
        if directory:
            self.output_dir = directory
            self.output_path_var.set(directory)
            self.output_manually_set = True  # Mark that user manually set the output
    
    def update_file_count(self):
        """Count and display the number of TIFF files in the input directory"""
        if not self.input_dir:
            return
        
        tiff_files = [f for f in os.listdir(self.input_dir) if f.lower().endswith(('.tiff', '.tif'))]
        self.file_count_var.set(str(len(tiff_files)))
    
    def start_processing(self):
        """Start the batch processing of TIFF files"""
        if not self.input_dir:
            messagebox.showerror("Error", "Please select an input directory.")
            return
            
        if not self.output_dir:
            messagebox.showerror("Error", "Please select an output directory.")
            return
        
        # Get the method and create a method-specific subfolder
        method = self.method_var.get()
        method_subfolder = os.path.join(self.output_dir, method)
        
        # Create output directory with method subfolder if it doesn't exist
        os.makedirs(method_subfolder, exist_ok=True)
        
        # Update the output path to use the method subfolder
        self.method_output_dir = method_subfolder
        
        # Update UI state
        self.processing = True
        self.start_button.config(state=tk.DISABLED)
        self.cancel_button.config(state=tk.NORMAL)
        
        # Start processing in a separate thread to keep UI responsive
        self.process_thread = Thread(target=self.process_files)
        self.process_thread.daemon = True
        self.process_thread.start()
    
    def cancel_processing(self):
        """Cancel the ongoing processing operation"""
        if self.processing:
            self.processing = False
            self.status_var.set("Cancelling...")
    
    def process_files(self):
        """Main processing function that handles batches of TIFF files"""
        method = self.method_var.get()
        tiff_files = sorted([f for f in os.listdir(self.input_dir) if f.lower().endswith(('.tiff', '.tif'))])
        total_files = len(tiff_files)
        
        if total_files == 0:
            self.update_status("No TIFF files found in the selected directory.")
            self.reset_ui()
            return
        
        batch_index = 1
        processed_count = 0
        
        # Process files in batches of BATCH_SIZE
        for i in range(0, total_files, BATCH_SIZE):
            if not self.processing:
                break
                
            batch_files = tiff_files[i:i + BATCH_SIZE]
            depths = []
            batch_start = i + 1
            batch_end = min(i + BATCH_SIZE, total_files)
            self.update_batch_info(f"Preparing Batch {batch_index}: Files {batch_start}-{batch_end} of {total_files}")
            
            # Process each file in the current batch
            for file_idx, file in enumerate(batch_files):
                if not self.processing:
                    break
                    
                self.update_status(f"Processing: {file} ({processed_count + 1}/{total_files})")
                tiff_path = os.path.join(self.input_dir, file)
                
                # Update batch progress bar
                self.update_batch_progress((file_idx / len(batch_files)) * 100)
                
                try:
                    # Upscale the image using selected method
                    upscaled = self.upscale_image(tiff_path, method)
                    depths.append(upscaled)
                except Exception as e:
                    self.update_status(f"Error processing {file}: {str(e)}")
                    continue
                
                # Update overall progress
                processed_count += 1
                self.update_progress(processed_count / total_files * 100)
            
            if not depths or not self.processing:
                continue
            
            # Stack processed images into a batch array
            depths_array = np.stack(depths)
            frames = np.array([os.path.splitext(f)[0] for f in batch_files])
            count = len(batch_files)
            
            # Show saving status
            self.update_batch_info(f"Saving Batch {batch_index}: Creating NPZ file with {count} frames")
            
            # Create NPZ data structure with required fields
            npz_data = {
                "depths": depths_array,
                "poses": np.zeros((count, 6), dtype=np.float32),  # Placeholder poses
                "frames": frames,
                "distances": np.zeros(count, dtype=np.float32),   # Placeholder distances
                "actions": np.zeros(count, dtype=np.int64),       # Placeholder actions
                "victim_dirs": np.zeros((count, 3), dtype=np.float32),  # Placeholder directions
                "split": "test"
            }
            
            # Save the batch as an NPZ file
            output_name = f"batch{batch_index:05d}.npz"
            output_path = os.path.join(self.method_output_dir, output_name)
            np.savez(output_path, **npz_data)
            
            # Show detailed batch save information
            file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
            self.update_batch_info(f"Saved Batch {batch_index}: '{output_name}' ({count} frames, {file_size_mb:.2f} MB)")
            self.update_status(f"Saved: {output_name} with {count} frames")
            
            # Reset batch progress bar to 100% when batch is complete
            self.update_batch_progress(100)
            
            batch_index += 1
        
        # Show final status message
        if self.processing:
            self.update_status(f"Processing complete. {processed_count}/{total_files} files processed.")
            self.update_batch_info(f"All batches completed. Created {batch_index-1} batch files.")
        else:
            self.update_status(f"Processing cancelled. {processed_count}/{total_files} files processed.")
            self.update_batch_info(f"Processing cancelled. Created {batch_index-1} batch files.")
        
        self.reset_ui()
    
    def upscale_image(self, tiff_path, method):
        """
        Upscale a depth image using the selected method (cv2 or EDSR).
        Returns a normalized, cropped, and resized depth image.
        """
        # Read TIFF image and normalize
        depth_img = Image.open(tiff_path)
        depth_np = np.array(depth_img).astype(np.float32)
        depth_norm = (depth_np - depth_np.min()) / np.ptp(depth_np)
        
        if method == 'edsr':
            # Use EDSR neural network for 4x upscaling
            temp_png = os.path.join(self.method_output_dir, 'temp_depth.png')
            cv2.imwrite(temp_png, (depth_norm * 255).astype(np.uint8))
            sr = dnn_superres.DnnSuperResImpl_create()
            sr.readModel(MODEL_PATH)
            sr.setModel("edsr", 4)
            image = cv2.imread(temp_png)
            upscaled = sr.upsample(image)
            upscaled_gray = cv2.cvtColor(upscaled, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
            # Clean up temp file
            if os.path.exists(temp_png):
                os.remove(temp_png)
        else:
            # Use OpenCV cubic interpolation for upscaling
            upscaled_gray = cv2.resize(depth_norm, (1024, 768), interpolation=cv2.INTER_CUBIC)
        
        # Crop the center portion and resize to 512x512
        h, w = upscaled_gray.shape
        cx = w // 2
        cropped = upscaled_gray[:, cx - 256:cx + 256]  # Center crop
        cropped_square = cv2.resize(cropped, (512, 512), interpolation=cv2.INTER_CUBIC)
        return cropped_square.astype(np.float32)
    
    def update_status(self, message):
        """Update the status message in the UI thread-safely"""
        self.root.after(0, lambda: self.status_var.set(message))
    
    def update_progress(self, value):
        """Update the overall progress bar in the UI thread-safely"""
        self.root.after(0, lambda: self.progress_var.set(value))
    
    def update_batch_progress(self, value):
        """Update the batch progress bar in the UI thread-safely"""
        self.root.after(0, lambda: self.batch_progress_var.set(value))
    
    def update_batch_info(self, message):
        """Update the batch info message in the UI thread-safely"""
        self.root.after(0, lambda: self.batch_info_var.set(message))
    
    def reset_ui(self):
        """Reset UI elements to their initial state after processing"""
        self.root.after(0, lambda: self.start_button.config(state=tk.NORMAL))
        self.root.after(0, lambda: self.cancel_button.config(state=tk.DISABLED))
        self.root.after(0, lambda: self.batch_progress_var.set(0))
        self.processing = False
    
    def on_method_change(self, *args):
        """Handler for upscaling method change"""
        # Update output directory when method changes
        if self.input_dir:
            self.update_output_dir()
    
    def update_output_dir(self):
        """Auto-update the output directory based on the input directory"""
        # Don't update if the user has manually set the output directory
        if self.output_manually_set:
            return
            
        # Use a simple "output" folder name without method suffix
        # since we now create method-specific subfolders at processing time
        default_output = os.path.join(self.input_dir, "output")
        self.output_dir = default_output
        self.output_path_var.set(default_output)

# --- MAIN APP ---
if __name__ == "__main__":
    root = tk.Tk()
    app = TiffProcessorApp(root)
    root.mainloop()
