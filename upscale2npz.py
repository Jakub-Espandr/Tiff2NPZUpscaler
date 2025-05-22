import os
import cv2
import numpy as np
from PIL import Image
from tkinter import filedialog, Tk, StringVar, OptionMenu, Button, Label
from cv2 import dnn_superres

# --- FIXED SETTINGS ---
# Number of TIFF files to process in each batch
BATCH_SIZE = 10
# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
# Path to the EDSR super-resolution model file
MODEL_PATH = os.path.join(SCRIPT_DIR, "EDSR_x4.pb")  # model in same dir as script

# --- STEP 1: GUI METHOD SELECTION ---
def start_processing():
    """
    Main processing function that runs after user selects an upscaling method.
    Handles folder selection, batch processing, and saving output files.
    """
    # Get the selected upscaling method from the GUI
    method = method_var.get()
    # Close the GUI window
    root.destroy()

    # Folder picker
    Tk().withdraw()  # Hide the main Tkinter window
    input_dir = filedialog.askdirectory(title="Select Folder with TIFF Files")
    if not input_dir:
        print("❌ No folder selected.")
        return

    # Create output folder named output_<method> inside the input directory
    output_dir = os.path.join(input_dir, f"output_{method}")
    os.makedirs(output_dir, exist_ok=True)

    def upscale_image(tiff_path):
        """
        Upscale a TIFF depth image using the selected method.
        
        Args:
            tiff_path: Path to the TIFF file
            
        Returns:
            A 512x512 cropped, upscaled depth image as a float32 numpy array
        """
        # Open the TIFF file and convert to numpy array
        depth_img = Image.open(tiff_path)
        depth_np = np.array(depth_img).astype(np.float32)
        # Normalize depth values to 0-1 range
        depth_norm = (depth_np - depth_np.min()) / np.ptp(depth_np)

        if method == 'edsr':
            # Using EDSR super-resolution model
            # Save normalized image as temporary PNG
            temp_png = 'temp_depth.png'
            cv2.imwrite(temp_png, (depth_norm * 255).astype(np.uint8))
            # Initialize super-resolution model
            sr = dnn_superres.DnnSuperResImpl_create()
            sr.readModel(MODEL_PATH)
            sr.setModel("edsr", 4)  # EDSR model with 4x upscaling
            # Read the temporary image and upscale it
            image = cv2.imread(temp_png)
            upscaled = sr.upsample(image)
            # Convert to grayscale and normalize
            upscaled_gray = cv2.cvtColor(upscaled, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        else:
            # Using CV2 cubic interpolation method
            upscaled_gray = cv2.resize(depth_norm, (1024, 768), interpolation=cv2.INTER_CUBIC)

        # Extract dimensions of upscaled image
        h, w = upscaled_gray.shape
        # Calculate center x-coordinate
        cx = w // 2
        # Crop 512 pixels centered horizontally
        cropped = upscaled_gray[:, cx - 256:cx + 256]
        # Resize to final 512x512 square
        cropped_square = cv2.resize(cropped, (512, 512), interpolation=cv2.INTER_CUBIC)
        return cropped_square.astype(np.float32)

    # --- Batch processing loop ---
    # Get all TIFF files from input directory and sort them
    tiff_files = sorted([f for f in os.listdir(input_dir) if f.lower().endswith('.tiff')])
    batch_index = 1

    # Process files in batches of BATCH_SIZE
    for i in range(0, len(tiff_files), BATCH_SIZE):
        batch_files = tiff_files[i:i + BATCH_SIZE]
        depths = []

        # Process each file in the current batch
        for file in batch_files:
            print(f"🔄 Processing: {file}")
            tiff_path = os.path.join(input_dir, file)
            try:
                upscaled = upscale_image(tiff_path)
                depths.append(upscaled)
            except Exception as e:
                print(f"⚠️ Skipping {file}: {e}")
                continue

        # Skip empty batches
        if not depths:
            continue

        # Stack processed images into a 3D array
        depths_array = np.stack(depths)
        # Extract frame names without extensions
        frames = np.array([os.path.splitext(f)[0] for f in batch_files])
        count = len(batch_files)

        # Create NPZ data structure with required fields
        npz_data = {
            "depths": depths_array,            # Processed depth images
            "poses": np.zeros((count, 6), dtype=np.float32),  # Placeholder pose data
            "frames": frames,                  # Frame identifiers
            "distances": np.zeros(count, dtype=np.float32),   # Placeholder distances
            "actions": np.zeros(count, dtype=np.int64),       # Placeholder actions
            "victim_dirs": np.zeros((count, 3), dtype=np.float32),  # Placeholder victim directions
            "split": "test"                    # Dataset split identifier
        }

        # Save batch as NPZ file with sequential numbering
        output_name = f"batch{batch_index:05d}.npz"
        output_path = os.path.join(output_dir, output_name)
        np.savez(output_path, **npz_data)
        print(f"✅ Saved: {output_path}")
        batch_index += 1

# --- GUI WINDOW SETUP ---
root = Tk()
root.title("Select Upscaling Method")

# Create a variable to store the selected method, with default value "cv2"
method_var = StringVar(value="cv2")

# Add GUI components
Label(root, text="Choose upscaling method:").pack(pady=10)
OptionMenu(root, method_var, "cv2", "edsr").pack()
Button(root, text="Continue", command=start_processing).pack(pady=20)

# Start the GUI event loop
root.mainloop()
