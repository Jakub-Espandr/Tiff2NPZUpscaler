# TIFF to NPZ Depth Image Processor

This application processes TIFF depth images by upscaling them and converting them into NPZ (NumPy compressed) files in batches. It is designed for processing training datasets captured using iPhone LiDAR, offering both standard OpenCV interpolation and EDSR neural network upscaling methods.

## Requirements

- Python 3.7+
- OpenCV with contrib modules (for dnn_superres)
- NumPy
- Pillow (PIL)
- Tifffile

## Quick Start

```bash
# Clone the repository
git clone https://github.com/Jakub-Espandr/tiff2npz-upscaler.git
# Move to the directory
cd tiff2npz-upscaler
# Install dependencies
pip install -r requirements.txt
# Run the GUI application
python upscale2npz_gui.py
# Run the terminal application
python upscale2npz.py
```

## Usage

1. Run the application:
   ```bash
   python upscale2npz_gui.py
   ```

2. Select input directory containing TIFF files
3. Choose output directory (defaults to "output" subfolder in input directory)
4. Select upscaling method:
   - `cv2`: Faster but lower quality
   - `edsr`: Higher quality but slower
5. Click "Start Processing"
6. Monitor progress in the GUI
7. Find processed NPZ files in the output directory, in a method-specific subfolder

## Features

- Batch processing of TIFF depth images
- Two upscaling methods: 
  - `cv2`: Fast OpenCV cubic interpolation
  - `edsr`: High-quality neural network upscaling (4x)
- Automatic center cropping and resizing to 512x512
- Batch output to NPZ files with proper metadata structure
- Progress tracking with dual progress bars
- Separate thread for processing to maintain responsive UI

## How It Works

The application:

1. Loads TIFF depth images from an input directory
2. Processes them in batches (default: 10 images per batch)
3. For each image:
   - Normalizes the depth values
   - Upscales using the selected method
   - Center crops and resizes to 512x512
4. Creates NPZ files with the following structure:
   ```
   {
     "depths": [batch_size, 512, 512] array,
     "poses": placeholder zeros,
     "frames": original filenames,
     "distances": placeholder zeros,
     "actions": placeholder zeros,
     "victim_dirs": placeholder zeros,
     "split": "test"
   }
   ```
5. Organizes output in method-specific subfolders

## Output Format

Each batch is saved as a separate NPZ file with the following naming convention:
```
batch00001.npz, batch00002.npz, ...
```

The NPZ files contain structured data compatible with depth-based machine learning pipelines.

## Sample Data

Sample data is available in the `Test` folder. These data can be visualized using the Depth Image Viewer available in the Tools section of the [disaster-sim-coppeliasim repository](https://github.com/Jakub-Espandr/disaster-sim-coppeliasim).

## Notes

- The EDSR model (`EDSR_x4.pb`) must be in the same directory as the script
- The application automatically creates method-specific subfolders in the output directory
- Batch size is fixed at 10 files per batch
- The application normalizes depth values before processing

## Data Source

The TIFF depth images used in this application are captured using the [LiDAR Depth Map Capture for iOS](https://github.com/ioridev/LiDAR-Depth-Map-Capture-for-iOS) app. This app allows for capturing full-resolution, 32-bit floating-point depth maps using the LiDAR scanner on supported iPhone and iPad models. The depth maps preserve the original precision, making them suitable for high-quality upscaling and conversion processes.