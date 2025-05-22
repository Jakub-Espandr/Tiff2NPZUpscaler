# Changelog for TIFF to NPZ Depth Image Processor

## Version 0.1.0 - Initial Release

### Features
- Batch processing of TIFF depth images
- Two upscaling methods: OpenCV cubic interpolation and EDSR neural network upscaling
- Automatic center cropping and resizing to 512x512
- Batch output to NPZ files with metadata structure
- Progress tracking with dual progress bars
- Separate thread for processing to maintain responsive UI

### Usage
- Run the GUI application with `python upscale2npz_gui.py`
- Run the terminal application with `python upscale2npz.py`
- Select input and output directories
- Choose upscaling method (`cv2` or `edsr`)
- Monitor progress and find processed NPZ files in the output directory

### Notes
- EDSR model (`EDSR_x4.pb`) must be in the same directory as the script
- Application creates method-specific subfolders in the output directory
- Batch size is fixed at 10 files per batch
- Normalizes depth values before processing