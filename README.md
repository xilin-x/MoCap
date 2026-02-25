# MoCap: Multi-View Motion Capture & 3D Generation

MoCap is a high-performance toolkit for 3D human pose estimation and reconstruction based on the Segment Anything Model (SAM). It is optimized for processing synchronized multi-view exercise videos (Left, Center, and Right angles) to generate accurate 3D motion data and human meshes.

## 1. Key Features

* **Multi-View Fusion:** Seamlessly integrates synchronized video feeds from Left, Center, and Right perspectives for robust 3D spatial reconstruction.
* **SAM-Powered Segmentation:** Leverages SAM variants for sub-pixel precise mask extraction of the human subject, even in complex gym environments.
* **10-Action Optimization:** Pre-tuned for 10 standard fitness movements, ensuring high fidelity during high-intensity or occlusion-prone exercises.
* **End-to-End Pipeline:** Automated workflow from raw multi-view video input to 3D meshes (.ply).

## 2. Project Structure

```text
MoCap/
├── extract_frame.py                    # Utility for extracting frames from multi-view videos
├── detector.py                         # ViTDet-based human detection and segmentation module
├── segmentor.py                        # SAM-based segmentation wrapper for precise mask generation
├── mesh_generate.py                    # Module for generating 3D meshes from segmented masks
├── checkpoints/                        # Model weights directory
│   └── sam-3d-body-dinov3              # Optimized weights for human body detection, segmentation and 3D reconstruction
├── data/                               # Input data directory
│   └── vidoes/
│       └── ArmCircles/
│           ├── ArmCircles_A.mp4        # Left-side view
│           ├── ArmCircles_B.mp4        # Center/Front view
│           └── ArmCircles_C.mp4        # Right-side view
└── README.md
```

## 3. Requirements

- **Python:** 3.11+
- **OS:** Ubuntu 20.04+ (Recommended)
- **Hardware:** NVIDIA GPU with CUDA 11.0+ support
- **Dependencies:** OpenCV, NumPy, PyTorch (standard libraries only for the wrapper)

## 4. Usage

### 4.1 Installation & Setup

1. Clone the repository:
```bash
git clone https://github.com/xilin-x/MoCap.git
cd MoCap
```
2. Environment Setup (Conda recommended):

```bash
# Create and Activate Environment
conda create -n mocap python=3.11 -y
conda activate mocap
# Install PyTorch, Please install PyTorch following the official instructions.
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
# Install Python Dependencies
pip install pytorch-lightning pyrender opencv-python yacs scikit-image einops timm dill pandas rich hydra-core hydra-submitit-launcher hydra-colorlog pyrootutils webdataset chump networkx==3.2.1 roma joblib seaborn wandb appdirs appnope ffmpeg cython jsonlines pytest xtcocotools loguru optree fvcore black pycocotools tensorboard huggingface_hub
# Install Detectron2
pip install 'git+https://github.com/facebookresearch/detectron2.git@a1ce2f9' --no-build-isolation --no-deps
# Install MoGe (Optional)
pip install git+https://github.com/microsoft/MoGe.git
# Install SAM3 (Optional)
# this is a minimal installation of sam3 only to support its inference
git clone https://github.com/facebookresearch/sam3.git
cd sam3
pip install -e .
pip install decord psutil
```

3. Running the Tool

The ```extract_frame.py``` script is used to extract frames from the multi-view videos. It will automatically detect the three view files (```XXX_A.mp4```, ```XXX_B.mp4```, ```XXX_C.mp4```) within the target data directory and save the extracted frames in a structured format.

```bash
python extract_frame.py --data_dir ./data/videos/ArmCircles
```

The ```detector.py``` script is responsible for detecting human subjects in each extracted frame. It uses ViTDet for initial detection and SAM for precise segmentation.

```bash
python detector.py --data_dir ./data/videos/ArmCircles
```

The ```segmentor.py``` script takes the detected masks and generates refined segmentation masks using SAM, which are then used for 3D reconstruction.

```bash
python segmentor.py --data_dir ./data/videos/ArmCircles
```

The ```mesh_generate.py``` script processes the segmented masks to generate 3D meshes (.ply) based on the specified output format.

```bash
python mesh_generate.py --data_dir ./data/videos/ArmCircles
```

## 5. Output

Results are exported to the data/ directory:

- ```frames/``` - Extracted frames from the multi-view videos
- ```bboxes/``` - Detected bounding boxes for each frame
- ```masks/``` - Segmented masks for each frame
- ```meshes/``` - Generated 3D meshes in .ply format
