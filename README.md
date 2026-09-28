# MoCap: Multi-View Motion Capture & 3D Generation

Human motion capture pipeline:

```text
Video -> Frames -> BBox -> Mask -> MHR -> Motion -> FBX
```

The pipeline processes each camera view independently. It includes person detection, human segmentation, MHR pose estimation, mesh visualization, motion construction, and FBX export.

## 1. Key Features

- **End-to-End Pipeline:** Converts raw videos into frames, masks, 3D meshes, motion data, and FBX files.
- **Multiple Prompting Modes:** Supports bounding-box prompts and the text prompt `person` for mask extraction.
- **MHR Reconstruction:** Extracts body parameters, joint coordinates, rotations, and mesh geometry.
- **Resume Support:** Long-running stages can continue from existing frame, mask, mesh, and visualization outputs.
- **Blender Integration:** Exports the reconstructed motion to a template rig and can render FBX files to video.
- **Shared Utilities:** Common traversal, progress, and video-writing code is kept in `src/`.

## 2. Project Structure

```text
MoCap/
├── *.py                  # Runnable pipeline and utility scripts
├── run_all.sh            # Full pipeline entry point
├── src/
│   ├── script_utils.py   # Directory, path, and progress utilities
│   └── video_utils.py    # Shared video writer
├── archive/              # One-off test scripts
├── lib/                  # External model repositories and MHR assets
├── pyproject.toml        # Python and uv configuration
├── uv.lock
├── LICENSE
└── README.md
```

## 3. Requirements

The project uses Python 3.12, PyTorch 2.10.0, and CUDA 12.8.

```bash
uv python install 3.12
uv python pin 3.12

mkdir -p lib
git clone https://github.com/facebookresearch/sam-3d-body.git lib/body_model
git clone https://github.com/facebookresearch/detectron2.git lib/detectron2
git -C lib/detectron2 checkout a1ce2f9
git clone https://github.com/microsoft/MoGe.git lib/moge
git -C lib/moge checkout 07444410f1e33f402353b99d6ccd26bd31e469e8
git clone https://github.com/facebookresearch/sam3.git lib/sam3
uv sync --python 3.12

mkdir -p lib/mhr
curl -L https://github.com/facebookresearch/MHR/releases/download/v1.0.1/assets.zip -o /tmp/mhr_assets.zip
unzip -q /tmp/mhr_assets.zip -d lib/mhr
rm /tmp/mhr_assets.zip
```

The MHR Python package is not installed. Only the model assets in `lib/mhr/assets` are used.

If Detectron2 cannot detect the GPU during compilation, configure the CUDA toolkit and GPU architecture first:

```bash
export CUDA_HOME=<path_to_cuda>/cuda-12.8
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"
export TORCH_CUDA_ARCH_LIST=<your_gpu_compute_capability>
export FORCE_CUDA=1
CUDA_VISIBLE_DEVICES="" uv sync --python 3.12
unset FORCE_CUDA TORCH_CUDA_ARCH_LIST
```

## 4. Usage

### 4.1 Input Layout

`extract_frames.py` reads `.MP4` files organized by sequence:

```text
videos/
├── Arm Circles/
│   ├── Arm Circles (A).MP4
│   └── Arm Circles (B).MP4
└── Cossack Squats/
    └── Cossack Squats (A).MP4
```

Each video is rotated 90 degrees and exported as `frame_XXXX.jpg`. Later stages use the same `sequence/video/frame` directory structure.

### 4.2 Full Pipeline

Update `DATA`, `GPU`, and `TEMPLATE` in [run_all.sh](run_all.sh), then run the full pipeline:

```bash
bash run_all.sh
```

The full pipeline runs these stages in order:

1. Extract video frames
2. Extract human bounding boxes
3. Check for missing bounding boxes
4. Extract human masks with a box prompt
5. Extract MHR parameters and body meshes from masks
6. Generate mesh visualization images
7. Create a video from the visualization images
8. Build motion and skeleton files
9. Export an FBX file with Blender

Most stages support `--resume`, allowing interrupted jobs to continue from existing outputs.

### 4.3 Step-by-step

Set the dataset root before running the commands below:

```bash
export DATA=/path/to/mocap_data
```

#### 1. Extract frames

```bash
uv run python extract_frames.py \
    -i $DATA/videos \
    -o $DATA/results/frames \
    -w 8 \
    --resume
```

#### 2. Extract bounding boxes

```bash
uv run python extract_bbox.py \
    -i $DATA/results/frames \
    -o $DATA/results/bboxes \
    -v all \
    --resume
```

Replace `-v` with one or more video names to process selected videos. Omit it or use `all` to process every video.

Check for missing results:

```bash
uv run python check_bboxes.py \
    -i $DATA/results/frames \
    -b $DATA/results/bboxes \
    -v all
```

#### 3. Extract masks

The default mode uses a bounding box prompt:

```bash
uv run python extract_masks.py \
    -i $DATA/results/frames \
    -b $DATA/results/bboxes \
    -o $DATA/results/masks \
    -v all \
    --resume
```

You can also use the text prompt `person` directly:

```bash
uv run python extract_masks_text.py \
    -i $DATA/results/frames \
    -o $DATA/results/masks \
    -v all
```

#### 4. Extract MHR data and meshes

```bash
uv run python extract_meshes.py \
    -i $DATA/results/frames \
    -m $DATA/results/masks \
    -o $DATA/results/meshes \
    -v all \
    --resume
```

Each frame produces MHR parameters, a mesh, and a 2D visualization:

```text
meshes/<sequence>/<video>/
├── mhr_params/frame_0000.npz
├── meshes/frame_0000_mesh_000.ply
└── visualizations/frame_0000.jpg
```

#### 5. Visualize meshes

```bash
uv run python visualize_meshes.py \
    -i $DATA/results/frames \
    -m $DATA/results/meshes \
    -o $DATA/results/mesh_visualizations \
    -v all \
    --resume
```

Use `--face-step` to reduce triangle sampling density. `--elev` and `--azim` control the visualization camera.

#### 6. Build motion

```bash
uv run python build_motion.py \
    -i "$DATA/results/meshes/Arm Circles/Arm Circles (A)/mhr_params" \
    --fps 30
```

Outputs are written to the parent directory of `mhr_params`:

```text
meshes/<sequence>/<video>/
├── motion.npz
├── skeleton.npz
├── skeleton_motion.mp4
└── skeleton_visualizations/frame_0000.jpg
```

`--fps` should match the source video frame rate. MHR assets default to `lib/mhr/assets` and can be changed with `--assets`.

#### 7. Export FBX

Use a Monty FBX as the template and run Blender in background mode:

```bash
blender -b --python export_fbx_blender.py -- \
    --template "/path/to/TPose.fbx" \
    --motion "$DATA/results/meshes/Arm Circles/Arm Circles (A)/motion.npz" \
    --skeleton "$DATA/results/meshes/Arm Circles/Arm Circles (A)/skeleton.npz" \
    --output "$DATA/results/meshes/Arm Circles/Arm Circles (A)/Arm_Circles.fbx"
```

The exported rig, mesh, skin, and bone hierarchy come from the template. The `Monty_` prefix is removed from bone names; for example, `Monty_Hips` becomes `Hips`.

### 4.4 Optional Tools

Create a video from mesh visualization images:

```bash
uv run python images_to_video.py \
    -i $DATA/results/mesh_visualizations \
    -o $DATA/results/mesh_videos \
    --fps 30 \
    --resume
```

Create a video from a flat directory of image frames:

```bash
uv run python frames_to_video.py \
    --input /path/to/frame_directory \
    --output /path/to/output.mp4 \
    --fps 30
```

Render an FBX to video with Blender:

```bash
blender -b --python fbx_to_video.py -- \
    --input "/path/to/output.fbx" \
    --output "/path/to/output.mp4" \
    --fps 30 \
    --view front \
    --keep-frames
```

Use `inspect_fbx_bones.py` to list the FBX bone hierarchy and directions:

```bash
blender -b --python inspect_fbx_bones.py -- "/path/to/output.fbx"
```

## 5. Output

The full pipeline produces an output structure similar to this:

```text
results/
├── frames/<sequence>/<video>/frame_0000.jpg
├── bboxes/<sequence>/<video>/frame_0000.npy
├── masks/<sequence>/<video>/frame_0000.png
├── meshes/<sequence>/<video>/
│   ├── mhr_params/frame_0000.npz
│   ├── meshes/frame_0000_mesh_000.ply
│   ├── visualizations/frame_0000.jpg
│   ├── motion.npz
│   ├── skeleton.npz
│   └── skeleton_motion.mp4
├── mesh_visualizations/<sequence>/<video>/frame_0000.jpg
└── mesh_videos/<sequence>/<video>.mp4
```
