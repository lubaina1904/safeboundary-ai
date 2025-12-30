<div align="center">
  <img src="assets/images/banner.png" alt="SafeBoundary AI Banner" width="100%">
</div>

## SafeBoundary AI

Real-time bladder segmentation with multi-level danger-zone visualization, built on a U-Net++ backbone. The project includes training, evaluation, and inference utilities plus end-to-end automation for dataset creation and reporting.

### 🎯 Highlights
- **U-Net++ Architecture** with boundary-aware losses for robust bladder masks
- **Cross-Platform Support**: Apple Silicon (MPS), CUDA, and CPU
- **Real-time Visualization**: Multi-level danger zones with FPS/BPE metrics
- **Complete Pipeline**: Automated workflow from data to deployment
- **Professional Output**: Comparison videos and evaluation plots

### 📊 Example Output

<div align="center">
  <img src="assets/images/example.png" alt="Example Output" width="90%">
  <p><i>Left: Original surgical video | Right: AI-powered boundary detection with danger zones</i></p>
</div>

### 🏗️ Architecture

<div align="center">
  <img src="assets/images/architecture.png" alt="System Architecture" width="85%">
</div>

### 🔄 Complete Pipeline Workflow

<div align="center">
  <img src="assets/images/workflow.png" alt="Pipeline Workflow" width="95%">
</div>

### 📁 Project Layout
- `scripts/run_complete_pipeline.sh` – full pipeline: download → extract → annotate → train → evaluate.
- `src/data/` – dataset creation (`extract_frames.py`), PyTorch dataset/augmentations (`dataset.py`).
- `src/models/` – U-Net++ (`unet_plus.py`), loss functions (`losses.py`).
- `src/training/` – training loop (`train.py`), pseudo-labeling utilities.
- `src/inference/` – video processor with danger-zone visualizer (`video_processor.py`).
- `src/evaluation/` – quick visual eval (`evaluate_model.py`), plotting (`plot_metrics.py`).
- `src/utils/` – annotation tool (SAM-assisted), side-by-side `create_comparison.py`.
- `outputs/` – generated videos/reports (ignored in git).

### ⚙️ Setup
Prereqs: Python 3.10+, ffmpeg, Git, and a PyTorch build for your device (MPS/CUDA/CPU).

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
# Install torch for your platform; example for CPU:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install opencv-python albumentations scipy tqdm numpy matplotlib seaborn scikit-learn pyyaml
```

### 📦 Data
- Training expects `data/annotations/images/*.jpg|png` with matching `data/annotations/masks/<name>_mask.png`.
- Use `src/data/extract_frames.py` to sample frames from a video:
  ```bash
  python -m src.data.extract_frames --video data/raw/surgery_video.mp4 --output data/frames --num_frames 200 --quality_threshold 0.3 --visualize
  ```
- Semi-automatic annotation with SAM:
  ```bash
  python -m src.utils.annotation --frames data/frames --output data/annotations --sam_checkpoint models/sam/sam_vit_h_4b8939.pth --device mps
  ```

### 🚀 Train
```bash
python -m src.training.train \
  --data_dir data/annotations \
  --output_dir models/checkpoints \
  --epochs 25 \
  --batch_size 4 \
  --image_size 512 \
  --device cuda  # or mps/cpu
```
Checkpoints (including `best_model.pth`) land in `models/checkpoints/`.

### 📈 Evaluate & Visualize
- Quick visual eval and Dice stats (saves to `outputs/evaluation/`):
  ```bash
  python -m src.evaluation.evaluate_model --model models/checkpoints/best_model.pth --data_dir data/annotations --num_samples 5 --device mps
  ```
- Plot training curves:
  ```bash
  python -m src.evaluation.plot_metrics --log_dir models/checkpoints/logs --output outputs/reports/metrics.png
  ```
- Side-by-side comparison video:
  ```bash
  python -m src.utils.create_comparison --original data/raw/surgery_video.mp4 --processed outputs/visualizations/demo_final.mp4 --output outputs/visualizations/comparison.mp4
  ```

### 🎥 Inference on Video
```bash
python -m src.inference.video_processor \
  --video data/raw/surgery_video.mp4 \
  --model models/checkpoints/best_model.pth \
  --output outputs/visualizations/demo_overlaid.mp4 \
  --device mps \
  --show_dashboard --show_metrics
```
Produces an overlaid video with danger zones, boundary tracing, FPS, and risk indicator.

### 🔧 Full Pipeline (Optional)
The helper script wires everything together (download → extract → annotate → train → eval):
```bash
bash scripts/run_complete_pipeline.sh
```
Review the script to adjust URLs, counts, and hyperparameters before running.

### 📝 Notes
- Large artifacts, videos, and checkpoints are git-ignored by default.
- For best performance on Apple Silicon, use `--device mps`; otherwise choose `cuda` or `cpu`.
- Keep outputs under `outputs/` and `models/` to avoid polluting git history.

