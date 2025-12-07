#!/bin/bash

###############################################################################
# SafeBoundary AI - Complete Pipeline
# One script to run the entire project from start to finish
###############################################################################

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
VIDEO_URL="https://youtu.be/TUKr2C5E8jA"
DATA_DIR="data"
RAW_DIR="$DATA_DIR/raw"
FRAMES_DIR="$DATA_DIR/frames"
ANNOTATIONS_DIR="$DATA_DIR/annotations"
PSEUDO_LABELS_DIR="$DATA_DIR/pseudo_labels"
MODELS_DIR="models"
CHECKPOINTS_DIR="$MODELS_DIR/checkpoints"
OUTPUT_DIR="outputs"
SAM_MODEL="$MODELS_DIR/sam/sam_vit_h_4b8939.pth"

# Training parameters
NUM_FRAMES=200
EPOCHS_INITIAL=50
EPOCHS_FINETUNE=30
BATCH_SIZE=8

###############################################################################
# Helper Functions
###############################################################################

print_header() {
    echo -e "${BLUE}"
    echo "============================================================================"
    echo "  $1"
    echo "============================================================================"
    echo -e "${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

check_command() {
    if ! command -v $1 &> /dev/null; then
        print_error "$1 is not installed"
        exit 1
    fi
}

###############################################################################
# Pre-flight Checks
###############################################################################

preflight_checks() {
    print_header "Pre-flight Checks"
    
    # Check Python
    check_command python3
    print_success "Python 3 found"
    
    # Check pip
    check_command pip
    print_success "pip found"
    
    # Check ffmpeg
    if ! command -v ffmpeg &> /dev/null; then
        print_warning "ffmpeg not found, installing..."
        brew install ffmpeg
    fi
    print_success "ffmpeg ready"
    
    # Check PyTorch
    python3 -c "import torch" 2>/dev/null || {
        print_error "PyTorch not installed"
        exit 1
    }
    print_success "PyTorch installed"
    
    # Check MPS (Mac GPU)
    if python3 -c "import torch; assert torch.backends.mps.is_available()" 2>/dev/null; then
        print_success "Apple Silicon GPU (MPS) available"
        DEVICE="mps"
    else
        print_warning "MPS not available, using CPU"
        DEVICE="cpu"
    fi
    
    echo ""
}

###############################################################################
# Step 1: Download and Extract Frames
###############################################################################

download_and_extract() {
    print_header "Step 1: Download Video and Extract Frames"
    
    # Create directories
    mkdir -p $RAW_DIR $FRAMES_DIR
    
    # Download video
    if [ ! -f "$RAW_DIR/surgery_video.mp4" ]; then
        echo "Downloading video..."
        python3 src/data/download_video.py \
            --url "$VIDEO_URL" \
            --output "$RAW_DIR/surgery_video.mp4"
        print_success "Video downloaded"
    else
        print_warning "Video already exists, skipping download"
    fi
    
    # Extract frames
    if [ -z "$(ls -A $FRAMES_DIR)" ]; then
        echo "Extracting frames..."
        python3 src/data/extract_frames.py \
            --video "$RAW_DIR/surgery_video.mp4" \
            --output "$FRAMES_DIR" \
            --num_frames $NUM_FRAMES \
            --quality_threshold 0.3 \
            --visualize
        print_success "Frames extracted: $NUM_FRAMES frames"
    else
        print_warning "Frames already exist, skipping extraction"
    fi
    
    echo ""
}

###############################################################################
# Step 2: Annotate Frames
###############################################################################

annotate_frames() {
    print_header "Step 2: Semi-Automatic Annotation with SAM"
    
    # Check if SAM model exists
    if [ ! -f "$SAM_MODEL" ]; then
        print_warning "SAM model not found, downloading..."
        mkdir -p $(dirname "$SAM_MODEL")
        curl -L "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth" \
            -o "$SAM_MODEL"
        print_success "SAM model downloaded"
    fi
    
    # Run annotation
    if [ -z "$(ls -A $ANNOTATIONS_DIR 2>/dev/null)" ]; then
        echo "Starting semi-automatic annotation..."
        echo "This will take 2-4 hours (150 frames to review)"
        echo ""
        echo "Instructions:"
        echo "  Y = Accept annotation"
        echo "  N = Reject (skip frame)"
        echo "  Q = Quit"
        echo ""
        
        python3 src/utils/annotation.py \
            --frames "$FRAMES_DIR" \
            --output "$ANNOTATIONS_DIR" \
            --sam_checkpoint "$SAM_MODEL" \
            --device "$DEVICE"
        
        print_success "Annotation complete"
    else
        print_warning "Annotations already exist, skipping"
    fi
    
    echo ""
}

###############################################################################
# Step 3: Initial Training
###############################################################################

train_initial() {
    print_header "Step 3: Initial Model Training"
    
    echo "Training U-Net++ model for $EPOCHS_INITIAL epochs..."
    echo "Batch size: $BATCH_SIZE"
    echo "Device: $DEVICE"
    echo ""
    
    python3 src/training/train.py \
        --data_dir "$ANNOTATIONS_DIR" \
        --output_dir "$CHECKPOINTS_DIR" \
        --epochs $EPOCHS_INITIAL \
        --batch_size $BATCH_SIZE \
        --device "$DEVICE" \
        --save_freq 10 \
        --val_split 0.15
    
    print_success "Initial training complete"
    echo ""
}

###############################################################################
# Step 4: Generate Pseudo-Labels
###############################################################################

generate_pseudo_labels() {
    print_header "Step 4: Generate Pseudo-Labels"
    
    echo "Generating pseudo-labels for unlabeled frames..."
    echo "This expands the training set without manual annotation"
    echo ""
    
    mkdir -p "$PSEUDO_LABELS_DIR"
    
    python3 src/training/pseudo_label.py \
        --model "$CHECKPOINTS_DIR/best_model.pth" \
        --unlabeled_frames "$FRAMES_DIR" \
        --labeled_frames "$ANNOTATIONS_DIR" \
        --output "$PSEUDO_LABELS_DIR" \
        --confidence_threshold 0.85 \
        --device "$DEVICE"
    
    print_success "Pseudo-labels generated"
    echo ""
}

###############################################################################
# Step 5: Fine-tune with Pseudo-Labels
###############################################################################

finetune_model() {
    print_header "Step 5: Fine-tune with Pseudo-Labels"
    
    echo "Fine-tuning model with expanded dataset..."
    echo ""
    
    python3 src/training/train.py \
        --data_dir "$ANNOTATIONS_DIR" \
        --pseudo_labels "$PSEUDO_LABELS_DIR" \
        --output_dir "$CHECKPOINTS_DIR" \
        --epochs $EPOCHS_FINETUNE \
        --batch_size $BATCH_SIZE \
        --device "$DEVICE" \
        --resume "$CHECKPOINTS_DIR/best_model.pth" \
        --learning_rate 0.00005 \
        --save_freq 5
    
    print_success "Fine-tuning complete"
    echo ""
}

###############################################################################
# Step 6: Run Inference on Full Video
###############################################################################

run_inference() {
    print_header "Step 6: Generate Demo Video"
    
    mkdir -p "$OUTPUT_DIR/visualizations"
    mkdir -p "$OUTPUT_DIR/predictions"
    
    echo "Processing full video with danger zone visualization..."
    echo ""
    
    python3 src/inference/video_processor.py \
        --video "$RAW_DIR/surgery_video.mp4" \
        --model "$CHECKPOINTS_DIR/best_model.pth" \
        --output "$OUTPUT_DIR/visualizations/safeboundary_demo.mp4" \
        --device "$DEVICE" \
        --show_dashboard \
        --show_metrics \
        --danger_zones
    
    print_success "Demo video created: $OUTPUT_DIR/visualizations/safeboundary_demo.mp4"
    echo ""
}

###############################################################################
# Step 7: Evaluate Model
###############################################################################

evaluate_model() {
    print_header "Step 7: Model Evaluation"
    
    mkdir -p "$OUTPUT_DIR/reports"
    
    echo "Calculating metrics (IoU, Dice, BPE, FPS)..."
    echo ""
    
    python3 src/evaluation/evaluator.py \
        --model "$CHECKPOINTS_DIR/best_model.pth" \
        --test_data "$ANNOTATIONS_DIR" \
        --output "$OUTPUT_DIR/reports" \
        --device "$DEVICE"
    
    print_success "Evaluation complete"
    
    # Display results
    if [ -f "$OUTPUT_DIR/reports/evaluation_report.txt" ]; then
        echo ""
        cat "$OUTPUT_DIR/reports/evaluation_report.txt"
    fi
    
    echo ""
}

###############################################################################
# Step 8: Create Presentation Materials
###############################################################################

create_presentation() {
    print_header "Step 8: Create Presentation Materials"
    
    mkdir -p "$OUTPUT_DIR/presentation"
    
    echo "Generating presentation materials..."
    
    # Create comparison video
    python3 src/utils/create_comparison.py \
        --original "$RAW_DIR/surgery_video.mp4" \
        --processed "$OUTPUT_DIR/visualizations/safeboundary_demo.mp4" \
        --output "$OUTPUT_DIR/presentation/comparison.mp4"
    
    # Create metrics visualization
    python3 src/evaluation/plot_metrics.py \
        --report "$OUTPUT_DIR/reports/evaluation_report.json" \
        --output "$OUTPUT_DIR/presentation/metrics.png"
    
    print_success "Presentation materials ready in $OUTPUT_DIR/presentation/"
    echo ""
}

###############################################################################
# Main Pipeline
###############################################################################

main() {
    echo -e "${GREEN}"
    cat << "EOF"
    ____        __     ____                        __                 
   / __/__  __/ /_   / __ )____  __  ______  ____/ /___ ________  __
  / /_/ _ \/ / __/  / __  / __ \/ / / / __ \/ __  / __ `/ ___/ / / /
 / __/  __/ / /_   / /_/ / /_/ / /_/ / / / / /_/ / /_/ / /  / /_/ / 
/_/  \___/_/\__/  /_____/\____/\__,_/_/ /_/\__,_/\__,_/_/   \__, /  
                                                            /____/   
                 AI - Complete Pipeline                              
EOF
    echo -e "${NC}"
    
    echo "This script will run the complete SafeBoundary AI pipeline"
    echo "Estimated time: 8-10 hours (depending on your Mac)"
    echo ""
    echo "Pipeline steps:"
    echo "  1. Download video and extract frames (30 min)"
    echo "  2. Semi-automatic annotation with SAM (2-4 hours)"
    echo "  3. Initial model training (2-3 hours)"
    echo "  4. Generate pseudo-labels (30 min)"
    echo "  5. Fine-tune model (1-2 hours)"
    echo "  6. Generate demo video (30 min)"
    echo "  7. Evaluate model (15 min)"
    echo "  8. Create presentation materials (15 min)"
    echo ""
    read -p "Continue? (y/n) " -n 1 -r
    echo ""
    
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Aborted."
        exit 1
    fi
    
    # Record start time
    START_TIME=$(date +%s)
    
    # Run pipeline
    preflight_checks
    download_and_extract
    annotate_frames
    train_initial
    generate_pseudo_labels
    finetune_model
    run_inference
    evaluate_model
    create_presentation
    
    # Calculate total time
    END_TIME=$(date +%s)
    TOTAL_TIME=$((END_TIME - START_TIME))
    HOURS=$((TOTAL_TIME / 3600))
    MINUTES=$(((TOTAL_TIME % 3600) / 60))
    
    # Final summary
    print_header "Pipeline Complete! 🎉"
    
    echo "Total time: ${HOURS}h ${MINUTES}m"
    echo ""
    echo "📦 Deliverables:"
    echo "   ✓ Trained model: $CHECKPOINTS_DIR/best_model.pth"
    echo "   ✓ Demo video: $OUTPUT_DIR/visualizations/safeboundary_demo.mp4"
    echo "   ✓ Evaluation report: $OUTPUT_DIR/reports/evaluation_report.txt"
    echo "   ✓ Presentation materials: $OUTPUT_DIR/presentation/"
    echo ""
    echo "🎬 Next steps:"
    echo "   1. Watch demo video: open $OUTPUT_DIR/visualizations/safeboundary_demo.mp4"
    echo "   2. Review metrics: cat $OUTPUT_DIR/reports/evaluation_report.txt"
    echo "   3. Prepare presentation using materials in $OUTPUT_DIR/presentation/"
    echo ""
    echo "🏆 You're ready for the hackathon!"
    echo ""
}

# Run main function
main