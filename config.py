import os

# Paths
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
TRAIN_CSV = os.path.join(DATA_DIR, "train_fold.csv")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")

# Data
IMG_SIZE = (576, 864)       # preserve original 2:3 aspect ratio, both multiples of 16
MASK_SIZE = (350, 525)      # Submission required mask size
NUM_CLASSES = 4
LABELS = ['Fish', 'Flower', 'Gravel', 'Sugar']

# Training
BATCH_SIZE = 4
NUM_EPOCHS = 50
LR = 2e-4
WEIGHT_DECAY = 0.01
WARMUP_STEPS = 1000
EMA_DECAY = 0.9999
ACCUMULATION_STEPS = 2      # effective batch size = 4 * 2 = 8

# Loss weights (following 1st place: 0.75 BCE + 0.25 Dice)
BCE_WEIGHT = 0.75
DICE_WEIGHT = 0.25
CLS_WEIGHT = 0.1
IGNORE_NEGATIVE = True      # only compute seg loss on classes with clouds

# Post-processing
CLS_THRESHOLD = 0.6         # grid search optimal on val set
PIXEL_THRESHOLD = 0.4

# Fold
IDX_FOLD = 0
NUM_FOLDS = 5

# Model - DINOv3 SAT backbone
DINO_MODEL = "facebook/dinov3-vitl16-pretrain-sat493m"
DINO_FEATURE_DIM = 1024     # DINOv3-Large hidden dim
DINO_LAYERS = [6, 12, 18, 24]  # layers to extract features from (1-indexed)
BACKBONE_UNFREEZE_LAYERS = 24  # unfreeze all transformer blocks (full fine-tuning)
BACKBONE_LR = 1e-5             # lower lr to protect shallow features
DECODER_DIM = 256

# Workers
NUM_WORKERS = 8
