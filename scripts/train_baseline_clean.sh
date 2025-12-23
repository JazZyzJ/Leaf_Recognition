#!/bin/bash

# Configuration
CONFIGS=("configs/effnet_b4.yaml" "configs/resnet50d_baseline.yaml" "configs/resnet200d.yaml")
TRAIN_CLEAN_CSV="train_clean.csv"

# Check if train_clean.csv exists
if [ ! -f "$TRAIN_CLEAN_CSV" ]; then
    echo "Error: $TRAIN_CLEAN_CSV not found. Please run src/find_label_errors.py first."
    exit 1
fi

echo "Starting training on clean data..."

for CONFIG in "${CONFIGS[@]}"; do
    if [ ! -f "$CONFIG" ]; then
        echo "Warning: $CONFIG not found, skipping..."
        continue
    fi

    # Create a clean version of the config
    BASE_NAME=$(basename "$CONFIG" .yaml)
    CLEAN_CONFIG="configs/${BASE_NAME}_clean.yaml"
    
    echo "Creating clean config: $CLEAN_CONFIG"
    
    # Replace train_csv and update experiment name
    # We use a more specific regex to avoid matching 'model_name'
    cat "$CONFIG" | \
    sed "s/^  name: .*/  name: ${BASE_NAME}_clean/" | \
    sed "s/train_csv: .*/train_csv: $TRAIN_CLEAN_CSV/" > "$CLEAN_CONFIG"


    echo "Running training for $CLEAN_CONFIG..."
    python src/train.py --config "$CLEAN_CONFIG"
    
    if [ $? -eq 0 ]; then
        echo "Successfully finished training for $BASE_NAME"
    else
        echo "Error: Training failed for $BASE_NAME"
        exit 1
    fi
done

echo "All baseline clean training runs completed!"
