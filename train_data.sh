#!/bin/bash

BASE_YAML="options/train/InRetouch_Optimize_Single2.yml"
TEMP_YAML="options/train/temp.yml"
#change this path
DATA_ROOT="/NTIRE2026/C11_RetouchTransfer/testing_data/Subjective_Evaluation_Data"

for SAMPLE_DIR in $DATA_ROOT/sample*; do

    SAMPLE=$(basename $SAMPLE_DIR)

    BEFORE="$SAMPLE_DIR/${SAMPLE}_before.jpg"
    AFTER="$SAMPLE_DIR/${SAMPLE}_after.jpg"
    INPUT="$SAMPLE_DIR/${SAMPLE}_input.jpg"

    EXP_NAME="InRetouch_${SAMPLE}"

    echo "Processing $SAMPLE"

    cp $BASE_YAML $TEMP_YAML

    # change experiment name
    sed -i "s|name:.*|name: ${EXP_NAME}|g" $TEMP_YAML

    # change paths
    sed -i "s|style_natural:.*|style_natural: ['$BEFORE']|g" $TEMP_YAML
    sed -i "s|style_output:.*|style_output: ['$AFTER']|g" $TEMP_YAML
    sed -i "s|inp_natural:.*|inp_natural: '$INPUT'|g" $TEMP_YAML

    python -m basicsr.train_INR -opt $TEMP_YAML

done
