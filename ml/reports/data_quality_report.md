# SwarmAgri AI - Data Quality Report

## Executive Summary
- **Overall Quality Status**: **PASSED**
- **Total Matched Pairs**: 1,300
- **Corrupted Image Files**: 0
- **Orphaned Images**: 0
- **Orphaned Labels**: 0
- **Total Bounding Boxes**: 2,072
- **Valid Bounding Boxes**: 2,072 (100.0%)
- **Invalid Bounding Boxes**: 0

## Image Integrity
- Every JPEG file was decoded using Pillow raster verification.
- Zero decoding failures, zero truncations, zero missing headers.
- All 1,300 images decoded into standard RGB raster arrays.

## Annotation Integrity
- Every YOLO row was validated for exactly 5 tokens, valid integer class IDs, finite normalized coordinates in [0, 1], and positive width/height.
- All bounding box coordinates remain within valid image bounds (maximum floating point excess: 5.0e-7, originating from 6-digit decimal quantization).
- Coordinates clamped strictly to `[0.0, 1.0]` upon export to `data/processed/yolo_dataset/`.

## Duplicate & Near-Duplicate Analysis
- **Exact File Byte Duplicates**: 0 groups
- **Exact Pixel Buffer Duplicates**: 0 groups
- **Identical Perceptual dHash Groups**: 95 groups
- **Potential Near-Duplicates (Hamming dist <= 2)**: 494 pairs
- **Leakage Prevention**: All duplicates and connected components were clustered and assigned atomically to identical splits. Cross-split duplicate leakage is **0.0%**.
