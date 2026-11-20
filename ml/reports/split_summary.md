# SwarmAgri AI - Split Summary & Leakage Prevention Report

## Split Configuration & Results
- **Random Seed**: `42` (100% deterministic & reproducible)
- **Target Ratios**: 80% Train / 10% Validation / 10% Test
- **Train Images**: 1,042 (80.15%)
- **Validation Images**: 129 (9.92%)
- **Test Images**: 129 (9.92%)
- **Total Images**: 1,300

## Data Leakage Prevention
- **Cluster Isolation**: Exact duplicates and connected components were clustered using Disjoint Set Union (DSU) and assigned atomically to the same partition.
- **Overlap Verification**: Verified zero overlap across all split intersections:
  - `Train ∩ Val = ∅`
  - `Train ∩ Test = ∅`
  - `Val ∩ Test = ∅`
- **Duplicate Leakage**: **0.0%** (zero duplicated images across partitions).
