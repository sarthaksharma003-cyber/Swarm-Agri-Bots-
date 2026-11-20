# SwarmAgri AI - Class Frequency & Distribution Report

## Class Mapping & Totals
| Class ID | Class Name | Total Bounding Boxes | Percentage |
|:---:|:---:|:---:|:---:|
| 0 | crop | 1,212 | 58.49% |
| 1 | weed | 860 | 41.51% |
| **Total** | - | **2,072** | **100.0%** |

## Stratified Split Distribution
| Split | Images | Class 0 (crop) | Class 1 (weed) | Total Boxes | Crop Ratio | Weed Ratio |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Train (80%)** | 1,042 | 945 | 691 | 1,636 | 0.91 | 0.66 |
| **Val (10%)** | 129 | 117 | 82 | 199 | 0.91 | 0.64 |
| **Test (10%)** | 129 | 150 | 87 | 237 | 1.16 | 0.67 |

## Class Co-Occurrence
- Images with **Crop Only**: 633
- Images with **Weed Only**: 665
- Images with **Both Crop & Weed**: 2 (agri_0_1852, agri_0_3820)
