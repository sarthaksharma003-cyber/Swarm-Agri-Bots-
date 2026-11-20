# SwarmAgri AI — YOLO11n Training Summary

## Training Configuration
- **Model**: YOLO11n (Pretrained: `yolo11n.pt`)
- **Dataset**: `/home/smasher/Desktop/swarmagri-ai/data/processed/yolo_dataset/data.yaml`
- **Classes**: `{0: 'crop', 1: 'weed'}`
- **Epochs Requested**: 10 (Completed: 10, Patience: 5)
- **Batch Size**: 4
- **Image Size**: 512 x 512
- **Compute Device**: `cpu`
- **Random Seed**: `42`
- **Duration**: 2670.65 s (44.51 min)

## Model Checkpoints
- **Run Directory**: `/home/smasher/Desktop/swarmagri-ai/ml/models/yolo11n/run_10epochs`
- **Best Weights**: `/home/smasher/Desktop/swarmagri-ai/ml/models/yolo11n/run_10epochs/weights/best.pt`
- **Last Weights**: `/home/smasher/Desktop/swarmagri-ai/ml/models/yolo11n/run_10epochs/weights/last.pt`
- **Primary Checkpoint**: `/home/smasher/Desktop/swarmagri-ai/ml/models/yolo11n/best.pt`

## Per-Epoch Metrics Progression
| Epoch | Train (Box/Cls) | Val (Box/Cls) | Precision | Recall | mAP@50 | mAP@50-95 |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | 1.3443 / 2.0742 | 1.4648 / 1.6611 | 0.5782 | 0.6558 | **0.6184** | **0.3164** |
| 2 | 1.3742 / 1.3687 | 1.5438 / 1.3799 | 0.6897 | 0.6247 | **0.7130** | **0.3911** |
| 3 | 1.3372 / 1.2756 | 1.5585 / 1.5303 | 0.5390 | 0.7016 | **0.5749** | **0.2974** |
| 4 | 1.3005 / 1.2092 | 1.3694 / 1.1668 | 0.8479 | 0.6780 | **0.7768** | **0.4558** |
| 5 | 1.2615 / 1.0957 | 1.4193 / 1.0934 | 0.7779 | 0.7406 | **0.8266** | **0.4814** |
| 6 | 1.2201 / 1.0121 | 1.2954 / 1.0726 | 0.8986 | 0.6924 | **0.8422** | **0.5191** |
| 7 | 1.1462 / 0.9460 | 1.2591 / 1.0739 | 0.7965 | 0.7663 | **0.8192** | **0.5121** |
| 8 | 1.1066 / 0.8751 | 1.2095 / 0.9499 | 0.8232 | 0.8246 | **0.8600** | **0.5404** |
| 9 | 1.1034 / 0.8157 | 1.2169 / 0.9708 | 0.8314 | 0.8334 | **0.8546** | **0.5457** |
| 10 | 1.0447 / 0.7773 | 1.1986 / 0.9647 | 0.7992 | 0.7948 | **0.8425** | **0.5393** |

