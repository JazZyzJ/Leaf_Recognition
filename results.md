| model_name | img_size | aug_notes | epochs | batch_size | CV_acc | LB_public | LB_private | notes |
|------------|---------:|-----------|-------:|-----------:|-------:|----------:|-----------:|-------|
| resnet200d | 512 | RRC + flips + rotation | 25 | 8 | 0.9769 | TBD | TBD | bs=8 |
| tf_efficientnet_b4_ns | 380 | RRC + flips + rotation (380) | 25 | 16 | 0.9767 | TBD | TBD | mixup prototype |
| resnet50d | 512 | RRC + flips + rotation | 20 | 32 | 0.9744 | 0.9818 | 0.9798 | - |
