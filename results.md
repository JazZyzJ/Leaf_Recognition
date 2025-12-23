| model_name | img_size | aug_notes | epochs | batch_size | CV_acc | LB_public | LB_private | notes |
|------------|---------:|-----------|-------:|-----------:|-------:|----------:|-----------:|-------|
| tf_efficientnet_b4_ns | 380 | Finetune no-mixup | 5 | 16 | 0.9779 | 0.9827 | 0.9877 | finetune effnet with sweep script |
| tf_efficientnet_b4_ns | 380 | Finetune lr=1e-4 ep=5 | 5 | 16 | 0.9777 | TBD | TBD | ft sweep |
| tf_efficientnet_b4_ns | 380 | Finetune lr=1e-4 ep=3 | 3 | 16 | 0.9778 | TBD | TBD | ft sweep |
| tf_efficientnet_b4_ns | 380 | Finetune lr=5e-5 ep=5 | 5 | 16 | 0.9777 | TBD | TBD | ft sweep |
| tf_efficientnet_b4_ns | 380 | Finetune lr=5e-5 ep=3 | 3 | 16 | 0.9775 | TBD | TBD | ft sweep |
| tf_efficientnet_b4_ns+resnet50d+resnet200d | [380, 512, 512] | RRC + flips + rotation (380) | RRC + flips + rotation | RRC + flips + rotation | [25, 20, 25] | [16, 32, 8] | 0.9767 | 0.9832 | 0.9879 | with tta |
| tf_efficientnet_b4_ns+resnet50d+resnet200d | [380, 512, 512] | RRC + flips + rotation (380) | RRC + flips + rotation | RRC + flips + rotation | [25, 20, 25] | [16, 32, 8] | 0.9767 | 0.9834 | 0.9879 | without tta |
| tf_efficientnet_b4_ns+resnet50d | [380, 512] | RRC + flips + rotation (380) | RRC + flips + rotation | [25, 20] | [16, 32] | 0.9767 | 0.9839 | 0.9882 | - |
| resnet200d | 512 | RRC + flips + rotation | 25 | 8 | 0.9769 | TBD | TBD | bs=8 |
| tf_efficientnet_b4_ns | 380 | RRC + flips + rotation (380) | 25 | 16 | 0.9767 | TBD | TBD | mixup prototype |
| tf_efficientnet_b4_ns | 380 | RRC + flips + rotation (380) | 25 | 16 | 0.9767 | TBD | TBD | - |
| resnet50d | 512 | RRC + flips + rotation | 20 | 32 | 0.9744 | 0.9798 | 0.9818 | - |
