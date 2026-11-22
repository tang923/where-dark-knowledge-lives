# Sources and attribution

This is a compact implementation of the supplied ECE661 proposal. The loss implementation uses the mathematical definitions below; it does not import mdistiller as a dependency.

- Hinton, Vinyals, Dean, *Distilling the Knowledge in a Neural Network* (2015): https://arxiv.org/abs/1503.02531
- Yuan et al., *Revisit Knowledge Distillation: a Teacher-free Framework* (2019): https://arxiv.org/abs/1909.11723
- Zhao et al., *Decoupled Knowledge Distillation*, CVPR 2022: https://openaccess.thecvf.com/content/CVPR2022/html/Zhao_Decoupled_Knowledge_Distillation_CVPR_2022_paper.html
- Official DKD implementation/configuration, consulted 2026-11-08: https://github.com/megvii-research/mdistiller
  - `configs/cifar100/dkd/res56_res20.yaml`: 240 epochs, SGD 0.05, batch 64, momentum 0.9, weight decay 0.0005, LR milestones 150/180/210, beta=2.
  - `mdistiller/engine/cfg.py`: alpha=1, T=4, 20-epoch warmup.
  - `mdistiller/dataset/cifar100.py`: CIFAR-100 normalization and augmentation.
  - `mdistiller/models/cifar/resnet.py`: architecture and checkpoint naming compatibility (projection shortcuts).

Our changes include a held-out stratified validation split, equal CE weights across methods, shared distillation warmup, stable log-domain loss decomposition, final-epoch selection for omitted-class experiments, and a small common training loop. Do not describe this as an exact reproduction of DKD Table 1.

No public teacher weights are bundled. Any supplied teacher must be attributed with `--teacher-source`; its SHA-256 and validation accuracy are stored in the run result.

The upstream mdistiller repository is MIT-licensed, copyright (c) 2022 MEGVII Research, and attributes RepDistiller (copyright (c) 2020 Yonglong Tian). See its [license](https://github.com/megvii-research/mdistiller/blob/master/LICENSE). This repository does not bundle upstream source files or third-party datasets. Before publishing, the project authors should choose a license for their own code and preserve attribution for any subsequently incorporated code or checkpoints.
