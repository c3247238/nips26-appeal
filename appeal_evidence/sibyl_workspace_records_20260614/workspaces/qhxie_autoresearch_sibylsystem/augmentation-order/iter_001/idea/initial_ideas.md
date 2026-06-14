1. Train the same model (ResNet-18 on CIFAR-10) with augmentation pipelines in different orderings (geometric first vs. color first vs. mixed)
2. Compare: fixed order vs. random order shuffling at each step
3. Extend to ViT-small to check if architecture changes the sensitivity to ordering
4. Hypothesis: geometric transformations (crop, flip) should precede color transformations for better performance