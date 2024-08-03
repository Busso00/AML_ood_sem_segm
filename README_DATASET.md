### Prepare directory of Training [Cityscapes](https://www.cityscapes-dataset.com) and Validation (All validation dataset) in a ./datasets dir with following structure

```plaintext
datasets
├── Train_Dataset
│   ├── gtFine
│   │   ├── test
│   │   ├── train
│   │   └── val
│   └── leftImg8bit
│       ├── test
│       ├── train
│       └── val
└── Validation_Dataset
    ├── FS_LostFound_full
    │   ├── images
    │   └── labels_masks
    ├── fs_static
    │   ├── images
    │   └── labels_masks
    ├── RoadAnomaly
    │   ├── images
    │   └── labels_masks
    ├── RoadAnomaly21
    │   ├── images
    │   └── labels_masks
    └── RoadObsticle21
        ├── images
        └── labels_masks
```

