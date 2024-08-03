prepare directory of Training (Cityscapes) and Validation (All validation dataset)
in a ./dataset dir with following structure

dataset
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

