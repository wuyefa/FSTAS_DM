DATASET = dict(
    temporal_clip_batch_size = 1,
    video_batch_size = 1,
    num_workers = 2,
    train = dict(
        name = "FeatureStreamSegmentationDataset",
        data_prefix = "./",
        file_path = "./data/breakfast/splits/train.split1.bundle",
        feature_path = "./data/breakfast/features",
        gt_path = "./data/breakfast/groundTruth",
        actions_map_file_path = "./data/breakfast/mapping.txt",
        dataset_type = "breakfast",
        train_mode = True,
        sliding_window = 1
    ),
    test = dict(
        name = "FeatureStreamSegmentationDataset",
        data_prefix = "./",
        file_path = "./data/breakfast/splits/test.split1.bundle",
        feature_path = "./data/breakfast/features",
        gt_path = "./data/breakfast/groundTruth",
        actions_map_file_path = "./data/breakfast/mapping.txt",
        dataset_type = "breakfast",
        train_mode = False,
        sliding_window = 1
    )
)

METRIC = dict(
    TAS = dict(
    name = "TASegmentationMetric",
    overlap = [.1, .25, .5],
    actions_map_file_path = "./data/breakfast/mapping.txt",
    file_output = False,
    score_output = False),
)