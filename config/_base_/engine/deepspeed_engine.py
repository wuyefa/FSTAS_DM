ENGINE = dict(
    name = "DeepSpeedDistributedDataParallelEngine",
    record = dict(
        name = "ValueRecord"
    ),
    iter_method = dict(
        name = "EpochMethod",
        epoch_num = 80,
        batch_size = 1,
        test_interval = 1,
        criterion_metric_name = "F1@0.50"
    ),
    checkpointor = dict(
        name = "DeepSpeedCheckpointor"
    )
)
