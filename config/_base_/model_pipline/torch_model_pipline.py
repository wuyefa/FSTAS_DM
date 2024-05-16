MODEL_PIPLINE = dict(
    name = "TorchModelPipline",
    grad_accumulate = dict(
        name = "GradAccumulate",
        accumulate_type = "conf"
    ),
    model = None,
    post_processing = None,
    criterion = None,
    optimizer = dict(
        name = "AdamWOptimizer",
        learning_rate = 0.0005,
        weight_decay = 1e-4,
        betas = (0.9, 0.999),
        finetuning_scale_factor=0.5,
        no_decay_key = [],
        finetuning_key = [],
        freeze_key = [],
    ),
    lr_scheduler = dict(
        name = "MultiStepLR",
        step_size = [50],
        gamma = 0.1,
    )
)