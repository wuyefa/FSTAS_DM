COLLATE = dict(
    train = dict(
        name = "BatchCompose",
        to_tensor_keys = ["imgs", "flows", "res", "feature", "labels", "masks", "precise_sliding_num"],
        compress_keys = ["sliding_num", "current_sliding_cnt"],
        ignore_index = -100
    ),
    test = dict(
        name = "BatchCompose",
        to_tensor_keys = ["imgs", "flows", "res", "feature", "labels", "masks", "precise_sliding_num"],
        compress_keys = ["sliding_num", "current_sliding_cnt"],
        ignore_index = -100
    ),
    infer = dict(
        name = "BatchCompose",
        to_tensor_keys = ["imgs", "flows", "res", "feature", "labels", "masks", "precise_sliding_num"],
        compress_keys = ["sliding_num", "current_sliding_cnt"],
        ignore_index = -100
    )
)