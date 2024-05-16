COLLATE = dict(
    train=dict(
        name = "StreamBatchCompose",
        to_tensor_keys = ["imgs", "flows", "res", "labels", "masks", "precise_sliding_num"]
    ),
    test=dict(
        name = "StreamBatchCompose",
        to_tensor_keys = ["imgs", "flows", "res", "labels", "masks", "precise_sliding_num"]
    ),
    infer=dict(
        name = "StreamBatchCompose",
        to_tensor_keys = ["imgs", "flows", "res", "labels", "masks", "precise_sliding_num"]
    )
)