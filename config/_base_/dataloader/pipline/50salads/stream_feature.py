sample_rate = 1
clip_seg_num = 256
in_channels = 2048
sliding_window = clip_seg_num * sample_rate


DATASETPIPELINE = dict(
    train = dict(
        name = "BasePipline",
        decode = dict(
            name='FeatureDecoder',
            backend=dict(
                    name='NPYContainer',
                    is_transpose=False,
                    temporal_dim=-1,
                    revesive_name=[(r'(mp4|avi)', 'npy')]
                 )
        ),
        sample = dict(
            name = "FeatureStreamSampler",
            is_train = True,
            sample_rate_dict={"feature":sample_rate, "labels":sample_rate},
            clip_seg_num_dict={"feature":clip_seg_num, "labels":clip_seg_num},
            sliding_window_dict={"feature":sliding_window, "labels":sliding_window},
            sample_add_key_pair={"frames":"feature"},
            feature_dim_dict={"feature":in_channels},
            sample_mode = "uniform"
        ),
        transform = dict(
            name = "FeatureStreamTransform",
            transform_dict = dict(
                feature = [dict(XToTensor = None)]
            )
        )
    ),
    test = dict(
        name = "BasePipline",
        decode = dict(
            name='FeatureDecoder',
            backend=dict(
                    name='NPYContainer',
                    is_transpose=False,
                    temporal_dim=-1,
                    revesive_name=[(r'(mp4|avi)', 'npy')]
                 )
        ),
        sample = dict(
            name = "FeatureStreamSampler",
            is_train = False,
            sample_rate_dict={"feature":sample_rate, "labels":sample_rate},
            clip_seg_num_dict={"feature":clip_seg_num, "labels":clip_seg_num},
            sliding_window_dict={"feature":sliding_window, "labels":sliding_window},
            sample_add_key_pair={"frames":"feature"},
            feature_dim_dict={"feature":in_channels},
            sample_mode = "uniform"
        ),
        transform = dict(
            name = "FeatureStreamTransform",
            transform_dict = dict(
                feature = [dict(XToTensor = None)]
            )
        )
    )
)