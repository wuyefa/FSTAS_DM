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
            name = "FeatureSampler",
            is_train = True,
            sample_rate_dict={ "feature": 1,"labels": 1 },
            sample_add_key_pair={ "frames": "feature" },
            sample_mode = "uniform",
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
            name = "FeatureSampler",
            is_train = False,
            sample_rate_dict={ "feature": 1,"labels": 1 },
            sample_add_key_pair={ "frames": "feature" },
            sample_mode = "uniform",
        ),
        transform = dict(
            name = "FeatureStreamTransform",
            transform_dict = dict(
                feature = [dict(XToTensor = None)]
            )
        )
    )
)