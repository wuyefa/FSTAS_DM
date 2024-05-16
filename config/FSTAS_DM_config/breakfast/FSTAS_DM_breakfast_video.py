_base_ = [
    '../../_base_/dataloader/collater/stream_compose.py',
    '../../_base_/engine/standaline_engine.py',
    '../../_base_/logger/python_logger.py',
]

split = 1
num_classes = 48
ignore_index = -100
epochs = 1000
batch_size = 1
in_channels = 2048
clip_seg_num_list = [64, 128, 256]
sample_rate_list = [8]
sample_rate = 8
clip_seg_num = 128
sliding_window = clip_seg_num * sample_rate
sigma = 1
model_name = "FSTAS_DM_breakfast_split" + str(split)

ENGINE = dict(
    name = "TorchDistributedDataParallelEngine",
    record = dict(
        name = "StreamValueRecord"
    ),
    iter_method = dict(
        name = "StreamEpochMethod",
        epoch_num = epochs,
        batch_size = batch_size,
        test_interval = 1,
        criterion_metric_name = "F1@0.50",
        logger_iter_interval = 100,
        save_interval = 50,
    ),
    checkpointor = dict(
        name = "TorchCheckpointor",
        # load_path = "the path of the weights" # testing
    )
)

LOGGER_LIST = dict(
    PythonLoggingLogger = dict(
        name = "SVTAS"
    ),
    TensboardLogger = dict(
        name = "SVTAS_tensorboard"
    )
)

MODEL_PIPLINE = dict(
    name = "TorchModelPipline",
    grad_accumulate = dict(
        name = "GradAccumulate",
        accumulate_type = "conf"
    ),
    model = dict(
        name = "TemporalActionSegmentationDDIMModel",
        prompt_net = dict(
            name = "StreamVideoSegmentation",
            architecture_type ='3d',
            addition_loss_pos = 'with_backbone_loss',
            backbone = dict(
                name = "SwinTransformer3D",
                pretrained = "./data/checkpoint/swin_base_patch244_window877_kinetics600_22k.pth",
                pretrained2d = False,
                patch_size = [2, 4, 4],
                embed_dim = 128,
                depths = [2, 2, 18, 2],
                num_heads = [4, 8, 16, 32],
                window_size = [8,7,7],
                mlp_ratio = 4.,
                qkv_bias = True,
                qk_scale = None,
                drop_rate = 0.,
                attn_drop_rate = 0.,
                drop_path_rate = 0.2,
                patch_norm = True,
                use_checkpoint = True,
            ),
            neck = dict(
                name = "TaskFusionPoolNeck",
                num_classes=num_classes,
                in_channels = 1024,
                need_pool = True
            ),
            head = dict(
                name = "PhaseChangeModel",
                exponential_boundary = 3,
                deta_dilation = 8,
                kernel_size = 5,
                num_stages = 4,
                num_layers = 10,
                num_f_maps = 128,
                dim = 1024,
                num_classes = num_classes,
                sample_rate = sample_rate * 2,
                out_feature = True,
                out_dict = True
            ),
        ),
        unet = dict(
            name = "TASDiffusionConditionUnet",
            num_layers = 10,
            num_f_maps = 128,
            dim = num_classes,
            num_classes = num_classes,
            condition_dim = 128,
            time_embedding_dim = 512,
            condtion_res_layer_idx = [2, 3, 4, 5, 7],
            sample_rate = sample_rate
        ),
        scheduler = dict(
            name = "DiffsusionActionSegmentationScheduler",
            num_train_timesteps = 1000,
            num_inference_steps = 25,
            ddim_sampling_eta = 1.0,
            snr_scale = 0.5,
            timestep_spacing = 'linspace',
            infer_region_seed = 8
        )
    ),
    post_processing = dict(
        name = "StreamScorePostProcessing",
        ignore_index = ignore_index
    ),
    criterion = dict(
        name = "TASDiffusionStreamSegmentationLoss",
        prompt_net_loss_cfg = dict(
            name = "SegmentationLoss",
            num_classes = num_classes,
            sample_rate = sample_rate,
            smooth_weight = 0.25,
            ignore_index = ignore_index
        ),
        unet_loss_cfg = dict(
            name = "SegmentationLoss",
            num_classes = num_classes,
            sample_rate = sample_rate,
            smooth_weight = 0.25,
            ignore_index = ignore_index
        ),
        prompt_backbone_loss_cfg = dict(
            name = "SegmentationLoss",
            num_classes = num_classes,
            sample_rate = sample_rate,
            smooth_weight = 0.0,
            ignore_index = ignore_index
        ),
    ),
    optimizer = dict(
        name = "AdamWOptimizer",
        learning_rate = 0.0001,
        weight_decay = 1e-4,
        betas = (0.9, 0.999),
        finetuning_scale_factor=0.02,
        no_decay_key = [],
        finetuning_key = ["prompt_net.backbone"],
        freeze_key = [],
    ),
    lr_scheduler = dict(
        name = "CosineAnnealingLR",
        T_max = 100,
        eta_min = 0.00003,
    )
)

DATALOADER = dict(
    name = "TorchStreamDataLoader",
    temporal_clip_batch_size = 3,
    video_batch_size = batch_size,
    num_workers = 8
)

COLLATE = dict(
    train=dict(
        name = "StreamBatchCompose",
        to_tensor_keys = ["imgs", "flows", "res", "labels", "masks", "precise_sliding_num", "labels_onehot", "boundary_prob"]
    ),
    test=dict(
        name = "StreamBatchCompose",
        to_tensor_keys = ["imgs", "flows", "res", "labels", "masks", "precise_sliding_num", "labels_onehot", "boundary_prob"]
    ),
)

DATASET = dict(
    train = dict(
        name = "DiffusionRawFrameDynamicStreamSegmentationDataset",
        data_prefix = "./",
        file_path = "./data/breakfast/splits/train.split" + str(split) + ".bundle",
        videos_path = "./data/breakfast/Videos_mp4",
        gt_path = "./data/breakfast/groundTruth",
        actions_map_file_path = "./data/breakfast/mapping.txt",
        dataset_type = "breakfast",
        train_mode = True,
        dynamic_stream_generator=dict(
            name = "MultiEpochStageDynamicStreamGenerator",
            multi_epoch_list = [40, 80],
            strategy_list = [
                dict(name = "ListRandomChoiceDynamicStreamGenerator",
                     clip_seg_num_list = [128], # [128]
                     sample_rate_list = sample_rate_list),
                dict(name = "ListRandomChoiceDynamicStreamGenerator",
                     clip_seg_num_list = [128, 256],
                     sample_rate_list = sample_rate_list),
                dict(name = "ListRandomChoiceDynamicStreamGenerator",
                     clip_seg_num_list = [128, 256, 64],
                     sample_rate_list = sample_rate_list),
            ]
        )
    ),
    test = dict(
        name = "DiffusionRawFrameStreamSegmentationDataset",
        data_prefix = "./",
        file_path = "./data/breakfast/splits/test.split" + str(split) + ".bundle",
        videos_path = "./data/breakfast/Videos_mp4",
        gt_path = "./data/breakfast/groundTruth",
        actions_map_file_path = "./data/breakfast/mapping.txt",
        dataset_type = "breakfast",
        train_mode = False,
        sliding_window = sliding_window
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

DATASETPIPLINE = dict(
    train = dict(
        name = "BaseDatasetPipline",
        decode = dict(
            name="VideoDecoder",
            backend=dict(
                    name='DecordContainer')
        ),
        sample = dict(
            name = "VideoDynamicStreamSampler",
            is_train = True,
            sample_rate_name_dict={"imgs":'sample_rate', "labels":'sample_rate'},
            clip_seg_num_name_dict={"imgs": 'clip_seg_num', "labels": 'clip_seg_num'},
            ignore_index=ignore_index,
            sample_add_key_pair={"frames":"imgs"},
            sample_mode = "uniform"
        ),
        transform = dict(
            name = "VideoTransform",
            transform_dict = dict(
                imgs = [
                dict(ResizeImproved = dict(size = 256)),
                dict(RandomCrop = dict(size = 224)),
                dict(RandomHorizontalFlip = None),
                dict(PILToTensor = None),
                dict(ToFloat = None),
                dict(Normalize = dict(
                    mean = [0.4245283568405083 * 255, 0.3904851168609079 * 255, 0.33709139617292494 * 255],
                    std = [0.26207845745959846 * 255, 0.26008439810422 * 255, 0.24623600365905168 * 255]
                ))],
                labels = dict(
                    labels_onehot = dict(
                        name = 'direct_transform',
                        transforms_op_list = [
                            dict(XToTensor = None),
                            dict(LabelsToOneHot = dict(
                                num_classes = num_classes,
                                sample_rate = sample_rate,
                                ignore_index = ignore_index
                            ))
                        ]
                    ),
                    boundary_prob = dict(
                        name = 'direct_transform',
                        transforms_op_list = [
                            dict(XToTensor = None),
                            dict(SegmentationLabelsToBoundaryProbability = dict(
                                sigma = sigma,
                                need_norm = True
                            ))
                        ]
                    )
                )
            )
        )
    ),
    test = dict(
        name = "BaseDatasetPipline",
        decode = dict(
            name="VideoDecoder",
            backend=dict(
                    name='DecordContainer')
        ),
        sample = dict(
            name = "VideoStreamSampler",
            is_train = False,
            sample_rate_dict={"imgs":sample_rate,"labels":sample_rate},
            clip_seg_num_dict={"imgs":clip_seg_num ,"labels":clip_seg_num},
            sliding_window_dict={"imgs":sliding_window,"labels":sliding_window},
            sample_add_key_pair={"frames":"imgs"},
            sample_mode = "uniform"
        ),
        transform = dict(
            name = "VideoTransform",
            transform_dict = dict(
                imgs = [
                    dict(ResizeImproved = dict(size = 256)),
                    dict(CenterCrop = dict(size = 224)),
                    dict(PILToTensor = None),
                    dict(ToFloat = None),
                    dict(Normalize = dict(
                        mean = [0.4245283568405083 * 255, 0.3904851168609079 * 255, 0.33709139617292494 * 255],
                        std = [0.26207845745959846 * 255, 0.26008439810422 * 255, 0.24623600365905168 * 255]
                    ))],
                labels = dict(
                    labels_onehot = dict(
                        name = 'direct_transform',
                        transforms_op_list = [
                            dict(XToTensor = None),
                            dict(LabelsToOneHot = dict(
                                num_classes = num_classes,
                                sample_rate = sample_rate,
                                ignore_index = ignore_index
                            ))
                        ]
                    ),
                    boundary_prob = dict(
                        name = 'direct_transform',
                        transforms_op_list = [
                            dict(XToTensor = None),
                            dict(SegmentationLabelsToBoundaryProbability = dict(
                                sigma = sigma,
                                need_norm = True
                            ))
                        ]
                    )
                )
            )
        )
    )
)