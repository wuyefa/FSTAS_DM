import torch
from svtas.utils.logger import get_logger
from svtas.utils import mkdir
from svtas.utils import AbstractBuildFactory
from svtas.engine import BaseEngine
from ..utils.collect_env import collect_env

@torch.no_grad()
def test(local_rank,
         nprocs,
         cfg,
         args):
    logger = get_logger("SVTAS")
    model_name = cfg.model_name
    output_dir = cfg.get("output_dir", f"./output/{model_name}")
    mkdir(output_dir)
    # env info logger
    # env_info_dict = collect_env()
    # env_info = '\n'.join([f'{k}: {v}' for k, v in env_info_dict.items()])
    # dash_line = '-' * 60 + '\n'
    # logger.info('Environment info:\n' + dash_line + env_info + '\n' +
    #             dash_line)

    # 2. build metirc
    metric_cfg = cfg.METRIC
    metrics = dict()
    for k, v in metric_cfg.items():
        v['train_mode'] = False
        metrics[k] = AbstractBuildFactory.create_factory('metric').create(v)
    
    # 3. construct model_pipline
    model_pipline = AbstractBuildFactory.create_factory('model_pipline').create(cfg.MODEL_PIPLINE)

    # 4. Construct Dataset
    temporal_clip_batch_size = cfg.DATALOADER.get('temporal_clip_batch_size', 3)
    video_batch_size = cfg.DATALOADER.get('video_batch_size', 8)
    test_pipeline = AbstractBuildFactory.create_factory('dataset_pipline').create(cfg.DATASETPIPLINE.test)
    test_dataset_config = cfg.DATASET.test
    test_dataset_config['pipeline'] = test_pipeline
    test_dataset_config['temporal_clip_batch_size'] = temporal_clip_batch_size
    test_dataset_config['video_batch_size'] = video_batch_size * nprocs
    test_dataset_config['local_rank'] = local_rank
    test_dataset_config['nprocs'] = nprocs
    test_dataloader_config = cfg.DATALOADER
    test_dataloader_config['dataset'] = AbstractBuildFactory.create_factory('dataset').create(test_dataset_config)
    test_dataloader_config['collate_fn'] = AbstractBuildFactory.create_factory('dataset_pipline').create(cfg.COLLATE.test)
    test_dataloader = AbstractBuildFactory.create_factory('dataloader').create(test_dataloader_config)
    
    # 5. build engine
    engine_config = cfg.ENGINE
    engine_config['logger_dict'] = cfg.LOGGER_LIST
    engine_config['metric'] = metrics
    engine_config['model_pipline'] = model_pipline
    engine_config['model_name'] = model_name
    test_engine: BaseEngine = AbstractBuildFactory.create_factory('engine').create(engine_config)
    test_engine.set_dataloader(test_dataloader)
    test_engine.running_mode = 'test'
    
    # 6. resume engine
    if cfg.ENGINE.checkpointor.get('load_path', None) is not None:
        test_engine.resume()
        
    # 7. run engine
    test_engine.init_engine()
    test_engine.run()
    test_engine.shutdown()
    