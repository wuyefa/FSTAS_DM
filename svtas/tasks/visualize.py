import os
import sys
path = os.path.join(os.getcwd())
sys.path.append(path)
from svtas.utils.logger import get_logger
from svtas.utils import AbstractBuildFactory

def visualize(local_rank,
              nprocs,
              cfg,
              args):
    logger = get_logger("SVTAS")
    model_name = cfg.model_name

    # construct model
    model_pipline = AbstractBuildFactory.create_factory('model_pipline').create(cfg.MODEL_PIPLINE)

    # construct dataloader
    temporal_clip_batch_size = cfg.DATALOADER.get('temporal_clip_batch_size', 3)
    video_batch_size = cfg.DATALOADER.get('video_batch_size', 8)
    test_pipeline = AbstractBuildFactory.create_factory('dataset_pipline').create(cfg.DATASETPIPLINE)
    test_dataset_config = cfg.DATASET.config
    test_dataset_config['pipeline'] = test_pipeline
    test_dataset_config['temporal_clip_batch_size'] = temporal_clip_batch_size
    test_dataset_config['video_batch_size'] = video_batch_size
    test_dataloader_config = cfg.DATALOADER
    test_dataloader_config['dataset'] = AbstractBuildFactory.create_factory('dataset').create(test_dataset_config)
    test_dataloader_config['collate_fn'] = AbstractBuildFactory.create_factory('dataset_pipline').create(cfg.COLLATE.test)
    test_dataloader = AbstractBuildFactory.create_factory('dataloader').create(test_dataloader_config)
    
    engine_config = cfg.ENGINE
    engine_config['logger_dict'] = cfg.LOGGER_LIST
    engine_config['model_pipline'] = model_pipline
    engine_config['model_name'] = model_name + "_visualize"
    visualize_engine = AbstractBuildFactory.create_factory('engine').create(engine_config)
    visualize_engine.set_dataloader(test_dataloader)

    visualize_engine.init_engine()
    visualize_engine.run()
    logger.info("Finish all extracting!")
    visualize_engine.shutdown()
