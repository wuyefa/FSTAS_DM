export CUDA_VISIBLE_DEVICES=0
export DECORD_EOF_RETRY_MAX=20480

### gtea ###
python tools/launch.py --mode train -c config/FSTAS_DM_config/gtea/FSTAS_DM_gtea_video.py --seed 0

### 50salads ###
# python tools/launch.py --mode train -c config/FSTAS_DM_config/50salads/FSTAS_DM_50salads_video.py --seed 0

### breakfast ###
# python tools/launch.py --mode train -c config/FSTAS_DM_config/breakfast/FSTAS_DM_breakfast_video.py --seed 0