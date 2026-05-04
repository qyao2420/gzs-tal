import numpy as np

# 加载 .npy 文件
i3d_data = np.load('/data/qianyihao/Datasets/ActivityNet13/i3d_features/v_8uP35-qttBo.npy')
clip_data = np.load('/data/qianyihao/Datasets/ActivityNet13/CLIP_feature/v_8uP35-qttBo.npy')
iv_data = np.load('/data/qianyihao/Datasets/ActivityNet13/InternVideo_feat_16frames/v_8uP35-qttBo.npy')

# 打印数据的形状
print(f'i3d : {i3d_data.shape}')
print(f'clip: {clip_data.shape}')
print(f'iv  : {iv_data.shape}')
