import numpy as np

iv_feat_path = '/data-store/zengrh/qianyihao/Datasets/ActivityNet13/InternVideo_16frames/anet_mae_hugek700/v_-06dWmCzbxY.npy'
clip_feat_path = '/data-store/zengrh/qianyihao/Datasets/ActivityNet13/CLIP_feature/v_-06dWmCzbxY.npy'

iv_tensor = np.load(iv_feat_path)
clip_tensor = np.load(clip_feat_path)

print("iv_feat_shape: ", iv_tensor.shape)
print("clip_feat_shape: ", clip_tensor.shape)

'''import json

annotations_file_path = '/data-store/zengrh/qianyihao/GAP-main/data/Thumos14/Thumos14_annotations.json'  # 替换为你的 annotations.json 文件路径
info_file_path = '/data-store/zengrh/qianyihao/GAP-main/data/Thumos14/CLIP/Thumos14_info_8frame.json'  # 替换为你的 info.json 文件路径
out_annotations_file_path = '/data-store/zengrh/qianyihao/af_ttt_v1/data/thumos14/thumos14_clip_annotations.json'

with open(annotations_file_path, 'r') as f:
    annotations_data = json.load(f)

with open(info_file_path, 'r') as f:
    info_data = json.load(f)

# 更新 annotations.json 中的 fps 并检查视频名
count = 0
for video_name, attributes in annotations_data['database'].items():
    if video_name in info_data:
        video_fps = info_data[video_name]['video_fps']
        if video_fps != attributes['fps']:
            print(video_name)
            count += 1
            attributes['fps'] = video_fps  # 更新 fps
    else:
        print(f"视频 {video_name} 在 info.json 中未找到")
print(count)

# 将更新后的 annotations_data 写回文件
with open(out_annotations_file_path, 'w') as f:
    json.dump(annotations_data, f, indent=4)

print("更新完成！")'''

