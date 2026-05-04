import json

in_file_path = '/data-store/zengrh/qianyihao/af_ttt_v1/data/thumos14/thumos14_clip_annotations.json'
out_file_path = '/data-store/zengrh/qianyihao/af_ttt_v1/data/thumos14/thumos14_clip_annotations.json'

# 读取 aa.json 文件
with open(in_file_path, 'r') as f:
    data = json.load(f)

'''labels_to_remove = ["Ambiguous"]
for video_key, video_info in list(data['database'].items()):
    # 筛选出不在 labels_to_remove 中的注释
    filtered_annotations = [
        annotation for annotation in video_info['annotations']
        if annotation['label'] not in labels_to_remove
    ]
    
    # 更新视频信息中的注释
    video_info['annotations'] = filtered_annotations

    # 如果没有剩余的注释，删除该视频
    if not video_info['annotations']:
        del data['database'][video_key]'''

# 对 classes 进行排序
data['classes'].sort()

# 创建一个 label 到 id 的映射
label_to_id = {label: idx for idx, label in enumerate(data['classes'])}

# 为 annotations 中的每个 segment 增加 label_id
for entry in data['database'].values():
    for annotation in entry['annotations']:
        annotation['label_id'] = label_to_id[annotation['label']]

# 将修改后的内容写回 aa.json 文件
with open(out_file_path, 'w') as f:
    json.dump(data, f, indent=4)

print("Successfully updated aa.json!")
