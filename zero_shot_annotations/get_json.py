import json

json_path = '/data-store/zengrh/qianyihao/af_ttt_v1/data/anet13/anet1.3_clip_filtered.json'
list_path = '/data-store/zengrh/qianyihao/af_ttt_v1/zero_shot_annotations/label_list/activitynet/train_50.list'
output_path = '/data-store/zengrh/qianyihao/af_ttt_v1/zero_shot_annotations/annotations/activitynet/50_test.json'

# 读取 json_path 文件
with open(json_path, 'r') as f:
    data = json.load(f)

## get ['classes']
with open(list_path, 'r') as f:
    labels_to_remove = {line.strip() for line in f.readlines()}  # 使用集合以便快速查找
    #print(labels_to_remove)

if "classes" in data and isinstance(data["classes"], list):
    #print(data["classes"])
    data["classes"] = [label for label in data["classes"] if label not in labels_to_remove]
    #print(data["classes"])

## get ['database']
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
        del data['database'][video_key]

## sort label_id
remaining_labels = set()
for video_info in data['database'].values():
    for annotation in video_info['annotations']:
        remaining_labels.add(annotation['label'])

# 对剩下的标签进行排序
sorted_labels = sorted(remaining_labels)

# 创建一个新的 label_id 映射，从 0 开始
label_id_mapping = {label: idx for idx, label in enumerate(sorted_labels)}

# 更新所有剩下的 label_id
for video_info in data['database'].values():
    for annotation in video_info['annotations']:
        annotation['label_id'] = label_id_mapping[annotation['label']]

# 将结果写回到 aa_filtered.json 文件
with open(output_path, 'w') as f:
    json.dump(data, f, indent=4)

print("Finished")

