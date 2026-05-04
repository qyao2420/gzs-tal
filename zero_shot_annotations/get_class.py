import json

# 文件路径
json_file_path = '/data-store/zengrh/qianyihao/Ego/af_ttt/zero_shot_annotations/annotations/activitynet/50_train.json'
list_file_path = '/data-store/zengrh/qianyihao/Ego/af_ttt/zero_shot_annotations/label_list/activitynet/test_50.list'

# 读取 aa.list 文件
with open(list_file_path, 'r') as list_file:
    labels_to_remove = set(line.strip() for line in list_file)

# 读取 JSON 文件
with open(json_file_path, 'r') as json_file:
    data = json.load(json_file)

# 检查并删除 classes 中存在于 aa.list 的标签
if "classes" in data and isinstance(data["classes"], list):
    data["classes"] = [label for label in data["classes"] if label not in labels_to_remove]

# 将修改后的数据写回 JSON 文件
with open(json_file_path, 'w') as json_file:
    json.dump(data, json_file, indent=4)

print("Specified labels removed successfully.")
