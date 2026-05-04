import json

in_file_path = '/data-store/zengrh/qianyihao/af_ttt_v1/data/thumos14/Thumos14_annotations.json'
out_file_path = '/data-store/zengrh/qianyihao/af_ttt_v1/data/thumos14/Thumos14_annotations.json'

# 读取 aa.json 文件
with open(in_file_path, 'r') as f:
    data = json.load(f)

# 将修改后的内容写回 aa.json 文件
with open(out_file_path, 'w') as f:
    json.dump(data, f, indent=4)

print("Successfully updated aa.json!")
