import json

aa_json = '/data-store/zengrh/qianyihao/af_ttt_v1/data/anet13/anet1.3_i3d_filtered.json'
bb_json = '/data-store/zengrh/qianyihao/af_ttt_v1/data/anet13/anet1.3_clip_annotations.json'
cc_json = '/data-store/zengrh/qianyihao/af_ttt_v1/data/anet13/anet1.3_clip_filtered.json'

# 读取 aa.json 和 bb.json
with open(aa_json, 'r') as f:
    aa_data = json.load(f)

with open(bb_json, 'r') as f:
    bb_data = json.load(f)

# 获取 aa.json 中存在的视频 ID
aa_video_ids = set(aa_data['database'].keys())
aa_video_ids = {'v_'+video_id for video_id in aa_video_ids}

'''print(aa_video_ids)
print(len(aa_video_ids))'''

# 筛选 bb.json 中存在于 aa.json 的视频及其内容
'''filtered_data = {
    "database": {
        video_id: bb_data['database'][video_id]
        for video_id in bb_data['database'] if video_id in aa_video_ids
    }
}'''
filtered_data = bb_data
filtered_data['database'] = {
    video_id: bb_data['database'][video_id]
    for video_id in bb_data['database'] if video_id in aa_video_ids
}

# 将筛选后的数据写入 cc.json
with open(cc_json, 'w') as f:
    json.dump(filtered_data, f, indent=4)

print("筛选完成，结果已写入 cc.json")
