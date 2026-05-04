import json

def write_results_json(results, class_dict, out_file_path, map=None, avg=None, tiou=None):
    
    reversed_class_dict = {v: k for k, v in class_dict.items()}

    # 创建最终的结果结构
    if tiou is None:
        formatted_results = {
            "version": "VERSION 1.0",
            "results": {}
        }
    else:
        # 构建结果字符串
        map_results = []
        for tiou_val, mAP_val in zip(tiou, map):
            map_results.append(f"|tIoU = {tiou_val:.2f}: mAP = {mAP_val*100:.2f} (%)")
        map_results.append(f"Average mAP: {avg*100:.2f} (%)")

        formatted_results = {
            "version": "VERSION 1.0", 
            "eval": map_results, 
            "results": {}
        }

    # 遍历 'video-id' 来填充结果
    for i in range(len(results['video-id'])):
        video_id = results['video-id'][i]
        
        # 创建当前视频的预测字典
        prediction = {
            "score": float(results['score'][i]),
            "segment": [float(results['t-start'][i]), float(results['t-end'][i])],
            "label": reversed_class_dict.get(results['label'][i])
        }
        
        # 如果视频ID不在结果中则初始化一个空列表
        if video_id not in formatted_results["results"]:
            formatted_results["results"][video_id] = []
        
        # 将当前预测添加到视频的预测列表中
        formatted_results["results"][video_id].append(prediction)

    print(f'num_of_video: {len(formatted_results["results"])}')
    
    # 将结果保存为 JSON 文件
    with open(out_file_path, 'w') as json_file:
        json.dump(formatted_results, json_file, indent=4)

    print(f'结果已保存到 {out_file_path}')

if __name__ == "__main__":

    results = {
        'video-id': ['video_test_0000292', 'video_test_0000292'],  # 示例视频ID，可以包含多个视频
        't-start': [58.27053464000001, 59.9478064],  # 示例开始时间
        't-end': [61.102297359999994, 63.7503772],  # 示例结束时间
        'label': [0, 1],  # 示例标签
        'score': [1.00808375e-05, 1.01391215e-05]  # 示例分数
    }

    class_dict = {'GolfSwing':0, 'VolleyballSpiking':1}

    out_file_path = '/data-store/zengrh/qianyihao/af_ttt_v1/results/test.json'

    write_results_json(results, class_dict, out_file_path)
