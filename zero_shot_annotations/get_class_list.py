# merge_and_sort.py

def merge_and_sort_lists(file1, file2, output_file):
    # 读取两个文件的内容
    with open(file1, 'r', encoding='utf-8') as f1, open(file2, 'r', encoding='utf-8') as f2:
        labels1 = f1.readlines()
        labels2 = f2.readlines()

    # 合并并去掉重复项
    all_labels = set(labels1 + labels2)

    # 去掉每个标签的换行符，并排序
    sorted_labels = sorted(label.strip() for label in all_labels)

    # 将排序后的标签写入输出文件
    with open(output_file, 'w', encoding='utf-8') as output:
        for label in sorted_labels:
            output.write(label + '\n')

# 使用示例
if __name__ == '__main__':
    merge_and_sort_lists('/data-store/zengrh/qianyihao/Code/af_ttt_v6/splits/train_75_test_25/ActivityNet/train/split_0.list', '/data-store/zengrh/qianyihao/Code/af_ttt_v6/splits/train_75_test_25/ActivityNet/test/split_0.list', 'merged_sorted.list')
