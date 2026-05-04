#!/usr/bin/python2.7
# adapted from: https://github.com/colincsl/TemporalConvolutionalNetworks/blob/master/code/metrics.py

from ast import Str
import numpy as np
import argparse
from itertools import groupby
import os
import json


def read_file(path):
    with open(path, 'r') as f:
        content = f.read()
        f.close()
    return content


def get_labels_start_end_time(frame_wise_labels, bg_class=["background"]):
    labels = []
    starts = []
    ends = []
    last_label = frame_wise_labels[0]
    if frame_wise_labels[0] not in bg_class:
        labels.append(frame_wise_labels[0])
        starts.append(0)
    for i in range(len(frame_wise_labels)):
        if frame_wise_labels[i] != last_label:
            if frame_wise_labels[i] not in bg_class:
                labels.append(frame_wise_labels[i])
                starts.append(i)
            if last_label not in bg_class:
                ends.append(i)
            last_label = frame_wise_labels[i]
    if last_label not in bg_class:
        ends.append(i + 1)
    return labels, starts, ends


def levenstein(p, y, norm=False):
    m_row = len(p)    
    n_col = len(y)
    D = np.zeros([m_row+1, n_col+1], np.float64)
    for i in range(m_row+1):
        D[i, 0] = i
    for i in range(n_col+1):
        D[0, i] = i

    for j in range(1, n_col+1):
        for i in range(1, m_row+1):
            if y[j-1] == p[i-1]:
                D[i, j] = D[i-1, j-1]
            else:
                D[i, j] = min(D[i-1, j] + 1,
                              D[i, j-1] + 1,
                              D[i-1, j-1] + 1)
    
    if norm:
        score = (1 - D[-1, -1]/max(m_row, n_col)) * 100
    else:
        score = D[-1, -1]

    return score


def edit_score(recognized, ground_truth, norm=True, bg_class=["background"]):
    # modified edit_score to remove consecutive duplicates after filtering out background
    recognized_no_bg = [a for a in recognized if not a in bg_class]
    ground_truth_no_bg = [a for a in ground_truth if not a in bg_class]
    P = [k for k, g in groupby(recognized_no_bg)]
    Y = [k for k, g in groupby(ground_truth_no_bg)]
    #P, _, _ = get_labels_start_end_time(recognized, bg_class)
    #Y, _, _ = get_labels_start_end_time(ground_truth, bg_class)
    return levenstein(P, Y, norm)

def edit_score_json(segments, ground_truth, norm=True, bg_class=["background"]):
    # modified edit_score to remove consecutive duplicates after filtering out background
    # recognized_no_bg = [a for a in recognized if not a in bg_class]
    recognized_no_bg = []
    for i in range(len(segments)):
        recognized_no_bg.append(segments[i]['label'])
    ground_truth_no_bg = [a for a in ground_truth if not a in bg_class]
    P = [k for k, g in groupby(recognized_no_bg)]
    Y = [k for k, g in groupby(ground_truth_no_bg)]
    #P, _, _ = get_labels_start_end_time(recognized, bg_class)
    #Y, _, _ = get_labels_start_end_time(ground_truth, bg_class)
    return levenstein(P, Y, norm)

def f_score(recognized, ground_truth, overlap, bg_class=["background"]):
    p_label, p_start, p_end = get_labels_start_end_time(recognized, bg_class)
    y_label, y_start, y_end = get_labels_start_end_time(ground_truth, bg_class)

    tp = 0
    fp = 0

    hits = np.zeros(len(y_label))

    for j in range(len(p_label)):
        intersection = np.minimum(p_end[j], y_end) - np.maximum(p_start[j], y_start)
        union = np.maximum(p_end[j], y_end) - np.minimum(p_start[j], y_start)
        IoU = (1.0*intersection / union)*([p_label[j] == y_label[x] for x in range(len(y_label))])
        # Get the best scoring segment
        idx = np.array(IoU).argmax()

        if IoU[idx] >= overlap and not hits[idx]:
            tp += 1
            hits[idx] = 1
        else:
            fp += 1
    fn = len(y_label) - sum(hits)
    return float(tp), float(fp), float(fn)

def f_score_json(segments, ground_truth, overlap, bg_class=["background"], fps=15):
    # p_label, p_start, p_end = get_labels_start_end_time(recognized, bg_class)
    p_label = []
    p_start = []
    p_end = []
    for i in range(len(segments)):
        p_label.append(segments[i]['label'])
        p_start.append(int(segments[i]['segment'][0]*fps))
        p_end.append(int(segments[i]['segment'][1]*fps))
    y_label, y_start, y_end = get_labels_start_end_time(ground_truth, bg_class)

    tp = 0
    fp = 0

    hits = np.zeros(len(y_label))

    for j in range(len(p_label)):
        intersection = np.minimum(p_end[j], y_end) - np.maximum(p_start[j], y_start)
        union = np.maximum(p_end[j], y_end) - np.minimum(p_start[j], y_start)
        IoU = (1.0*intersection / union)*([p_label[j] == y_label[x] for x in range(len(y_label))])
        # Get the best scoring segment
        idx = np.array(IoU).argmax()

        if IoU[idx] >= overlap and not hits[idx]:
            tp += 1
            hits[idx] = 1
        else:
            fp += 1
    fn = len(y_label) - sum(hits)
    return float(tp), float(fp), float(fn)

def evaluate_json(dataset, json_path, split, data_dir, fps):
    # 真实标签目录
    ground_truth_path = data_dir + dataset + '/groundTruth/'
    # 预测标签目录
    recog_path = json_path
    
    with open(recog_path, 'r') as f:
        data = json.load(f)

    # .txt / .bundle 获取文件列表
    file_list = data_dir + dataset + "/splits/test.split" + split + ".bundle"
    list_of_videos = read_file(file_list).split('\n')[:-1]  # 文件列表      
    
    overlap = [.1, .25, .5]
    tp, fp, fn = np.zeros(3), np.zeros(3), np.zeros(3)
    bg_class = ['BG'] if dataset in ['ptg', 'coffee', 'tea', 'pinwheels', 'oatmeal', 'quesadilla'] else ['background']
    edit = 0
    
    # 遍历每个视频并生成标签
    for video_name, segments in data['results'].items():
        if not video_name.endswith('.txt'):
            video_name = video_name + '.txt'
        gt_file = ground_truth_path + video_name
        gt_content = read_file(gt_file).split('\n')[0:-1]
        
        edit += edit_score_json(segments, gt_content, bg_class=bg_class)
        
        for s in range(len(overlap)):
            tp1, fp1, fn1 = f_score_json(segments, gt_content, overlap[s], bg_class, fps)
            tp[s] += tp1
            fp[s] += fp1
            fn[s] += fn1
            
    edit = (1.0*edit)/len(list_of_videos)
    res_list = [edit]

    #print("Acc: %.4f" % (100*float(correct)/total))
    #print('Edit: %.4f' % ((1.0*edit)/len(list_of_videos)))
    for s in range(len(overlap)):
        precision = tp[s] / float(tp[s]+fp[s])
        recall = tp[s] / float(tp[s]+fn[s])
    
        f1 = 2.0 * (precision*recall) / (precision+recall)

        f1 = np.nan_to_num(f1)*100
        #print('F1@%0.2f: %.4f' % (overlap[s], f1))
        res_list.append(f1)
    # print(dataset, ' '.join(['{:.2f}'.format(r) for r in res_list]))
    result_metrics = {'Edit': f"{res_list[-4]:.2f}", 'F1@10': f"{res_list[-3]:.2f}", 'F1@25': f"{res_list[-2]:.2f}", 'F1@50': f"{res_list[-1]:.2f}"}
    # print(result_metrics)
    
    return res_list

# 数据集；结果所在目录；
def evaluate(gt_dir, dataset, result_dir, split, data_dir=None):
    # 真实标签目录
    ground_truth_path = gt_dir
    # 预测标签目录
    recog_path = result_dir
    
    '''# .txt / .bundle 获取文件列表
    file_list = data_dir + dataset + "/splits/test.split" + split + ".bundle"
    list_of_videos = read_file(file_list).split('\n')[:-1]  # 文件列表'''
    
    list_of_videos = []
    for filename in os.listdir(result_dir):
        list_of_videos.append(filename)

    overlap = [.1, .25, .5]
    tp, fp, fn = np.zeros(3), np.zeros(3), np.zeros(3)

    correct = 0
    total = 0
    correct_wo_bg = 0
    total_wo_bg = 0
    edit = 0
    bg_class = ['BG'] if dataset in ['ptg', 'coffee', 'tea', 'pinwheels', 'oatmeal', 'quesadilla'] else ['background']

    for vid in list_of_videos:
        if not vid.endswith('.txt'):
            vid = vid + '.txt'
        gt_file = ground_truth_path + vid
        gt_content = read_file(gt_file).split('\n')[0:-1]
        
        recog_file = recog_path + vid
        recog_content = read_file(recog_file).split('\n')[0:-1]
        
        # 用BG_class填充预测标签的长度
        while len(recog_content) < len(gt_content):
            recog_content.append(bg_class[0])

        for i in range(len(gt_content)):
            if gt_content[i] not in bg_class:
                total_wo_bg += 1
                if gt_content[i] == recog_content[i]:
                    correct_wo_bg += 1
            total += 1
            if gt_content[i] == recog_content[i]:
                correct += 1
        
        edit += edit_score(recog_content, gt_content, bg_class=bg_class)

        for s in range(len(overlap)):
            tp1, fp1, fn1 = f_score(recog_content, gt_content, overlap[s], bg_class)
            tp[s] += tp1
            fp[s] += fp1
            fn[s] += fn1
            
    acc = 100*float(correct)/total
    acc_wo_bg = 100*float(correct_wo_bg)/total_wo_bg
    # print(str(correct) + ' ' + str(total) + ' ' + str(correct_wo_bg) + ' ' + str(total_wo_bg))
    edit = (1.0*edit)/len(list_of_videos)
    res_list = [acc, acc_wo_bg, edit]

    #print("Acc: %.4f" % (100*float(correct)/total))
    #print('Edit: %.4f' % ((1.0*edit)/len(list_of_videos)))
    for s in range(len(overlap)):
        precision = tp[s] / float(tp[s]+fp[s])
        recall = tp[s] / float(tp[s]+fn[s])
    
        f1 = 2.0 * (precision*recall) / (precision+recall)

        f1 = np.nan_to_num(f1)*100
        #print('F1@%0.2f: %.4f' % (overlap[s], f1))
        res_list.append(f1)
    # print(dataset, ' '.join(['{:.2f}'.format(r) for r in res_list]))
    result_metrics = {'Acc': acc,  'Acc-bg': acc_wo_bg, 'Edit': edit, 
                    'F1@10': res_list[-3], 'F1@25': res_list[-2], 'F1@50': res_list[-1]}
    # print(result_metrics)
    
    return res_list
    '''result_path = os.path.join(recog_path, 'split'+split+'.eval.json')
    with open(result_path, 'w') as fw:
        json.dump(result_metrics, fw, indent=4)'''

def generate_frame_level_labels(json_file, txt_dir, gt_dir, bg_class, fps):
    # 读取 JSON 文件
    with open(json_file, 'r') as f:
        data = json.load(f)

    # 遍历每个视频并生成标签
    for video_name, segments in data['results'].items():
        # 找到视频的总时长
        '''total_duration = max(segment['segment'][1] for segment in segments)
        total_frames = int(total_duration * fps)'''
        total_frames = len(read_file(gt_dir + video_name + '.txt').split('\n')[0:-1])

        # 初始化每帧的标签为 'background'
        labels_per_frame = bg_class * total_frames
        score_per_frame = [0] * total_frames
        
        # 填充标签
        for segment in segments:
            start_time, end_time = segment['segment']
            score = segment['score']
            label = segment['label']

            # 转换时间到帧
            start_frame = int(start_time * fps)
            end_frame = int(end_time * fps)

            # 填充标签，考虑重叠部分
            for frame in range(max(0, start_frame), min(end_frame, total_frames)):
                # 如果当前帧是 'background' 或者当前标签的分数更高，则更新标签
                if labels_per_frame[frame] == bg_class or score > score_per_frame[frame]:
                    labels_per_frame[frame] = label
                    score_per_frame[frame] = score

        # 写入到 TXT 文件
        if not os.path.exists(txt_dir):
            os.makedirs(txt_dir)
        output_file = txt_dir + video_name + '.txt'  # 以视频名称作为文件名
        with open(output_file, 'w') as f:
            for label in labels_per_frame:
                f.write(label + '\n')

def main():
    
    parser = argparse.ArgumentParser()

    # 前**个参数需要修改
    parser.add_argument('--dataset', default='gtea')  # 修改
    parser.add_argument('--version', default='_clip')
    parser.add_argument('--fps', default="15", type=int)
    parser.add_argument('--bg_class', default=['background'])
    parser.add_argument('--split', default='1')
    #parser.add_argument('--data_dir', default='/data-store/zengrh/qianyihao/Ego/ProTAS-main/data/')
    
    args = parser.parse_args()
    
    result_dir = '/data-store/zengrh/qianyihao/Code/af_ttt_tpt/results/gtea_iv_50_train_iv' + '/'   # 修改
    #result_dir = '/data-store/zengrh/qianyihao/GAP-main/results/gtea7/Train_gtea-10_50gzs' + '/'
    frame_dir = './Eval_tas/frame_level_results/' + args.dataset + args.version + '/'
    #gt_dir = args.data_dir + args.dataset + '/groundTruth/'
    gt_dir = './Eval_tas/groundTruth/' + args.dataset + '/'
    if not os.path.exists(frame_dir):
        os.makedirs(frame_dir)
    
    epoch = 10
    step = 5

    best_epoch = 1
    max_acc = 0
    max_res_list = [0, 0, 0, 0, 0, 0]

    best_epoch_bg = 1
    max_acc_bg = 0
    max_res_list_bg = [0, 0, 0, 0, 0, 0]
    while True:
        json_path = result_dir + f'epoch_{epoch:03}.pth.json'
        #json_path = result_dir + f'detection_validation_{epoch}_raw.json'
        
        if os.path.exists(json_path):
            try:
                generate_frame_level_labels(json_path, frame_dir, gt_dir, args.bg_class, args.fps)
                # print('epoch: ' + str(epoch))
                res_list = evaluate(gt_dir, args.dataset, frame_dir, args.split)
            except Exception as e:
                epoch += step
                continue
            
            # best_acc
            if res_list[-6] > max_acc:
                max_acc = res_list[-6]
                max_res_list = res_list
                best_epoch = epoch
                
            # best_acc-bg
            if res_list[-5] > max_acc_bg:
                max_acc_bg = res_list[-5]
                max_res_list_bg = res_list
                best_epoch_bg = epoch
        else:
            break
        epoch += step
    # ->frame_result后计算
    print('total_epochs: ' + str(epoch-1))
    print('acc acc-bg edit f1@10 f1@25 f1@50')
    print(best_epoch, args.dataset, ' '.join(['{:.2f}'.format(r) for r in max_res_list]))
    print(best_epoch_bg, args.dataset, ' '.join(['{:.2f}'.format(r) for r in max_res_list_bg]))
    '''result_metrics = {'best_epoch': best_epoch, 
                    'Acc': f"{max_res_list[-6]:.2f}",  
                    'Acc-bg': f"{max_res_list[-5]:.2f}", 
                    'Edit': f"{max_res_list[-4]:.2f}", 
                    'F1@10': f"{max_res_list[-3]:.2f}", 
                    'F1@25': f"{max_res_list[-2]:.2f}", 
                    'F1@50': f"{max_res_list[-1]:.2f}"}
    print(result_metrics)
    
    # 直接.json计算
    best_json_path = result_dir + 'proposal_pred_test_' + str(best_epoch) + '.json'
    segment_res_list = evaluate_json(args.dataset, best_json_path, args.split, args.data_dir, args.fps)
    segment_result_metrics = {'best_epoch': best_epoch,  
                    'Edit': f"{segment_res_list[-4]:.2f}", 
                    'F1@10': f"{segment_res_list[-3]:.2f}", 
                    'F1@25': f"{segment_res_list[-2]:.2f}", 
                    'F1@50': f"{segment_res_list[-1]:.2f}"}
    print(segment_result_metrics)'''

if __name__ == '__main__':
    main()
