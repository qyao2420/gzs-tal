
def data_split_dir(file, task, data_split, mode, split_num):
    ## file_dir = "./splits/train_{r1}_test_{r2}/THUMOS14/{mode}/split_{r3}.list"
    # train_100_test_100
    if task == 'close_set':
        if mode == 'train':
            file_dir = file.format(mode='train',r1=100, r2=100, r3=f'{data_split}_{split_num}')
        else:
            file_dir = file.format(mode='test',r1=100, r2=100, r3=f'{data_split}_{split_num}')
    # train_75_test_25
    elif task == 'zero_shot':
        if mode == 'train':
            file_dir = file.format(mode='train',r1=data_split, r2=100-data_split, r3=split_num)
        else:
            file_dir = file.format(mode='test',r1=data_split, r2=100-data_split, r3=split_num)
    # train_75_test_100
    elif task == 'gzs':
        if mode == 'train':
            file_dir = file.format(mode='train',r1=data_split, r2=100, r3=split_num)
        else:
            file_dir = file.format(mode='test',r1=data_split, r2=100, r3=split_num)
        
    return file_dir

