import os

dir_path = 'train_pic/neymar'

def rename_pic(dir_path):
    i = 0
    for pic in os.listdir(dir_path):
        new_name = 'pic_' + str(i)+'.jpg'
        src_path = str(dir_path)+'/'+pic
        dst_path = str(dir_path)+'/'+new_name
        os.rename(src_path, dst_path)
        i += 1

if __name__ == '__main__':
    rename_pic(dir_path)