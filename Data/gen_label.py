import os

img_dir = r'D:/ZJU/research/datasets/retrieval/Sketchy (low)/Sketchy/ambiguity'

output_cate = r'D:\ZJU\research\datasets\retrieval\Sketchy (low)\Sketchy\zeroshot2\cname_cid.txt'
output_label = r'D:\ZJU\research\datasets\retrieval\Sketchy (low)\Sketchy\zeroshot2\all_photo_filelist_train.txt'

category_list = []
label_list = []
category = 0
for dir in os.listdir(img_dir):

    img_dir_dir = os.path.join(img_dir, dir)
    category_list.append(dir + ' ' + str(category) + '\n')

    for img in os.listdir(img_dir_dir):
        img_path = os.path.join(img_dir_dir.split('/')[-1], img)

        label_list.append(img_path + ' ' + str(category) + '\n')
    category += 1

print(category_list)
print(label_list)

# with open(output_cate, "w") as f:
#     for i in category_list:
#         f.write(i)

with open(output_label, "w") as f:
    for i in label_list:
        f.write(i)
