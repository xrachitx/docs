import os 
import shutil
import csv
from tqdm import tqdm

base_dir = "./data/"
img_dir = base_dir+"Images/"
gt_dir = base_dir+"Skeletons/"
print(img_dir)
categories = os.listdir(img_dir)
gt_sav_dir = "./save/"
txt_file = []

for category in categories:
    save_cat = gt_sav_dir+"Skeletons/"+category+"/"
    if not os.path.exists(save_cat): 
        os.makedirs(save_cat)
    image_category_dir = img_dir+category+"/"
    gt_category_dir = gt_dir+category+"/"
    images = os.listdir(image_category_dir)
    for idx1 in tqdm(range(len(images))):
        image1 = image_category_dir+ images[idx1]
        gt1 = gt_category_dir+ images[idx1]
        for idx2 in range(idx1+1,len(images)):
            image2 = image_category_dir+ images[idx2]
            gt2 = gt_category_dir+ images[idx2]
            
            shutil.copy(gt1,save_cat+images[idx1][:-4]+"_"+images[idx2])
            shutil.copy(gt2,save_cat+images[idx2][:-4]+"_"+images[idx1])
            txt_file.append([images[idx1][:-4],images[idx2][:-4], images[idx1][:-4]+"_"+images[idx2][:-4],images[idx2][:-4]+"_"+images[idx1][:-4]])

with open("f.txt", 'w') as csvfile: 
    # creating a csv writer object 
    csvwriter = csv.writer(csvfile) 
        
    # writing the fields 
    # csvwriter.writerow(fields) 
        
    # writing the data rows 
    csvwriter.writerows(txt_file)