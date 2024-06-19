# freetalk_senior dataset의 
# 1.Training- labeled_data- 1.AI챗봇 label 과 (340851)
# 1.Training- raw_data- 1.AI챗봇 audio 개수가 한 개 차이남. (340850)

# whisper_kor env 필요

import os
import glob
import re
from tqdm import tqdm
from datasets import load_dataset, Features, Audio

# File path
file_dir = "/data/freetalk_senior/1.Training/"
raw_data_files = sorted(glob.glob(os.path.join(file_dir, "raw_data/**/*.zip"), recursive=True))
labeled_data_files_1 = sorted(glob.glob(os.path.join(file_dir, "labeled_data/1.*/**/*.json"), recursive=True))

data_files = {
    "train_input": raw_data_files,    
    "train_label": labeled_data_files_1,  
}

# raw file은 zip 이라 load_dataset 필요
def load_raw_data(data_files=data_files):
    features = Features({
        'audio': Audio()
    })

    # [0:2] = 1.AI chatbot datas
    dataset_input = load_dataset("audiofolder", data_files={"train": data_files["train_input"][0:2]}, 
                                                features=features)
    return dataset_input

def check_data(raw_dataset, labeled_data_files = labeled_data_files_1):
    # audio, label 개수 검사
    pattern1 = r'(\d{2}_F_\d{10}_\d{2})'  # 07_F_1522434093_60
    pattern2 = r'_(\d{5})\.'           # _08580.

    for i, data in enumerate(tqdm(labeled_data_files)):
        raw = raw_dataset['train'][i]['audio']['path']
        
        match1 = re.search(pattern1, raw)
        match_json1 = re.search(pattern1, data)
        if (match1.group() != match_json1.group()):
            msg = f"{i} is not matching"
            print(msg)
            os.system("msg > unmatched_data.out")

        match2 = re.search(pattern2, raw)
        match_json2 = re.search(pattern2, data)
        if (match2.group() != match_json2.group()):
            msg = f"{i} is not matching"
            print(msg)
            os.system("msg > unmatched_data.out")

def main():
    raws = load_raw_data()
    print(raws)
    check_data(raws)

    return

if __name__ == "__main__":
    main()