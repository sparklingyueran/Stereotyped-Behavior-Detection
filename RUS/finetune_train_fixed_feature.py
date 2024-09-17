import torch
from torch.utils.data import Dataset, DataLoader
import pickle
from torchvision import transforms
from finetune_dataset_feature import *
from transformers import Blip2Processor
from finetune_model_feature import Blip2FinetuneFeature
import datetime
from PIL import Image

# print('Now',datetime.datetime.now().strftime("%Y%m%d%H%M%S"))


# 0.Processor
model_name="/home/pany/WorkSpace/EILEV/eilev-blip2-flan-t5-xl"
processor_name=None
device="cpu"

if processor_name is None:
    processor_name = model_name
# processor = Blip2Processor.from_pretrained(processor_name)

# 1. 定义transform
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

Q1 = "What is the child doing?"
Q2 = "Is the child doing repetitive actions?"
prompt = Q2

# 2. 创建数据集和数据加载器


#####以下是对SSBD-Pro拓展原始数据集
DATA_ROOT = '/AsdData/SSBD-Pro/Results'
# GROUP_NAME = 'SSBDPro-rawFPS-3s-1soverlap'
# GROUP_NAME = 'SSBDPro-rawFPS-3s-0soverlap'
# GROUP_NAME = 'SSBDPro-rawFPS-2s-0soverlap'
# GROUP_NAME = 'SSBDPro-rawFPS-3s-2soverlap'
GROUP_NAME = 'SSBDPro-rawFPS-1s-0soverlap'

# GROUP_NAME = 'SSBDPro-rawFPS-2s-1soverlap'
train_txt_file = os.path.join(DATA_ROOT,GROUP_NAME+'-label.txt')  # 替换为你的训练数据的文件路径
# test_txt_file = '/AsdData/SSBD-Pro/Results/SSBDPro-rawFPS-3s-0soverlap-label.txt'  # 替换为你的训练数据的文件路径
feature_save_folder = os.path.join(DATA_ROOT,GROUP_NAME+'-eilev-feature')
#####以上是对SSBD-Pro拓展原始数据集



#####以下是对SSBD原始数据集

# DATA_ROOT = '/AsdData/SSBD/Results'
# GROUP_NAME = 'SSBD-rawFPS-1s-0soverlap'
# GROUP_NAME = 'SSBD-rawFPS-3s-0soverlap'
# GROUP_NAME = 'SSBD-rawFPS-2s-0soverlap'
# GROUP_NAME = 'SSBD-rawFPS-3s-1soverlap'
# GROUP_NAME = 'SSBD-rawFPS-3s-2soverlap'
# GROUP_NAME = 'SSBD-rawFPS-2s-1soverlap'
# train_txt_file = os.path.join(DATA_ROOT,GROUP_NAME+'-label.txt')  # 替换为你的训练数据的文件路径
# test_txt_file = '/AsdData/SSBD-Pro/Results/SSBDPro-rawFPS-3s-0soverlap-label.txt'  # 替换为你的训练数据的文件路径
# feature_save_folder = os.path.join(DATA_ROOT,GROUP_NAME+'-eilev-feature')
#####以上是对SSBD原始数据集

#######以下是对kling数据集
# train_txt_file = '/AsdData/Kling/Kling-raw-speedup1.8-3s-label/kling-yes-label.txt'
# feature_save_folder = '/AsdData/Kling/Kling-raw-speedup1.8-3s-label-Results/kling-yes-eilev-feature'

# train_txt_file = '/AsdData/Kling/Kling-raw-speedup1.8-3s-label/kling-no-label.txt'
# feature_save_folder = '/AsdData/Kling/Kling-raw-speedup1.8-3s-label-Results/kling-no-eilev-feature'

# train_txt_file = '/AsdData/Kling/Kling-raw-speedup1.8-3s-label/kling-yes-anti-label.txt'
# feature_save_folder = '/AsdData/Kling/Kling-raw-speedup1.8-3s-label-Results/kling-yes-anti-eilev-feature'

# train_txt_file = '/AsdData/Kling/Kling-raw-speedup1.8-3s-label/kling-yes-notsure-label.txt'
# feature_save_folder = '/AsdData/Kling/Kling-raw-speedup1.8-3s-label-Results/kling-yes-notsure-eilev-feature'
#######以上是对kling数据集


############################################################################################
#==========================================================================================#
############################################################################################

print(feature_save_folder)


if not os.path.exists(feature_save_folder):
    print('save path not exist')
    os.mkdir(feature_save_folder)

# 创建训练数据集和加载器
train_dataset = VideoDataset(train_txt_file, transform, prompt, device)
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=0)
print('finish loading data')
# 创建测试数据集和加载器
# test_dataset = VideoDataset(test_txt_file, transform, prompt, device)
# test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)



# 3. 模型
# 3.1 获取模型
model = VideoBlipForConditionalGeneration.from_pretrained(model_name)
# model = Blip2FinetuneFeature(model).to(device)
model = model.to(device)
model.eval()


num_epochs = 1

print('finish loading model')
current_time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
for epoch in range(num_epochs):
    print('epoch',epoch)
    for i, (inputs_process,inputs, labels,video_paths) in enumerate(train_loader):
        inputs_process.data['pixel_values']=inputs_process.data['pixel_values'].squeeze(0)
        inputs_process.data['input_ids']=inputs_process.data['input_ids'].squeeze(0)
        inputs_process.data['attention_mask'] = inputs_process.data['attention_mask'].squeeze(0)
        model.to(device)

        # outputs = model(inputs_process)
        outputs = model.generate(
            **inputs_process,
            num_beams=4,
            max_new_tokens=128,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.5,
            do_sample=True,
        )

        outputs_array = outputs.detach().cpu().numpy()
        # print(type(video_paths[0]))
        # print(str(video_paths[0]))


        video_path_str = str(video_paths[0])
        output_name = ((str(video_paths[0]).split('/'))[-1])[:-4]+'.npy'
        output_path = os.path.join(feature_save_folder,output_name)
        np.save(output_path, outputs_array)

        # print(output_path)
        print(outputs.shape,labels,video_paths)
        # # outputs = model(inputs)[0]


# wandb.finish()

'''
sac eil-env
python samples/finetune_train_fixed_feature.py
'''