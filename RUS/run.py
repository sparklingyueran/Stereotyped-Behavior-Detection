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
model_name="..."
processor_name=None
device="cpu"

if processor_name is None:
    processor_name = model_name
# processor = Blip2Processor.from_pretrained(processor_name)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

Q2 = "Is the child doing repetitive actions?"
prompt = Q2


DATA_ROOT = '...'
GROUP_NAME = '...'

train_txt_file = os.path.join(DATA_ROOT,GROUP_NAME+'-label.txt') 
feature_save_folder = os.path.join(DATA_ROOT,GROUP_NAME+'-eilev-feature')



if not os.path.exists(feature_save_folder):
    print('save path not exist')
    os.mkdir(feature_save_folder)

train_dataset = VideoDataset(train_txt_file, transform, prompt, device)
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=0)
print('finish loading data')


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


        video_path_str = str(video_paths[0])
        output_name = ((str(video_paths[0]).split('/'))[-1])[:-4]+'.npy'
        output_path = os.path.join(feature_save_folder,output_name)
        np.save(output_path, outputs_array)

        print(outputs.shape,labels,video_paths)
