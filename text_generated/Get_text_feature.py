import clip
import torch    
import os, json
import numpy as np  


device="cuda" if torch.cuda.is_available() else "cpu"
def feature_extractor(text):
    tokenized_text = clip.tokenize(text,truncate=True).cuda()
    with torch.no_grad():
        text_features = model.encode_text(tokenized_text)
        text_features /= text_features.norm(dim=-1, keepdim=True)
    return text_features

def get_text(dataset:str):
    description = None
    if dataset=='cifar100':
        text_json_path = '/mnt/Data/wangshilong/SAVC/text_generated/cifar100_llm_outputs.json'
    elif dataset=='imagenet':
        text_json_path = '/mnt/Data/wangshilong/SAVC/text_generated/imagenet_llm_outputs.json'
    else:
        text_json_path='/mnt/Data/wangshilong/SAVC/text_generated/cub200_llm_outputs.json'

    if os.path.exists(text_json_path):
        with open(text_json_path, 'r') as f:
             description = json.load(f)
    else:
        print("No text description found!")
    return description


if __name__ == '__main__':

    dataset="mini_imagenet" #cub200, mini_imagenet, cifar100

    model,_=clip.load("ViT-B/32",device=device)
    model.eval()

    texts=get_text(dataset) 

    feature=[]
    for i in range(len(texts)):
        feat=feature_extractor(texts[str(i)])
        feature.append(feat.cpu().numpy())
    
    all_class_features=np.stack(feature,axis=0)

    save_path=f'/mnt/Data/wangshilong/SAVC/text_generated/{dataset}_all_classText_feature.npy'    
    np.save(save_path,all_class_features)   
    print(f'{dataset} text feature saved in {save_path} Successfully!')