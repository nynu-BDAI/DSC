import os
import pickle
import json
from torchvision.datasets.utils import check_integrity  
from wordnet import *
from text_generater import *

api_key="" 
base_url=""  

class LLMPrompt:
    
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name
        self.root = "/mnt/Data/wangshilong/self_datasets/"
        if dataset_name == 'cifar100':
            self.base_folder = 'cifar-100-python'
            self.meta = {
            'filename': 'meta',
            'key': 'fine_label_names',
            'md5': '7973b15100ade9c7d40fb424638fde48',
            }
            self.class_to_idx_cifar100()

        elif dataset_name == 'cub200':
            self.base_folder='CUB_200_2011'
            self.class_to_idx_cub200()
        elif dataset_name == 'mini_imagenet':
            self.base_folder='miniimagenet'
            self.class_to_idx_miniimagenet()

    def class_to_idx_cifar100(self):
        path = os.path.join(self.root, self.base_folder, self.meta['filename'])
        if not check_integrity(path, self.meta['md5']):
            raise RuntimeError('Dataset metadata file not found or corrupted.' +
                               ' You can use download=True to download it')
        with open(path, 'rb') as infile:
            data = pickle.load(infile, encoding='latin1')
            self.classes = data[self.meta['key']]
            self.class_to_idx = { i:_class for i, _class in enumerate(self.classes)}
        return self.class_to_idx
    
    def class_to_idx_cub200(self):
        path = os.path.join(self.root, self.base_folder, 'classes_mapping.txt')
        self.class_to_idx = {}
        with open(path, 'r') as f:
            for line in f:
                idx, name = line.strip().split(' ', 1)
                self.class_to_idx[int(idx)] = name
        return self.class_to_idx
    
    def class_to_idx_miniimagenet(self):
        path = os.path.join(self.root, self.base_folder, 'id2textlabel.json')
        self.class_to_idx = {}
        with open(path, 'r') as f:
            id2textlabel = json.load(f)
        for k, v in id2textlabel.items():
            self.class_to_idx[int(k)] = v.split(',')[0]  

    def build_prompt_from_wordnet(self, class_name_dict, manual_override=None,use_wordnet=True):
        prompts,metas=build_prompts_for_dict(class_name_dict, manual_override=manual_override,use_wordnet=use_wordnet)
        return prompts,metas

    def get_DescriptionGenerator(self, api_key, base_url):  
        return DescriptionGenerator(api_key=api_key, base_url=base_url)
    
    
if __name__ == "__main__":
    dataset_name='mini_imagenet'  # 'cifar100', 'cub200', 'mini_imagenet'
    if dataset_name=='cifar100':
        LLMprompt = LLMPrompt(dataset_name=dataset_name)
        cifar_dict= LLMprompt.class_to_idx
        cifar100_prompts,_=LLMprompt.build_prompt_from_wordnet(cifar_dict, manual_override=None, use_wordnet=False)
        text_generter = LLMprompt.get_DescriptionGenerator(api_key=api_key, base_url=base_url)
        
        results = {}
        for class_id, prompt in cifar100_prompts.items():
            # if class_id==2:
            #     break
            print(f"正在生成类别 {class_id} 的描述...")
            output = text_generter.get_text_answer(prompt, system=None)
            results[class_id] = output
            
        # 保存到 JSON 文件
        output_file = "cifar100_llm_outputs.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=4)

        print(f"已将所有类别的 LLM 输出保存到 {output_file}")

    elif dataset_name=='cub200':
        LLMprompt = LLMPrompt(dataset_name=dataset_name)
        cub_dict= LLMprompt.class_to_idx
        cub200_prompts,_=LLMprompt.build_prompt_from_wordnet(cub_dict, manual_override=None,use_wordnet=False)
        text_generter = LLMprompt.get_DescriptionGenerator(api_key=api_key, base_url=base_url)
        
        results = {}
        for class_id, prompt in cub200_prompts.items():
            # if class_id==2:
            #     break
            print(f"正在生成类别 {class_id} 的描述...")
            output = text_generter.get_text_answer(prompt, system=None)
            results[class_id] = output
            
        # 保存到 JSON 文件
        output_file = "cub200_llm_outputs.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=4)
        print(f"已将所有类别的 LLM 输出保存到 {output_file}")

    elif dataset_name=='mini_imagenet':
        LLMprompt= LLMPrompt(dataset_name=dataset_name)  
        mini_dict= LLMprompt.class_to_idx
        mini_prompts,_=LLMprompt.build_prompt_from_wordnet(mini_dict, manual_override=None,use_wordnet=False)
        text_generter = LLMprompt.get_DescriptionGenerator(api_key=api_key, base_url=base_url)
        
        results = {}
        for class_id, prompt in mini_prompts.items():
            # if class_id==2:
            #     break
            print(f"正在生成类别 {class_id} 的描述...")
            output = text_generter.get_text_answer(prompt, system=None)
            results[class_id] = output
            
        # 保存到 JSON 文件
        output_file = "mini_imagenet_llm_outputs.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=4)
        print(f"已将所有类别的 LLM 输出保存到 {output_file}")
    


   