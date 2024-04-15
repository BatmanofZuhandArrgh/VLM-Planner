# Util functions
import re
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
import random

def locate_agent(admissible_actions):
    close_actions = [x for x in admissible_actions if 'close ' in x]
    if len(close_actions) == 0:
        return ''
    elif len(close_actions) == 1:
        return close_actions[0].split('close ')[-1]
    else: raise ValueError('Shouldnt be more than 1 close action')

def action_formatting(action_str, admissible_actions, cur_loc, cur_obj):
    '''
    Format action 'go to fridge' to 'go to fridge 1'
        or 'take apple' to 'take apple 1 from {cur_loc}'

    action_str: take apple
    admissible_action: info['admissible_actions'], output of env.step
    cur_loc: current location of the agent, like fridge 1
    cur_obj: current object being held by the agent
    '''
    
    action_str = action_str.strip().lower()
    action_str = re.sub(r'\d+', '', action_str)


    
    #Get all objects admissible
    # pattern = r'(\b\w+\b) (\d+)'
    # all_objects = [re.findall(pattern, x)[0] for x in admissible_actions]
    # print(all_objects)

    if any(keyword in action_str for keyword in ['go to', 'use', 'open', 'close']): 
        #Get all actions admissible similar to x
        all_similar_actions = [x for x in admissible_actions if action_str in x]
    
        if len(all_similar_actions) == 0:
            #Completely error generated plan
            return action_str
        output = random.sample(all_similar_actions, 1)[0]
        
    elif 'take' in action_str:
        #Take is usually not in the perms
        tar = action_str.split(' ')[-1]
        all_similar_actions = [x for x in admissible_actions if action_str in x]

        output = 'take ' + tar + ' 1' +' from ' + cur_loc

    elif 'put' in action_str:
        # It does seem that the plans has coherence, it only puts down what it already has
        #Take is usually not in the perms
        tar = action_str.split(' ')[-1]
        output = 'put ' +  cur_obj + ' in ' + tar + ' 1' #Both in/on works
        
    print('Output plan: ', action_str, output)

    return output

 
# def mapping_admissable_commands(action_str, admissable_command):
def action_mapping(action_str):
    '''
    Mapping action vocab from alfworld==0.2.2 to alfworld==0.3.3
    '''
    if "Navigation" in action_str:
        return action_str.replace('Navigation', 'go to')
    
    elif "PickupObject " in action_str:
        return action_str.replace('PickupObject', 'take')

    elif "PutObject " in action_str:
        return action_str.replace('PutObject', 'put')

    elif "OpenObject " in action_str:
        return action_str.replace('OpenObject', 'open')

    elif "CloseObject " in action_str:
        return action_str.replace('CloseObject', 'close')

    elif "ToggleObjectOn " in action_str:
        return action_str.replace('ToggleObjectOn', 'use')

    elif "ToggleObjectOff " in action_str:
        return action_str.replace('ToggleObjectOff', 'use')

    elif "SliceObject " in action_str:
        return action_str.replace('SliceObject', 'slice')
    # Below is not used by LLM-Planner
    elif "HeatObject " in action_str:
        return action_str.replace('HeatObject', 'heat')
    elif "CoolObject " in action_str:
        return action_str.replace('CoolObject', 'cool')
    elif "CleanObject " in action_str:
        return action_str.replace('CleanObject', 'clean')
    elif "Inventory" in action_str:
        return action_str.replace('Inventory', 'inventory')
    elif "ExamineObject " in action_str:
        return action_str.replace('ExamineObject', 'examine')

    elif "LookObject" in action_str:
        return action_str.replace('LookObject', 'look')
    # else:
    #     return action_str.replace('Navigation', 'go to')
    

def embed_sentence(sentence, model="paraphrase-MiniLM-L6-v2"):
    # From https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
    
    #Mean Pooling - Take attention mask into account for correct averaging
    def mean_pooling(model_output, attention_mask):
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    # Load model from HuggingFace Hub
    tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/paraphrase-MiniLM-L6-v2')
    model = AutoModel.from_pretrained('sentence-transformers/paraphrase-MiniLM-L6-v2')

    # Tokenize sentences
    encoded_input = tokenizer(sentence, padding=True, truncation=True, return_tensors='pt')

    # Compute token embeddings
    with torch.no_grad():
        model_output = model(**encoded_input)

    # Perform pooling
    sentence_embedding = mean_pooling(model_output, encoded_input['attention_mask'])

    # Normalize embeddings
    sentence_embedding = F.normalize(sentence_embedding, p=2, dim=1)

    return sentence_embedding




if __name__ == '__main__':


    # sentence = ['This is an example sentence']

    # print(embed_sentence(sentence))
    print(mapping_action_str('Navigation Batman'))