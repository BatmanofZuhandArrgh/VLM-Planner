# GPT-3 HLP generator

import pandas as pd
import openai
import random
import re

from llm import llm
from ast import literal_eval
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim


ACT_TO_STR = {
    'OpenObject': "Open",
    'CloseObject': "Close",
    'PickupObject': "Pickup",
    'PutObject': "Put",
    'ToggleObjectOn': "Toggle on",
    'ToggleObjectOff': "Toggle off",
    'SliceObject': "Slice",
    'Navigation': "Navigate"
}

TO_BE_REMOVE = ['ToggleObjectOff', "Toggle off"] #Remove because I'm not sure it's in the command

def vlm_detect(model, processor, image,  prompt = "USER: <image>\nWhat objects are in this pic?\nASSISTANT:"):
    #Try to get every objects, even tho it produces more objects
    answer = vlm(model, processor, text=prompt, image=image)
    print(answer)
    pattern =  r'\ba[n]?\s+([\w\s]+?(?=[,.]))' #r'\ba[n]?\s+(\w+)'
    matches = re.findall(pattern, answer.split('ASSISTANT')[-1])
    
    return list(set(matches))
    
class LLM_HLP_Generator():
    def __init__(self, knn_data_path, emb_model_name='paraphrase-MiniLM-L6-v2', debug=False):
        self.sentence_embedder = SentenceTransformer(emb_model_name)
        from transformers import GPT2Tokenizer
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.knn_set = pd.read_pickle(knn_data_path)
        self.debug=debug


    def knn_retrieval(self, curr_task, k):
        # Find K train examples with closest sentence embeddings to test example
        
        traj_emb = self.sentence_embedder.encode(curr_task["task_instr"])
        topK = []
        for idxTrain, trainItem in self.knn_set.iterrows():

            train_emb = self.sentence_embedder.encode(trainItem["task_instr"])

            dist = -1 * cos_sim(traj_emb, train_emb)

            if len(topK) < k:
                topK.append((trainItem["task"], dist))
                topK = sorted(topK, key = lambda x : x[1])
            else:
                if float(dist) < topK[-1][1]:
                    if (trainItem["task"], dist) not in topK:
                        topK.append((trainItem["task"], dist))
                        topK = sorted(topK, key = lambda x : x[1])
                        topK = topK[:k]

        return [entry[0] for entry in topK]


    def generate_prompt(self, curr_task, k, vision = False, removeNav=False, naturalFormat=False, includeLow=False):
        #header
        prompt = "Create a high-level plan for completing a household task using the allowed actions and visible objects."

        if vision:
            prompt = "Create a high-level plan in the format <Allow Action> <Target Object>, <Allow Action> <Target Object>,... for completing a household task using the field-of-view in simulated environment, the allowed actions and visible and other objects in the room."
        
        if naturalFormat:
            prompt += f"\n\n\nAllowed actions: {', '.join([x for x in ACT_TO_STR.values() if x not in TO_BE_REMOVE])}" 
        else:
            prompt += f"\n\n\nAllowed actions: {', '.join([x for x in ACT_TO_STR.keys() if x not in TO_BE_REMOVE])}" 
        
        #prompt += "Valid objects in the environment: f{}"

        # Run KNN retrieval
        knn_retrieved_examples = self.knn_retrieval(curr_task, k)

        # Add in-context examples from knn retrieval
        for retrieved_task in knn_retrieved_examples:
            trainTaskRow = self.knn_set.loc[self.knn_set["task"] == retrieved_task]
            trainTaskRow = trainTaskRow.iloc[0]


            step_list = [literal_eval(listItem) for rowItem in trainTaskRow["gold_traj"] for listItem in rowItem]

            #REMOVE NAVIGATION STEPS if the flag is set
            if removeNav:
                stepListCleaned = []
                for listItem in step_list:
                    if "Navigation" not in listItem:
                        stepListCleaned.append(listItem)
                step_list = stepListCleaned
            
            # Format action names to be more natural
            if naturalFormat:
                stepListCleaned = []
                for listItem in step_list:
                    listItem = list(listItem)
                    act_str = ACT_TO_STR[listItem[0]] 
                    listItem[0] = act_str
                    stepListCleaned.append(tuple(listItem))
                step_list = stepListCleaned
            
            # Split past and next plans randomly
            planSplit = random.sample(range(len(step_list)),1)[0]
            
            step_list = [x for x in step_list if x[0] not in TO_BE_REMOVE]

            # In-context examples components
            high_level_str = str(trainTaskRow["task_instr"])
            step_by_step_str = '. '.join(trainTaskRow["step_instr"])
            past_plan_str = self.format_plan_str(step_list[:planSplit])
            next_plans_str = self.format_plan_str(step_list[planSplit:])
            in_context_obj_str = self.format_object_str(trainTaskRow["vis_objs"])

            # In-context examples
            prompt += "\n\nTask description: " + high_level_str \
                    
            # Include low-level instructions
            if includeLow:
                prompt += "\nStep by step instructions: " + step_by_step_str

            prompt +=  "\nCompleted plans: " + past_plan_str  \
                    + "\nOther objects in the room are " + in_context_obj_str \
                    + "\nNext Plans: " + next_plans_str
                    
        
        # Add the task prompt for GPT-3
        ## In-context examples components
        completed_plans = curr_task["completed_plans"]
        vis_objs = curr_task["vis_objs"]

        task_high_level_str = str(curr_task["task_instr"][0])
        task_step_by_step_str = '. '.join(curr_task["step_instr"])
        task_past_plan_str = self.format_plan_str(completed_plans)
        task_obj_str = self.format_object_str(vis_objs)

        # Example for above strings
        # task_high_level_str = 'Cook the potato and put it into the recycle bin.'
        # task_step_by_step_str = '. '.join(curr_task["step_instr"])
        # task_past_plan_str = self.format_plan_str(completed_plans)
        # task_obj_str = 'microwave, fridge, potato, garbagecan'

        prompt += "\n\nTask description: " + task_high_level_str \
                

        if includeLow:
            prompt += "\nStep by step instructions: " + task_step_by_step_str

        prompt += "\nCompleted plans: " + task_past_plan_str \
                + "\nOther objects in the room are " + task_obj_str \
                + "\nNext Plans:"
                
        curr_task["Prompts"] = prompt
        curr_task["vis_objs"] = vis_objs
        
        return prompt

    def get_logit_biases(self, logit_bias_text):
        #identify tokens for which to increase logit bias
        
        logit_biases = {}

        try:
            tokens = self.tokenizer.encode(logit_bias_text)
            for token in tokens:
                logit_biases[token]= .1 #logit bias 
        except Exception as e:
            print('Error in tokenizer doing logit bias')
            pass
        
        return logit_biases

    #run GPT-3 on specified test set using the KNN prompts
    def run_gpt3(self, prompt, logit_bias_text, engine='text-davinci-003', max_tokens=200):
        
        #GENERATE Relation Extraction PREDICTIONS
        gpt3_output = []

        #identify tokens for which to increase logit bias
        logit_biases = self.get_logit_biases(logit_bias_text)

        if self.debug:
            print("\n---------------Prompt----------------")
            print(prompt)

            print("\n---------------Logit Bias Objects----------------")
            print(logit_bias_text)

        sample = openai.Completion.create(engine=engine,
                                        prompt=prompt,
                                        max_tokens=max_tokens,
                                        temperature=0,
                                        logit_bias = logit_biases)

        gpt3_output.append(sample)
        prediction = sample['choices'][0]['text']

        return prediction, gpt3_output   

    # Main point of entry for LLM HLP generator
    def generate_hlp(self, curr_task, k, vision = False):


        prompt = self.generate_prompt(curr_task, k, removeNav=False, naturalFormat=False, vision = vision, includeLow=False)
        
        if self.debug:
            print("\n---------------Prompt----------------")
            print(prompt)
        
        logit_bias = self.get_logit_biases(curr_task['vis_objs'])
        logit_bias = {}

        # generated_hlp, gpt3_output = self.run_gpt3(prompt, curr_task["vis_objs"])
        generated_hlp = llm(prompt, engine='gpt-3', logit_bias=logit_bias,  images=[], stop=['\n'])

        return generated_hlp 

    # Main point of entry for LLM HLP prompt generator, use in run_eval
    def generate_gpt_prompt(self, curr_task, k, vision = False):

        prompt = self.generate_prompt(curr_task, k, removeNav=False, vision = vision, naturalFormat=False, includeLow=False)

        return prompt 

    
    # Below are helper functions 

    # Change object list into object string:
    ## Example: ['Drawer', 'ButterKnife'] -> Drawer, ButterKnife
    def format_object_str(self, obj_list):

        if not obj_list:
            return ""

        obj_str = ", ".join(obj_list).lower()
        return obj_str

    # Change plan list into plan string:
    ## Example: [('Navigation','Shelf'), ('PickupObject', 'knife')] -> Navigation Shelf. PickupObject Knife
    def format_plan_str(self, plan_list):

        if not plan_list:
            return ""

        # Lowercase object names in (action, plan) tuple
        lowercased_plan_list = []
        for item in plan_list:
            item_list = list(item)
            item_list[1] = item_list[1].lower()
            if len(item_list) > 2:
                item_list[2] = item_list[2].lower()
            lowercased_plan_list.append(tuple(item_list))

        plans = [" ".join(item) for item in lowercased_plan_list]
        plan_str = ', '.join(plans)

        return plan_str


if __name__=='__main__':

    # Example task format
    curr_task = {
                    "task_instr": ["Cook the potato and put it into the recycle bin."],
                    "step_instr": ["Go to the potato near the sink", "Pick up the potato", "Go to the microwave next to the fridge.", "Open the microwave", "Cook the potato in the microwave", "Take out the potato", "Go to the recycle bin", "Throw the potato in the recycle bin"],
                    "vis_objs": ["cup", "microwave", "fridge", "garbagecan"], "completed_plans": [("Navigation", "Countertop"),("PickupObject", "Potato"), ("Navigation", "Microwave")]
                }
    
    print("\n---------------Example Task----------------")
    for key, value in curr_task.items():
        print(f"{key}: {value}")


    hlp_generator = LLM_HLP_Generator(knn_data_path="knn_set.pkl", emb_model_name="paraphrase-MiniLM-L6-v2", debug=True)

    generated_plan = hlp_generator.generate_hlp(curr_task, k=9, vision = True)

    print("\n---------GPT3 generated HLP-------------")
    print(generated_plan)
