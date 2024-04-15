# ALL in one script to run ALFRED with ALFWORLD backbone

import os
import base64
import sys
import json
import yaml
import cv2
import argparse
import pprint
from tqdm import tqdm
from openai import OpenAI

from utils import action_mapping, action_formatting, locate_agent
import alfworld.agents.environment
from hlp_planner import LLM_HLP_Generator

CLIENT = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

def llm(prompt, engine, images=None, stop=["\n"]):
    
    if engine == 'gpt-4':
        response = CLIENT.chat.completions.create(
            model='gpt-4',
            messages=[
                {"role": "system", "content": "You are a helpful assistant that can plan household tasks."},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
            max_tokens=100,
            top_p=1,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            stop=stop
        )
        return response.choices[0].message.content
    
    elif engine == 'gpt-4v':

        response = CLIENT.chat.completions.create(
            model="gpt-4-vision-preview",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{images[0]}"
                            }
                        },
                        {
                            "type": "text",
                            "text": prompt,
                        },
                    ],
                }
            ],
            max_tokens=300,
        )
        return response.choices[0].message.content

    else:
        response = CLIENT.completions.create(
            model="gpt-3.5-turbo-instruct", #"text-davinci-003",
            prompt=prompt,
            temperature=0,
            max_tokens=100,
            top_p=1,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            stop=stop
        )
        return response.choices[0].text


def process_ob(ob):
    if ob.startswith('You arrive at loc '):
        ob = ob[ob.find('. ')+2:]    
    return ob


def encode_image(numpy_img):
    _, JPEG = cv2.imencode('.jpeg', numpy_img)
    return base64.b64encode(JPEG).decode('utf-8')


def eval_single_task(prompt, env, hlp_generator, curr_task, engine, dynamic=True, to_print=True, vision=False,  ob='', info = ''):
    # print('Input params', prompt, env, hlp_generator, curr_task, engine, dynamic, to_print, vision,  ob)

    # Get initial frames if GPT-4V is used
    if engine == "gpt-4v":
        vision == True

    encoded_frames = []
    if vision:
        init_frames = env.get_frames()
        encoded_frames = [encode_image(frame) for frame in init_frames]

    completed_plans = []
    seen_objs = [obj["objectType"] for obj in env.envs[0].env.last_event.metadata['objects'] if obj['visible']]
    
    # Get initial high-level plan
    # print("Prompt: ", prompt)
    # print('---End of prompt---')
    llm_out = llm(prompt, engine=engine, images=encoded_frames, stop=['\n'])
    high_level_plans = llm_out.split(',')
    high_level_plans = [action_mapping(x) for x in high_level_plans]

    cur_loc = locate_agent(admissible_actions=info['admissible_commands'][0]) #Current location/recep of the agent
    cur_obj = '' #Object being held by the agent right now
    prev_fail_plan = ''


    # Run until high-level plans are exhausted
    while high_level_plans:
        print('===========NEW STEP==============')
        print('Admissible', info['admissible_commands'][0])
        
        print('Generated plans: ', high_level_plans)
        plan = high_level_plans.pop(0).strip()
        plan_old_format = plan

        plan = action_formatting( 
            plan, 
            admissible_actions=info['admissible_commands'][0],
            cur_loc=cur_loc,
            cur_obj=cur_obj,
            )

        # print('Plan: ', plan)
        if plan != prev_fail_plan:
            observation, reward, done, info = env.step([plan])
            # print("Output of step: observations:", observation,'reward', reward,'done', done)
            # print(info)

            # raise KeyboardInterrupt
            observation, reward, done = process_ob(observation[0]), info['won'][0], done[0]
        else: observation = 'Nothing happens.'
        
        if done:
            return reward
        # raise KeyboardInterrupt
        # High level plan has an error
        if observation == "Nothing happens.":
            #Assign this current plan as failure
            prev_fail_plan = plan
            print(f'----------------{prev_fail_plan}')

            # Dynamic re-planning       
            if dynamic:
                curr_vis_objs = [obj["objectType"] for obj in env.envs[0].env.last_event.metadata['objects'] if obj['visible']]
                seen_objs += curr_vis_objs
                print('Visible object', init_vis_objs)

                curr_task = {
                        "task_instr": high_instrs[0],
                        "step_instr": step_instrs[0],
                        "vis_objs": init_vis_objs, 
                        "completed_plans": []
                    }
                curr_task["vis_objs"] = curr_vis_objs
                curr_task["completed_plans"] = completed_plans
                new_prompt = hlp_generator.generate_gpt_prompt(curr_task, k=9)
                # print("Prompt: ", prompt)
                # print('---End of prompt---')
                
                if vision:
                    curr_frames = env.get_frames()
                    encoded_frames = [encode_image(frame) for frame in curr_frames]

                # Generate new plans if dynamic
                llm_out = llm(new_prompt, images=encoded_frames, engine = engine,stop=['\n'])
                high_level_plans = llm_out.split(',')
                high_level_plans = [action_mapping(x) for x in high_level_plans]
                print('New plan!!!', high_level_plans)

        else:
            prev_fail_plan = '' #A failed plan may not be a failure everywhere
            if 'go to ' in plan:
                cur_loc = plan.split('go to')[-1].strip() #Current location/recep of the agent
                print(f'--Currently at {cur_loc}')
            elif 'take ' in plan:
                cur_obj = plan.split('take')[-1].split('from')[0].strip() #Object being held by the agent right now
                print(f'--Currently holding {cur_loc}')
            elif 'put ' in plan:
                cur_obj = ''
                print(f'--Put down--')

            completed_plans.append(plan_old_format)
        
        if to_print:
            print(f'Act {i}: {plan}\nObs {i}: {observation}')
            sys.stdout.flush()
            
    return 0


if __name__ == '__main__':

    # Read config
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", help="path to config file")
    args = parser.parse_args()

    with open(args.config_file) as reader:
        config = yaml.safe_load(reader)
    
    split = "eval_out_of_distribution"

    # Start simulator and set up environment
    env = getattr(alfworld.agents.environment, config["env"]["type"])(config, train_eval=split)
    env = env.init_env(batch_size=1)
    num_games = len(env.json_file_list)

    prefixes = {
        'pick_and_place': 'put',
        'pick_clean_then_place': 'clean',
        'pick_heat_then_place': 'heat',
        'pick_cool_then_place': 'cool',
        'look_at_obj': 'examine',
        'pick_two_obj': 'puttwo'
    }
    cnts = [0] * 6
    rs = [0] * 6

    hlp_generator = LLM_HLP_Generator(knn_data_path=config["llm_planner"]["knn_dataset_path"], emb_model_name=config["llm_planner"]["emb_model_name"], debug=config["llm_planner"]["debug"])

    # Main eval loop
    for _ in range(num_games):
        ob, info = env.reset()
        
        # print('visible object')
        # print(env.envs[0].controller.visible_objects)
        # print('receps')
        # print(env.envs[0].controller.curr_recep)

        # raise KeyboardInterrupt

        ob = '\n'.join(ob[0].split('\n\n')[1:])
        name = '/'.join(info['extra.gamefile'][0].split('/')[-2:-1])

        # print(ob, info)
        with open(os.path.join(info['extra.gamefile'][0], "traj_data.json"), "r") as f:
            traj_data = json.load(f)

        # print(traj_data)
        # import pprint
        # pprint.pprint(traj_data['plan'])
        # print(traj_data['plan'].keys())
        # raise KeyboardInterrupt
        # Retrieve all instructions for the current enviornment setup
        high_instrs = [ann["task_desc"] for ann in traj_data["turk_annotations"]["anns"]]
        step_instrs = [ann["high_descs"] for ann in traj_data["turk_annotations"]["anns"]]
        
        # print('high_instrs')
        # print('step_instrs')
        
        init_vis_objs = [obj["objectType"] for obj in env.envs[0].env.last_event.metadata['objects'] if obj['visible']]
        # print(env.envs[0].env.last_event.metadata['objects'])
        print('Visible object', init_vis_objs)
        # raise KeyboardInterrupt

        # If there are multiple annotations for the same environment setup, evaluate on all of them
        for i, high_instr in enumerate(high_instrs):
            curr_task = {
                        "task_instr": [high_instr],
                        "step_instr": step_instrs[i],
                        "vis_objs": init_vis_objs, 
                        "completed_plans": []
                    }
            # print("Cur_task:")
            # pprint.pprint(curr_task)
            
            # print('init prompt')
            init_prompt = hlp_generator.generate_gpt_prompt(curr_task, k=config["llm_planner"]["num_in_context_examples"])
            # print(init_prompt)
            # print('---end of prompt---')
            
            for i, (k, v) in enumerate(prefixes.items()):
                if name.startswith(k):
                    # print('k', 'v', k, v)
                    r = eval_single_task(init_prompt, env, hlp_generator, curr_task, config["llm_planner"]["engine"], dynamic=config["llm_planner"]["dynamic"], ob=ob, info = info)
                    rs[i] += r
                    cnts[i] += 1
                    break
            print(_+1, 'r', r, 'rs', rs, 'cnts', cnts, 'sum(rs)/sum(cnts)', sum(rs) / sum(cnts))
            print('------------\n')
