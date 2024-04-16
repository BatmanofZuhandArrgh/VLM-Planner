# ALL in one script to run ALFRED with ALFWORLD backbone

import os
import base64
import sys
import json
import yaml
import cv2
import argparse
import pprint
import re
from tqdm import tqdm


import alfworld.agents.environment
from utils import action_mapping, action_formatting, locate_agent
from hlp_planner import LLM_HLP_Generator
# from vlm import vlm, get_model
from llm import llm

def weird_fix(llm_output = ' c l o s e   d r a w e r, e a t   a s s, g o   t o   d e s k'):
    llm_output = re.sub(r'\s{2,}', '@@@', llm_output.strip()) #Replace any more than 2 space characters with special chars
    llm_output = llm_output.replace(' ', '')
    llm_output = llm_output.replace('@@@', ' ')
    return llm_output

def encode_image(numpy_img):
    _, JPEG = cv2.imencode('.jpeg', numpy_img)
    return base64.b64encode(JPEG).decode('utf-8')

def process_ob(ob):
    if ob.startswith('You arrive at loc '):
        ob = ob[ob.find('. ')+2:]    
    return ob

def get_similar_objects(object):
    '''
    Return synonyms, singular or subword of "object"
    '''
    
    prompt = f'''
    Find synonyms and subword for "coffee machines": [coffee machine, coffee maker, coffee brewer, coffee, machine, drink]
    Find synonyms and subword for "pencils" : [pen, pencil, writing instrument]
    Find synonyms and subword for "chair" : [armchair, bench, sofa]
    Find synonyms and subword for "{object}":  
    '''
    llm_out = llm(prompt, engine='gpt-3', images=[], stop=['\n'])
    candidate_objects = [x.strip() for x in llm_out[1: -1].split(',')]
    candidate_objects = [x for x in candidate_objects if x != object]
    return candidate_objects

def detect(env, engine):
    if engine == "gpt-4v":
        cur_frames = env.get_frames()
        encoded_frames = [encode_image(frame) for frame in cur_frames]
        llm_out = llm('List all objects in this image? Answer in format [banana, apple, orange]', engine=engine, images=encoded_frames, stop=['\n'])
        observed_objects = [x.strip() for x in llm_out[1: -1].split(',')]
        
    else:
        observed_objects = [obj["objectType"] for obj in env.envs[0].env.last_event.metadata['objects'] if obj['visible']]

    return observed_objects

def explore(env, engine, admissible_actions, num_explore_locs = 3):
    #Go to all locations to see what objects are available
    all_objects = []
    
    # Get current frame at current location
    observed_objects = detect(env, engine)
    all_objects.extend(observed_objects)
    
    # Go to all possible location
    print('Explore to get all objects in the room, as much as one can: ')
    #Only first 3, might take too long for a case
    for action in tqdm(admissible_actions[:num_explore_locs]):
        if 'go to ' in action:
            observation, reward, done, info = env.step([action])
            observed_objects = detect(env, engine)
            all_objects.extend(observed_objects)

    return list(set(all_objects))

def eval_single_task(
        prompt, 
        env, 
        hlp_generator,
        curr_task,
        engine, 
        dynamic=True, 
        to_print=True, 
        vision=False,  
        ob='', 
        info = '',
        max_num_error_trials = 10, #Too many failed execution, escape to avoid repeated exact plans
        max_num_trials = 40        #Too many trials, escape to avoid loops
        ):
    # print('Input params', prompt, env, hlp_generator, curr_task, engine, dynamic, to_print, vision,  ob)

    # Get initial frames if GPT-4V is used
    if engine == "gpt-4v":
        vision == True

    encoded_frames = []
    if vision:
        init_frames = env.get_frames()
        encoded_frames = [encode_image(frame) for frame in init_frames]

    completed_plans = []
    recorded_actions= []

    # Get initial high-level plan
    # print("Prompt: ", prompt)
    # print('---End of prompt---')
    llm_out = llm(prompt, engine=engine, images=encoded_frames, stop=['\n'])
    # print(llm_out)


    high_level_plans_old_format = llm_out.split(',')
    high_level_plans = [action_mapping(x) for x in high_level_plans_old_format]
    recorded_actions.append('---Plan: ' + ', '.join(high_level_plans))

    error_gen = 0
    while None in high_level_plans:
        print('problem here', 'trying to fix')
        llm_out = llm(new_prompt, engine=engine, images=encoded_frames, stop=['\n'])
        print(llm_out)
        llm_out = weird_fix(llm_out)
        print(llm_out)
        high_level_plans_old_format = llm_out.split(',')
        high_level_plans = [action_mapping(x) for x in high_level_plans_old_format]

        error_gen += 1
        if error_gen == 10:
            print('LLM Generation Error')
            return 0, 0, []
        
    cur_loc = locate_agent(admissible_actions=info['admissible_commands'][0]) #Current location/recep of the agent, just for command formatting, this information does not go to the agent
    cur_obj = '' #Object being held by the agent right now
    # prev_fail_plan = '' #Last failed plan, just 1


    # Run until high-level plans are exhausted
    error_exe = 0
    exe = 0
    while high_level_plans:
        print(f'===========NEW STEP===at {cur_loc} holding {cur_obj}===')
        # print('Admissible', info['admissible_commands'][0])
        
        # print('Generated plans: ', high_level_plans)
        plan = high_level_plans.pop(0).strip()
        plan_old_format = high_level_plans_old_format.pop(0).strip()

        plan = action_formatting( 
            plan, 
            admissible_actions=info['admissible_commands'][0],
            cur_loc=cur_loc,
            cur_obj=cur_obj,
            )
        recorded_actions.append('---Admissible: ' + ', ' .join(info['admissible_commands'][0]))
        # print('Plan: ', plan)
        # if plan_old_format != prev_fail_plan: #If the first step is to do the previous mistaken plan, just skip it
        observation, reward, done, info = env.step([plan])
        exe += 1
        # print("Output of step: observations:", observation,'reward', reward,'done', done)
        # print(info)

        # print('Get info self._feedback, self._done, acs, won, goal_condition_success_rate, expert_actions')
        # print(env.envs[0].get_info())
        
        # raise KeyboardInterrupt
        observation, reward, done = process_ob(observation[0]), info['won'][0], done[0]
        # else: observation = 'Nothing happens.'
        _, _, _, _, goal_condition_success_rate, _ = env.envs[0].get_info()
        success = 0 if observation == "Nothing happens." else 1
        recorded_actions.append('++'.join([plan,str(success)]))

        if done or exe == max_num_trials:
            return reward, goal_condition_success_rate, recorded_actions
        
        # raise KeyboardInterrupt
        # High level plan has an error
        if observation == "Nothing happens.":
            #Assign this current plan as failure
            # prev_fail_plan = plan
            # print(f'----------------{prev_fail_plan}')
            # print(completed_plans)

            #At 10 consecutive failed executions, exit
            error_exe += 1
            if error_exe == max_num_error_trials:
                return reward, goal_condition_success_rate, recorded_actions

            # Dynamic re-planning       
            if dynamic:
                
                seen_objs = detect(env, engine)
                # print('Visible object', seen_objs)
                recorded_actions.append('---Visible:' + ', '.join(seen_objs))

                curr_task["vis_objs"] = seen_objs 
                curr_task["completed_plans"] = completed_plans 
                # print(curr_task)
                new_prompt = hlp_generator.generate_gpt_prompt(curr_task, k=9)
                # print("New Prompt: ", new_prompt)
                # print('---End of prompt---')
                
                if vision:
                    curr_frames = env.get_frames()
                    encoded_frames = [encode_image(frame) for frame in curr_frames]

                # Generate new plans if dynamic
                llm_out = llm(new_prompt, engine=engine, images=encoded_frames, stop=['\n'])
                # print(llm_out)

                high_level_plans_old_format = llm_out.split(',')
                high_level_plans = [action_mapping(x) for x in high_level_plans_old_format]
                recorded_actions.append('---Replan: ' + ', '.join(high_level_plans))

                #Catch OpenAI error plan generation for connection reasons
                # Sometimes llm output  'g o   t o   s a f e, o p e n   s a f e, p u t   c d   i n   s a f e, c l o s e   s a f e'
                #Which after mapping turns to [None, None, None]
                error_gen = 0                
                while None in high_level_plans:
                    print('problem here', 'trying to fix')
                    llm_out = llm(new_prompt, engine=engine, images=encoded_frames, stop=['\n'])
                    print(llm_out)
                    llm_out = weird_fix(llm_out)
                    print(llm_out)
                    high_level_plans_old_format = llm_out.split(',')
                    high_level_plans = [action_mapping(x) for x in high_level_plans_old_format]

                    error_gen += 1
                    if error_gen == 10:
                        print('LLM Generation Error')
                        return 0, goal_condition_success_rate, recorded_actions
                    
                # print('New plan!!!', high_level_plans_old_format, '|', high_level_plans)

        else:
            # prev_fail_plan = '' #A failed plan may not be a failure everywhere
            
            # Reset number of trials
            error_exe = 0

            if 'go to ' in plan:
                cur_loc = plan.split('go to')[-1].strip() #Current location/recep of the agent
                print(f'--Currently at {cur_loc}')
            elif 'take ' in plan:
                cur_obj = plan.split('take')[-1].split('from')[0].strip() #Object being held by the agent right now
                print(f'--Currently holding {cur_obj}')
            elif 'put ' in plan:
                cur_obj = ''
                print(f'--Put down--')

            completed_plan_tup = tuple(plan_old_format.split(' '))
            completed_plans.append(completed_plan_tup)
        
        if to_print:
            print(f'Act {i}: {plan}\nObs {i}: {observation}')
            sys.stdout.flush()
            
    return 0, goal_condition_success_rate, recorded_actions


if __name__ == '__main__':

    # Read config
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", help="path to config file")
    parser.add_argument("--output_path", default="../out/runs")
    args = parser.parse_args()

    with open(args.config_file) as reader:
        config = yaml.safe_load(reader)
    
    output_path = args.output_path

    split = "eval_out_of_distribution" #for unseen val
    # split = "eval_in_distribution" #for seen val

    # Start simulator and set up environment
    env = getattr(alfworld.agents.environment, config["env"]["type"])(config, train_eval=split)
    env = env.init_env(batch_size=1)
    num_games = len(env.json_file_list)
    engine = config["llm_planner"]["engine"]
    
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

    # model, processor = get_model()

    hlp_generator = LLM_HLP_Generator(knn_data_path=config["llm_planner"]["knn_dataset_path"], emb_model_name=config["llm_planner"]["emb_model_name"], debug=config["llm_planner"]["debug"])

    # print(env.json_file_list)
    
    # Main eval loop
    all_gc = []
    for index in range(num_games):
        ob, info = env.reset(index)
        print('Task id: ', info['extra.gamefile'][0])        
        
        full_output_path = os.path.join(output_path, os.path.dirname(info['extra.gamefile'][0].split('alfworld/json_2.1.1/')[-1]))
        os.makedirs(full_output_path, exist_ok = True)

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
        # print(high_instrs)
        # print(step_instrs)
        # pprint.pprint(traj_data)
        # print('Get info self._feedback, self._done, acs, won, goal_condition_success_rate, expert_actions')
        # print(env.envs[0].get_info())
        # raise KeyboardInterrupt
    
        if engine == "gpt-4v":
            init_vis_objs = explore(env=env, engine=engine, admissible_actions=info['admissible_commands'][0])
        else:
            init_vis_objs = [obj["objectType"] for obj in env.envs[0].env.last_event.metadata['objects'] if obj['visible']]

        # print('Detected object', init_vis_objs)        
        
        # If there are multiple annotations for the same environment setup, evaluate on all of them
        for j, high_instr in enumerate(high_instrs):
            curr_task = {
                        "task_instr": [high_instr],
                        "step_instr": step_instrs[j],
                        "vis_objs": init_vis_objs, 
                        "completed_plans": []
                    }
            # print("Cur_task:")
            # pprint.pprint(curr_task)
            output_predfile = os.path.join(full_output_path, high_instr.replace(' ', '_') + '+pred' + '.txt')
            output_truefile = os.path.join(full_output_path, high_instr.replace(' ', '_') + '+true' + '.txt')
            
            # print('init prompt')
            init_prompt = hlp_generator.generate_gpt_prompt(curr_task, k=config["llm_planner"]["num_in_context_examples"])
            # print(init_prompt)
            # print('---end of prompt---')
            
            for i, (k, v) in enumerate(prefixes.items()):
                if name.startswith(k):
                    print('Task type: ', high_instr)
                    # print(k, v)
                    r, gc, recorded_actions = eval_single_task(init_prompt, env, hlp_generator, curr_task, config["llm_planner"]["engine"], dynamic=config["llm_planner"]["dynamic"], ob=ob, info = info)
                    # r = 0
                    rs[i] += r
                    cnts[i] += 1
                    all_gc.append(gc)
                    break
            print(index+1, 'Success Rate: r', r, 'rs', rs, 'cnts', cnts, 'sum(rs)/sum(cnts)', sum(rs) / sum(cnts))
            print('Goal-condition Success Rate', gc, 'Accumulate Average GC:', sum(all_gc)/len(all_gc))
            print('------------\n')
            
            recorded_actions = ['Success: ' + str(r) + '| GC: '+ str(gc)] +  recorded_actions
            with open(output_predfile, 'w') as f:
                print(f'Writing prediction at {output_predfile}')
                f.write('\n'.join(recorded_actions))
            
            with open(os.path.join('../out/runs/valid_unseen', 'cont_eval.txt'), 'a+') as f:
                f.write(str(index+1) + ' Success Rate: r ' + str(r) + ' rs ' + str(rs) + ' cnts ' + str(cnts) +  ' sum(rs)/sum(cnts) ' + str(sum(rs) / sum(cnts)))
                f.write('\n')
                f.write('Goal-condition Success Rate ' +  str(gc) + ' | Accumulate Average GC: ' +  str(sum(all_gc)/len(all_gc)))
                f.write('\n')