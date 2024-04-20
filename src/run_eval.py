# ALL in one script to run ALFRED with ALFWORLD backbone
'''
hlp: high level plan
gc: goal-condition success rate
'''
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
from alfworld.gen.constants import OBJECTS_SINGULAR
from utils import action_mapping, action_formatting, locate_agent
from hlp_planner import LLM_HLP_Generator
# from vlm import vlm, get_model
from llm import llm

CANDIDATE_VOCAB = {} 

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
    if object not in CANDIDATE_VOCAB.keys():
        llm_out = llm(prompt, engine='gpt-3', images=[], stop=['\n'])
        candidate_objects = [x.strip().lower() for x in llm_out[1: -1].split(',')]
        candidate_objects = [x for x in candidate_objects if x != object]
        CANDIDATE_VOCAB[object] = candidate_objects
    else:
        candidate_objects = CANDIDATE_VOCAB[object]
    
    return candidate_objects

def get_potential_commands(command):
    '''
    Command here is in old format, like Navigation fridge 
    '''
    
    parts = command.strip().split(' ')
    if len(parts) < 2: return [command]
    
    tar = parts[1]
    sim_objs = get_similar_objects(tar)
    candidates = list(set([tar] + sim_objs).intersection(OBJECTS_SINGULAR))
    
    if len(candidates) == 0:
        return [command]
    else:
        return [command.replace(tar, x) for x in candidates]

def detect(env, engine):
    if engine == "gpt-4v":
        cur_frames = env.get_frames()
        encoded_frames = [encode_image(frame) for frame in cur_frames]
        llm_out = llm('List all objects in this image? Answer in format [banana, apple, orange]', engine=engine, images=encoded_frames, stop=['\n'])
        observed_objects = [x.strip() for x in llm_out[1: -1].split(',')]
        
    else:
        observed_objects = [obj["objectType"] for obj in env.envs[0].env.last_event.metadata['objects'] if obj['visible']]

    return observed_objects

def explore(env, engine, admissible_actions):
    #Go to all locations to see what objects are available
    all_objects = []
    
    # Get current frame at current location
    observed_objects = detect(env, engine)
    all_objects.extend(observed_objects)
    
    # Go to all possible location
    print('Explore to get all objects in the room, as much as one can: ')
    #Just go to the location with index 1, since shelf 1,2,3,... are usually next to each other
    admissible_actions = [x for x in admissible_actions if ' 1' in x]

    for action in tqdm(admissible_actions):
        if 'go to ' in action:
            observation, reward, done, info = env.step([action])
            observed_objects = detect(env, engine)
            all_objects.extend(observed_objects)

    return list(set([x.lower().replace(' ', '') for x in all_objects]))

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
        max_num_trials = 40,        #Too many trials, escape to avoid loops
        temperature = 0.0,
        ):
    
    # Get initial frames if GPT-4V is used
    if engine == "gpt-4v":
        vision = True

    encoded_frames = []
    if vision:
        init_frames = env.get_frames()
        encoded_frames = [encode_image(frame) for frame in init_frames]

    completed_plans = []
    recorded_actions= []
    all_objs = curr_task['vis_objs']

    # Get initial high-level plan
    # print("Prompt: ", prompt)
    # print('---End of prompt---')

    # At a beginning of task, no need for vision, it's unlikely the robot is looking at a correct direction
    text_engine = 'gpt-3' if vision else engine
    logit_bias = hlp_generator.get_logit_biases(all_objs)
    llm_out = llm(prompt, engine=text_engine, temperature = temperature, logit_bias=logit_bias, images=[], stop=['\n'])

    hlp_old_format = llm_out.split(',')
    hlp = [action_mapping(x) for x in hlp_old_format]
    recorded_actions.append('---Plan: ' + ', '.join(hlp))

    error_gen = 0
    while None in hlp:
        llm_out = llm(new_prompt, engine=engine, temperature = temperature, logit_bias=logit_bias, images=encoded_frames, stop=['\n'])
        llm_out = weird_fix(llm_out)
        hlp_old_format = llm_out.split(',')
        hlp = [action_mapping(x) for x in hlp_old_format]

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
    while hlp:
        print(f'===========NEW STEP===at {cur_loc} holding {cur_obj}===')
        # print('Admissible', info['admissible_commands'][0])
        
        # print('Generated plans: ', hlp)
        plan_old_format = hlp_old_format.pop(0).strip()
        hlp.pop(0).strip()

        potential_plans = get_potential_commands(plan_old_format)
        print(potential_plans)

        for cur_plan in potential_plans:
            print('Trying ', cur_plan)
            cur_plan = action_formatting( 
                        action_mapping(cur_plan), 
                        admissible_actions=info['admissible_commands'][0],
                        cur_loc=cur_loc,
                        cur_obj=cur_obj,
                        )
                
            recorded_actions.append('---Admissible: ' + ', ' .join(info['admissible_commands'][0]))
            print('---Admissible: ', info['admissible_commands'][0])
            # print('Plan: ', plan)
            observation, reward, done, info = env.step([cur_plan])
            observation, reward, done = process_ob(observation[0]), info['won'][0], done[0]
            success = 0 if observation == "Nothing happens." else 1

            if success:
                break

        exe += 1
        _, _, _, _, gc, _ = env.envs[0].get_info()
        recorded_actions.append('++'.join([cur_plan,str(success)]))

        if done or exe == max_num_trials:
            return reward, gc, recorded_actions
        
        # High level plan has an error
        if observation == "Nothing happens.":
            #Assign this current plan as failure
            # prev_fail_plan = plan
        
            #At 10 consecutive failed executions, exit
            error_exe += 1
            if error_exe == max_num_error_trials:
                return reward, gc, recorded_actions

            # Dynamic re-planning       
            if dynamic:
                
                seen_objs = detect(env, engine) +  [obj["objectType"] for obj in env.envs[0].env.last_event.metadata['objects'] if obj['visible']]
                # print('Visible object', seen_objs)
                recorded_actions.append('---Visible:' + ', '.join(seen_objs))
                
                #VERY NAIVE CONTROL HERE:
                # If put/take fails and last successful plan is Navigation, means you're at a wrong place
                # Remove navigation as the last successful plan, so agent can go elsewhere
                if completed_plans != []:
                    if ('take ' in cur_plan or 'put ' in cur_plan or 'open ' in cur_plan) and 'Navigation' in completed_plans[-1]:
                        completed_plans.pop(-1)
                    
                #All objs in FOV + in the room, except for the obj that has been tried and failed
                curr_task["vis_objs"] = [x for x in set(seen_objs + all_objs) if x != re.sub(r'[\d\s]+', '', cur_obj)]
                curr_task["completed_plans"] = completed_plans 

                logit_bias = hlp_generator.get_logit_biases(curr_task["vis_objs"])
                new_prompt = hlp_generator.generate_gpt_prompt(curr_task, vision = vision, k=9)
                # print("New Prompt: ", new_prompt)
                # print('---End of prompt---')
                
                if vision:
                    curr_frames = env.get_frames()
                    encoded_frames = [encode_image(frame) for frame in curr_frames]

                # Generate new plans if dynamic
                llm_out = llm(new_prompt, engine=engine, temperature = temperature, images=encoded_frames, logit_bias=logit_bias, stop=['\n'])
                # print(llm_out)

                hlp_old_format = llm_out.split(',')
                hlp = [action_mapping(x) for x in hlp_old_format]
                recorded_actions.append('---Replan: ' + ', '.join(hlp))

                #Catch OpenAI error plan generation for connection reasons
                # Sometimes llm output  'g o   t o   s a f e, o p e n   s a f e, p u t   c d   i n   s a f e, c l o s e   s a f e'
                #Which after mapping turns to [None, None, None]
                error_gen = 0                
                while None in hlp:
                    print('problem here', 'trying to fix')
                    llm_out = llm(new_prompt, engine=engine, temperature = temperature, logit_bias=logit_bias, images=encoded_frames, stop=['\n'])
                    print(llm_out)
                    llm_out = weird_fix(llm_out)
                    print(llm_out)
                    hlp_old_format = llm_out.split(',')
                    hlp = [action_mapping(x) for x in hlp_old_format]

                    error_gen += 1
                    if error_gen == 10:
                        print('LLM Generation Error')
                        return 0, gc, recorded_actions
                    
                print('New plan!!!', hlp_old_format, '|', hlp)

        else:
            # prev_fail_plan = '' #A failed plan may not be a failure everywhere
            
            # Reset number of trials
            error_exe = 0

            if 'go to ' in cur_plan:
                cur_loc = cur_plan.split('go to')[-1].strip() #Current location/recep of the agent
                print(f'--Currently at {cur_loc}')
            elif 'take ' in cur_plan:
                cur_obj = cur_plan.split('take')[-1].split('from')[0].strip() #Object being held by the agent right now
                print(f'--Currently holding {cur_obj}')
            elif 'put ' in cur_plan:
                cur_obj = ''
                print(f'--Put down--')

            completed_plan_tup = tuple(plan_old_format.split(' '))
            completed_plans.append(completed_plan_tup)
        
        if to_print:
            print(f'Act {i}: {cur_plan}\nObs {i}: {observation}')
            sys.stdout.flush()
            
    return 0, gc, recorded_actions


if __name__ == '__main__':

    # Read config
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", help="path to config file")
    parser.add_argument("--output_path", default="../out/runs", help="output of predicted trajectory files")
    parser.add_argument('--split', default='valid_unseen', help = "data portion to be evaluated on, either valid_seen or valid_unseen")
    args = parser.parse_args()

    with open(args.config_file) as reader:
        config = yaml.safe_load(reader)
    
    output_path = args.output_path
    split = "eval_in_distribution" if args.split == 'valid_seen' else "eval_out_of_distribution"
    
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

    hlp_generator = LLM_HLP_Generator(knn_data_path=config["llm_planner"]["knn_dataset_path"], emb_model_name=config["llm_planner"]["emb_model_name"], debug=config["llm_planner"]["debug"])
    
    # Main eval loop
    all_gc = []
    for index in range(num_games):
        ob, info = env.reset(index)
        print('Task id: ', info['extra.gamefile'][0])        
        
        full_output_path = os.path.join(output_path, os.path.dirname(info['extra.gamefile'][0].split('alfworld/json_2.1.1/')[-1]))
        os.makedirs(full_output_path, exist_ok = True)

        ob = '\n'.join(ob[0].split('\n\n')[1:])
        name = '/'.join(info['extra.gamefile'][0].split('/')[-2:-1])

        # print(ob, info)
        with open(os.path.join(info['extra.gamefile'][0], "traj_data.json"), "r") as f:
            traj_data = json.load(f)

        # Retrieve all instructions for the current enviornment setup
        high_instrs = [ann["task_desc"] for ann in traj_data["turk_annotations"]["anns"]]
        step_instrs = [ann["high_descs"] for ann in traj_data["turk_annotations"]["anns"]]
        
        # print('high_instrs')
        # print('step_instrs')
        # print(high_instrs)
        # print(step_instrs)
        # pprint.pprint(traj_data)
        # print('Get info self._feedback, self._done, acs, won, gc, expert_actions')
        # print(env.envs[0].get_info())
        # raise KeyboardInterrupt
    
        if engine == "gpt-4v":
            init_vis_objs = explore(env=env, engine=engine, admissible_actions=info['admissible_commands'][0])
        else:
            init_vis_objs = [obj["objectType"] for obj in env.envs[0].env.last_event.metadata['objects'] if obj['visible']]
        
        # If there are multiple annotations for the same environment setup, evaluate on all of them
        for j, high_instr in enumerate(high_instrs):
            curr_task = {
                        "task_instr": [high_instr],
                        "step_instr": step_instrs[j],
                        "vis_objs": init_vis_objs, 
                        "completed_plans": []
                    }
            output_predfile = os.path.join(full_output_path, high_instr.replace(' ', '_') + '+pred' + '.txt')
            output_truefile = os.path.join(full_output_path, high_instr.replace(' ', '_') + '+true' + '.txt')
            init_prompt = hlp_generator.generate_gpt_prompt(curr_task, k=config["llm_planner"]["num_in_context_examples"])
            
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
            
            with open(os.path.join('../out/runs', 'cont_eval.txt'), 'a+') as f:
                f.write(str(index+1) + ' Success Rate: r ' + str(r) + ' rs ' + str(rs) + ' cnts ' + str(cnts) +  ' sum(rs)/sum(cnts) ' + str(sum(rs) / sum(cnts)))
                f.write('\n')
                f.write('Goal-condition Success Rate ' +  str(gc) + ' | Accumulate Average GC: ' +  str(sum(all_gc)/len(all_gc)))
                f.write('\n')