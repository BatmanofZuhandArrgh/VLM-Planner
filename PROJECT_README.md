CS6224 Project:

To install, please refer to the README.md

To download the inference results, use this [link](https://drive.google.com/file/d/1MEhHJdOtSkV80GqY81-JQaPJ9rBuq69H/view?usp=sharing).

1. runs: results containing planner's output
    - llm_valid_seen        : reproduced llm-planner evaluated on ALFRED's valid_seen
    - llm_valid_unseen      : reproduced llm-planner evaluated on ALFRED's valid_unseen
    - llm_valid_unseen_mod  : reproduced llm-planner evaluated on ALFRED's valid_unseen, also modified with logit_bias, output text matching,..
    - vlm_valid_unseen      : vlm-planner evaluated on ALFRED's valid_unseen

The structure of content within each task file:
- Success | GC  : Output from Alfworld
- Plan          : Plan generated from the planner
- Admisible     : Admissible action, needs to be input exactly for the simulation to execute
- <action, object>++<boolean value>  : Action input, and a boolean value: 1 for successful execution and 0 for failed 

2. json_2.1.1: Images and actions for several trials for each tasks
    - valid_seen
    - valid_unseen

3. report_samples
- Each run's testing log and continuous evaluation
- Several samples showing the issues mentioned in the project report

