import requests
from PIL import Image
import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration

def get_model(model_id = "bczhou/tiny-llava-v1-hf"):
  model = LlavaForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
  ).to(0)
  processor = AutoProcessor.from_pretrained(model_id)
  return model, processor

def vlm(model, processor, text, image):
  inputs = processor(text, image, return_tensors='pt').to(0, torch.float16)
  output = model.generate(**inputs, max_new_tokens=200, do_sample=False)

  return processor.decode(output[0][2:], skip_special_tokens=True)

if __name__ == '__main__':
  model, processor = get_model()
  raw_image = Image.open('../out/samples/0.png')
  prompt ='''USER: <image>\n
Prompt:  Create a high-level plan for completing a household task using the allowed actions and visible objects.
      
Allowed actions: OpenObject, CloseObject, PickupObject, PutObject, ToggleObjectOn, ToggleObjectOff, SliceObject, Navigation

Task description: place a cooled tomato slice inside of the microwave
Step by step instructions: move to the left a bit to face the kitchen sink. grab the butter knife out of the kitchen sink. slice the tomato to the left of the kitchen sink. place the knife in the kitchen sink. grab a tomato slice off of the kitchen counter to the left of the sink. turn right and walk over to the fridge on the left. place the tomato slice inside of the fridge, let it cool, then pick it back up and close the door. turn left and walk over to the microwave ahead. place the tomato slice inside of the microwave
Completed plans: Navigation sinkbasin, PickupObject butterknife, SliceObject tomato, PutObject butterknife sink, PickupObject tomatosliced, Navigation fridge, OpenObject fridge, PutObject tomatosliced fridge, CloseObject fridge, OpenObject fridge, PickupObject tomatosliced, CloseObject fridge, Navigation microwave, OpenObject microwave, PutObject tomatosliced microwave
Visible objects are butterknife, sink, fridge, tomatosliced, microwave
Next Plans: CloseObject microwave

Task description: Put a chilled tomato slice into the microwave.
Step by step instructions: Take a step forward, then turn left and walk over to the toaster on the counter.. Pick up the red tomato on the counter to the right of the stove.. Turn around begin walking across the room, hang left at the fridge and walk up to the kitchen island.. Put the tomato onto the island below the butter knife.. Pick up the butter knife off of the kitchen island.. Slice up the tomato on the kitchen island.. Place the butter knife onto the island to the right of the sliced tomato.. Pick up a tomato slice off of the kitchen island.. Turn around and walk up to the fridge.. Open the fridge and put the tomato slice on the bottom shelf, then close the door, after a couple seconds open the fridge and remove the tomato slice then close the door.. Turn right and walk over to the oven, then turn right again and begin across the room, hang a left and walk up to the microwave.. Open the microwave door and place the tomato slice inside the microwave in front of the egg.
Completed plans: Navigation countertop, PickupObject tomato, Navigation countertop, PutObject tomato countertop, PickupObject butterknife
Visible objects are countertop, tomato, butterknife, fridge, tomatosliced, microwave
Next Plans: SliceObject tomato, PutObject butterknife countertop, PickupObject tomatosliced, Navigation fridge, OpenObject fridge, PutObject tomatosliced fridge, CloseObject fridge, OpenObject fridge, PickupObject tomatosliced, CloseObject fridge, Navigation microwave, OpenObject microwave, PutObject tomatosliced microwave, CloseObject microwave


Task description: Place a chilled tomato in a microwave.
Step by step instructions: Walk ahead and veer left after getting to the counter on the left, Walk to the sink on the left counter.. Pick up the tomato in the sink.. Turn right and walk to the fridge.. Place the tomato in the fridge, in front of the center brown bowl. Close the door, wait a moment, and take it out again.. Turn right and walk to the stove, facing it.. Place the tomato in the microwave above the stove.
Completed plans: 
Visible objects are chair, houseplant
Next Plans: \

  \nASSISTANT: '''

  # raw_image = Image.open(requests.get(image_file, stream=True).raw)
  
  print(vlm(model, processor, text=prompt, image=raw_image))