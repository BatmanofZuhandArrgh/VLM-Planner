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

Task description: To move the CD from the drawer to the safe.
Completed plans: Navigation drawer, OpenObject drawer, PickupObject cd, CloseObject drawer, Navigation safe
Visible objects are safe, cd
Next Plans: OpenObject safe, PutObject cd safe, CloseObject safe

Task description: Move a laptop from a coffee table to an armchair.
Completed plans: Navigation coffeetable, CloseObject laptop, PickupObject laptop
Visible objects are armchair, laptop
Next Plans: Navigation armchair, PutObject laptop armchair

Task description: Place the two laptops on the ottoman.
Completed plans: Navigation coffeetable, CloseObject laptop, PickupObject laptop, Navigation ottoman, PutObject laptop ottoman, Navigation sofa, CloseObject laptop, PickupObject laptop
Visible objects are laptop, ottoman
Next Plans: Navigation ottoman, PutObject laptop ottoman

Task description: Move two salt shakers from the cabinet above the stove to the drawer.
Completed plans: Navigation cabinet, OpenObject cabinet, PickupObject saltshaker, CloseObject cabinet, Navigation drawer
Visible objects are saltshaker, drawer
Next Plans: OpenObject drawer, PutObject saltshaker drawer, CloseObject drawer, Navigation cabinet, OpenObject cabinet, PickupObject saltshaker, CloseObject cabinet, Navigation drawer, OpenObject drawer, PutObject saltshaker drawer, CloseObject drawer

Task description: Put two salt shakers in the drawer.
Completed plans: Navigation cabinet, OpenObject cabinet, PickupObject saltshaker, CloseObject cabinet, Navigation drawer, OpenObject drawer
Visible objects are saltshaker, drawer
Next Plans: PutObject saltshaker drawer, CloseObject drawer, Navigation countertop, PickupObject saltshaker, Navigation drawer, OpenObject drawer, PutObject saltshaker drawer, CloseObject drawer

Task description: Put two candles in the drawer under the sink.
Completed plans: Navigation candle
Visible objects are candle, drawer
Next Plans: PickupObject candle, Navigation drawer, OpenObject drawer, PutObject candle drawer, CloseObject drawer, Navigation countertop, PickupObject candle, Navigation drawer, OpenObject drawer, PutObject candle drawer, CloseObject drawer

Task description: Place a box with the credit card on the blue sofa chair.
Completed plans: Navigation box, PickupObject box, Navigation armchair
Visible objects are armchair, creditcard, box
Next Plans: PutObject box armchair, Navigation coffeetable, PickupObject creditcard, Navigation box, PutObject creditcard box, CloseObject box

Task description: Put watches on a shelf.
Completed plans: Navigation diningtable, PickupObject watch, Navigation shelf, PutObject watch shelf, Navigation diningtable, PickupObject watch, Navigation shelf
Visible objects are watch, shelf
Next Plans: PutObject watch shelf

Task description: Move the white towel from the wall to the cabinet under the sink on the left
Completed plans: Navigation handtowelholder, PickupObject handtowel
Visible objects are handtowel, cabinet
Next Plans: Navigation cabinet, OpenObject cabinet, PutObject handtowel cabinet, CloseObject cabinet

Task description: Transfer the two CDs from the desk to the vault.
Completed plans: 
Visible objects are mirror, safe
Next Plans:

Complete the next plans for the last steps
  \nASSISTANT: '''

  # raw_image = Image.open(requests.get(image_file, stream=True).raw)
  
  print(vlm(model, processor, text=prompt, image=raw_image))