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
  inputs = processor(prompt, raw_image, return_tensors='pt').to(0, torch.float16)
  output = model.generate(**inputs, max_new_tokens=200, do_sample=False)

  return processor.decode(output[0][2:], skip_special_tokens=True)

if __name__ == '__main__':
  model, processor = get_model()
  image_file = "http://images.cocodataset.org/val2017/000000039769.jpg"

  prompt = "USER: <image>\nWhat are these?\nASSISTANT:"
  raw_image = Image.open(requests.get(image_file, stream=True).raw)
  
  print(vlm(model, processor, text=prompt, image=raw_image))