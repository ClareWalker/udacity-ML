# Imports here

from ml_functions import load_checkpoint, predict
import json
import torch
import argparse

# Set arguments
parser = argparse.ArgumentParser()
parser.add_argument('image_path')
parser.add_argument('checkpoint')
parser.add_argument('--top_k', default=1)
parser.add_argument('--category_name', default='cat_to_name.json')
parser.add_argument('--gpu', action='store_true')

args = parser.parse_args()

# Load trained model from checkpoint
model = load_checkpoint(args.checkpoint)

# Use GPU if it's available
if args.gpu:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    device = "cpu"
    
model.to(device)

# Load mapping dictionaries and get name
class_to_idx = model.class_to_idx
with open(args.category_name, 'r') as f:
    cat_to_name = json.load(f)
  
# Get predictions
probs, classes = predict(args.image_path, model, device, top_k=int(args.top_k))

# Print result
for i in range(int(args.top_k)):
    prob = round(probs[i]*100)
    name = cat_to_name[classes[i]]
    print(f'{i+1}:   {name} with probability {prob}%')