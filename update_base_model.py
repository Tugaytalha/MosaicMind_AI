# Updates the Base Model in order to implement Federated Learning
# python3 update_base_model.py `json file name`


import numpy as np
import json
import torch
import sys


if len(sys.argv) != 2:
    sys.stderr.write("Enter json file\n")
    exit(1)

json_file_path = sys.argv[1]

with open(json_file_path) as f:
    json_file = json.load(f)





# Loading the model
model1 = torch.jit.load(json_file['model1'], map_location=torch.device('cpu'))
model2 = torch.jit.load(json_file['model2'], map_location=torch.device('cpu'))
model3 = torch.jit.load(json_file['model3'], map_location=torch.device('cpu'))
model4 = torch.jit.load(json_file['model4'], map_location=torch.device('cpu'))
base_model = torch.jit.load(json_file['base_model'], map_location=torch.device('cpu'))


model1_num = json_file['model1_num']
model2_num = json_file['model2_num']
model3_num = json_file['model3_num']
model4_num = json_file['model4_num']

num_sum = model1_num + model2_num + model3_num + model4_num


params_model1 = list(model1.parameters())
params_model2 = list(model2.parameters())
params_model3 = list(model3.parameters())
params_model4 = list(model4.parameters())


avg_params = []
for param1, param2, param3, param4 in zip(params_model1, params_model2, params_model3, params_model4):
    avg_param = param1*(model1_num/num_sum) + param2*(model2_num/num_sum) + param3*(model3_num/num_sum) + param4*(model4_num/num_sum)
    avg_params.append(avg_param)

# Set parameters of model3 to the average
with torch.no_grad():
    for base_model_param, avg_param in zip(base_model.parameters(), avg_params):
        base_model_param.copy_(avg_param)




# Save the trained model
base_model.save('model_scripted.pt')
