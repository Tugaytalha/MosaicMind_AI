# Generates caption for given image.
# Get a signle image at a time
# Give the file name of the file as argument


import numpy as np
import cv2
import torch
import torchvision.transforms as transforms
import sys


MAX_LEN = 34


argc = len(sys.argv)
if argc == 1:
    sys.stderr.write('Please enter filename of the image as an argument!\n')
    exit(1)
elif argc > 2:
    sys.stderr.write('Too many arguments. Please enter single file at a time!\n')
    exit(1)

# Loading Image
file_path = sys.argv[1]
try:
    img = cv2.imread(file_path)
except:
    sys.stderr.write('The file ' + file_path + ' cannot be opened!\n')
    exit(1)


# Preprocessing Image
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (224, 224))
img = img / 255 # normalize




# Loading the model
model = torch.jit.load('model_scripted.pt', map_location=torch.device('cpu'))




# Function that loads our word to index map
def load_word_to_idx(filename):
    word_to_idx = {}
    with open(filename, 'r') as file:
        while True:
            line = file.readline()
            line_splitted = line.split(',')
            if len(line_splitted) < 2:
                break
            word = line_splitted[0]
            # not include new line char and cast idx to int
            idx = int(line_splitted[1][:-1]) 
            word_to_idx[word] = idx
    return word_to_idx


# Load the word to index map
word_to_idx = load_word_to_idx('word_to_idx.txt')





# Function loads the vocabulary
def load_vocabulary_list(filename):
    vocabulary = []
    with open(filename, 'r') as file:
        while True:
            line = file.readline()
            line_splitted = line.split(',')
            if len(line_splitted) < 2:
                break
            word = line_splitted[0]
            vocabulary.append(word)
    return vocabulary


# Load the vocabulary
vocabulary = load_vocabulary_list('word_to_idx.txt')




# Takes one image at a time
def generate_caption(model, img, max_len=34):
    # initialize the y_in
    y_in = [word_to_idx['<pad>'] for i in range(max_len-1)] + [word_to_idx['<start>']]
    y_in = torch.tensor(y_in)
    
    # Convert img type from numpy tensor to pytorch tensor
    transform = transforms.Compose([transforms.ToTensor()])
    x = transform(img).type(torch.float32)


    model.eval()
    model = model.to(torch.device('cpu'))
    caption = []

    # add a batch dimension
    x = x.unsqueeze(0)
    y_in = y_in.unsqueeze(0)

    for _ in range(max_len-1):
        y_out = model(x, y_in)
        y_out = y_out.squeeze(0)
        y_out = torch.argmax(y_out).item()
        y_in = y_in.squeeze(0).tolist()
        y_in = y_in[1:] + [y_out]
        y_in = torch.tensor(y_in)
        y_in = y_in.unsqueeze(0)
        if y_out == word_to_idx['<end>']:
            break


    y_in = y_in.squeeze(0).tolist()
    for idx in y_in:
        if idx == word_to_idx['<pad>'] or idx == word_to_idx['<start>'] or idx == word_to_idx['<end>']:
            continue
        caption.append(vocabulary[idx])


    return " ".join(caption)



caption = generate_caption(model, img, MAX_LEN)


output_filename = file_path.split('.')[0] + '_output.txt'
f = open(output_filename, 'w')
f.write(caption + '\n')
f.close()
