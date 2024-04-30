# Trains the model for a given single image
# Enter a image filename and caption
# Example:
#     Image filename: `image.png`
#     caption: "<start> Kid playing <end>"
#     python3 train_model image.png '<start> Kid playing <end>'


import numpy as np
import cv2
import torch
import torchvision.transforms as transforms
import sys
import string


MAX_LEN = 34


if len(sys.argv) != 3:
    sys.stderr.write('Please enter image filename and caption filename as arguments!\n')
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
transform = transforms.Compose([transforms.ToTensor()])
x = transform(img).type(torch.float32)


# The caption
caption = sys.argv[2]



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



def preprocess(caption):
    # prepare translation table for removing punctuation
    table = str.maketrans('', '', string.punctuation)

    # tokenize
    caption = caption.split()

    # remove <start> and <end> token from list
    caption = caption[1:-1]

    # convert to lower case
    caption = [word.lower() for word in caption]
    # remove punctuation from each token
    caption = [w.translate(table) for w in caption]
    # remove hanging 's' and 'a'
    caption = [word for word in caption if len(word)>1]
    # remove tokens with numbers in them
    caption = [word for word in caption if word.isalpha()]

    caption = ['<start>'] + caption + ['<end>']

    return ' '.join(caption)



def create_y_in_and_y_out(caption):
    caption = preprocess(caption)
    caption_list = caption.split()

    y = []
    for token in caption_list:
        if token in word_to_idx:
            y.append(word_to_idx[token])
        else:
            y.append(word_to_idx['<unknown>'])
    

    y_in = []
    y_out = []
    for i in range(1, len(y)):
        y_in.append( [word_to_idx['<pad>'] for j in range(MAX_LEN - i)] + y[:i])
        y_out.append(y[i])


    y_in = torch.tensor(y_in).to(torch.device('cpu'))
    y_out = torch.tensor(y_out).to(torch.device('cpu'))

    return y_in, y_out


# Preparing data for training
y_in, y_out = create_y_in_and_y_out(caption)
X = np.asarray([x for i in range(len(y_in))])
X = torch.from_numpy(X).type(torch.float32)



# Loading the model
model = torch.jit.load('model_scripted.pt', map_location=torch.device('cpu'))



def train_model(model, X, y_in, y_out, epochs=1, learning_rate=1e-4):
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    model.train()
    for _ in range(epochs):
        outputs = model(X, y_in)
        loss = torch.nn.functional.cross_entropy(outputs, y_out)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


train_model(model, X, y_in, y_out)


# Save the trained model
model.save('model_scripted.pt')