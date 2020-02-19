# Functions follow camelCase
# Variables follow PascalCase
# Methods follow lowe_snake_case
# Fields follow lower_snake_case for better SQL compatibility


from __future__ import unicode_literals, print_function, division
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import torch.nn as nn
from io import open
import unicodedata
import string
import pandas
import random
import torch
import glob
import math
import time
import os

########## GLOBAL IO CONSTANTS #############
WORKING_DIR = '/Users/Joseph/Desktop/Software/Python/pytorch_nlp_tutorial/'
LANGUAGES_FILES = WORKING_DIR + 'languages/names/*.txt'
MODEL_SAVE_FILE = "geology_model.pt"
PRINT_EVERY = 5000
PLOT_EVERY = 1000

########## CHARACTER CONSTANTS #############
all_letters = string.ascii_letters + ".,;'"

########## CATEGORY GLOBAL VARIABLES #######
category_lines = {}
all_categories = []

########## TRAINING CONSTANTS ##############
LEARNING_RATE = 0.005
NUM_ITERS = 100000
NUM_CONFUSION = 10000
NUM_HIDDEN = 128
criterion = nn.NLLLoss()

########## DATA CONSTANTS ##################
PERCENT_TRAINING_DATA = 0.75



"""
Returns all paths that match the given pattern
"""
def findFiles(path): 
    return glob.glob(path)

"""
Converts unicode encoding into ASCII for use by the RNN
"""
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )

"""
Reads in a given file and returns the file
"""
def readLines(filename):
    df = pandas.read_excel(filename, index_col=None, header=None)
    # Removes all rows that have any empty cells
    df.dropna(axis=0, how='any', thresh=None, subset=None, inplace=True)
    # Extract pandas columns as <n x 1> dataframes
    names = df[[0]]
    categories = df[1].tolist()
    return names, categories

""" 
===================================
=========== DATA INPUT ============ 
===================================
"""
# Opens and loads all of the data from the excel file
filename = findFiles('geology\\*.xlsx')[0]
names, all_categories = readLines(filename)
testing_categories = []
category_set = [] # Used to remove all duplicates from list of categories

# Create a dictionary of category and the names belonging to that category
category_lines, testing_category_lines = {}, {}
num_categories, num_testing_categories = 0, 0
for i in range(len(names)):
    name = names.iloc[i, 0]
    for c in str(name):
        # Creates a list of all of the characters seen in data since not all are contained in ASCII
        if c not in all_letters:
            all_letters += c
    category = all_categories[i]

    chance = random.random()
    if chance > PERCENT_TRAINING_DATA:
        # Testing Data
        if category in testing_category_lines:
            testing_category_lines[category].append(name)
        else:
            testing_categories.append(category)
            num_testing_categories += 1
            testing_category_lines[category] = [name]
    else:
        # Training Data
        if category in category_lines:
            category_lines[category].append(name)
        else:
            category_set.append(category)
            num_categories += 1
            category_lines[category] = [name]
all_categories = category_set
num_letters = len(all_letters)
print(f"Gathered {num_categories} categories from xlsx file. {num_letters} characters in total")


"""
Returns index of the given letter in our letter bank
"""
def letterToIndex(letter):
    return all_letters.find(letter)

"""
One-hot encoding of a string, returns a <line_length x 1 x num_letters>
"""
def lineToTensor(line):
    tensor = torch.zeros(len(line), 1, num_letters)
    for li, letter in enumerate(line):
        tensor[li][0][letterToIndex(letter)] = 1
    return tensor

"""
Translates the output tensor into category names
"""
def categoryFromOutput(output):
    top_n, top_i = output.topk(1)
    category_i = top_i[0].item()
    return all_categories[category_i], category_i

"""
Returns a random integer between 0 and the last index of l
"""
def randomChoice(l):
    res = None
    while type(res) is not str:
        res = l[random.randint(0, len(l) - 1)]
    return res

"""
Grabs a random line to generate a training example from the list of training data
"""
def randomTrainingExample():
    category = randomChoice(all_categories)
    line = randomChoice(category_lines[category])
    category_tensor = torch.tensor([all_categories.index(category)], dtype=torch.long)
    
    line_tensor = lineToTensor(line)
    return category, line, category_tensor, line_tensor

"""
Grabs a random line to generate a testing example from the list of testing data
"""
def randomTestingExample():
    category = randomChoice(testing_categories)
    line = randomChoice(testing_category_lines[category])
    category_tensor = torch.tensor([testing_categories.index(category)], dtype=torch.long)
    
    line_tensor = lineToTensor(line)
    return category, line, category_tensor, line_tensor


def train(category_tensor, line_tensor):
    hidden = rnn.initHidden()

    rnn.zero_grad()

    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)

    
    loss = criterion(output, category_tensor)
    loss.backward()
    

    # Add parameters' gradients to their values, multiplied by learning rate
    for p in rnn.parameters():
        p.data.add_(-LEARNING_RATE, p.grad.data)

    return output, loss.item()

# Keep track of the losses for plotting
current_loss = 0
all_losses = []
def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

# Just return an output given a line
def evaluate(line_tensor):
    hidden = rnn.initHidden()

    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)

    return output

"""
Takes in a line and returns the top 3 predictions
"""
def predict(input_line, n_predictions=3):
    print('\n> %s' % input_line)
    with torch.no_grad():
        output = evaluate(lineToTensor(input_line))

        # Get top N categories
        topv, topi = output.topk(n_predictions, 1, True)
        predictions = []

        for i in range(n_predictions):
            value = topv[0][i].item()
            category_index = topi[0][i].item()
            print('(%.2f) %s' % (value, all_categories[category_index]))
            predictions.append([value, all_categories[category_index]])

"""
Continuously predicts based on the user's input
"""
def userInput():
    while True:
        user_input = input("Enter the word to predict, or exit() to exit: ")
        if (user_input == "exit()"):
            exit()
        predict(user_input)


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size

        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden
    

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)




""" 
==================================================
=========== NETWORK INIT AND TRAINING ============ 
==================================================
""" 
rnn = RNN(num_letters, NUM_HIDDEN, num_categories)

if os.path.exists(MODEL_SAVE_FILE):
    print("Loading file...")
    rnn = torch.load(MODEL_SAVE_FILE)
    rnn.eval()
    userInput()

print("File not found!")

input = lineToTensor('Asia')
hidden = torch.zeros(1, NUM_HIDDEN)

output, next_hidden = rnn(input[0], hidden)
print(categoryFromOutput(output))


start = time.time()
for iter in range(1, NUM_ITERS + 1):
    category, line, category_tensor, line_tensor = randomTrainingExample()
    
    output, loss = train(category_tensor, line_tensor)
    current_loss += loss
    if type(current_loss) is not float:
        print(current_loss)
        print(loss)

    # Print iter number, loss, name and guess
    if iter % PRINT_EVERY == 0:
        guess, guess_i = categoryFromOutput(output)
        correct = 'Y' if guess == category else 'N (%s)' % category
        print('%d %d%% (%s) %.4f %s / %s %s' % \
            (iter, iter / NUM_ITERS * 100, timeSince(start), float(loss), line, guess, correct))

    # Add current loss avg to list of losses
    if iter % PLOT_EVERY == 0:
        all_losses.append(current_loss / PLOT_EVERY)
        current_loss = 0

plt.figure()
plt.plot(all_losses)
plt.show()

torch.save(rnn, MODEL_SAVE_FILE)

userInput()
