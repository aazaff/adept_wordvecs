# Functions follow camelCase
# Variables follow PascalCase
# Methods follow lowe_snake_case
# Fields follow lower_snake_case for better SQL compatibility

# Python has no Multi-Value logic implementation. Therefore, pre-indexing arrays to hold outputs
# using functions like numpy.empty are avoided as much as possible to avoid uncatchable errors.

# This script is a modification of the basic neural network tutorial provided by pytorch.org
# It is mostly identical, but I changed the namespace a bit to be somewhat clearer
# https://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html

#############################################################################################################
############################################## CONFIGURATION, SCRIPT ########################################
#############################################################################################################    
# a __future__ statement is different than a regular import that seems to handle high-level configuration
# and inter-version compatiblity. They always need to go at the top of a file
from __future__ import unicode_literals, print_function, division

# All of these imports assume prior installation
from io import open
import glob # glob is essentially a way to use unix shell syntax (nice!)
import os
import unicodedata
import string

# Load in the pytorch and numpy
import torch
# import numpy, commented out because not actually used
import torch.nn as nnet
import random

# These are used simply for benchmarking and progbar on the nnet
import time
import math

# Set a working directory. Currently local, but can be reconfigured later to hit the GitHub repo
Working = '/Users/azaffos/Box Sync/GitRepositories/pytorch_nlp_tutorial/data/names/'

#############################################################################################################
####################################### LOAD DATA FUNCTIONS, TEMPLATE #######################################
#############################################################################################################
# A simple function for displaying stdout
def findFiles(path): return glob.glob(path)

# A function to turn unicode to ASCII from https://stackoverflow.com/a/518232/2809427
# I am not even gonna pretend to try and figure out how this one works today
def convertUnicode(char_string):
    return ''.join(
            c for c in unicodedata.normalize('NFD',char_string)
            if unicodedata.category(c) != 'Mn'
            and c in string.ascii_letters + " .,;'"
    )

# Read in the file - a lot of hard coded stuff such as the delimeter in .split
# probably should design a more flexible version in the future
def readLines(file):
    lines = open(file, encoding='utf-8').read().strip().split('\n')
    return [convertUnicode(line) for line in lines]

######################################### LOAD DATA SCRIPT, TEMPLATE ########$$##############################
# Generate a random numpy array
# RandNump = numpy.random.rand(2,2)
# equivalent of runif() is random_sample
# equivalent of rnorm() is randn
# equivalent of sample() is choice

# Generate a random torch tensor
# RandTorch = torch.rand(2,2)

# Move between torch and numpy
# TorchNump = RandTorch.numpy() 
# NumpTorch = torch.from_numpy(RandNump)

# This is a script to load in the training sets provided in the demo
# that are stored in the Working directory. I would prefer if this was functionalized
Names = {}
Languages = []
for filename in findFiles(Working+'/*.txt'):
    language = os.path.splitext(os.path.basename(filename))[0]
    Languages.append(language)
    lines = readLines(filename)
    Names[language] = lines

#############################################################################################################
####################################### SHAPE DATA FUNCTIONS, TEMPLATE ######################################
#############################################################################################################
# This is something called a "one-hot vector" where the index position indicates the letter
# For example, in an array from 1:26, a 1 at 26 indicates a Z. I'm not sure if this is
# a common NLP tool or specific to Tensors/PyTorch
def hotVec(text):
     charspace = string.ascii_letters + " .,;'" # this is the character space - my little joke on namespace
     tensor = torch.zeros(len(text),1,len(charspace))
     for index, value in enumerate(text):
         tensor[index][0][charspace.find(value)] = 1
     return tensor

####################################### SHAPE DATA SCRIPT, TEMPLATE #########################################
# No script at this time

#############################################################################################################
#################################### BUILD TRAINING SET FUNCTIONS, TEMPLATE #################################
#############################################################################################################
# Generate random samples of the training data
def randomChoice(vector):
    return vector[random.randint(0, len(vector) - 1)] # lol at zero indexing

def randomTrainingExample():
    language = randomChoice(Languages)
    name = randomChoice(Names[language])
    language_tensor = torch.tensor([Languages.index(language)], dtype=torch.long)
    name_tensor = hotVec(name)
    return language, name, language_tensor, name_tensor

#############################################################################################################
################################# CONFIGURE HYPERPARAMS FUNCTIONS, TEMPLATE #################################
#############################################################################################################
for i in range(10):
    language, name, language_tensor, name_tensor = randomTrainingExample()
    print('language =', language, '/ name =', name)
     
####################################### SHAPE DATA SCRIPT, TEMPLATE #########################################
# We define a series of arbitrary hyperparameters that specify model run_time, iterations, and other configuration
# values  

n_hidden = 128 # number of hidden layers
n_iters = 100000 # number of iterations of the model
n_chars = len(string.ascii_letters + " .,;'") # the length of hotvectors... ugh global vs local in python sucks
criterion = nnet.NLLLoss() # the Rish temptation is to think of NLLLoss as a function, but it is actually establishing a new class
# Hence, the temptation to just bypass criterion and call nnet.NLLLoss woudln't work unless if I did
# nnet.NLLLoss()(Input,Output)



#############################################################################################################
####################################### NEURAL NET FUNCTIONS, TEMPLATE ######################################
#############################################################################################################
# We are creating a Recurrent Nueral Network
class RNN(nnet.Module):
     # the __init__ is where you essentially set up the modules, essentailly a mini configuration for the class
     # Essentially here you are definingboth hidden and explicit layers. The if2h and if2o and softmax
     # terminology are hard to decipher at this time, but I suspect it is because this is a recurrent
     # rather than regular neural network... incidentally recurrent makes 100% perfect sense in the case of
     # sequential analyases of charcers like in NLP
     def __init__(self,input_size,hidden_size,output_size):
         super(RNN,self).__init__()
         self.hidden_size = hidden_size
         self.i2h = nnet.Linear(input_size + hidden_size, hidden_size)
         self.i2o = nnet.Linear(input_size + hidden_size, output_size)
         self.softmax = nnet.LogSoftmax(dim=1)
         
     def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden

     def initHidden(self):
        return torch.zeros(1, self.hidden_size)

# Interpret the output of the network. 
# Use Tensor.topk to get the index of the greatest likelihood value
def categoryFromOutput(output):
    top_n, top_i = output.topk(1)
    category_i = top_i[0].item()
    return Languages[category_i], category_i

# Define the training function
def train(language_tensor, name_tensor):
    hidden = rnn.initHidden()

    rnn.zero_grad()

    for i in range(name_tensor.size()[0]):
        output, hidden = rnn(name_tensor[i], hidden)

    loss = criterion(output, language_tensor)
    loss.backward()

    # Add parameters' gradients to their values, multiplied by learning rate
    for p in rnn.parameters():
        p.data.add_(-0.005, p.grad.data)

    return output, loss.item()

# progress bar function
def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


####################################### NEURAL NET SCRIPT, TEMPLATE #########################################
# Execute the model
start = time.time() # specify a star time
rnn = RNN(n_chars, n_hidden, len(Languages))
current_loss = 0 # set the initial loss function value to zero
all_losses = [] # create a blank array to hold all loss values as iterated on through the nnet

for iter in range(1, n_iters + 1):
    language, name, language_tensor, name_tensor = randomTrainingExample()
    output, loss = train(language_tensor, name_tensor)
    current_loss += loss

    # Print iter number, loss, name and guess
    # I hard-coded 500 as the print parameters
    if iter % 5000 == 0:
        guess, guess_i = categoryFromOutput(output)
        correct = '✓' if guess == language else '✗ (%s)' % language
        print('%d %d%% (%s) %.4f %s / %s %s' % (iter, iter / n_iters * 100, timeSince(start), loss, name, guess, correct))

    # Add current loss avg to list of losses
    # I hardcoded print_every at 1000
    if iter % 1000 == 0:
        all_losses.append(current_loss / 1000)
        current_loss = 0 # reset loss function