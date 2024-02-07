import numpy as np  # numerical calculations in python
import pandas as pd # data analysis library
import matplotlib.pyplot as plt  # plotting similar to matlab
import torch  # PyTorch: the general machine learning framework in Python
import torch.optim as optim  # contains optimizers for the backpropagation
import torch.nn as nn  # the artificial neural network module in PyTorch
from tqdm import tqdm  # produces progress bars for for-loops in Python
from sklearn.model_selection import train_test_split  # randomly splits a dataset
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, RobustScaler  # scaling algorithm for preprocessing
import seaborn as sns # Statistical data visualization package
import os
os.environ["CUDA_VISIBLE_DEVICES"]="" #Forces pytorch to use CPU, ensuring reproducibility

# Setting up a seed 

seed = 2024
torch.manual_seed(seed)                     # Sets seed for pytorch
torch.cuda.manual_seed(seed)                # Sets seed for cuda
np.random.seed(seed)                        # Sets seed for random number generator
torch.backends.cudnn.deterministic = True   # Forces cuda to use deterministic approach

# Loading the dataset
datafile = './FTS_exp_results_Fernandez_et_al.csv'
fts_data = pd.read_csv(datafile, sep=',', header=0)
fts_data

# In case you want to visualize the pairplots
#sns.pairplot(data=fts_data)
#plt.show()

# Extracting data
x_all = fts_data[["P_{T}",	"H_{2}:CO", "U/W"]].values
y_all = fts_data[["X_{CO}", "LG",	"LPG",	"Gas",	"Dies",	"O_{2+}", "O_{4+}",	"O_{9+}"]].values

# Data splitting
frac_train = 0.70  # Ratio of first splitting of dataset (15 train / 5 test)
x_train, x_test, y_train, y_test = train_test_split(x_all, y_all, train_size = frac_train, random_state=seed)

# Data normalization

# Creating the normalization functions for the entries and outputs of the network
norm_entries = MinMaxScaler().fit(x_train)
norm_output = MinMaxScaler().fit(y_train)

# Implementing normalization functions to the datasets
norm_x_train = norm_entries.transform(x_train)
norm_y_train = norm_output.transform(y_train)
norm_x_test = norm_entries.transform(x_test)
norm_y_test = norm_output.transform(y_test)

# Creating the tensors for the training and test datasets
dtype = torch.float
xt_train = torch.tensor(norm_x_train, dtype=dtype).float()
xt_test = torch.tensor(norm_x_test, dtype=dtype).float()
yt_train = torch.tensor(norm_y_train, dtype=dtype).float()
yt_test = torch.tensor(norm_y_test, dtype=dtype).float()

# Defining the neural network class for this problem

class NeuralNetwork(nn.Module):
    def __init__(self, n_input, n_hidden, n_output):

        ''' Sets up the layout of the neural network.  
        
        n_iput: number of input neurons
        n_hidden: number of neurons in the hidden layers
        n_output: number of output neurons

        '''
        # Unpacking topology

        n_hidden1, n_hidden2, n_hidden3 = n_hidden      #unpacking topology of layers

        super(NeuralNetwork, self).__init__()
        self.architecture = nn.Sequential(
            # sequential model definition: add up layers & activation functions
            nn.Linear(in_features=n_input, out_features=n_hidden1, bias=True),  # input layer
            nn.ReLU(), # activation function
            
            # First hidden layer

            nn.Linear(in_features=n_hidden1, out_features=n_hidden2, bias=True),  # hidden layer
            nn.ReLU(), # activation function of hidden layer

            # Second hidden layer

            nn.Linear(in_features=n_hidden2, out_features=n_hidden3, bias=True),  # hidden layer
            nn.ReLU(), # activation function of hidden layer

        # Third hidden layer

            nn.Linear(in_features=n_hidden3, out_features=n_output, bias=True),  # hidden layer
            nn.ReLU(), # activation function of hidden layer

            # Output layer
            nn.Linear(in_features=n_output, out_features=n_output, bias=True),   # output layer

        )
    def forward(self, input): # feed forward path
        output = self.architecture(input)
        return output

hidden_size = [20, 25, 25] # number of neurons in h1, h2, h3
learning_rate = 0.5e-3

# neural network training
net = NeuralNetwork(3, hidden_size, 8) # create instance of neural network (inputs, hidden layer topology, outputs)
optimizer = optim.Adam(net.parameters(), lr=learning_rate) # choose optimizer and learning rate
loss_fun = nn.MSELoss() # define loss function
epochs = 150000 # set number of epochs
net             # prints the network topology

# Training

# Lists to store the loss values during training and testing
train_loss = []
val_loss = []

# train the network
for epoch in tqdm(range(epochs)):

    # training data
    optimizer.zero_grad() # clear gradients for next training epoch
    y_pred = net(xt_train) # forward pass: prediction y based on input x
    loss = loss_fun(y_pred, yt_train)  # compare true y and predicted y to get the loss
    loss.backward() # backpropagation, compute gradients
    optimizer.step() # apply gradients to update weights and biases
    train_loss.append(loss.item()) # save loss for later evaluation

    # validation data
    y_val = net(xt_test) # prediction of y based on input from validation set
    loss = loss_fun(y_val, yt_test) # compare true y and predicted y to get the loss
    val_loss.append(loss.item()) # save loss for later evaluation

# Training and validation graph

plt.figure(figsize=(10,4))
plt.plot(train_loss, label='Training')
plt.plot(val_loss, label='validation', linestyle='--')
#plt.yscale('log')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend()
plt.title('Loss plot')
plt.show()
print('Final training loss: ')
print(train_loss[-1])

# Summary

print('Hidden layer size: ', hidden_size)
print('Learning rate: ', learning_rate)
print('Validation MSE: {:.2f}'.format(val_loss[-1])) # last element from the validation loss

# Defining functions for computing the error 

def rsme(prediction, true_result):
    return np.sqrt(((prediction - true_result)**2).mean())

def ope(prediction, true_result):
    ''' Computes the absolute error of the predictions '''

    return np.sum(np.abs(prediction - true_result))

def mpe(prediction, true_result):
    ''' Relative error [%] according to Fernandes et al., 2006 '''

    return np.sum(np.abs(prediction - true_result)/true_result)/np.size(true_result)*100

# IMPLEMENTATION 

# Comparison of the prediction and ground truth values for the 1st run in Fernandes et al., 2006
op_conditions = np.array([1.5, 2.0, 1.5]).reshape(-1,1)

# Normalization of operational conditions
norm_op_conditions = norm_entries.transform(op_conditions.T)  # The normalized entries

# Converting operational conditions to tensor
xt_op = torch.tensor(norm_op_conditions, dtype=torch.float)

# Predicting product distribution with NN
with torch.no_grad():
    sol = net(xt_op)

# Extracting the prediction
prediction = sol.detach().numpy()
rescaled_prediction = norm_output.inverse_transform(prediction) # Rescaling the output

# Result suggested by Fernandes et al., 2006
# Order: xco, LG, LPG, Gasoline, Diesel, O2+, O4+, O9+
expected_result = np.array(([46.2, 0.226, 0.069, 0.101, 0.070, 0.140, 0.339, 0.055]))

# Calculating the RSME of the prediction
error = rsme(rescaled_prediction, expected_result)

#Calculating the MPE and OPE according to the definitions of Fernandes et al., 2006
error_mpe = mpe(rescaled_prediction, expected_result)
error_ope = ope(rescaled_prediction, expected_result)

# Printing the results 
print('The operating conditions : \n \
      Pressure: {:.2f} MPa \n \
      H2:CO ratio: {:.2f}  \n \
      Space velocity: {:.2f} \n \
      result in the following values for: \n \
       - OPE: {:.2f} \n \
       - MPE {:.2f} %'.format(op_conditions[0,0], 
                              op_conditions[1,0], 
                              op_conditions[2,0], 
                               error_ope, error_mpe))

# Optimal conditions reported by Fernandes et al., 2006 (P [MPa], H2:CO ratio, phi_cat [10^3 N m3 / g_cat / s])
opt_conditions = torch.tensor(np.array(([2.51, 1.75, 0.50])), dtype=torch.float).unsqueeze(-1)
opt_conditions.shape

# Normalizing the operating conditions for the NN
norm_opt_conditions = norm_entries.transform(opt_conditions.T)  # The normalized entries

# Creates the tensor of the optimal conditions
xt_opt = torch.tensor(norm_opt_conditions, dtype=torch.float)

# Implementing the network to predict product distribution
with torch.no_grad():
    sol = net(xt_opt)

# Extracting the prediction and rescale it 
prediction = sol.detach().numpy()
rescaled_prediction = norm_output.inverse_transform(prediction)

# Post-processing the prediction according to the variables reported by Fernandes et al., 2006
# i.e., olefins O2+ and O4+ are combined 
opt_prediction = np.append(rescaled_prediction[0,:4], 
                                [rescaled_prediction[0,5]+ rescaled_prediction[0,6],
                                rescaled_prediction[0,7]])

# Results for the optimum conditions for producing gasoline according to Fernandes et al., 2006
# i.e., Xco, LG, LPG, Gasoline, Diesel, O2+ + O4+ , O9+
expected_result = np.array(([80.3, 0.287, 0.124, 0.156, 0.308, 0.017])) 

# Calculating OPE
error_ope = ope(opt_prediction, expected_result)

# Calculating MPE
error_mpe = mpe(opt_prediction, expected_result)

# Printing the results 
print('The operating conditions : \n \
      Pressure: {:.2f} MPa \n \
      H2:CO ratio: {:.2f}  \n \
      Space velocity: {:.2f} \n \
      result in the following values for: \n \
       - OPE: {:.2f} \n \
       - MPE {:.2f} %'.format(op_conditions[0,0], 
                              op_conditions[1,0], 
                              op_conditions[2,0], 
                               error_ope, error_mpe))


