## NIRANJAN_ANANDAKUMAR_BATCH_2_ASSIGNMENT2
### Assignment 2A and 2B ..
- [Github - Stallon-Niranjan,  Assignment Uploads.](https://github.com/Stallon-niranjan/Inkers_EIP_WDocs)

```python
# Simple Neural Network in python 
import numpy as np
import pandas as pd
```


```python
#First Line of The Codes from Reference and 
# Assignment 2A, in th Last Assignment is in Concise Codes , 
#For Assignment 2B.. 
```


```python
#Step 0 : Read Input and Output
eip_in = np.array([[1,0,1,0],[1,0,1,1],[0,1,0,1]])
eip = np.array([[1],[1],[0]])
print(eip_in)
print(eip)
```
Input
    [[1 0 1 0]
     [1 0 1 1]
     [0 1 0 1]]
     
     Output 
    [[1]
     [1]
     [0]]
    


```python

```


```python
#Step 1: Initialize weights and biases with random values
##(There are methods to initialize weights and biases but for now initialize with random values)

mlblr = np.array([[0.42,0.88,0.55],[0.1,0.73,0.68],[0.6,0.18,0.47],[0.92,0.11,0.52]])
print("Weights ",mlblr)
print("-----------------")
bh = np.array([0.46,0.72,0.08])
print("Bias ",bh)
```

    Weights 
    [[ 0.42  0.88  0.55]
     [ 0.1   0.73  0.68]
     [ 0.6   0.18  0.47]
     [ 0.92  0.11  0.52]]
    -----------------
    Bias  [ 0.46  0.72  0.08]
    


```python
#Step 2 Calculate hidden layer input:
mlblr_in = np.dot(eip_in,mlblr) + bh
print("Hidden Layer Inputs\n",mlblr_in)
```

    Hidden Layer Inputs
     [[ 1.48  1.78  1.1 ]
     [ 2.4   1.89  1.62]
     [ 1.48  1.56  1.28]]
    


```python
#Step 3: Perform non-linear transformation on hidden linear input
#hiddenlayer_activations = sigmoid(hidden_layer_input)
def sigmoid(x, derivative=False):
 return x*(1-x) if derivative else 1/(1+np.exp(-x))

mlblr_activations = sigmoid(mlblr_in)
print("Hidden Layer Activations\n",mlblr_activations)
```

    Hidden Layer Activations
     [[ 0.81457258  0.85569687  0.75026011]
     [ 0.9168273   0.86875553  0.83479513]
     [ 0.81457258  0.82635335  0.78244978]]
    


```python
#Step 4: Perform linear and non-linear transformation of hidden layer activation at output layer
#output_layer_input = matrix_dot_product (hiddenlayer_activations * wout ) + bout
#output = sigmoid(output_layer_input)
wout = np.array([[0.3],[0.25],[0.23]])
bout = np.array([0.69])

mlblr_out = np.dot(mlblr_activations,wout) + bout
eip_out = sigmoid(mlblr_out)
print("Output :\n",eip_out)

```

    Output :
     [[ 0.78932406]
     [ 0.79806432]
     [ 0.78933532]]
    


```python
#Step 5: Calculate gradient of Error(E) at output layer
#Error = y - output

E = eip - eip_out
print(E)

```

    [[ 0.21067594]
     [ 0.20193568]
     [-0.78933532]]
    


```python
#Step 6: Compute slope at output and hidden layer
#`Slope_output_layer= derivatives_sigmoid(output)

#Slope_hidden_layer = derivatives_sigmoid(hidden_layer_activations)`
#Sigmoid Function
def derivative_sigmoid(x):
    return x * (1 - x)
#`Slope_output_layer = derivatives_sigmoid(output)
Slope_eip_out = derivative_sigmoid(eip_out)
print("Slope_output_layer \n",Slope_eip_out)

#Slope_hidden_layer = derivatives_sigmoid(hidden_layer_activations)`
Slope_mlblr =  derivative_sigmoid(mlblr_activations)
print("\nSlope_hidden_layer \n",Slope_mlblr)
```

    Slope_output_layer 
     [[ 0.16629159]
     [ 0.16115766]
     [ 0.16628507]]
    
    Slope_hidden_layer 
     [[ 0.15104409  0.12347974  0.18736988]
     [ 0.076255    0.11401936  0.13791222]
     [ 0.15104409  0.14349349  0.17022212]]
    


```python
#Step 7: Compute delta at output layer
# d_output = E * slope_output_layer*lr
lr = 0.01 # learning Rate
d_eip_out = E * Slope_eip_out
print("delta at output layer \n", d_eip_out)
```

    delta at output layer 
     [[ 0.03503364]
     [ 0.03254348]
     [-0.13125468]]
    


```python
#Step 8: Calculate Error at hidden layer
#Error_at_hidden_layer = matrix_dot_product(d_output, wout.Transpose)
E_mlblr_in = np.dot(d_eip_out, wout.T)
print("Error_at_hidden_layer \n",E_mlblr_in)
```

    Error_at_hidden_layer 
     [[ 0.01051009  0.00875841  0.00805774]
     [ 0.00976304  0.00813587  0.007485  ]
     [-0.0393764  -0.03281367 -0.03018858]]
    


```python
#Step 9: Compute delta at hidden layer
#d_hiddenlayer = Error_at_hidden_layer * slope_hidden_layer

d_mlblr_in = E_mlblr_in * Slope_mlblr
print(" delta at hidden layer \n",d_mlblr_in)


```

     delta at hidden layer 
     [[ 0.00158749  0.00108149  0.00150978]
     [ 0.00074448  0.00092765  0.00103227]
     [-0.00594757 -0.00470855 -0.00513876]]
    


```python
#Step 10: Update weight at both output and hidden layer
#wout = wout + matrix_dot_product (hiddenlayer_activations.Transpose, d_output) * learning_rate
wout = wout + np.dot(mlblr_activations.T,d_eip_out)*lr
print("New Weights at the Output Layer \n",wout,"\n---------------\n")

#wh = wh+ matrix_dot_product (X.Transpose,d_hiddenlayer) * learning_rate
mlblr = mlblr + np.dot(eip_in.T,d_mlblr_in) *lr
print("New Weights in Hidden Layer \n",mlblr)

```

    New Weights at the Output Layer 
     [[ 0.29951458]
     [ 0.24949788]
     [ 0.22950751]] 
    ---------------
    
    New Weights in Hidden Layer 
     [[ 0.42002332  0.88002009  0.55002542]
     [ 0.09994052  0.72995291  0.67994861]
     [ 0.60002332  0.18002009  0.47002542]
     [ 0.91994797  0.10996219  0.51995894]]
    


```python
#Step 11: Update biases at both output and hidden layer
#bout = bout + sum(d_output, axis=0)*learning_rate
bout = bout + np.sum(d_eip_out,axis=0, keepdims=True)*lr
print("New Bias at the Output Layer \n",bout,"\n---------------\n")
#bh = bh + sum(d_hiddenlayer, axis=0) * learning_rate
bh = bh + np.sum(d_mlblr_in, axis=0)*lr
print("New Bias in Hidden Layer \n",bh)

```

    New Bias at the Output Layer 
     [[ 0.68936322]] 
    ---------------
    
    New Bias in Hidden Layer 
     [ 0.45996384  0.71997301  0.07997403]
    


```python
### Assignment 2B : Python File in Concise Program..
import numpy as np

#Input array
X = np.array([[1,0,1,0],[1,0,1,1],[0,1,0,1]])
print("X \n",X)
#Output
y = np.array([[1],[1],[0]])
print("y \n",y)
#Sigmoid Function
def sigmoid (x):
    return 1/(1 + np.exp(-x))

#Derivative of Sigmoid Function
def derivatives_sigmoid(x):
    return x * (1 - x)

#Variable initialization
##----------------------###
#Setting training iterations
epochs = 3

#Setting learning rate
lr1 = 0.1 

#number of features in data set
inputlayer_neurons = X.shape[1]

#number of hidden layers neurons
hiddenlayer_neurons = 3 

#number of neurons at output layer
output_neurons = 1

#weight and bias initialization
print("Random Weights and Bias numbers are choosen \n")
wh1 = np.random.uniform(size=(inputlayer_neurons,hiddenlayer_neurons))
print("wh1 \n",wh1)
bh1 = np.random.uniform(size=(1,hiddenlayer_neurons))
print("bh1 \n",bh1)
wout1 = np.random.uniform(size=(hiddenlayer_neurons,output_neurons))
print("Wout1 \n",wout1)
bout1 = np.random.uniform(size=(1,output_neurons))
print("Bout1 \n",bout1)

for i in range(epochs):
    print("\n Epoch No : ",i+1)
    #Forward Propogation
    hidden_layer_input1 = np.dot(X,wh1)
    
    hidden_layer_input = hidden_layer_input1 + bh1
    print("hidden layer input \n",hidden_layer_input)

    hiddenlayer_activations = sigmoid(hidden_layer_input)
    print("hiddenlayer activations \n",hiddenlayer_activations)

    output_layer_input1 = np.dot(hiddenlayer_activations,wout1) 
        
    output_layer_input = output_layer_input1+ bout1
    print(" output layer input \n",output_layer_input)
    
    output = sigmoid(output_layer_input)
    print("output \n",output)
    
    #Backpropagation
    E = y-output
    print("Error or Loss \n",E)
    
    slope_output_layer = derivatives_sigmoid(output)
    print("slope output layer \n",slope_output_layer)
    
    slope_hidden_layer = derivatives_sigmoid(hiddenlayer_activations)
    print("slope hidden layer \n",slope_hidden_layer)

    d_output = E * slope_output_layer
    print("delta at output \n", d_output)

    Error_at_hidden_layer = d_output.dot(wout1.T)
    print("Error at hidden layer \n",Error_at_hidden_layer)

    d_hiddenlayer = Error_at_hidden_layer * slope_hidden_layer
    print("delta at hidden layer \n",wout1)

    wout1 += hiddenlayer_activations.T.dot(d_output) *lr1
    print("New Weights at Output Layer\n",wout1)

    bout1 += np.sum(d_output, axis=0,keepdims=True) *lr1
    print(" New Bias at Output Layer \n",bout1)
    
    wh1 += X.T.dot(d_hiddenlayer) *lr1
    print("New Weights at Hidden Layer\n",wh1)
    bh1 += np.sum(d_hiddenlayer, axis=0,keepdims=True) *lr1
    print(" New Bias at Hidden Layer \n",bh1)

print ("\n Final Output \n",output)
```

    X 
     [[1 0 1 0]
     [1 0 1 1]
     [0 1 0 1]]
    y 
     [[1]
     [1]
     [0]]
    Random Weights and Bias numbers are choosen 
    
    wh1 
     [[ 0.98843715  0.73626613  0.66268227]
     [ 0.86914818  0.97597204  0.7479577 ]
     [ 0.61006716  0.86879147  0.71780495]
     [ 0.2937741   0.6825182   0.0073115 ]]
    bh1 
     [[ 0.79383757  0.29845415  0.12919281]]
    Wout1 
     [[ 0.91897081]
     [ 0.99616049]
     [ 0.16463027]]
    Bout1 
     [[ 0.59926697]]
    
     Epoch No :  1
    hidden layer input 
     [[ 2.39234188  1.90351176  1.50968003]
     [ 2.68611597  2.58602995  1.51699153]
     [ 1.95675985  1.95694439  0.88446201]]
    hiddenlayer activations 
     [[ 0.91624147  0.87028847  0.81901378]
     [ 0.93620239  0.92995706  0.82009504]
     [ 0.87618187  0.87620188  0.707746  ]]
     output layer input 
     [[ 2.44304759]
     [ 2.52100859]
     [ 2.39380665]]
    output 
     [[ 0.92005155]
     [ 0.92560154]
     [ 0.91635381]]
    Error or Loss 
     [[ 0.07994845]
     [ 0.07439846]
     [-0.91635381]]
    slope output layer 
     [[ 0.0735567 ]
     [ 0.06886333]
     [ 0.07664951]]
    slope hidden layer 
     [[ 0.07674304  0.11288645  0.14823021]
     [ 0.05972747  0.06513693  0.14753917]
     [ 0.1084872   0.10847214  0.2068416 ]]
    delta at output 
     [[ 0.00588074]
     [ 0.00512333]
     [-0.07023807]]
    Error at hidden layer 
     [[ 0.00540423  0.00585817  0.00096815]
     [ 0.00470819  0.00510365  0.00084345]
     [-0.06454673 -0.06996839 -0.01156331]]
    delta at hidden layer 
     [[ 0.91897081]
     [ 0.99616049]
     [ 0.16463027]]
    New Weights at Output Layer
     [[ 0.91383515]
     [ 0.99099445]
     [ 0.16056101]]
     New Bias at Output Layer 
     [[ 0.59334357]]
    New Weights at Hidden Layer
     [[ 0.98850674  0.73636551  0.66270906]
     [ 0.86844793  0.97521308  0.74771852]
     [ 0.61013676  0.86889085  0.71783174]
     [ 0.29310197  0.68179248  0.00708477]]
     New Bias at Hidden Layer 
     [[ 0.79320692  0.29779456  0.12898043]]
    
     Epoch No :  2
    hidden layer input 
     [[ 2.39185041  1.90305092  1.50952123]
     [ 2.68495238  2.5848434   1.516606  ]
     [ 1.95475681  1.95480012  0.88378372]]
    hiddenlayer activations 
     [[ 0.91620374  0.87023644  0.81899024]
     [ 0.93613286  0.92987973  0.82003815]
     [ 0.8759644   0.8759691   0.70760569]]
     output layer input 
     [[ 2.42450014]
     [ 2.50198649]
     [ 2.37552503]]
    output 
     [[ 0.91867658]
     [ 0.92428096]
     [ 0.91494182]]
    Error or Loss 
     [[ 0.08132342]
     [ 0.07571904]
     [-0.91494182]]
    slope output layer 
     [[ 0.07470992]
     [ 0.06998566]
     [ 0.07782328]]
    slope hidden layer 
     [[ 0.07677445  0.11292498  0.14824523]
     [ 0.05978813  0.06520341  0.14757558]
     [ 0.10865077  0.10864723  0.20689988]]
    delta at output 
     [[ 0.00607567]
     [ 0.00529925]
     [-0.07120378]]
    Error at hidden layer 
     [[ 0.00555216  0.00602095  0.00097552]
     [ 0.00484264  0.00525152  0.00085085]
     [-0.06506851 -0.07056255 -0.01143255]]
    delta at hidden layer 
     [[ 0.91383515]
     [ 0.99099445]
     [ 0.16056101]]
    New Weights at Output Layer
     [[ 0.90865068]
     [ 0.98577872]
     [ 0.15645474]]
     New Bias at Output Layer 
     [[ 0.58736068]]
    New Weights at Hidden Layer
     [[ 0.98857832  0.73646774  0.66273608]
     [ 0.86774096  0.97444644  0.74748198]
     [ 0.61020834  0.86899308  0.71785876]
     [ 0.29242395  0.68106008  0.00686078]]
     New Bias at Hidden Layer 
     [[ 0.79257152  0.29713015  0.12877091]]
    
     Epoch No :  3
    hidden layer input 
     [[ 2.39135818  1.90259098  1.50936575]
     [ 2.68378212  2.58365105  1.51622653]
     [ 1.95273642  1.95263667  0.88311367]]
    hiddenlayer activations 
     [[ 0.91616594  0.87018449  0.81896719]
     [ 0.93606285  0.92980195  0.81998215]
     [ 0.87574471  0.87573386  0.70746703]]
     output layer input 
     [[ 2.40577614]
     [ 2.4827839 ]
     [ 2.35707309]]
    output 
     [[ 0.9172667 ]
     [ 0.92292606]
     [ 0.91349479]]
    Error or Loss 
     [[ 0.0827333 ]
     [ 0.07707394]
     [-0.91349479]]
    slope output layer 
     [[ 0.0758885 ]
     [ 0.07113355]
     [ 0.07902205]]
    slope hidden layer 
     [[ 0.07680591  0.11296344  0.14825993]
     [ 0.05984919  0.06527028  0.14761143]
     [ 0.10881591  0.10882407  0.20695743]]
    delta at output 
     [[ 0.00627851]
     [ 0.00548254]
     [-0.07218624]]
    Error at hidden layer 
     [[ 0.00570497  0.00618922  0.0009823 ]
     [ 0.00498172  0.00540457  0.00085777]
     [-0.06559207 -0.07115965 -0.01129388]]
    delta at hidden layer 
     [[ 0.90865068]
     [ 0.98577872]
     [ 0.15645474]]
    New Weights at Output Layer
     [[ 0.90341743]
     [ 0.98051324]
     [ 0.15231155]]
     New Bias at Output Layer 
     [[ 0.58131817]]
    New Weights at Hidden Layer
     [[ 0.98865195  0.73657293  0.66276331]
     [ 0.86702721  0.97367205  0.74724824]
     [ 0.61028197  0.86909827  0.71788598]
     [ 0.29174001  0.68032097  0.00663971]]
     New Bias at Hidden Layer 
     [[ 0.79193141  0.29646096  0.1285644 ]]
    
     Final Output 
     [[ 0.9172667 ]
     [ 0.92292606]
     [ 0.91349479]]
    


```python

```


```python

```
