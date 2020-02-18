#################Implementation of Discrete CMAC###################
import numpy as np
import matplotlib.pyplot as plt
import random
import math

# defining a class for Associative cells
class association:
    def __init__(self,index,weight):
        self.index = 0
        self.weight = []



#preparing the data
freq_size = 100
freq = 1
sample_size = 100

x = np.arange(sample_size)
y = np.sin(2 * np.pi * freq * x / freq_size)# (freq * x / freq_size) -> gives the total frequency

#print(x)
#print('\n')
#print(y)       

plt.plot(x, y, 'c')
plt.show()


fn = np.stack((x.T, y.T), axis=0)
#print(x.T,'\n', y.T)
#print(x.T.shape)
#print(fn)
#print(fn.shape)
data = fn.T
#print(data)

#train_data = random.sample(data,70)
#print(train_data)

np.random.shuffle(data)

train_data = data[:70]
test_data = data[70:]

                
#Training data 
X = train_data[:, 0]
Y = train_data[:, 1]


#Testing data
X_test = test_data[:,0]
Y_test = test_data[:,1]

plt.figure(1)
plt.plot(X,Y,'c*',label = 'input data')
plt.legend()
plt.show()

#To check if the test cases are random and
#not being repeated--->
print(X_test)
print('\n')
print(Y_test)


bet = 5
assoc_num = 35

weights = np.ones((35,1))
rate = 1


def assoc_values(ind, bet):
    weights = []
    b = (bet//2)
    for i in range(ind - b, ind + b + 1):
        weights.append(i)
    return weights

def assoc_index(i,bet,assoc_num,sample):
    i = int(i)
    a_ind = bet//2 + ((assoc_num - 2*(bet//2))*i)/sample
    return math.floor(a_ind)

def meanSqEr(weights, synapse_weight,X,Y):
    meansq =0
    for i in range(0,len(synapse_weight)):
        sum_syn = 0
        for j in synapse_weight[i]:
            sum_syn = sum_syn + weights[j]
        meansq += (sum_syn - Y[i])**2
    return meansq

def test(weights,synapse_weight):
    output = []
    for i in range(0,len(synapse_weight)):
        sum_syn = 0
        for j in synapse_weight[i]:
            sum_syn += weights[j]
        output.append(sum_syn)
    return output


synapse = association([],[])
synapse_test = association([],[])


#Creating the Associative cells


for ind in X:
    synapse.index = assoc_index(ind, bet , assoc_num, sample)
    synapse.weight.append(assoc_values(synapse.index, bet))

for ix in X_test:
    synapse_test.index = (assoc_index(ix, bet , assoc_num, sample))
    synapse_test.weight.append(assoc_values(synapse_test.index, bet))
    


err_lst = []
err_plt = []

prevError = 0
currentError = 10
iterations = 0

while iterations < 100 and abs(prevError - currentError) > 0.00001:
    prevError = currentError
    #print(abs(prevError- currentError))
    for i in range(0,len(synapse.weight)):
        sum_syn = 0
        for j in synapse.weight[i]:
            sum_syn += weights[j]
            #print(sum_syn)
        error = sum_syn - Y[i]
        #print(error)
        correction  = error/bet
        #print(correction)
        for j in synapse.weight[i]:
            weights[j] -= rate*correction
            #print(correction)
    currentError = float(meanSqEr(weights,synapse.weight,X,Y))
    #print(currentError)
    err_lst.append(currentError)
    iterations += 1
    err_plt.append(iterations)

plt.figure(2)
plt.plot(np.asarray(err_plt), np.asarray(err_lst), 'b--',label = 'error convergence')
plt.legend()
plt.show()



#Testing the trained synapses
output = test(weights, synapse_test.weight)

plt.figure(3)
plt.plot(X,Y,'y*',label = 'training data')
plt.plot(X_test,Y_test,'c*',label = 'test data')
plt.plot(X_test,np.asarray(output),'k.', label = 'predicted outputs')
plt.legend()
plt.show()



plt.plot(X_test,Y_test,'m*',label = 'test data')
plt.legend()
plt.show()
