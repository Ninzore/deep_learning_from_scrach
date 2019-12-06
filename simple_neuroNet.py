import numpy
import scipy.special
import matplotlib.pyplot
import pandas

class neuralNet:
    def __init__(self, in_node, out_node, hid_node, learning_rate):
        self.in_node = in_node
        self.hid_node = hid_node
        self.out_node = out_node
        
        #learning rate
        self.learning_rate = learning_rate

        #weights
        self.w_i2h = numpy.random.normal(0,  pow(self.hid_node, -0.5), (self.hid_node, self.in_node))
        self.w_h2o = numpy.random.normal(0,  pow(self.out_node, -0.5), (self.out_node, self.hid_node))

        #activation function
        self.activation = lambda S: scipy.special.expit(S)

        pass

    def forward(self, in_list, target_list):
        #convert inputs list to 2d array
        in_arr = numpy.array(in_list, ndmin=2).T
        tar_arr = numpy.array(target_list, ndmin=2).T

        #calculate i2h
        hid_in = numpy.dot(self.w_i2h, in_arr)
        #i2h activation function
        hid_out = self.activation(hid_in)
        #calculate h2o
        final_in = numpy.dot(self.w_h2o, hid_out)
        #i2h activation function
        final_out = self.activation(final_in)

        #calculate errors
        out_error = tar_arr - final_out
        #hidden error is the output error splits by weights
        hid_error = numpy.dot(self.w_h2o.T, out_error)

        #update hidden output weight
        self.w_h2o += self.learning_rate * numpy.dot((out_error * final_out * (1 - final_out)), numpy.transpose(hid_out))
        #update input hidden weight
        self.w_i2h += self.learning_rate * numpy.dot((hid_error * hid_out * (1 - hid_out)), numpy.transpose(in_arr))

        pass

    #query the neural network
    def query(self, in_list):
        #convert input into 2D array
        in_arr = numpy.array(in_list, ndmin=2).T

        #calculate i2h
        i2h_i = numpy.dot(self.w_i2h, in_arr)
        #i2h activation function
        i2h_o = self.activation(i2h_i)

        #calculate h2o
        h2o_i = numpy.dot(self.w_h2o, i2h_o)
        #i2h activation function
        h2o_o = self.activation(h2o_i)

        return h2o_o

# train_data = pandas.read_csv('F:/Coding/Python/NeuroNet/mnist_train_100.csv', header=None,sep=',').to_numpy()
train_data = numpy.genfromtxt('F:/Coding/Python/NeuroNet/mnist_train_100.csv', delimiter=',')
# image_arr = numpy.asfarray(train_data[0,1:].reshape(28,28))
# print(scaled_input)
# matplotlib.pyplot.imshow(image_arr ,cmap='Greys')
# matplotlib.pyplot.show()

in_node = 784
out_node = 10
hidden = 100
learning_rate = 0.2
N = neuralNet(in_node, out_node, hidden, learning_rate)

for record in train_data:
    scaled_input = numpy.asfarray(record[1:]) / 255 * 0.99 + 0.01
    target = numpy.zeros(out_node) + 0.01
    target[int(record[0])] = 0.99
    N.forward(scaled_input, target)


test_data = numpy.genfromtxt('F:/Coding/Python/NeuroNet/mnist_test_10.csv', delimiter=',')
test = test_data[8]
image_arr = numpy.asfarray(test[1:].reshape(28,28))
matplotlib.pyplot.imshow(image_arr ,cmap='Greys')
matplotlib.pyplot.show()

N_out = N.query((numpy.asfarray(test[1:]) / 255 * 0.99 + 0.01))
print(N_out)
# print(numpy.where(N_out == numpy.amax(N_out))[0])
print(numpy.argmax(N_out))