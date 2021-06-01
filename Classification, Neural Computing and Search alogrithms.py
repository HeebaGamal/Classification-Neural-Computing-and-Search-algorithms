from math import sqrt
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from random import seed
from random import randrange
from csv import reader
from math import sqrt
import random
import math
import queue
import collections
from collections import deque


# region SearchAlgorithms
class Positions:
    def __init__(self, r, c):
        self.row = r
        self.col = c

class Node:
    id = None
    up = None
    down = None
    left = None
    right = None
    previousNode = None
    position_node = None

    def __init__(self, value):
        self.value = value
class SearchAlgorithms:
    path = []  # Represents the correct path from start node to the goal node.
    fullPath = []  # Represents all visited nodes from the start node to the goal node.

    maze = []
    number_of_col = 0
    number_of_row = 0

    start_point_row = 0
    start_point_col = 0
             #U  D  L   R
    rowNum = [-1, 1, 0, 0]
    colNum = [0, 0, -1, 1]
    def __init__(self, mazeStr):
        ''' mazeStr contains the full board The board is read row wise, the nodes are numbered 0-based starting    the leftmost node'''
        self.mazeStr = mazeStr
        self.length = len(mazeStr)


    def Create_maze(self):
        row = []
        id = 0
        for i in range(self.length):
            if self.mazeStr[i] != " " and self.mazeStr[i] != ",":
                node = Node(self.mazeStr[i])
                node.id = id
                id+=1
                row.append(node)
            elif self.mazeStr[i] == " ":
                self.maze.append(row)
                row = []
        self.maze.append(row)

        self.number_of_col = self.maze[0]
        self.number_of_row = self.maze

        self.maze[0][0].position_node = Positions(0, 0)
        for i in range(len(self.maze)):
            for j in range(len(self.maze[i])):
               #print(self.maze[i][j].value, end=" ")
               self.maze[i][j].position_node = Positions(i, j)

               for d in range(4):
                   row = self.maze[i][j].position_node.row + self.rowNum[d]
                   col = self.maze[i][j].position_node.col + self.colNum[d]

                   if self.Valid_direction(row, col):
                       if d == 0:
                           self.maze[i][j].up = self.maze[row][col]
                       elif d == 2:
                           self.maze[i][j].left = self.maze[row][col]
                       elif d == 3:
                           self.maze[i][j].right = self.maze[row][col]
                       elif d == 1:
                           self.maze[i][j].down = self.maze[row][col]
               '''for d in range(4):
                   row = self.maze[i][j].position_node.row + self.rowNum[d]
                   col = self.maze[i][j].position_node.col + self.colNum[d]

                   if self.Valid_direction(row, col):
                       if d == 0:
                           print(self.maze[i][j].up,end=" ")
                       elif d == 1:
                           print(self.maze[i][j].left,end=" ")
                       elif d == 2:
                           print(self.maze[i][j].right,end=" ")
                       elif d == 3:
                           print(self.maze[i][j].down,end=" ")
               print()'''
            #print()

    def Valid_direction(self, i, j):
        if not (0 <= i and i < len(self.maze) and 0 <= j and j < len(self.maze[0])):
            return False
        return True

    def Find_Start(self):
        for i in range(len(self.maze)):
            for j in range(len(self.maze[i])):
                if self.maze[i][j].value == "S":
                    self.start_point_row = i
                    self.start_point_col = j

    def Find_end(self, i, j):
        if self.maze[i][j].value == "E":
            print("Found")
            return True
        return False

    def Valid_move(self, i, j, visited, node):
        if not (0 <= i and i < len(self.maze) and 0 <= j and j < len(self.maze[0])):
            return False
        elif self.maze[i][j].value == "#":
            return False
        elif visited[i][j]:
            return False
        elif node == None:
            return False
        return True

    def BFS(self):
        '''Implement Here'''
        self.path.clear()
        self.fullPath.clear()
        self.maze.clear()
        self.Create_maze()
        self.Find_Start()

        queue_direction = deque()

        visited = [[False for i in range(len(self.number_of_col))] for j in range(len(self.number_of_row))]

        visited[self.start_point_row][self.start_point_col] = True
        self.maze[self.start_point_row][self.start_point_col].position_node = Positions(self.start_point_row,
                                                                                         self.start_point_col)
        self.maze[self.start_point_row][self.start_point_col].previousNode = Positions(-1, -1)

        queue_direction.append(self.maze[self.start_point_row][self.start_point_col])

        #self.fullPath.append(self.maze[self.start_point_row][self.start_point_col].id)

        while queue_direction:
            currend_node = queue_direction.popleft()

            if(self.Find_end(currend_node.position_node.row, currend_node.position_node.col)):
                self.fullPath.append(currend_node.id)
                prev_position = Positions(0, 0)
                previous_node = currend_node
                while previous_node.value != 'S':
                    r = previous_node.position_node.row
                    c = previous_node.position_node.col
                    self.path.append(self.maze[r][c].id)
                    previous_node = self.maze[r][c].previousNode
                self.path.append(previous_node.id)
                break
            cur_pos_row = currend_node.position_node.row
            cur_pos_col = currend_node.position_node.col

            if self.Valid_move(cur_pos_row - 1, cur_pos_col, visited, currend_node.up) :
                self.maze[cur_pos_row - 1][cur_pos_col].previousNode = currend_node
                visited[cur_pos_row- 1][cur_pos_col] = True
                queue_direction.append(self.maze[cur_pos_row- 1][cur_pos_col])

            if self.Valid_move(cur_pos_row + 1, cur_pos_col, visited, currend_node.down):
                self.maze[cur_pos_row + 1][cur_pos_col].previousNode = currend_node
                visited[cur_pos_row + 1][cur_pos_col] = True
                queue_direction.append(self.maze[cur_pos_row + 1][cur_pos_col])

            if self.Valid_move(cur_pos_row, cur_pos_col - 1, visited, currend_node.left):
                self.maze[cur_pos_row][cur_pos_col - 1].previousNode = currend_node
                visited[cur_pos_row][cur_pos_col - 1] = True
                queue_direction.append(self.maze[cur_pos_row][cur_pos_col - 1])
            if self.Valid_move(cur_pos_row, cur_pos_col + 1, visited, currend_node.right):
                self.maze[cur_pos_row][cur_pos_col + 1].previousNode = currend_node
                visited[cur_pos_row][cur_pos_col + 1] = True
                queue_direction.append(self.maze[cur_pos_row][cur_pos_col + 1])

            '''for i in range(4):
                row = currend_node.position_node.row + self.rowNum[i]
                col = currend_node.position_node.col + self.colNum[i]

                if self.Valid_move(row, col, visited):
                    visited[row][col] = True
                    self.maze[row][col].position_node = Positions(row, col)
                    self.maze[row][col].previousNode = Positions(currend_node.position_node.row, currend_node.position_node.col)
                    queue_direction.append(self.maze[row][col])'''
            self.fullPath.append(currend_node.id)
        self.path.reverse()
        return self.fullPath, self.path

# endregion

# region NeuralNetwork
class NeuralNetwork():

    def __init__(self, learning_rate, threshold):
        self.learning_rate = learning_rate
        self.threshold = threshold
        np.random.seed(1)
        self.synaptic_weights = 2* np.random.random((2, 1)) - 1

    def step(self, x):
        if x > float(self.threshold):
            return 1
        else:
            return 0

    def train(self, training_inputs, training_outputs, training_iterations):
        '''print("1", self.synaptic_weights)
        weight = [[-1.5]]
        temp_synaptic_weights = np.array
        #for w in self.synaptic_weights:
        self.synaptic_weights= np.append(self.synaptic_weights, weight)
        #temp_synaptic_weights.append(self.synaptic_weights, weight)
        print("2", self.synaptic_weights)
        print(training_inputs)'''

        basis = [1]
        temp  = []
        for row in training_inputs:
            modified_row = np.append(row, basis)
            temp.append(modified_row)
        temp = np.array(temp)
        training_inputs_basis = temp

        for iteration in range(training_iterations):
            output = self.learn(training_inputs_basis)
            error = training_outputs - output
            adjustments = np.dot(training_inputs_basis.T, error * self.learning_rate)
            temp_adjustments = []
            temp_adjustments.append(adjustments[0])
            temp_adjustments.append(adjustments[1])
            self.synaptic_weights += temp_adjustments

    def learn (self, inputs):
        weights = self.synaptic_weights.tolist()
        inputs = inputs.astype(float)
        temp_w = -5
        w = 0
        for i in inputs:
            c = 0
            for j in i:
                if c == 2:
                    w += j * temp_w
                    continue
                w += j * weights[c][0]
                c += 1
        output = self.step(w)
        return output

    def think(self, inputs):
        weights = self.synaptic_weights.tolist()
        inputs = inputs.astype(float)
        temp_w = -5
        w = 0
        c = 0
        for i in range(len(inputs)+1):
            if c == 2:
                w += 2 * temp_w
                continue
            w += inputs[i] * weights[c][0]
            c += 1
        #print("w = ", w)
        output = self.step(w)
        return output
# endregion

# region ID3



class item:
    ###123
    def __init__(self, age, prescription, astigmatic, tearRate, diabetic, needLense):
        self.age = age
        self.prescription = prescription
        self.astigmatic = astigmatic
        self.tearRate = tearRate
        self.diabetic = diabetic
        self.needLense = needLense #decision

    def getDataset():
        data = []
        labels = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0] #conclusion of desition tree
        data.append(item(0, 0, 0, 0, 1, labels[0]))
        data.append(item(0, 0, 0, 1, 1, labels[1]))
        data.append(item(0, 0, 1, 0, 1, labels[2]))
        data.append(item(0, 0, 1, 1, 1, labels[3]))
        data.append(item(0, 1, 0, 0, 1, labels[4]))
        data.append(item(0, 1, 0, 1, 1, labels[5]))
        data.append(item(0, 1, 1, 0, 1, labels[6]))
        data.append(item(0, 1, 1, 1, 1, labels[7]))
        data.append(item(1, 0, 0, 0, 1, labels[8]))
        data.append(item(1, 0, 0, 1, 1, labels[9]))
        data.append(item(1, 0, 1, 0, 1, labels[10]))
        data.append(item(1, 0, 1, 1, 0, labels[11]))
        data.append(item(1, 1, 0, 0, 0, labels[12]))
        data.append(item(1, 1, 0, 1, 0, labels[13]))
        data.append(item(1, 1, 1, 0, 0, labels[14]))
        data.append(item(1, 1, 1, 1, 0, labels[15]))
        data.append(item(1, 0, 0, 0, 0, labels[16]))
        data.append(item(1, 0, 0, 1, 0, labels[17]))
        data.append(item(1, 0, 1, 0, 0, labels[18]))
        data.append(item(1, 0, 1, 1, 0, labels[19]))
        data.append(item(1, 1, 0, 0, 0, labels[20]))
        return data

class Feature:
    def __init__(self, name):
        self.name = name
        self.visited = -1
        self.infoGain = -1
        self.col = []
class Node_Feature:
    name_node = None
    left_node = None
    right_node = None
class ID3:
    Total_entropy = 0
    max_name = ""
    max_gain = -1
    col_age         = []
    col_prescription= []
    col_astigmatic  = []
    col_tearRate    = []
    col_diabetic    = []
    col_needLense   = []
    #
    dictionary_nodes = {}
    rec_col_zero_age = []
    rec_col_one_age = []


    def __init__(self, features):
        self.features = features

    def set_all_data(self):
        trained_data = item.getDataset()
        for i in range(len(trained_data)):
            self.col_age.append(trained_data[i].age)
            self.col_prescription.append(trained_data[i].prescription)
            self.col_astigmatic.append(trained_data[i].astigmatic)
            self.col_tearRate.append(trained_data[i].tearRate)
            self.col_diabetic.append(trained_data[i].diabetic)
            self.col_needLense.append(trained_data[i].needLense)

    def entropy(self, column_of_features):
        values_zero = 0
        values_one = 0
        entropy = 0
        for i in range(0,len(column_of_features)):
            if column_of_features[i]==0:
                values_zero+=1
            elif column_of_features[i]==1:
                values_one+=1
        length=len(column_of_features)
        #print("l  = ", length)

        if (values_zero/length) == 0 and (values_one/length) != 0:
            entropy = -(0 + ((values_one / length) * math.log2(values_one / length)))
        elif values_zero/length != 0 and values_one/length == 0:
            entropy = -(((values_zero / length) * math.log2(values_zero / length))  + 0)
        elif values_zero / length == 0 and values_one / length == 0:
            entropy = 0
        else:
            entropy = -(((values_zero / length) * math.log2(values_zero / length)) +
                        ((values_one / length) * math.log2(values_one / length)))

        return entropy

    def gain(self, column_of_features, needLense_desicion_col):
        #print("start gain")
        length_features = len(column_of_features)
        zero_feature_count = 0
        one_feature_count = 0
        zero_feature_list = []
        one_feature_list = []
        for i in range(0, length_features):
            if column_of_features[i] == 0:
                zero_feature_count += 1
                zero_feature_list.append(needLense_desicion_col[i])
            elif column_of_features[i] == 1:
                one_feature_count += 1
                one_feature_list.append(needLense_desicion_col[i])
                #print("-->", one_feature_list)
        Total_entropy = self.entropy(needLense_desicion_col)
        zero_feature_entropy= self.entropy(zero_feature_list)
        one_feature_entropy = self.entropy(one_feature_list)
        gain_feature = Total_entropy - ( ((zero_feature_count / length_features) * zero_feature_entropy)
                                        + ((one_feature_count / length_features) * one_feature_entropy) )
        return gain_feature
    def id3_main_algorithim(self, id3):
        new_node = None
        age_zero = []
        prescription_zero = []
        astigmatic_zero = []
        tearRate_zero = []
        diabetic_zero = []
        needLense_zero = []

        age_one = []
        prescription_one = []
        astigmatic_one = []
        tearRate_one = []
        diabetic_one = []
        needLense_one = []

        for i in range(0, len(self.features)):
            if self.features[i].visited == -1:
                if self.features[i].name =='age':
                    #print(self.features[i].name)
                    gain_out = self.gain(self.col_age, self.col_needLense)
                    #print("2", self.features[i].name)
                    name_feature = 'age'
                if self.features[i].name =='prescription':
                    gain_out = self.gain(self.col_prescription, self.col_needLense)
                    name_feature = 'prescription'
                    #print(name_feature)
                if self.features[i].name =='astigmatic':
                    gain_out = self.gain(self.col_astigmatic, self.col_needLense)
                    name_feature = 'astigmatic'
                    #print(name_feature)
                if self.features[i].name =='tearRate':
                    gain_out = self.gain(self.col_tearRate, self.col_needLense)
                    name_feature = 'tearRate'
                    #print(name_feature)
                if self.features[i].name =='diabetic':
                    gain_out = self.gain(self.col_diabetic, self.col_needLense)
                    name_feature = 'diabetic'
                    #print(name_feature)

                if gain_out > self.max_gain:
                    self.max_gain = gain_out
                    self.max_name = name_feature

        #the non basic code of id3 starting from here
        '''global node_obj
        if id3 == 2:
            node_obj = Node_Feature()
            node_obj.name = self.max_name
            self.dictionary_nodes[node_obj.name] = node_obj
            new_node = node_obj
        elif id3 == 1:
            node_obj = self.dictionary_nodes.get(new_node.name)
            node_obj.right = self.max_name
            self.dictionary_nodes[new_node.name] = node_obj
            node_obj = Node_Feature()
            node_obj.name = self.max_name
            new_node = node_obj
            dictionary_nodes[self.max_name] = node_obj
        elif id3 == 0:
            node_obj = dictionary_nodes.get(new_node.name)
            node_obj.left = max_name
            dictionary_nodes[new_node.name] = node_obj
            node_obj = Node_Feature()
            node_obj.name = max_name
            new_node = node_obj
            dictionary_nodes[max_name] = node_obj

        c = 0
        for i in range(0, len(self.features)):
            if self.max_name == self.features[i].name:
                self.features[i].visited = 0
                c += 1
        if self.max_name == 'age':
            arr = self.col_age.copy()
        elif self.max_name == 'prescription':
            arr = self.col_prescription.copy()
        elif self.max_name == 'astigmatic':
            arr = self.col_astigmatic.copy()
        elif self.max_name == 'tearRate':
            arr = self.col_tearRate.copy()
        elif self.max_name == 'diabetic':
            arr = self.col_diabetic.copy()

        zero = 0
        one = 0
        age_zero.clear()
        age_one.clear()
        prescription_zero.clear()
        prescription_one.clear()
        astigmatic_zero.clear()
        astigmatic_one.clear()
        tearRate_zero.clear()
        tearRate_one.clear()
        diabetic_one.clear()
        diabetic_zero.clear()
        needLense_zero.clear()
        needLense_one.clear()
        for i in range(0,len(arr)):
            if arr[i]==0:
                age_zero.append(self.col_age[i])
                prescription_zero.append(self.col_prescription[i])
                astigmatic_zero.append(self.col_astigmatic[i])
                tearRate_zero.append(self.col_tearRate[i])
                diabetic_zero.append(self.col_diabetic[i])
                needLense_zero.append(self.col_needLense[i])
                zero+=1
            elif arr[i]==1:
                age_one.append(self.col_age[i])
                prescription_one.append(self.col_prescription[i])
                astigmatic_one.append(self.col_astigmatic[i])
                tearRate_one.append(self.col_tearRate[i])
                diabetic_one.append(self.col_diabetic[i])
                needLense_one.append(self.col_needLense[i])
                one+=1
        a=np.unique(needLense_zero)
        # print(a)
        if len(a) > 1:
            self.col_age.clear()
            self.col_prescription.clear()
            self.col_astigmatic.clear()
            self.col_tearRate.clear()
            self.col_diabetic.clear()
            self.col_needLense.clear()
            for i in len(0, len(age_zero)):
                self.col_age[i] = age_zero[i]
                self.col_prescription[i] = prescription_zero[i]
                self.col_astigmatic[i] = astigmatic_zero[i]
                self.col_tearRate[i] = tearRate_zero[i]
                self.col_diabetic[i] = diabetic_zero[i]
                self.col_needLense[i] = needLense_zero[i]
            self.id3_main_algorithim(0)
        else:
            node_obj.left_node = a[0];
        a = np.unique(needLense_one)
        if len(a) > 1:
            self.col_age.clear()
            self.col_prescription.clear()
            self.col_astigmatic.clear()
            self.col_tearRate.clear()
            self.col_diabetic.clear()
            self.col_needLense.clear()
            for i in range(0, len(age_one)):
                self.col_age.append(age_one[i])
                self.col_prescription.append(prescription_one[i])
                self.col_astigmatic.append(astigmatic_one[i])
                self.col_tearRate.append(tearRate_one[i])
                self.col_diabetic.append(diabetic_one[i])
                self.col_needLense.append(needLense_one[i])
            self.id3_main_algorithim(1)
        else:
            node_obj.right_node = a[0]'''





    def classify(self, input):
        # takes an array for the features ex. [0, 0, 1, 1, 1]
        # should return 0 or 1 based on the classification
        for i in range(0, len(self.features)):
            self.features[i].visited = -1

        self.set_all_data()
        self.id3_main_algorithim(2)
        N = Node_Feature()
        for i in self.dictionary_nodes:
            N = self.dictionary_nodes[i]
            break
        while True:
            if N.name_node == 'diabetic':
                if input[4] == 1:
                    if N.right_node == 0 or N.right_node == 1:
                        return N.right_node
                    else:
                        N = self.dictionary_nodes.get(N.right_node)
                elif input[3] == 0:
                    if N.left_node == 0 or N.left_node == 1:
                        return N.left_node
                    else:
                        N = self.dictionary_nodes.get(N.left_node)
            if N.name_node == 'tearRate':
                if input[3] == 1:
                    if N.right_node == 0 or N.right_node == 1:
                        return N.right_node
                    else:
                        N = self.dictionary_nodes.get(N.right_node)
                elif input[3] == 0:
                    if N.left_node == 0 or N.left_node == 1:
                        return N.left_node
                    else:
                        N = self.dictionary_nodes.get(N.left_node)
            elif N.name_node == 'astigmatic':
                if input[2] == 1:
                    if N.right_node == 0 or N.right_node == 1:
                        return N.right_node
                    else:
                        N = self.dictionary_nodes.get(N.right_node)
                elif input[2] == 0:
                    if N.left_node == 0 or N.left_node == 1:
                        return N.left_node
                    else:
                        N = dictionary_nodes.get(N.left_node)
            elif N.name_node == 'prescription':
                if input[1] == 1:
                    if N.right_node == 0 or N.right_node == 1:
                        return N.right_node
                    else:
                        N = self.dictionary_nodes.get(N.right_node)
                elif input[1] == 0:
                    if N.left_node == 0 or N.left_node == 1:
                        return N.left_node
                    else:
                        N = dictionary_nodes.get(N.left_node)
            elif N.name_node == 'age':
                if input[0] == 1:
                    if N.right_node == 0 or N.right_node == 1:
                        return N.right_node
                    else:
                        N = self.dictionary_nodes.get(N.right_node)
                elif input[0] == 0:
                    if N.left_node == 0 or N.left_node == 1:
                        return N.left_node
                    else:
                        N = self.dictionary_nodes.get(N.left_node)

# endregion

#################################### Algorithms Main Functions #####################################
# region Search_Algorithms_Main_Fn

def SearchAlgorithm_Main():
    searchAlgo = SearchAlgorithms('S,.,.,#,.,.,. .,#,.,.,.,#,. .,#,.,.,.,.,. .,.,#,#,.,.,. #,.,#,E,.,#,.')
    fullPath, path = searchAlgo.BFS()
    print('**BFS**\n Full Path is: ' + str(fullPath) + "\n Path: " + str(path))

# endregion

# region Neural_Network_Main_Fn
def NN_Main():
    learning_rate = 0.1
    threshold = -0.2
    neural_network = NeuralNetwork(learning_rate, threshold)

    print("Beginning Randomly Generated Weights: ")
    print(neural_network.synaptic_weights)

    training_inputs = np.array([[0, 0],
                                [0, 1],
                                [1, 0],
                                [1, 1]])

    training_outputs = np.array([[0, 0, 0, 1]]).T

    neural_network.train(training_inputs, training_outputs, 100)

    print("Ending Weights After Training: ")
    print(neural_network.synaptic_weights)

    inputTestCase = [1, 1]

    print("Considering New Situation: ", inputTestCase[0], inputTestCase[1], end=" ")
    print("New Output data: ", end=" ")
    print(neural_network.think(np.array(inputTestCase)))
    print("Wow, we did it!")


# endregion
# region ID3_Main_Fn
def ID3_Main():
    dataset = item.getDataset()
    features = [Feature('age'), Feature('prescription'), Feature('astigmatic'), Feature('tearRate'),
                Feature('diabetic')]
    id3 = ID3(features)
    cls = id3.classify([0, 0, 1, 1, 1])
    print('testcase 1: ', cls)
    cls = id3.classify([1, 1, 0, 0, 0])
    print('testcase 2: ', cls)
    cls = id3.classify([1, 1, 1, 0, 0])
    print('testcase 3: ', cls)
    cls = id3.classify([1, 1, 0, 1, 0])
    print('testcase 4: ', cls)


# endregion

######################## MAIN ###########################33
if __name__ == '__main__':

    SearchAlgorithm_Main()
    NN_Main()
    ID3_Main()
