import numpy as np
import matplotlib.pyplot as plt
clean_data = np.loadtxt('co395-cbc-dt/wifi_db/clean_dataset.txt')
noisy_data = np.loadtxt('co395-cbc-dt/wifi_db/noisy_dataset.txt')

def sum_label(dataset):
    sum_1 = [0,0,0,0]
    for i in range(dataset.shape[0]):
        if dataset[i][7] == 1:
            sum_1[0]+=1
        if dataset[i][7] == 2:
            sum_1[1]+=1
        if dataset[i][7] == 3:
            sum_1[2]+=1
        if dataset[i][7] == 4:
            sum_1[3]+=1
    return sum_1

def calentropy(dataset, sum_array):
    class_labels = np.unique(dataset[:,7])
    # print dataset.shape, class_labels.shape
    H = 0
    for label in class_labels:
        # print sum_array[int(label)-1]
        pk = sum_array[int(label)-1]/float(dataset.shape[0])
        if pk != 0:
            h = pk * np.log2(pk)
            H -= h
    return H

def find_split(training_dataset):
    sum_array = sum_label(training_dataset)
    #print sum_array
    dataset_entropy = calentropy(training_dataset, sum_array)
    max_Gain = {'Wifi': 0 , 'Cut_point': 0.0, 'Gain': 0.0}
    for wifi in range(7):
        wifi_length_unique = np.unique(training_dataset[:, wifi])
        # wifi_midpoint=[]
        # for i in range(len(wifi_length_unique)-1):
        #     midpoint = (wifi_length_unique[i]+ wifi_length_unique[i+1])/2
        #     wifi_midpoint.append(midpoint)
        # print wifi_length_unique
        for cut_point in wifi_length_unique:
            left_index = np.where(training_dataset[:,wifi] <= cut_point)
            right_index = np.where(training_dataset[:,wifi] > cut_point)
            left_dataset = training_dataset[left_index]
            left_sum_array = sum_label(left_dataset)
            right_dataset = training_dataset[right_index]
            right_sum_array = sum_label(right_dataset)
            # print left_dataset.shape, right_dataset.shape
            # print left_dataset.shape, right_dataset.shape
            remainder = left_dataset.shape[0]/float(training_dataset.shape[0]) * calentropy(left_dataset, left_sum_array)\
            + right_dataset.shape[0]/float(training_dataset.shape[0]) * calentropy(right_dataset, right_sum_array)
            Gain = dataset_entropy - remainder
            # print Gain
            if Gain > max_Gain['Gain'] :
                max_Gain['Wifi'] = wifi+1
                max_Gain['Cut_point'] = cut_point
                max_Gain['Gain'] = Gain
    return max_Gain

def decision_tree_learning(training_dataset, depth):
    if len(np.unique(training_dataset[:,7])) == 1 :
        # leaf
        Attribute = np.unique(training_dataset[:,7])

        dt={'Attribute': int(Attribute), 'Value': 0, 'Left': None, 'Right': None, 'isLeaf': 1}
        return dt , depth
    else:
        split = find_split(training_dataset)
        cut_point = split['Cut_point']
        wifi = split['Wifi']
        left_index = np.where(training_dataset[:,wifi-1] <= cut_point)
        right_index = np.where(training_dataset[:,wifi-1] > cut_point)
        left_dataset = training_dataset[left_index]
        right_dataset = training_dataset[right_index]
        left_branch, left_depth = decision_tree_learning(left_dataset, depth+1)
        right_branch, right_depth = decision_tree_learning(right_dataset, depth+1)

        dt = {'Attribute': wifi, 'Value': cut_point, 'Left': left_branch, 'Right': right_branch, 'isLeaf': 0}
        return dt, max(left_depth, right_depth)
# print find_split(clean_data)
# decision_tree_learning(clean_data[:1000],0)




dt, depth = decision_tree_learning(clean_data, 0)
print depth
print dt

# def visualize(dt):
#     for key in dt.keys():
#         if type(dt[key]).__name__ == 'dict':
#             visualize(dt[key])
#         else:
#             print key, ":", dt[key]
#
# visualize(dt)
test_data = [-63,-56,-63,-65,-72,-82,-89,1]
def evaluation(test_data, dt):
    Attribute = dt.get('Attribute')
    value = dt.get('Value')
    print Attribute, value
    

evaluation(test_data, dt)
# for key in dt.keys():
#     if type(dt[key]).__name__ == 'dict':
#     if key =='Left' or key == 'Right'
#     print key
#     print dt[key]


#
# desisionNode = dict(boxstyle='sawtooth', fc = "0.8")
# leafNode = dict(boxstyle='round4', fc = '0.8')
# arrow_args = dict(arrow_args = "<-")
# numleafs = 0
# def getNumleafs(decision_tree):
#     global numleafs
#     for key, values in decision_tree.items():
#         if type(values).__name__ == 'dict':
#             getNumleafs(values)
#         elif key =='isLeaf':
#             numleafs+=values
#
# def plotMidText(cntrPt, parentPt, txtString):
#     xMid = (parentPt[0] - cntrPt[0])/2.0 + cntrPt[0]
#     yMid = (parentPt[1] - cntrPt[1])/2.0 + cntrPt[1]
#     creatPlot.ax1.text(xMid, yMid, txtString)
#
# def plotTree(myTree, parentPt, nodeName, numleafs, depth):
#     firstStr = list(myTree.keys())[0]
#     cntrPt = (plotTree.xOff+(0.5/plotTree.totalw+float(numleafs)/2.0/plotTree.totalw), plotTree.yOff)
#     plotMidText(cntrPt, parentPt, nodeName)
#     plotNode(firstStr, cntrPt, parentPt, decisionNode)
#     secondDict = myTree[firstStr]
#     plotTree.yOff = plotTree.yOff - 1.0/plotTree.totalD
#     for key in secondDict.keys():
#         if type(secondDict[key]).__name__=='dict':
#             plotTree(secondDict[key], cntrPt, str(key))
#         else:
#             plotTree.xOff = plotTree.xOff + 1.0/plotTree.totalw
#             plotNode(secondDict[key], (plotTree.xOff, plotTree.yOff), cntrPt, leafNode)
#             plotMidText((plotTree.xOff, plotTree.yOff), cntrPt, str(key))
#     plotTree.yOff = plotTree.yOff + 1.0/plotTree.totalD
#
# def creatPlot(inTree):
#     fig = plt.figure(1, facecolor='white')
#     fig.clf()
#     axprops = dict(xticks=[], yticks=[])
#     creatPlot.ax1 = plt.subplot(111, frameon=False, **axprops)
#     plotTree.totalw = float(numleafs)
#     plotTree.totalD = float(depth)
#     plotTree.xOff = -0.5/plotTree.totalw
#     plotTree.yOff = 1.0
#     plotTree(inTree, (0.5,1.0), '', numleafs, depth)
#     plt.show()
#
# getNumleafs(dt)
# creatPlot(dt)
