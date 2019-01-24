import numpy as np
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
        dt={'Attribute': Attribute, 'Value': 0, 'Left': None, 'Right': None, 'isLeaf': 1}
        return dt , depth
    else:
        split = find_split(training_dataset)
        cut_point = split['Cut_point']
        wifi = split['Wifi']
        print 'result =', wifi, cut_point, split['Gain']
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

def visualize(decision_tree):
    pass
dt, depth = decision_tree_learning(clean_data, 0)
print depth
