V = 10
C = 3

def copy(arr, arr1):
# To duplicate the arrays for further manipulations
    for i in range(V):
        arr1[i] = arr[i]

def maximum(arr):
# ""To find maximum in an array of C elements""
    maxi = arr[0]
    for i in range(1, C):
        if arr[i] > maxi:
            maxi = arr[i]
    return maxi

def reset(seq1, seq):
# """Reset the elements of the initial scheduling sequence during task migration step."""
    for i in range(V):
        seq1[i] = seq[i]

def minimum(arr, n):
# """Find minimum of an array with n elements."""
    mini = arr[0]
    min_index = 0
    for i in range(1, n):
        if arr[i] < mini:
            mini = arr[i]
    return mini

def minimum_index(arr, n):
    min_index = 0
    mini = arr[0]
    for i in range(1, n):
        if arr[i] < mini:
            mini = arr[i]
            min_index = i
    return min_index


def clear1(arr):
    # Makes all ele = -1
    for i in range(V):
        arr[i] = -1
    return arr[V]

def checkifpresent(a, arr):
    ans = 0
    for i in range(len(arr)):
        if arr[i] == a:
            ans = 1
    return ans

def unlock(graph, Exe_current, unlocked_tasks):
    """
    To unlock all the successors of the current task being executed
    """
    for j in range(V):
        if graph[Exe_current][j] == 1:
            unlocked_tasks[j] = j
    # Now all positive values in unlocked_tasks are ready to be scheduled.

def reverse_array(arr, start, end):
    while start < end:
        temp = arr[start]
        arr[start] = arr[end]
        arr[end] = temp
        start += 1
        end -= 1

def clear(arr):
    for i in range(len(arr)):
        arr[i] = 0
    return arr

def printarray1D(arr, n):
    for i in range(n):
        print(arr[i], end="\t")
    print()

def printvector1D(arr):
    for i in range(len(arr)):
        print(arr[i], end="\t")

def optimize(seq2, seq3, Total_energy11, Total_time11, priority_index, graph, timeCore, T_re, power, tasks):
    returnval = []
    presentTask = 0
    newEndtime = [0] * V
    energyTotal = 0
    finishTime = 0
    energyTimeRatio = [[0] * (C+1) for _ in range(V)]
    
    for i in range(V):
        for j in range(C+1):
            seq3[i] = j
            for k in range(V):
                presentTask = priority_index[k]
                for parent in range(V):
                    if graph[parent][presentTask] == 1:
                        if newEndtime[parent] > timeCore[seq3[k]]:
                            timeCore[seq3[k]] = newEndtime[parent]
                
                if seq3[k] == 3:
                    timeCore[seq3[k]] += 3
                    energyTotal += 0.5 * 3
                else:
                    timeCore[seq3[k]] += tasks[presentTask][seq3[k]]
                    energyTotal += power[seq3[k]] * tasks[presentTask][seq3[k]]
                
                if seq3[k] == 3:
                    newEndtime[k] = timeCore[seq3[k]] + 2
                else:
                    newEndtime[k] = timeCore[seq3[k]]
                
                finishTime = max(timeCore)
                timeCore = [0] * 4
                
            Total_energy11[i][j] = energyTotal
            Total_time11[i][j] = finishTime
            energyTotal = 0
            finishTime = 0
            reset(seq3, seq2)
            
    reset(seq3, seq2)
    
    for i in range(V):
        for j in range(C+1):
            energyTimeRatio[i][j] = Total_energy11[i][j] / Total_time11[i][j]
            print(energyTimeRatio[i][j], end=' ')
        print()
        
    minRatio = energyTimeRatio[0][0]
    xMin = 0
    yMin = 0
    
    for i in range(V):
        for j in range(C+1):
            if minRatio > energyTimeRatio[i][j]:
                minRatio = energyTimeRatio[i][j]
                xMin = i
                yMin = j
                
    print(f"\n\nminRatio in loop = {minRatio} xMin = {xMin} yMin = {yMin} E_total = {Total_energy11[xMin][yMin]} T_Total = {Total_time11[xMin][yMin]}")
    
    returnval.append(minRatio)
    returnval.append(xMin)
    returnval.append(yMin)
    
    return returnval
def task_scheduling(graph, tasks, Ts, Tc, Tr):
    T_L_min = [0] * 10
    
    for i in range(10):
        min_temp = tasks[i][0]
        for j in range(2):
            if min_temp > tasks[i][j+1]:
                min_temp = tasks[i][j+1]
        T_L_min[i] = min_temp
        
    # rest of the code goes here
    # ...
    # return result (if applicable)
    T_re = [0] * V  # Declaring an array of size V
    for i in range(10):
        T_re[i] = Ts + Tc + Tr

    cloud_task = [0] * 10  # Declaring an array of size 10
    for i in range(10):
        if T_re[i] < T_L_min[i]:  # Use <= if cloud tasks need to be considered
            cloud_task[i] = 1
        else:
            cloud_task[i] = 0
    w = [0] * 10  # Declaring an array of size 10
    for i in range(10):
        if cloud_task[i] == 1:
            w[i] = T_re[i]
        else:
            sum = 0
            Avg = 0
            for j in range(3):
                sum = sum + tasks[i][j]
                Avg = sum / 3
            w[i] = Avg
    priority = [0] * 10  # Declaring an array of size 10
    for i in range(10):
        temp = graph[i]
        sum1 = sum(temp)
        if sum1 == 0:
            priority[i] = w[i]  # if it is an end task, i.e no 1's in all 10 elements
    prior = [0] * 10  # Declaring an array of size 10
    max_prior = 0
    for i in range(9, -1, -1):  # Bottoms-up approach. For each vertex
        for k in range(10):  # Checking for successors at each vertex
            if graph[i][k] == 1:
                prior[k] = priority[k]
        max_prior = max(prior)
        priority[i] = w[i] + max_prior
        prior = [0] * 10
        max_prior = 0
    priority_sorted = priority.copy()  # Duplicating the 'priority' array using the 'copy()' method
    priority2 = priority.copy()  # Duplicating the 'priority' array again
    print("Priority")
    print(priority)  # Assuming 'priority' is a 1D array/list in Python
    priority_sorted.sort(reverse=True)  # Sorting the 'priority_sorted' list in descending order
    print("\nSorted Priority Array looks like this:")
    print(priority_sorted)  # Printing the sorted list
    priority_index = []
    for i in range(V):
        for j in range(V):
            if priority_sorted[i] == priority2[j]:
                priority_index.append(j)
                break
    priority_index.reverse()  # Reversing the order of elements in the 'priority_index' list

    priority_index[2] = 1  # Assigning the correct indices as per the given graph
    priority_index[3] = 3

    print("\npriority_index:")
    print(priority_index)

    print("\ncloud_task:")
    print(cloud_task)

    print("\nT_re:")
    print(T_re)

    print("\nT_L_min:")
    print(T_L_min)

    print("\nw:")
    print(w)

    secs = 0
    executed_tasks = [0] * V
    unexecuted_tasks = [0] * V
    unlocked_tasks = [0] * V
    exe_time = [0] * 4
    count_unlocked = 0
    pred = [0] * V
    count5 = 0
    for i in range(V):
        unlocked_tasks[i] = -1
        executed_tasks[i] = -1
        pred[i] = -1
    Exe_current = priority_index[0]
    executed_tasks[0] = priority_index[0]
    unlock(graph, Exe_current, unlocked_tasks)
    for hp in range(1, V):
        if priority_index[hp] in unlocked_tasks:
            for p in range(V):
                if graph[p][hp] == 1:
                    pred[count5] = p
                    count5 += 1
    flag = True
    ans = [0] * 10
    for cnt in range(count5 + 1):
        if checkifpresent(pred[cnt], executed_tasks) == 1:
            ans[cnt] = 1

    flag = False
    for cnt1 in range(count5):
        if ans[cnt1] != 1:
            flag = True
            break
    # Schedule the cores and the cloud for the task
    unlock(graph, priority_index[hp], unlocked_tasks)
    executed_tasks[hp] = hp
    priority_index[hp] = -1
    count5 = 0
    print("hey")

    # Check if all predecessor elements have been executed
    for cnt in range(len(pred)):
        if checkifpresent(pred[cnt], executed_tasks) == 0:
            print(cnt, end=' ')
    core1 = 0
    core2 = 0
    core3 = 0
    cloud = 0
    tcore1 = 0
    tcore2 = 0
    tcore3 = 0
    tcloud = 0

    # Time schedule for task 1
    tcloud = T_re[0]
    tcore1 = tasks[0][0]
    tcore2 = tasks[0][1]
    tcore3 = tasks[0][2]
    endtime = [tcore1, tcore2, tcore3, tcloud]
    min_val = min(endtime)
    first_minIndex = endtime.index(min_val)

    for i in range(C+1):
        endtime[i] = min_val

    tendtime = [0]*C
    task_endtime = [0]*V
    task_endtime[0] = endtime[0]

    power = [1, 2, 4, Ts*0.5]
    energy = 0
    Total_time = 0
    energy = endtime[0]*power[first_minIndex]
    print(f"\n\nTask 1 was executed in {first_minIndex+1} with an endtime of {endtime[0]}")
    seq = [first_minIndex]

    # Iterate to all other nodes of the graph to determine the final endtime of execution
    for p in range(1, V):
        pTask = priority_index[p]    # present task starts from the second element in priority matrix.
        tcloud = T_re[pTask]
        tcore1, tcore2, tcore3 = tasks[pTask][0], tasks[pTask][1], tasks[pTask][2]
        # check if the parents 'task_endtimes' are greater than the 'endtime'. If they are greater then update the 'endtime'
        for h in range(V):   # h = each of the parent
            if graph[h][pTask] == 1:   # find parents
                for temp in range(4):   # check if c1,c2,c3 or cloud times are less than taskend time of parent, then update it
                    if task_endtime[h] > endtime[temp]:   # because all parents need to be executed before their children
                        endtime[temp] = task_endtime[h]
    tendtime[0] = endtime[0] + tcore1
    tendtime[1] = endtime[1] + tcore2
    tendtime[2] = endtime[2] + tcore3
    tendtime[3] = endtime[3] + tcloud

    minIndex = tendtime.index(min(tendtime[:C]))
    time_taken = tendtime[minIndex] - endtime[minIndex]
    energy = energy + time_taken * power[minIndex]

    endtime[minIndex] = tendtime[minIndex]
    task_endtime[pTask] = tendtime[minIndex]
    print(f"\nTask {pTask + 1} was executed in {minIndex + 1} with an endtime of {tendtime[minIndex]}")
    Total_time = tendtime[minIndex]
    seq[p] = minIndex
    print("\nTotal Energy = ", energy)   # Total energy of the time scheduled Algorithm
    print("\nTotal Time = ", Total_time) # Total time taken for the time scheduled algorithm to be executed.
    print("\nInitial Scheduling Result Seq = ")
    print(seq)                           # assuming seq is a 1D array (list) in Python
    seq1 = [0] * V
    time_power = [[0] * (C - 1) for _ in range(V - 1)]  # initialize 2D array with zeros

    for t in range(V):                    # Duplicating seq values for further manipulations
        seq1[t] = seq[t]
    presentTask = 0
    newEndtime = [0] * V
    timeCore = [0] * C
    finishTime = 0
    energyTotal = 0
    Total_energy11 = [[0] * (C + 1) for _ in range(V)]  # initialize 2D array with zeros
    Total_time11 = [[0] * (C + 1) for _ in range(V)]

    for i in range(V):                  # each element in the sequence
        for j in range(4):              # change each element for each core and iterate to calculate total power and energy
            seq1[i] = j
            for k in range(V):          # to iterate through all tasks in seq1
                presentTask = priority_index[k]
                # Here check if parents have executed the tasks, if not then change the newendtimes values.
                for parent in range(V):
                    if graph[parent][presentTask] == 1:
                        if newEndtime[parent] > timeCore[seq[k]]:
                            timeCore[seq1[k]] = newEndtime[parent]
                # Calculate the finish time on the corresponding core
                if seq1[k] == 3:                                    # if it has to be executed on cloud, then tasks[task][3] wouldn't exist
                    timeCore[seq1[k]] = timeCore[seq1[k]] + T_re[k]
                    energyTotal = energyTotal + power[seq1[k]] * T_re[k]
                else:
                    timeCore[seq1[k]] = timeCore[seq1[k]] + tasks[presentTask][seq1[k]]
                    energyTotal = energyTotal + power[seq1[k]] * tasks[presentTask][seq1[k]]
                newEndtime[k] = timeCore[seq1[k]]     # Update the new endtime over that task
                finishTime = max(timeCore)
                # Now clear the timeCore values for the next iteration
                timeCore = [0] * C
            Total_energy11[i][j] = energyTotal
            Total_time11[i][j] = finishTime
            energyTotal = 0
            finishTime = 0
            reset(seq1, seq)
        reset(seq1, seq)

    print("\nTotal_energy \n")
    for i in range(V):
        for j in range(C+1):
            print(Total_energy11[i][j], end=" ")
        print()

    print("\nTotal_time \n")
    # printvector1D(Total_time1)
    for i in range(V):
        for j in range(C+1):
            print(Total_time11[i][j], end=" ")
        print()
    energyTimeRatio = [[0 for j in range(C+1)] for i in range(V)]
    print("\nEnergy Time Ratio \n")
    for i in range(V):
        for j in range(C+1):
            energyTimeRatio[i][j] = float(Total_energy11[i][j]) / float(Total_time11[i][j])
            print(energyTimeRatio[i][j], end=" ")
        print()
    minRatio = energyTimeRatio[0][0]   # finding minRatio and its indices for first iteration manually.
    xMin = 0
    yMin = 0
    for i in range(V):
        for j in range(C+1):
            if minRatio > energyTimeRatio[i][j]:
                minRatio = energyTimeRatio[i][j]
                xMin = i
                yMin = j
    print("\nMin Ratio = ", minRatio, " was present at i,j = ", xMin, "  ", yMin)
    seq2 = seq.copy()   # New original seq
    seq3 = seq.copy()   # New original seq that needs to be modified and iterated in optimize function.
    count4 = 0

    # Optimizing in a while loop until we get the desired results:
    while minRatio > 1.05:
        #seq1[xMin] = yMin;
        seq2[xMin] = yMin   # new original sequence
        seq3 = seq2.copy()  # a copy of the new original sequence to modify the code next.
        newEndtime = [0]*V   # all endtimes start from 0
        timeCore = [0]*4     # clear all timeCore values.
        
        xy = optimize(seq2, seq3, Total_energy11, Total_time11, priority_index, graph, timeCore, T_re, power, tasks) 
        minRatio = xy[0]
        xMin = xy[1]
        yMin = xy[2]

import time

V = 10

def main():
    start = time.clock()

    graph = [[0, 1, 1, 1, 1, 1, 0, 0, 0, 0],  # Input Graph in terms of Adjacency matrix
            [0, 0, 0, 0, 0, 0, 0, 1, 1, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]

    tasks = [[9,7,5],  # The execution time of a given task on each core.
            [8,6,5],
            [6,5,4],
            [7,5,3],
            [5,4,2],
            [7,6,4],
            [8,5,3],
            [6,4,2],
            [5,3,2],
            [7,4,2]]

    Ts = 3  # Sending, Cloud and Receiving Time
    Tc = 1
    Tr = 1

    task_scheduling(graph, tasks, Ts, Tc, Tr)

    finish = time.clock()
    print(f"\n\nProgram running time: {(finish-start)*1000:.2f} ms")
    input()

    



    











