# Naive Bayes naive classifier
# Jun Sung Tak
# Dec 1st 2020

import math
import numpy as np

data_points = []
pdf_points = []
bird_pdf = []
airplane_pdf = []
TRANSMISSION = 0.9

# Loads numerical data into the program for calculation
def loader(file_name, arr):
    counter = 0
    with open(file_name, "r") as filestream:
        for line in filestream:
            curr_line = line.split(",")
            for elem in curr_line:
                arr.append(float(elem))
                counter += 1
    return counter

# Splits the pdf data set, 1 for bird and another for airplane
def split_pdf(arr1, arr2):
    for i in range(400):
        arr1.append(pdf_points[i])
        arr2.append(pdf_points[400 + i])

# Splits up observation data into 10 elements in a list
def split_up_data_points():
    main_list = []
    temp_list = []
    counter = 0
    for i in range(3000):
        temp_list.append(data_points[i])
        counter += 1
        if counter == 300:
            main_list.append(temp_list)
            temp_list = []
            counter = 0

    return main_list
        

# Returns probabilty of observing velocity given its class
# velocity: query velocity
# arr     : class
def get_vprob_given_c(velocity, arr):
    index = velocity * 2
    if math.isnan(index):
        return 0
    else:
        return arr[int(index)]

# Multiplies all the probabilities
# Due to underflow, the log-sum-exp trick was used
def mul_ET_probs(arr, klass):
    log_list = []
    for i in range(len(arr)):
        cond_prob = get_vprob_given_c(arr[i], klass)
        if cond_prob == 0:
            continue
        log_cond_prob = math.log(cond_prob) + math.log(TRANSMISSION) #P(Ot|Ct)P(Ct|Ct-1)
        log_list.append(log_cond_prob)
    max_prob = max(log_list)
    for i in range(len(log_list)):
        temp = math.exp(log_list[i] - max_prob)
        log_list[i] = temp
    total = math.log(sum(log_list)) + max_prob
    return total

# Helper function for accounting for standard deviation of each observation
def scale_standard_dev(v, klass):
    if v > 4:
        if klass == "bird":
            return 1
        else:
            return -1
    else:
        if klass == "bird":
            return -1
        else:
            return 1


# Main loop for functionality
def naive_bayes():
    loader("data.txt", data_points)
    loader("pdf.txt", pdf_points)
    split_pdf(bird_pdf, airplane_pdf)
    big_list = split_up_data_points()
    for i in range(len(big_list)):
        bird_prob = mul_ET_probs(big_list[i], bird_pdf)
        airplane_prob = mul_ET_probs(big_list[i], airplane_pdf)
        denom = bird_prob + airplane_prob
        b_prob = bird_prob - denom # The value here must be subtracted because of the log-sum-exp trick
        a_prob = airplane_prob - denom
        v = np.nanstd(big_list[i])
        b_prob = b_prob + scale_standard_dev(v, "bird") # Account for standard deviation of each set
        a_prob = a_prob + scale_standard_dev(v, "air")
        print(f"Observation: {i + 1}")
        print(f"Probability representation that it is a bird: {b_prob}")
        print(f"Probability representation that it is an airplane: {a_prob}")
        if b_prob < a_prob:
            print("Given the data and probability distribution function, it is probably an aircraft")
        else:
            print("Given the data and probability distribution function, it is probably a bird")
        print()

    return

naive_bayes()




