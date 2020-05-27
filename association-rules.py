#alankala



import pandas as pd
import numpy as np
from collections import Counter
import itertools
from pprint import pprint
import copy
from random import randint
import matplotlib.pyplot as plt
import sys

filename = sys.argv[1]
minsup = float(sys.argv[2])
minconf = float(sys.argv[3])

df = pd.read_csv(filename, keep_default_na=False)

df.replace({False : 0, True : 1}, inplace=True)
# df.replace(True, 1, inplace=True)
df = pd.get_dummies(df.astype(str), columns=['city', 'state', 'alcohol', 'noiseLevel', 'attire', 'priceRange'])
# print(df)

# boolean_headers = ['goodForGroups', 'open', 'delivery', 'waiterService', 'caters']

# for (columnName, columnData) in df.iteritems():
#    print('Colunm Name : ', columnName)
#    print('Column Contents : ', columnData.values)

delimiter = "&&"

# Get Candidate Sets and Prune them
def apriori_generator(L_k, k):
    Ck = {}
    keys = list(L_k.keys())

    # print(keys)

    total_keys = len(keys)

    # Joining L_k-1 with itself to get probable L_k

    for i in range(total_keys):
        for j in range(i+1, total_keys):
            cnt = 0
            temp_key_1 = keys[i].split(delimiter)
            temp_key_2 = keys[j].split(delimiter)

            for attr in temp_key_1:
                if attr in temp_key_2:
                    cnt += 1

            if cnt == k-1:
                new_set = set(temp_key_1 + temp_key_2)
                new_key = delimiter.join(list(new_set))
                Ck[new_key] = 0

      # Perform pruning
    candidate_keys = list(Ck)

    # print(L_k)
    # print(candidate_keys)
    for key in candidate_keys:
        keys = key.split(delimiter)

        subsets = list(set(itertools.combinations(keys,k)))
        
        for subset in subsets:
            sub_key = delimiter.join(subset)

            # print(sub_key)
            if sub_key not in L_k:
                del Ck[key]
                break
    # print(Ck)
    return Ck

def apriori(min_sup, df):
    L_k = []
    Li = {}

    array = df.values.astype('int32')
    # print(array)

    min_sup_count = int(min_sup*len(df))
    # print("Minimum Support Count: ", min_sup_count)

    for i in range(0, len(df.columns)):
        if np.sum(array[:, i]) >= min_sup_count:
            Li[df.columns[i]] = np.sum(array[:,i])

    L_k.append(Li)
    # print(Li)

    k = 1

    while len(L_k[k-1]) > 0:
        Li = {}
        Ck = apriori_generator(L_k[k-1], k)

        # Get support for each keyset
        candidate_keys = list(Ck) 
        for key in candidate_keys:
            intersection_set = np.ones(len(array))

            for item in key.split(delimiter):
                column_idx = df.columns.get_loc(item)
                intersection_set = np.logical_and(intersection_set, array[:, column_idx])

            Ck[key] += np.sum(intersection_set)    
            
            # print(np.sum(intersection_set))
            if Ck[key] < min_sup_count:
                del Ck[key]

        L_k.append(Ck)
        k += 1

    return L_k

L_k = apriori(minsup, df)

# for i in L_k:
#     print(i)

def apriori_confidence(L_k, min_conf):
    # For single variable consequents

    rules = []

    rhs = list(L_k[0].keys())

    for consequent in rhs:
        for i in range(0, len(L_k)-1):
            for itemsets in list(L_k[i].keys()):
                numerator = 0
                confidence = 0

                key_1 = itemsets + delimiter + consequent
                key_2 = consequent + delimiter + itemsets

                if key_1 in L_k[i+1]:
                    numerator = L_k[i+1][key_1]
                elif key_2 in L_k[i+1]:
                    numerator = L_k[i+1][key_2]

                if numerator != 0:
                    confidence = float(numerator)/L_k[i][itemsets]

                if confidence >= min_conf:
                    rules.append([(itemsets, consequent), confidence, numerator])
    return rules

rules = apriori_confidence(L_k, minconf)

rules_count = {}

for i in rules:
    splitted = i[0][0].split(delimiter)
    length = len(splitted)

    if str(length) not in rules_count:
        rules_count[str(length)] = 1
    else:    
        rules_count[str(length)] += 1

print()

for i in range(1, len(L_k)-1):
    print("FREQUENT-ITEMS " + str(i+1) + " "  + str(len(L_k[i])))

print()

for key in rules_count.keys():
    if rules_count[key] > 0 and int(key) > 1:
        print("ASSOCIATION-RULES " + key + " "  + str(rules_count[key]))

# to_plot = np.array([[0.25, 0.75]])

# items_cnt = []
# rules_cnt = []

# for entry in to_plot:
#     L_k = apriori(entry[0], df)
#     rules = apriori_confidence(L_k, entry[1])

#     total_item = 0
#     total_rule = 0

#     for i in range(1, len(L_k)-1):
#         total_item += len(L_k[i])

#     for i in rules:
#         splitted = i[0][0].split(delimiter)
#         length = len(splitted)

#         if str(length) not in rules_count:
#             rules_count[str(length)] = 1
#         else:    
#             rules_count[str(length)] += 1

#     for key in rules_count.keys():
#         if rules_count[key] > 0 and int(key) > 1:
#             total_rule += rules_count[key]

#     items_cnt.append(total_item)
#     # print(total_rule)
#     rules_cnt.append(total_rule)
#     # print(total_item, total_rule)

# print(items_cnt)
# print(rules_cnt)

# print((to_plot[:, 0], items_cnt))

# fig, ax = plt.subplots(figsize=(20, 10))
# ax.plot(to_plot[:, 0], items_cnt)
# ax.scatter(to_plot[:, 0], items_cnt)
# fig.suptitle('Values of |I| and |R| at Minsup = 25% and Minconf = 75%', fontsize=20)
# ax.set_xlabel('Minsup', fontsize=18)
# ax.set_ylabel('Item Count', fontsize=18)

# for i in range(1):
#     string = '|I|=' + str(items_cnt[i]) + ' ' + '|R|=' + str(rules_cnt[i]) 
#     ax.annotate(string, (to_plot[i][0], items_cnt[i]), size=14)
# plt.show()


# to_plot = np.array([[0.10, 0.75], [0.30, 0.75], [0.50, 0.75]])

# items_cnt = []
# rules_cnt = []

# for entry in to_plot:
#     L_k = apriori(entry[0], df)
#     rules = apriori_confidence(L_k, entry[1])

#     total_item = 0
#     total_rule = 0

#     for i in range(1, len(L_k)-1):
#         total_item += len(L_k[i])

#     for i in rules:
#         splitted = i[0][0].split(delimiter)
#         length = len(splitted)

#         if str(length) not in rules_count:
#             rules_count[str(length)] = 1
#         else:    
#             rules_count[str(length)] += 1

#     for key in rules_count.keys():
#         if rules_count[key] > 0 and int(key) > 1:
#             total_rule += rules_count[key]

#     items_cnt.append(total_item)
#     # print(total_rule)
#     rules_cnt.append(total_rule)
#     # print(total_item, total_rule)

# print(items_cnt)
# print(rules_cnt)

# fig, ax = plt.subplots(figsize=(20, 10))
# ax.plot(to_plot[:, 0], items_cnt)
# ax.scatter(to_plot[:, 0], items_cnt)
# fig.suptitle('Values of |I| and |R| over varying minsup', fontsize=20)
# ax.set_xlabel('Minsup', fontsize=18)
# ax.set_ylabel('Item Count', fontsize=18)

# for i in range(3):
#     string = '|I|=' + str(items_cnt[i]) + ' ' + '|R|=' + str(rules_cnt[i]) 
#     ax.annotate(string, (to_plot[i][0], items_cnt[i]), size=14)
# plt.show()

# to_plot = np.array([[0.25, 0.40], [0.25, 0.60], [0.25, 0.80]])

# items_cnt = []
# rules_cnt = []

# for entry in to_plot:
#     L_k = apriori(entry[0], df)
#     rules = apriori_confidence(L_k, entry[1])

#     total_item = 0
#     total_rule = 0

#     for i in range(1, len(L_k)-1):
#         total_item += len(L_k[i])

#     for i in rules:
#         splitted = i[0][0].split(delimiter)
#         length = len(splitted)

#         if str(length) not in rules_count:
#             rules_count[str(length)] = 1
#         else:    
#             rules_count[str(length)] += 1

#     for key in rules_count.keys():
#         if rules_count[key] > 0 and int(key) > 1:
#             total_rule += rules_count[key]

#     items_cnt.append(total_item)
#     # print(total_rule)
#     rules_cnt.append(total_rule)
#     # print(total_item, total_rule)

# print(items_cnt)
# print(rules_cnt)

# fig, ax = plt.subplots(figsize=(20, 10))
# ax.plot(to_plot[:, 1], items_cnt)
# ax.scatter(to_plot[:, 1], items_cnt)
# fig.suptitle('Values of |I| and |R| over varying minconf', fontsize=20)
# ax.set_xlabel('Minconf', fontsize=18)
# ax.set_ylabel('Item Count', fontsize=18)

# for i in range(3):
#     string = '|I|=' + str(items_cnt[i]) + ' ' + '|R|=' + str(rules_cnt[i]) 
#     ax.annotate(string, (to_plot[i][1], items_cnt[i]), size = 14)

# plt.show()
