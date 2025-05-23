from ihpo.search_spaces import HPOBSearchSpace
import json
import os

data_path = './data/hpob-data/'
surrogate_path = './data/hpob-surrogates/'

with open(os.path.join(data_path, 'meta-dataset-descriptors.json'), 'r') as f:
    search_spaces = json.load(f)

variable_names = []

for space_id in search_spaces.keys():
    space = HPOBSearchSpace(data_path, space_id)
    vars = list(space.get_search_space_definition().keys())
    if len(vars) > 3:
        variable_names.append((set(vars), space_id))
    

sorted_variable_names = sorted(variable_names, key=lambda element: len(element[0]))


candidate_spaces = []

for idx in range(len(sorted_variable_names)):
    curr_set = sorted_variable_names[idx][0]

    for compare_set, ssid in sorted_variable_names[(idx+1):]:
        intersect = curr_set.intersection(compare_set)
        if len(intersect) > 2 and len(intersect) < len(curr_set) and (len(intersect) / len(compare_set)) >= 0.5:
            candidate_spaces.append((curr_set, compare_set, sorted_variable_names[idx][1], ssid))

final_candidates = []
for third_set, id in sorted_variable_names:
    for set1, set2, ssid1, ssid2 in candidate_spaces:
        if id != ssid1 and id != ssid2:
            union = set1.union(set2)
            intersect = union.intersection(third_set)
            if len(intersect) > 2 and len(intersect) < len(third_set) and (len(intersect) / len(union)) >= 0.5:
                final_candidates.append((ssid1, ssid2, id, len(set1.union(set2).union(third_set)), len(set1.intersection(set2)), 
                                         len(set1.intersection(intersect)), len(set2.intersection(intersect)), len(intersect)))    

print(len(final_candidates))
for fc in final_candidates:
    print(fc)
