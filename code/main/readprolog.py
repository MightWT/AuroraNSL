import numpy as np
import re

rule_text_path = '~/rules.txt'
rules = []
predicates = []
file1 = open(rule_text_path, 'r')
Lines = file1.readlines()

for line in Lines:
    rules.append(line)
    # print("Line: {}".format( line.strip()))
patten1 = r'[(,)\.]'
for rule in rules:
    predicate = re.split(patten1,rule)
    predicates.append(predicate[0])

print(predicates)


