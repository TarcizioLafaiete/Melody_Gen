import sys
import json

data = {}
with open(sys.argv[1],'r') as file:
    data = json.load(file)

for key in data:
    data[key]['offset'] = [round(v,3) for v in data[key]['offset']]
    data[key]['notes'] = [f"{v[0]};{v[1]}" for v in data[key]['notes']]

with open(sys.argv[2],'w') as file:
    data = json.dump(data,file,indent=1)

