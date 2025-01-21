import sys
import json

data = {}
nData = {}
with open(sys.argv[1],'r') as file:
    data = json.load(file)

for key in data:
    nData[key] = {}
    nData[key]['notes'] = [f"{v[0]}" for v in data[key]['notes']]
    nData[key]['duration'] = [f"{v[1]}" for v in data[key]['notes']]

with open(sys.argv[2],'w') as file:
    json.dump(nData,file,indent=1)

