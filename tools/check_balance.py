import os

def count(p): return sum(f.lower().endswith(('.jpg','.png','.jpeg')) for f in os.listdir(p))
base = 'data/combined'
for split in ['train', 'test']:
    a=count(f'{base}/{split}/attentive')
    i=count(f'{base}/{split}/inattentive')
    total=a+i
    print(split, 'attentive:',a, 'inattentive:',i, 'ratio attentive:', a/max(1,total))