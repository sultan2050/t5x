import glob
from tqdm import tqdm
files=glob.glob("*oscar*")
with open('arabic_osca.txt','w',encoding='utf8') as out:
    for filename  in tqdm(files):
        with open(filename,encoding='utf-8') as f:
            out.write(f.read())

