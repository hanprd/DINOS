#!/bin/bash
kaggle datasets download changheonhan/dinos-diverse-industrial-operation-sounds

python3 -c "
import zipfile, glob, os, shutil
for f in glob.glob('*.zip'):
    with zipfile.ZipFile(f, 'r') as z:
        z.extractall()
    os.remove(f)

if os.path.isdir('DINOS'):
    for item in os.listdir('DINOS'):
        src = os.path.join('DINOS', item)
        dst = os.path.join('.', item)
        shutil.move(src, dst)
    os.rmdir('DINOS')
"
