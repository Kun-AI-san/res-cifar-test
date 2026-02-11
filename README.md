<h1>Steps for running the script:</h1>
<h2>1. Installing dependencies</h2>
```pip install -r requirements.txt```
<h2>2. For training,</h2>
from root, command: 
```
python ./training/training.py --batch-size 256 --epochs 50
```
<h2>3. For confusion matrix</h2>
from root, command: 
```
python ./test/test.py --batch-size 1024
```
