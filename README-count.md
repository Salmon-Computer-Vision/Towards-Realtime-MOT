# count.py

Run count for entire folder of videos:
```bash
python count.py --cfg weights/cfg/yolo3.cfg --weights weights/latest.pt recursive path/to/vid_folder
```

Parse countable csv files in folder:
```bash
python count.py count countables/
```

# Caveats

Install each package one at a time:
```bash
xargs -L 1 pip install < requirements.txt
```

May need to upgrade numpy if there is an error:
```bash
pip install -U numpy
```

Upgrade pytorch for your CUDA distribution (Eg. CUDA 11.3):
```bash
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
```
