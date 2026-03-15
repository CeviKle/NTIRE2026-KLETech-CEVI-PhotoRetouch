## 1. Create and Activate Conda Environment

```bash
conda create -n inretouch python=3.10 -y
conda activate inretouch
python --version
```
## 2. Clone the Repository
```bash
git clone https://github.com/CeviKle/NTIRE2026-KLETech-CEVI-PhotoRetouch.git
cd NTIRE2026-KLETech-CEVI-PhotoRetouch
```
## 3. Install PyTorch with CUDA 12.8
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```
## 4. Install Other Dependencies
```bash
pip install -r requirements.txt
```
## 5. Install kornia
```bash
pip install kornia
```
## 6. Verify Installation (Optional)
```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

Download the pretrained weights from [google drive](https://drive.google.com/drive/folders/1RmpdphmhG3GT9ZCPmBnzWCzNnD9Kxlaq?usp=drive_link) and put it in '**models**' folder

## Train

### Dataset structure
```
|----- sample1
|     |------ sample1_before.jpg  (Refence Before Editing)
|     |------ sample1_after.jpg     (Refence After Editing)
|     |______ sample1_input.jpg     (Input to be Edited)
|
|----- ......
|
|-----  sample#
|     |------ sample#_before.jpg  (Refence Before Editing)
|     |------ sample#_after.jpg     (Refence After Editing)
|     |______ sample#_input.jpg     (Input to be Edited)
|
|----- ......
|
|____ sampleN
      |------ sampleN_before.jpg  (Refence Before Editing)
      |------ sampleN_after.jpg     (Refence After Editing)
      |______ sampleN_input.jpg     (Input to be Edited)
```

### Run this code
```bash
chmod +x run_all_samples.sh
./train_data.sh
```
Change the folder path in train_data.sh file (In train_data.sh file DATA_ROOT = "ADD YOUR PATH HERE")
