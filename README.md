## 1. Create and Activate Conda Environment

```bash
conda create -n inretouch python=3.10 -y
conda activate inretouch
```
## 2. Clone the Repository
```bash
git clone https://github.com/CeviKle/NTIRE2026-KLETech-CEVI-PhotoRetouch.git
cd NTIRE2026-KLETech-CEVI-PhotoRetouch
```
## 3. Install PyTorch with CUDA 12.8
```bash
pip install torch==2.10.0 torchvision==0.25.0+cu128 torchaudio -f https://download.pytorch.org/whl/torch_stable.html
```
## 4. Install Other Dependencies
```bash
pip install -r requirements.txt
```
## 5. Verify Installation (Optional)
```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```
## 6. Install basicsr
```bash
python setup.py develop --no_cuda_ext
```

Download the pretrained weights from [google drive](https://drive.google.com/drive/folders/1RmpdphmhG3GT9ZCPmBnzWCzNnD9Kxlaq?usp=drive_link) and put it in '**models**' folder

## To run the code
```bash
chmod +x run_all_samples.sh
./run_all_samples.sh
```
#change the path in run_all_samples.sh file in the main directory
