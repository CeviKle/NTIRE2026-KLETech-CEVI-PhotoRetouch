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

## Download pretrained weights from This Google Drive and place under 'models' folder
