conda create -n teammlg-project
conda activate teammlg-project
conda install python==3.12
conda install nvidia::cuda
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
python -m pip install transformers==4.51
python -m pip install accelerate
python -m pip install bitsandbytes