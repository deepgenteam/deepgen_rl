conda create -n unirl_ocr python=3.10 -y
conda activate unirl_ocr
python -m pip install paddlepaddle-gpu==3.2.2 -i https://www.paddlepaddle.org.cn/packages/stable/cu126/

pip install -r requirements.txt