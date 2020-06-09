mel=$1
wav=$2

echo 'mel size is [80, length]'
CUDA_VISIBLE_DEVICES=-1 python3 /apdcephfs/share_1213607/zhxliu/WAVERNN/wavernn.multi_band/synthesize.py --mel $mel --wav $wav
