CUDA_VISIBLE_DEVICES=0 python test.py --cfg-path1 configs/eval/decode_esc50.yaml --options gpu 0 
CUDA_VISIBLE_DEVICES=0 python test.py --cfg-path1 configs/eval/decode_vggsound.yaml --options gpu 0  
CUDA_VISIBLE_DEVICES=0 python test.py --cfg-path1 configs/eval/decode_urban.yaml --options gpu 0 
CUDA_VISIBLE_DEVICES=0 python test.py --cfg-path1 configs/eval/decode_tut.yaml --options gpu 0 
# wait 

CUDA_VISIBLE_DEVICES=0 python test.py --cfg-path1 configs/eval/decode_bjo.yaml --options gpu 0 
CUDA_VISIBLE_DEVICES=0 python test.py --cfg-path1 configs/eval/decode_musiccap.yaml --options gpu 0 
CUDA_VISIBLE_DEVICES=0 python test.py --cfg-path1 configs/eval/decode_clothov2.yaml --options gpu 0 
CUDA_VISIBLE_DEVICES=0 python test.py --cfg-path1 configs/eval/decode_genres.yaml --options gpu 0 
# wait 

CUDA_VISIBLE_DEVICES=0 python test.py --cfg-path1 configs/eval/decode_fsd50k.yaml --options gpu 0 
CUDA_VISIBLE_DEVICES=0 python test.py --cfg-path1 configs/eval/decode_clothoaqa_v2.yaml --options gpu 0 
CUDA_VISIBLE_DEVICES=0 python test.py --cfg-path1 configs/eval/decode_audiocap.yaml --options gpu 0 
CUDA_VISIBLE_DEVICES=0 python test.py --cfg-path1 configs/eval/decode_dcase17.yaml --options gpu 0 
# wait  
CUDA_VISIBLE_DEVICES=0 python test.py --cfg-path1 configs/eval/decode_MMAU_mini.yaml --options gpu 0 
CUDA_VISIBLE_DEVICES=0 python test.py --cfg-path1 configs/eval/decode_AIRBench_Chat.yaml --options gpu 0 
CUDA_VISIBLE_DEVICES=0 python test.py --cfg-path1 configs/eval/decode_MMAU.yaml --options gpu 0
