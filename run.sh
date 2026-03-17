
# 彻底禁用 Torch 动态编译
export TORCH_COMPILE_DISABLE=1
# 增加系统库搜索路径（解决 -lcuda 找不到的问题）
export LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
# 1. 清除 Conda 可能设置的编译器变量
unset CC
unset CXX

# 2. 强制指定使用系统的 GCC/G++
export CC=/usr/bin/gcc
export CXX=/usr/bin/g++
# export VLLM_ENFORCE_EAGER=1
# export VLLM_WORKER_MULTIPROC_METHOD=spawn
# python run.py --data TempCompass_64frame --model Qwen3-VL-8B-Instruct --use-vllm
# python run.py --data Video-MME_64frame --model Qwen3-VL-4B-Instruct-Intruder-800-lr1e-6-beta004 --use-vllm
# python run.py --data Video-MME_64frame --model Qwen3-VL-8B-Instruct-reordertask-distactor --use-vllm
python run.py --data MVBench_MP4_1fps --model Qwen3-VL-4B-Instruct-mixed --use-vllm
# python run.py --data AoTBench_ReverseFilm_16frame AoTBench_UCF101_16frame AoTBench_Rtime_t2v_16frame AoTBench_Rtime_v2t_16frame AoTBench_QA_16frame --model Qwen3-VL-8B-Instruct --use-vllm
# python run.py --data FutureOmni_32frame --model Qwen3-VL-8B-Instruct --use-vllm
# python run.py --data TempCompass_64frame --model Qwen3-VL-8B-Instruct-NarrativeReorder-with-clues --use-vllm --work-dir /m2v_intern/xuboshen/zgw/eval
# python run.py --data TempCompass_64frame --model Qwen3-VL-8B-Instruct-NarrativeReorder-wo-clues --use-vllm --work-dir /m2v_intern/xuboshen/zgw/eval