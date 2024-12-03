# conda create --name llm \
#     python=3.11 \
#     pytorch-cuda=12.1 \
#     pytorch cudatoolkit xformers -c pytorch -c nvidia -c xformers \
#     -y

# conda activate llm

pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install FlagEmbedding sentence-transformers nvitop 
pip install --no-deps human-eval
pip install vllm
pip install matplotlib
pip install -U transformers
pip install deepspeed
pip install seqeval