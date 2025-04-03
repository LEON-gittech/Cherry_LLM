# conda create --name llm \
#     python=3.11 \
#     pytorch-cuda=12.1 \
#     pytorch cudatoolkit xformers -c pytorch -c nvidia -c xformers \
#     -y

# conda install python=3.11 \
#     pytorch-cuda=12.1 \
#     pytorch cudatoolkit xformers -c pytorch -c nvidia -c xformers \
#     -y

# pip install torch torchaudio torchvision xformers
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install FlagEmbedding sentence-transformers nvitop 
pip install --no-deps human-eval
pip install colorama
pip install openai
pip install vllm
pip install matplotlib
# pip install -U transformers
pip install deepspeed
pip install seqeval
pip install evaluate rouge_score
# pip install torch==2.1.* torchaudio torchvision transformers FlagEmbedding