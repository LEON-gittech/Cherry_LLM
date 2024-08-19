pip install polars
pip install nvitop
pip install bitsandbytes
pip install accelerate
sudo apt install -y ranger htop
pip install trl
# pip install "unsloth[cu121] @ git+https://github.com/unslothai/unsloth.git"
pip install "unsloth[cu121-ampere] @ git+https://github.com/unslothai/unsloth.git"
rm -rf /home/tiger/.local/lib/python3.9/site-packages/unsloth/
cp -r /mnt/bn/data-tns-live-llm/leon/unsloth /home/tiger/.local/lib/python3.9/site-packages/unsloth/
pip install torch torchvision torchaudio
pip install promptsource
pip install sentence-transformers
pip install matplotlib
pip install pygwalker
pip install evaluate
pip install nltk rouge_score
pip install FlagEmbedding
pip install umap-learn
pip install ordered_set
pip install -U pandas
sudo apt-get -y install jq bc
pip install deepspeed
sudo apt install -y libaio-dev
pip install seqeval
# pip install benepar spacy
# python3 -m spacy download en_core_web_md
pip install --force-reinstall "numpy<2.0.0"
pip install -U transformers peft
pip uninstall -y flash-attn
pip install vllm
pip install openai