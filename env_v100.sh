pip install polars
pip install nvitop
pip install bitsandbytes
pip install accelerate
sudo apt install -y ranger htop
pip install trl
pip install "unsloth[cu121] @ git+https://github.com/unslothai/unsloth.git"
rm -rf /home/tiger/.local/lib/python3.9/site-packages/unsloth/
cp -r /mnt/bn/data-tns-live-llm/leon/unsloth_v100 /home/tiger/.local/lib/python3.9/site-packages/unsloth/
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
pip install peft transformers==4.43.2
pip uninstall -y flash-attn
pip install vllm
pip install openai
pip install latex2sympy2
pip install httpx==0.23.3
pip install -e /mnt/bn/data-tns-live-llm/leon/human-eval

# ——————————————————————————————————github cli————————————————————————————————————————
(type -p wget >/dev/null || (sudo apt update && sudo apt-get install wget -y)) \
&& sudo mkdir -p -m 755 /etc/apt/keyrings \
&& wget -qO- https://cli.github.com/packages/githubcli-archive-keyring.gpg | sudo tee /etc/apt/keyrings/githubcli-archive-keyring.gpg > /dev/null \
&& sudo chmod go+r /etc/apt/keyrings/githubcli-archive-keyring.gpg \
&& echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | sudo tee /etc/apt/sources.list.d/github-cli.list > /dev/null \
&& sudo apt update \
&& sudo apt install gh -y
# ——————————————————————————————————github cli————————————————————————————————————————