# mlx-gpt

| [![](https://github.com/DiogoNeves/mlx-gpt/assets/178898/f726440c-e308-4f27-8ecc-3e761eab57db)](https://youtu.be/kCc8FmEb1nY?si=PRVcXtLSZFvnNHjx) | This learning project implements a GPT language model using Apple's MLX library, following Andrej Karpathy's [Let's build GPT](https://youtu.be/kCc8FmEb1nY?si=PRVcXtLSZFvnNHjx) video. |
| --- | --- |

## 🚀 Getting Started
I tried to stay as close as possible to the original material, so that it's easy to follow.  
I recommend watching the walkthrough if you haven't yet!

### Instalation
```bash
# Setup the environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## 🤖 Usage
### Train and run the Bigram model
At the moment the command below trains and runs the model straight away.  
__It will also download and cache the data if needed.__  
```bash
python bigram.py
```

**Validation**
Roughly comparing the results in the video with my results as validation.
| Video | MLX |
|---|---|
| ![CleanShot 2024-03-25 at 09 54 24@2x](https://github.com/DiogoNeves/mlx-gpt/assets/178898/e8a02433-017e-4363-bcdd-d16b97f8157b) | ![CleanShot 2024-03-25 at 09 51 19@2x](https://github.com/DiogoNeves/mlx-gpt/assets/178898/4cc8dcaa-9e3b-4e47-8b6b-248200e760b6) |
> Both converge to a similar value (Please ignore the formatting issues)


### Train and run the GPT model
Coming soon...

### Other
You can inspect the experimental notebook I created while following the video at [experiment.ipynb](./experiment.ipynb). More understandable if you follow the video.  
_Tested on Macbook Air M1._

## 📦 Dependencies
- [Apple's MLX](https://ml-explore.github.io/mlx/build/html/index.html)
- [requests](https://requests.readthedocs.io/en/latest/)
- [Tiny Shakespeare Dataset](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt)
- (Optional) [jupyter](https://docs.jupyter.org/en/latest/)

## 📜 License
[MIT License](./LICENSE)
