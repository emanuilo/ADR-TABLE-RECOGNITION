# ADR Table Recognition

## Dependencies
Python 3.7.7

## Installation

1. Clone the repo
```
  git clone https://github.com/emanuilo/ADR-TABLE-RECOGNITION.git
  cd ADR-TABLE-RECOGNITION
  git submodule init
  git submodule update
```
2. Install the virtual environment and requirements
```
  python3 -m venv venv
  source venv/bin/activate
  pip install -r requirements.txt
```
3. Install Darkflow   
```
  cd darkflow
  pip install -e .
```
4. Download model weights from [here](https://drive.google.com/drive/folders/1mW735di8mXFFkIBTRy-O051beePIm6FK?usp=sharing)
5. Put weights in ckpt directory


## Usage
```
    python main.py --ground-truth-dir test_images/ground_truth --image-dir test_images
```


## Docker
