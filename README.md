# ADR Table Recognition

## Dependencies
Python 3.7

## Installation (MacOS)

1. Clone the repo
```
    git clone https://github.com/emanuilo/ADR-TABLE-RECOGNITION.git
    cd ADR-TABLE-RECOGNITION
    git submodule update --init
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

## Installation (Docker)

1. Clone the repo
```
    git clone https://github.com/emanuilo/ADR-TABLE-RECOGNITION.git
    cd ADR-TABLE-RECOGNITION
    git submodule update --init
```
2. Download model weights from [here](https://drive.google.com/drive/folders/1mW735di8mXFFkIBTRy-O051beePIm6FK?usp=sharing)
3. Put weights in ckpt directory
4. Build Docker image
```
    docker build -t adr-table-extractor .
```
5. Run Docker container 
```
    docker run --rm -it -v $(pwd)/out:/usr/src/app/out adr-table-extractor bash 
```

## Usage
Basic test
```
    python main.py
```
Custom test images directory and ground truth directory
```
    python main.py --ground-truth-dir <ground_truth_dir> --image-dir <test_images_dir>
```
TestReport.pdf file will be generated in the out/ directory