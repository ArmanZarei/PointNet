# PointNet Classification

### Usage
- Download the dataset: `sh download_dataset.sh`
- Creating Virtual Environment: `python3 -m venv .venv`
    - Activate it using
        - Unix or MacOS: `source .venv/bin/activate`
        - Windows: `.env/Scripts/activate.bat`
- Installing requirements: `pip install -r requirements.txt`
- Training the model: `python3 main.py`
    - Or you can run it in the background: `nohup python3 main.py > output.log &`
