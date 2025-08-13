# Lane-Detection-YOLOv8
A deep learning model to detect and segment lane lines in road images, trained with YOLOv8-seg.
## How to Use
### 1. Setup
Clone this repository, navigate into the directory, and install the required packages:
```bash
git clone https://github.com/pradnya641/Lane-Detection-YOLOv8.git
cd Lane-Detection-YOLOv8
pip install -r requirements.txt

### 2. Download Data and Model
The dataset and trained model are hosted on Google Drive due to their size.

* **[Download the Processed Dataset (datasets.zip)]**(https://drive.google.com/file/d/1qTNfFpL8GL9Pg3IQe9EGlRMnRGXheHwb/view?usp=sharing)
***[Download the Trained Model (best.pt)]**(https://drive.google.com/file/d/1gnr167hBNsfv18nKiTLZxRIX_ywPl5cb/view?usp=sharing)

Place the `best.pt` file and the unzipped `datasets` folder in the main project directory.

### 3. Running the Scripts

To run prediction on a new image with the trained model, use:
```bash
python predict.py

### 4. Demo Video
* **[Watch a Demo Video of the Model in Action](https://drive.google.com/file/d/1N0785btHrZKads1te5KC0S3H1hxYdbJC/view?usp=sharing)**
