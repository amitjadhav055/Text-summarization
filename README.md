README.md
markdown
Copy code
# Text Summarization Project

This project implements a text summarization model using the BART architecture from Hugging Face's Transformers library. The model is fine-tuned on a custom dataset to generate concise summaries of input text.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Training the Model](#training-the-model)
- [Evaluation](#evaluation)
- [Results](#results)
- [License](#license)

## Installation

1. Clone the repository:

   git clone <your-repo-url>
   cd Text-Summarization
Create a virtual environment (optional but recommended):

Copy code
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
Install the required packages:


Copy code
pip install -r requirements.txt
Usage
To train the model, run:


Copy code
python src/train_model.py
To evaluate the model, run:


Copy code
python src/evaluate_model.py

##Training the Model
The training script utilizes mixed precision training for faster training times and lower memory usage. The model is trained on the provided dataset and saved after training.

##Evaluation
The evaluation script calculates the ROUGE score for the generated summaries compared to the reference summaries.

##Results
The results of the evaluation will be printed in the console after running the evaluation script.

##License
This project is licensed under the MIT License. See the LICENSE file for more details.
