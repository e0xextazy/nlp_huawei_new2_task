<p align="center" width="100%">
    <img width="65%" src="images/Huawei-Logo.wine-cropped.svg">
</p>

<div align="center">
    
  <a href="https://github.com/e0xextazy/nlp_huawei_new2_task/issues">![GitHub issues](https://img.shields.io/github/issues/e0xextazy/nlp_huawei_new2_task)</a>
  <a href="https://github.com/e0xextazy/nlp_huawei_new2_task/blob/master/LICENSE">![GitHub license](https://img.shields.io/github/license/e0xextazy/nlp_huawei_new2_task?color=purple)</a>
  <a href="https://github.com/psf/black">![Code style](https://img.shields.io/badge/code%20style-black-black)</a>
    
</div>

# Practical Assignment 2
### Text multiclass classification: movie's genre

## Contents
- [Practical Assignment 2](#practical-assignment-2)
    - [Text multiclass classification: movie's genre](#text-multiclass-classification-movies-genre)
  - [Contents](#contents)
  - [Description](#description)
  - [Data](#data)
  - [Evaluation](#evaluation)
  - [Submission File](#submission-file)
  - [Usage](#usage)
  - [Contributing](#contributing)
    - [Issue](#issue)
    - [Pull request](#pull-request)
  - [Authors](#authors)

## Description
Your task is to classify film descriptions into 6 classes (Horror, Kids, Mystery, Comedy, Action, Drama). The metric is **Accuracy**.

We present you 4 baseline solutions based on logistic regression, catboost, LSTM and Transformers. You can find them in their respective folders: `./baseline_tfidf_logreg`, `./<catboost_baseline>`, `./baseline_rnn` and `./<transformer_baseline>`. Each of these folders contains a file `requirements.txt` that will help you with the installation of the dependencies. To see the score and how many points you get if you can beat him, look at the table below:

| baseline    | Accuracy    | Points      |
| ----------- | ----------- | ----------- |
| LogReg      | 0.36294     | `<X>` points|
| CatBoost    | 0.59137     | `<Y>` points|
| LSTM        | 0.52664     | `<Z>` points|
| Transformer | WIP         | `<V>` points|

If you will be the first in your group, you'll get 3 bonus points.

Please **DO NOT** develop your solution as a fork of this repository. Also please do not make it publicly available.

## Data

The dataset presented here was collected from one of the public film rating resources. We have selected the 6 most popular movie genres and invite you to try to predict them.

- `train.csv` - The training set, comprising the `movie_name`, `movie_description` and `target` of each film, the latter of which is the genre of the film. `target` comprise the target for the competition. All columns are a string data type.
- `test.csv` - For the test data we give only the `movie_description` of an film together with its `movie_name`.
- `sample_submission.csv` - A submission file in the correct format.

You can download the dataset by following the [link](https://drive.google.com/drive/folders/1ZXq5wkOH4FdHD3HM3fzh61rSwqJaYT5k?usp=sharing).

## Evaluation

Submissions are scored using Accuracy error:

<p align="center" width="100%">
    <img width="65%" src="images/accuracy.png">
</p>

<p align="center" width="100%">
    <img width="65%" src="images/exact_match.png">
</p>

where $N$ is the number of samples in the test dataset.

## Submission File

For each row in the test set, you need to predict one of the 6 movie genres. The file should contain a header and have the following format:
```
id,target
d996f823,Horror
1cf01f9c,Horror
856ea05c,Horror
c97899ee,Horror
...
```

## Usage
1. Clonning repo: `git clone https://github.com/e0xextazy/nlp_huawei_new2_task.git`
2. `cd nlp_huawei_new2_task/`
3. Setup your baseline:
   1. TF-IDF + Logistic Regression: `./setup/setup_tf_idf_logreg.sh`
   2. Catboost: `./setup/setup_catboost.sh`
   3. LSTM: `./setup/setup_lstm.sh`
   4. Transformers: `./setup/setup_transformers.sh`
4. Activate virtual environment: `source venv/bin/activate`
5. Download data: `./setup/download_data.sh`
6. Enjoy!

## Contributing
Copy of the [`contributing.md`](https://github.com/e0xextazy/nlp_huawei_new2_task/blob/master/contributing.md).

### Issue
- If you see an open issue and are willing to do it, add yourself to the performers and write about how much time it will take to fix it. See the pull request module below.
- If you want to add something new or if you find a bug, you should start by creating a new issue and describing the problem/feature. Don't forget to include the appropriate labels.

### Pull request
How to make a pull request.
1. Clone the repository;
2. Create a new branch, for example `git checkout -b issue-id-short-name`;
3. Make changes to the code (make sure you are definitely working in the new branch);
4. `git push`;
5. Create a pull request to the `master` branch;
6. Add a brief description of the work done;
7. Expect comments from the authors.

## Authors
- [Mark Baushenko](https://github.com/e0xextazy)
- [Artem Boldinov](https://github.com/limpwinter)
- [Milana Shxanukova](https://github.com/MilanaShhanukova)
