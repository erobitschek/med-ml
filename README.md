# med-ml
A repository to develop code workflows for to analyze medical data to improve healthcare, for example by extracting insights that inform population health and health policy. 

## Table of Contents
- [Setup](#setup)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Notes](#notes)

## Setup
1. Clone the repository:
   ```shell
   git clone https://github.com/erobitschek/med-ml
   ```
2. Navigate to the project directory:
   ```shell
   cd med-ml
   ```
3. Create environment for analysis using the `environment.yml` file (requires [conda](https://docs.conda.io/en/latest/)):
   ```shell
   conda env create -f src/environment.yml
   ```
4. Activate the virtual environment:
   ```shell
   conda activate med-ml
   ```
To run the analysis you will need a config file like the `experiment_config_example.py` file. Ensure that the variables like the path to the data is correctly specified in this file. 

## Usage

### Data

 Many medical/EHR datasets have access restrictions for ethical and privacy reasons. This repository will be a workspace to create generalizable model frameworks to use on such data using simulated or 'synthetic' data. Obviously even with access to medical record datasets, none of the real data will ever interact directly with this repository as it is private and access is and should be tightly controlled. 

### Run

Run a command to activate the python environment and run the `run_analysis.py` script to reproduce the analysis.

To reproduce the analysis run the following<sup>1</sup>: 

```shell
conda activate med-ml
python run_analysis.py --config configs/experiment_config_example.py --setup --train_mode=train 
```

This will save logs, results, and other outputs related to the run to a `run_dir` in the `out` dir specified by  ` \ results \ <dataset name> \ <model_name> \ <run_name>`

To run other analysis with other models, make a new config file with a similar format to the `experiment_config_example.py` (there are two example configs in the config folder)


## Project Structure
```
.
├── data
├── out
├── src
│   ├── configs
│   │   └── experiment_config.py
│   ├── utils.py 
|   ├── data.py
│   ├── models.py
│   ├── train.py
│   ├── predict.py
│   ├── eval.py
│   ├── vis.py
│   ├── run_simple.py
│   ├── run_torch.py
│   ├── run_analysis.py
│   └── environment.yml
└── .gitignore
```

---
### To implement
This repository is a work in progress and will be updated regularly. I am currently working on: 
- Mini batch gradient descent 
- Add lightgbm model
- Adding other evaluation metrics
- Plotting of ROC
- Add code for confusion matrix
- For pytorch models: 
   - cross validation 
   - grid search 
   - regularization
- Implementing more complex models (e.g. RNNs) but this also requires additional preprocessing of the data as timeseries, and a different target question
- Methods for explainability and interpretability
---


## Notes

#### ICD-10 Codes
To make this generated data more relevant, I will draw the medical codes from actual "International Statistical Classification of Diseases and Related Health Problems 10th Revision" or "ICD-10" codes commonly used in medical practice for diagnosis. Some browsers for this type of data can be found [from WHO](https://icd.who.int/browse10/2019/en) and [from the Icelandic DoH](https://skafl.is/). (I'm a partciular fan of the Icelandic one).

#### Initial question for model testing
 *Can we predict the biological sex of the patient based on the signal from the codes in their medical record?*
 - This task was chosen for its relative interpretability and simplicity as a good 'low hanging fruit' prediction task, and because I could a priori choose medical features related to sex to model in the dataset (e.g. birth, prostate issues, etc).
 - In my synthetic data I can titrate that predictive signal in the generated records of the patients in a way that loosely mimics real health records, because I can ensure that only female (male) generated patients get codes associated with female (male) biology, and can adjust what proportion of the total codes are sex-related. 
   - This will faciliate downstream testing and sanity checks of some interpretability and explanability methods, as I should be able to recover some of these features as most predictive and relevant later.
 
 ### Other questions for model testing
 - As a later task, I will also want to predict more complex and time dependent conditions so I will also specify target codes based on a common health condition with that in mind. 
 - I am currently reviewing the literature re: longitudinal analysis (e.g. with transformers), respresentational methods (e.g. graph neural networks) and interpretability and explanability techniques in order to be able to apply these methods appropriately.

 ### Recent work I find particularly cool 
 - Prediction of future disease states via health trajectories in [pancreatic cancer](https://www.nature.com/articles/s41591-023-02332-5) 
 - Using graph neural networks to encode underlying relationships like [family history](https://arxiv.org/abs/2304.05010) to predict disease risk
 - Real time prediction of disease risk as in [acute kidney disease](https://www.nature.com/articles/s41746-020-00346-8), or to manage [ICU circulatory failure](https://www.nature.com/articles/s41591-020-0789-4).
 - Combining econometrics and machine learning to evaluate [physician decision making](https://academic.oup.com/qje/article/137/2/679/6449024) and to assess health policies and standards (e.g. in the case of [breast cancer screening](https://economics.mit.edu/sites/default/files/2022-08/Screening%20and%20Selection-%20The%20Case%20of%20Mammograms.pdf))
 - Using machine learning methods to reduce disparities in underserved populations (e.g. in [pain reduction](https://www.nature.com/articles/s41591-020-01192-7))
 - Understanding unintended consequences of algorithms (e.g. how ML models can [predict race from medical imaging](https://www.thelancet.com/journals/landig/article/PIIS2589-7500(22)00063-2/fulltext)) or [racial biases](https://www.science.org/doi/10.1126/science.aax2342) to make the best and most fair models that enhance patient health.

