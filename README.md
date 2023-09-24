# med-ml
This repository is dedicated to crafting robust code workflows for medical data analysis. 

> Like many scientists, I'm used to analyzing data in notebooks (cue all my software engineering friends shaking their heads in dismay 😅). 

>While I managed to make that approach work for many use cases, including from analysis of genes 🧬, mobile phone data 📱, survey responses 📊 and policy documents 📜—this repository is my chance to give learning how to code a bit more "properly" a shot. 

>I hope filling such knowledge gaps amplifies my ability to make meaningful strides in healthcare by enhancing my technical capabilities to extract actionable insights from data for personalized and population health or to inform health policy decisions. 🚀 🏥 🖥️ 


## Table of Contents
- [Setup](#setup)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Notes](#notes)
   -[To Implement](#to-implement)

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

## Usage

### Data

This repo focuses on frameworks for medical/EHR datasets using synthetic data due to data access and privacy constraints. Real medical datasets, given their sensitivity, won't interact with this repository.

### Run

After activating the environment, execute the run_analysis.py script from the `src` directory with the desired config file.

```shell
python run_analysis.py --config=configs/experiment_config_example.py --train_mode=train
```

Outputs (logs, results, etc.) are saved in a structured directory under out.

To explore other models, craft a new config file mirroring `experiment_config_example.py`. Consult `config_scaffold.py` for possible inputs. Ensure that the variables like the path to the data are correctly specified in this file. 


## Project Structure
```
.
├── data
├── out
├── src
│   ├── configs
│   │   └── config_scaffold.py
│   │   └── experiment_config.py files
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
## Ongoing Work
This repository serves as a dynamic space for both **analysis** and **development** learning and enhancements.

## Development

The development updates emphasize refining software engineering practices for enhanced maintainability and robustness.

| **Completed** | **In Progress** | **Future** |
|:-------------|:---------------|:----------|
| Incorporation of Python `dataclasses` & `enums` | Docstring format revision | Impressing my software engineering friends 😄 |
| Adopted `isort` for imports & utilized `typing` for type hints | Type hint validation with `mypi` | |
| Workflow streamlined with config files; see `config_scaffold.py` for reference | Run mode enhancements: introducing `resume` for `train_mode` and new flows (`preprocess` & `split`) for `data_state` | |
| Config files serve as analysis summaries | | |
| Automation in directory management, file organization, and output generation | | |
| Integrated `logging` module for efficient logging | | |

## Analysis

Analysis updates focus on broadening the model repertoire, enhancing preprocessing, and fine-tuning feature engineering.

| **Completed** | **In Progress** | **Future** |
|:-------------|:---------------|:----------|
| Developed a modular code framework | Introducing the lightgbm model | Advanced preprocessing techniques for longitudinal data |
| Comprehensive pipeline from data ingestion to model evaluation | Chunk-wise data preprocessing for efficiency | Expansion of evaluation metrics |
| Implemented logistic regression using PyTorch & sklearn | Integration of mini batch gradient descent | For PyTorch models: diving into cross validation, grid search, and regularization |
| Rolled out grid search for sklearn models | Incorporation of confusion matrix visualization | Designing complex models (like RNNs) which would necessitate timeseries preprocessing |
| Visualization tools for training & validation losses | ROC curve plotting tools | Delving into methods for model explainability & interpretability |
| Detailed evaluation summaries for all models | | |


## Notes

### ICD-10 Codes
I'm utilizing real "ICD-10" codes, used in medical diagnosis, to enhance the relevancy of our generated data. I recommend exploring ICD-10 code browsers like [WHO's version](https://icd.who.int/browse10/2019/en) and the [Icelandic DoH's variant](https://skafl.is/) – I personally prefer the latter. (Go Iceland!)

### Primary Model Testing Question
**Can patient biological sex be predicted from their medical record codes?**
- This straightforward prediction task offers clarity and allows us to include sex-related medical features (e.g., childbirth, prostate concerns) in our dataset.
- In my synthetic data, I can control the predictiveness of medical codes to resemble real-world health records. For instance, ensuring female-specific codes appear only for generated female patients.
  - This aids in evaluating certain interpretability methods, as I expect to identify these features as significant predictors later.

### Additional Model Testing Avenues
- Future goals include predicting complex, time-sensitive health conditions. Specific target codes will be chosen reflecting prevalent health conditions.
- I'm delving into literature on longitudinal analysis, representational techniques, and interpretability methods to harness them effectively.


 ### Recent Work I Find Cool 
 - Prediction of future disease states via health trajectories in [pancreatic cancer](https://www.nature.com/articles/s41591-023-02332-5) 
 - Using graph neural networks to encode underlying relationships like [family history](https://arxiv.org/abs/2304.05010) to predict disease risk
 - Real time prediction of disease risk as in [acute kidney disease](https://www.nature.com/articles/s41746-020-00346-8), or to manage [ICU circulatory failure](https://www.nature.com/articles/s41591-020-0789-4).
 - Combining econometrics and machine learning to evaluate [physician decision making](https://academic.oup.com/qje/article/137/2/679/6449024) and to assess health policies and standards (e.g. in the case of [breast cancer screening](https://economics.mit.edu/sites/default/files/2022-08/Screening%20and%20Selection-%20The%20Case%20of%20Mammograms.pdf))
 - Using machine learning methods to reduce disparities in underserved populations (e.g. in [pain reduction](https://www.nature.com/articles/s41591-020-01192-7))
 - Understanding unintended consequences of algorithms (e.g. how ML models can [predict race from medical imaging](https://www.thelancet.com/journals/landig/article/PIIS2589-7500(22)00063-2/fulltext)) or [racial biases](https://www.science.org/doi/10.1126/science.aax2342) to make the best and most fair models that enhance patient health.

