# dp-xai-implementation
Implementation of Privacy-Preserving Explainability through Knowledge Distillation
# Implementation for [Your Paper Title]

This repository contains the Python implementation for the research paper "DP in XAI" submitted to ISCS '25 and as Capstone Project.

The code demonstrates the training and evaluation of baseline models, DP-SGD (Differentially Private) models, and Knowledge Distillation (KD) student models for both classification and regression tasks.

## File Structure

* `classification_baseline.ipynb`: Trains and evaluates the baseline classifier and runs a Membership Inference Attack (MIA).
* `classification_dp.ipynb`: Trains the DP-SGD teacher classifier, distills its knowledge into a student model, and evaluates utility, privacy (MIA), and explainability (SHAP).
* `regression_baseline.ipynb`: Trains and evaluates the baseline regressor and runs a Model Inversion Attack (MIA).
* `regression_dp.ipynb`: Trains the DP-SGD teacher regressor, distills it into a student, and runs MIA on both.
* `datasets/`: Contains the raw `diabetic_data.csv` and `drug_overdose.csv` files.
* `requirements.txt`: A list of all necessary Python libraries.

## How to Run

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Savnii/dp-xai-implementation.git
    cd dp-xai-implementation
    ```

2.  **Create a virtual environment (Recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the notebooks:**
    You can run the notebooks in any order. The `baseline` files are independent of the `dp` files.

    ```bash
    # To run the notebooks
    jupyter notebook
    ```
    * Open `classification_baseline.ipynb` and run all cells.
    * Open `classification_dp.ipynb` and run all cells.
    * ...and so on for the regression files.

5.  **Review results:**
    Running the code will generate `artifacts/` and `test_cases/` folders, which will contain the saved models (`.pt`), SHAP values (`.csv`), and attack results (`.npy`).

    ---

## Experimental Validation (Ablation Studies)

This repository also includes the code used to generate the specific "Test Case" tables found in Chapter 4 (Results and Analysis) and the Appendix of the research paper.

The code for these experiments is located in the `/experiments` folder. Each notebook is designed to be run independently to reproduce a specific table from the report, such as the Epsilon-Utility trade-off (TC1) or the Optimizer Interaction (TC3).