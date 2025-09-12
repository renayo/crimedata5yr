import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.feature_selection import SelectKBest, f_regression
from tqdm import tqdm
import warnings

# By having __init__.py in the folder, Python now knows to look here for modules.
# This should resolve the import error in most environments.
try:
    from gem2 import ChicagoAstroCrimePredictor
except ImportError as e:
    print("❌ Critical ImportError. Python cannot find 'gem2'.")
    print("   Please ensure that __init__.py, gem2.py, and this script are all in the same folder.")
    print(f"   Original error: {e}")
    exit()

warnings.filterwarnings('ignore')


def load_model_and_get_params(filename="random_forest_sum_severity032.pkl"):
    """Loads the specified model and extracts its parameters."""
    try:
        with open(filename, 'rb') as f:
            model = pickle.load(f)
        print(f"✅ Successfully loaded model from '{filename}'")
        return model.get_params()
    except FileNotFoundError:
        print(f"❌ Error: Model file '{filename}' not found.")
        print("   Please ensure the trained model .pkl file is in the same directory.")
        return None
    except Exception as e:
        print(f"❌ An error occurred while loading the model: {e}")
        return None

def prepare_data_for_simulation(start_year, end_year, target_variable):
    """
    Replicates the exact data preparation pipeline from gem2.py to get the
    datasets used for training and testing the original model.
    """
    print("\n" + "="*70)
    print("STEP 1: REPLICATING ORIGINAL DATA PREPARATION")
    print("="*70)
    
    predictor = ChicagoAstroCrimePredictor()
    df = predictor.prepare_data(start_year=start_year, end_year=end_year)
    
    if df is None or df.empty:
        raise ValueError("Data preparation failed. Cannot proceed.")
        
    print("\nReplicating feature engineering and data splitting...")
    feature_cols = [col for col in df.columns if col not in ['datetime', 'mean_severity', 'median_severity', 'sum_severity', 'crime_count']]
    X = df[feature_cols]
    y = df[target_variable]
    
    tscv = TimeSeriesSplit(n_splits=5)
    train_idx, test_idx = list(tscv.split(X))[-1]
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    
    n_features = min(50, X_train.shape[1])
    selector = SelectKBest(f_regression, k=n_features)
    
    X_train_selected = selector.fit_transform(X_train, y_train)
    X_test_selected = selector.transform(X_test)
    
    selected_features = X.columns[selector.get_support()].tolist()
    
    X_train_final = pd.DataFrame(X_train_selected, columns=selected_features, index=X_train.index)
    X_test_final = pd.DataFrame(X_test_selected, columns=selected_features, index=X_test.index)

    print(f"✅ Data preparation complete.")
    print(f"   Training features shape: {X_train_final.shape}")
    print(f"   Testing features shape:  {X_test_final.shape}")
    
    return X_train_final, X_test_final, y_train, y_test

def run_monte_carlo_p_value_test(X_train, X_test, y_train, y_test, model_params, observed_r2, n_permutations):
    """
    Runs a Monte Carlo permutation test to calculate the p-value for an R-squared score.
    """
    print("\n" + "="*70)
    print("STEP 2: RUNNING MONTE CARLO PERMUTATION TEST")
    print("="*70)
    print(f"Observed R-squared to beat: {observed_r2:.4f}")
    print(f"Number of permutations: {n_permutations:,}")
    
    count_exceeding = 0
    permuted_scores = []
    
    for _ in tqdm(range(n_permutations), desc="Running Permutations"):
        y_train_shuffled = np.random.permutation(y_train)
        
        perm_model = RandomForestRegressor(**model_params)
        perm_model.fit(X_train, y_train_shuffled)
        
        y_pred_perm = perm_model.predict(X_test)
        
        perm_r2 = r2_score(y_test, y_pred_perm)
        permuted_scores.append(perm_r2)
        
        if perm_r2 >= observed_r2:
            count_exceeding += 1
            
    p_value = (count_exceeding + 1) / (n_permutations + 1)
    
    print("✅ Simulation complete.")
    return p_value, count_exceeding, permuted_scores

def main():
    """Main execution function."""
    
    MODEL_FILENAME = "random_forest_sum_severity032.pkl"
    OBSERVED_R2 = 0.317899
    N_PERMUTATIONS = 1000
    TARGET_VARIABLE = 'sum_severity'
    START_YEAR = 2022
    END_YEAR = 2024
    
    model_params = load_model_and_get_params(MODEL_FILENAME)
    if model_params is None:
        return

    try:
        X_train, X_test, y_train, y_test = prepare_data_for_simulation(
            start_year=START_YEAR, 
            end_year=END_YEAR,
            target_variable=TARGET_VARIABLE
        )
    except Exception as e:
        print(f"\n❌ An error occurred during data preparation: {e}")
        return

    p_value, count, scores = run_monte_carlo_p_value_test(
        X_train, X_test, y_train, y_test, 
        model_params=model_params,
        observed_r2=OBSERVED_R2, 
        n_permutations=N_PERMUTATIONS
    )
    
    print("\n" + "="*70)
    print("STEP 3: FINAL RESULTS & INTERPRETATION")
    print("="*70)
    print(f"Observed R-squared value: {OBSERVED_R2:.4f}")
    print(f"Number of permutations run: {N_PERMUTATIONS:,}")
    print(f"Permuted models exceeding observed R²: {count}")
    print(f"Highest R² from a permuted (random) model: {max(scores):.4f}")
    print(f"\nCalculated Monte Carlo p-value: {p_value:.6f}")
    
    print("\n--- Interpretation ---")
    if p_value < 0.05:
        print(f"The p-value ({p_value:.6f}) is less than the common significance level of 0.05.")
        print("This provides strong evidence to reject the null hypothesis.")
        print(f"Conclusion: Your model's performance is statistically significant. 🥳")
    else:
        print(f"The p-value ({p_value:.6f}) is not less than the common significance level of 0.05.")
        print("We fail to reject the null hypothesis.")
        print(f"Conclusion: Your model's performance may not be statistically significant. 🤔")

if __name__ == "__main__":
    np.random.seed(42) 
    main()

