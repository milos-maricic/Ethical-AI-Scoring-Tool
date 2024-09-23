
import pandas as pd
import numpy as np

# Function to check for data bias in protected attributes
def assess_data_bias(data, protected_columns):
    """
    Check for data bias in protected attributes.
    :param data: DataFrame containing the dataset.
    :param protected_columns: List of columns that are protected attributes.
    :return: Bias scores for each protected column.
    """
    bias_scores = {}
    
    for column in protected_columns:
        # Calculate the distribution of values in the protected column
        value_counts = data[column].value_counts(normalize=True) * 100
        max_percentage = value_counts.max()
        
        # Score bias (higher values indicate less bias)
        bias_score = 100 - max_percentage  # Closer to 50% is ideal
        bias_scores[column] = {
            "distribution": value_counts.to_dict(),
            "bias_score": bias_score
        }
    
    return bias_scores

# Function to assess model explainability (e.g., feature importance)
def assess_model_explainability(model, feature_names):
    """
    Assess model explainability based on feature importance.
    :param model: Trained model with a feature_importances_ attribute.
    :param feature_names: List of feature names.
    :return: Explainability score and feature importance details.
    """
    feature_importance = model.feature_importances_
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': feature_importance
    }).sort_values(by='importance', ascending=False)
    
    # Calculate an explainability score (higher values for more even distribution of importance)
    explainability_score = 100 - importance_df['importance'].std() * 100  # Lower variance = better explainability
    
    return {
        "importance": importance_df.to_dict(orient='records'),
        "explainability_score": explainability_score
    }

# Function to generate an overall ethical score
def generate_ethical_score(data, protected_columns, model=None, feature_names=None):
    """
    Generate an overall ethical AI score based on data bias, model explainability, and other criteria.
    :param data: DataFrame containing the dataset.
    :param protected_columns: List of protected attribute columns to assess bias.
    :param model: Trained model for assessing explainability (optional).
    :param feature_names: List of feature names for explainability (optional).
    :return: Overall ethical score and breakdown.
    """
    # Bias assessment
    bias_assessment = assess_data_bias(data, protected_columns)
    bias_scores = [v['bias_score'] for v in bias_assessment.values()]
    overall_bias_score = np.mean(bias_scores)
    
    # Explainability assessment (optional)
    explainability_score = 0
    if model and feature_names:
        explainability_assessment = assess_model_explainability(model, feature_names)
        explainability_score = explainability_assessment["explainability_score"]
    
    # Weighted average to calculate final ethical score
    weights = {
        'bias': 0.6,
        'explainability': 0.4
    }
    ethical_score = (overall_bias_score * weights['bias'] + explainability_score * weights['explainability'])
    
    return {
        "ethical_score": ethical_score,
        "bias_assessment": bias_assessment,
        "explainability_assessment": explainability_assessment if model else None
    }

# Example usage with simulated data
data = pd.DataFrame({
    'gender': np.random.choice(['male', 'female'], size=100),
    'race': np.random.choice(['group_1', 'group_2', 'group_3'], size=100),
    'age': np.random.randint(18, 70, size=100),
    'income': np.random.randint(20000, 100000, size=100)
})

# Simulated feature importance for a model (optional)
feature_importances = np.random.rand(4)
model = type('Model', (object,), {'feature_importances_': feature_importances})()

# Generate ethical score
protected_columns = ['gender', 'race']
ethical_report = generate_ethical_score(data, protected_columns, model, ['gender', 'race', 'age', 'income'])
