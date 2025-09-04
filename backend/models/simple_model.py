import numpy as np
from sklearn.ensemble import RandomForestClassifier

class SimpleHeartFailureModel:
    """
    Simplified machine learning model for heart failure prediction
    """
    
    def __init__(self):
        """
        Initialize the model
        """
        print("Initializing SimpleHeartFailureModel...")
        self.model = RandomForestClassifier(n_estimators=10, random_state=42)
        self.feature_names = ['age', 'sex', 'cholesterol', 'max_heart_rate']
        print("SimpleHeartFailureModel initialized!")
    
    def predict(self, features):
        """
        Make a simple prediction
        """
        print("Making prediction...")
        # Create a dummy prediction
        prediction = np.random.random()
        confidence = np.random.random()
        
        # Create dummy SHAP values
        shap_dict = {
            'base_value': 0.5,
            'values': [0.1, -0.05, 0.2, -0.1],
            'feature_names': self.feature_names
        }
        
        print("Prediction complete!")
        return prediction, confidence, shap_dict
