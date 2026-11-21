import joblib
import pandas as pd

#import the model from download in notebook
model = joblib.load('cpu_forecast_linear_model.pkl')

#Give random values to check
sample_input = pd.DataFrame({
    'active_vm_count': [120],
    'lag_1': [0.45],
    'lag_2': [0.42],
    'lag_3': [0.40],
    'lag_6': [0.38],
    'lag_12': [0.28]
})


prediction = model.predict(sample_input)
print(f"Predicted Next 5-Min CPU Utilization: {prediction[0]:.4f}")