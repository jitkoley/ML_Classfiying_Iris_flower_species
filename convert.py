import os
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import joblib

# Define directories
input_dir = 'app/models'  # Directory containing the saved models
output_dir = 'app/convert'  # Directory to save ONNX models

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Function to convert a model to ONNX and save it
def convert_to_onnx(input_path, output_path, input_dimension):
    cls = joblib.load(input_path)  # Load the model
    initial_type = [('float_input', FloatTensorType([None, input_dimension]))]
    try:
        onx = convert_sklearn(cls, initial_types=initial_type)  # Convert to ONNX
        with open(output_path, 'wb') as f:
            f.write(onx.SerializeToString())
        print(f"ONNX model exported successfully: {output_path}")
    except Exception as e:
        print(f"Error converting {input_path} to ONNX: {e}")

# Iterate over all model files in the input directory
for model_file in os.listdir(input_dir):
    if model_file.endswith('.pkl'):  # Process only .pkl files
        model_path = os.path.join(input_dir, model_file)
        onnx_path = os.path.join(output_dir, model_file.replace('.pkl', '.onnx'))
        convert_to_onnx(model_path, onnx_path, input_dimension=4)  # Assuming 4 input features
