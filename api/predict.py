from http.server import BaseHTTPRequestHandler
import json
import numpy as np
import joblib
import os

# 1. Locate the .pkl files in the root folder (one level up from /api)
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(base_dir, 'digit_classification_model.pkl')
scaler_path = os.path.join(base_dir, 'digit_scaler.pkl')

# 2. Load the model and scaler into memory
model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

# 3. Vercel's required handler format
class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        try:
            # Read the incoming JSON data from the frontend
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data)
            
            # Extract the 64-pixel array
            pixels = data.get('pixels', [])
            
            # Reshape it into a single row, scale it, and predict
            pixel_array = np.array(pixels).reshape(1, -1)
            scaled_pixels = scaler.transform(pixel_array)
            prediction = model.predict(scaled_pixels)
            
            # Send the successful response back to JavaScript
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            
            # Convert prediction to an integer so JSON can read it
            response = {'prediction': int(prediction[0])}
            self.wfile.write(json.dumps(response).encode('utf-8'))
            
        except Exception as e:
            # If anything breaks, send an error back safely
            self.send_response(500)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            error_response = {'error': str(e)}
            self.wfile.write(json.dumps(error_response).encode('utf-8'))