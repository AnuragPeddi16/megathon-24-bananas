from flask import Flask, request, jsonify
from flask_cors import CORS
import uuid
import ollama
import requests


app = Flask(__name__)
CORS(app)  # Allow all domains to make requests

# Sample in-memory storage for messages
messages = []

@app.route('/messages', methods=['GET'])
def get_messages():
    return jsonify(messages)

@app.route('/messages', methods=['POST'])
def send_message():
    data = request.json
    message = {
        'id': str(uuid.uuid4()),  # Generate a unique ID
        'text': data.get('text'),
        'author': data.get('author'),
        'createdAt': data.get('createdAt'),
    }
    messages.append(message)
    
    # Print the received message
        # Print the received message
    print("Received message:", message)
    custom = 'Analyse the given message for a project"'+message['text']+'". Return the results in the following format (Polarity,Extracted Concern,Category,Intensity out of 10). Examples: "(Negative,constantly worried,Health Anxiety,4), (I feel happy and excited lately.,Positive,happy and excited,Depression,1)". Don\'t format or give a breakdown (only one tuple)';
    response = ollama.chat(model='llama3.2', messages=[{ 'role': 'user', 'content': custom }])
    if 'message' in response and 'content' in response['message']:
        content = response['message']['content']
        # Check if a specific string (e.g., "analyze") is in the content
        if 'self-harm' in content.lower() or 'sucidal' in content.lower():    
            print("(Positive, Self-harm, Suicidal, 10)")
            content="(Positive, Self-harm, Suicidal, 10)"
        else:   
            print(content)
    else:
        print("Prediction failed.")
    
    # Send the received message back as part of the response
    return jsonify(content), 201

def get_public_ip():
    try:
        response = requests.get('https://api.ipify.org?format=json')
        if response.status_code == 200:
            return response.json().get("ip")
        else:
            print("Error fetching IP:", response.status_code)
            return None
    except requests.RequestException as e:
        print("Request failed:", e)
        return None

if __name__ == '__main__':
    public_ip = get_public_ip()
    print("Public IP Address:", public_ip)
    app.run(host='0.0.0.0',port=5001)