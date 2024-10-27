# from flask import Flask, render_template, request, jsonify

# app = Flask(__name__)

# @app.route("/")
# def home_page():  # Change the function name to avoid any conflicts
#     return render_template("chat.html")

# @app.route("/about")
# def about_page():  # Another unique function name
#     return render_template("about.html")

# @app.route("/reach_us")
# def reach_us_page():  # Another unique function name
#     return render_template("reach_us.html")

# # Backend processing for chat
# @app.route("/chat", methods=["POST"])
# def chat():
#     user_message = request.json.get("message", "")
#     # Simple echo response
#     response = f"Server says: {user_message}"
#     return jsonify({"response": response})

# if __name__ == "__main__":
#     app.run(debug=True)

from flask import Flask, render_template, request, jsonify
import requests  # To make requests to the server

app = Flask(__name__)

@app.route("/")
def home_page():
    return render_template("chat.html")

@app.route("/about")
def about_page():
    return render_template("about.html")

@app.route("/reach_us")
def reach_us_page():
    return render_template("reach_us.html")

# Backend processing for chat
@app.route("/chat", methods=["POST"])
def chat():
    user_message = request.json.get("message", "")
    
    # Send the message to the server
    server_url = 'http://127.0.0.1:5001/messages'  # Adjust this URL if needed
    response = requests.post(server_url, json={"text": user_message, "author": "User", "createdAt": "now"})
    
    if response.status_code == 201:
        server_response = response.json()
        return jsonify({"response": {"text": server_response}})
    else:
        return jsonify({"response": "Error contacting server"}), 500

if __name__ == "__main__":
    app.run( port=5000)  # Make sure the port is set correctly
