from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

@app.route("/")
def home_page():  # Change the function name to avoid any conflicts
    return render_template("chat.html")

@app.route("/about")
def about_page():  # Another unique function name
    return render_template("about.html")

@app.route("/reach_us")
def reach_us_page():  # Another unique function name
    return render_template("reach_us.html")

# Backend processing for chat
@app.route("/chat", methods=["POST"])
def chat():
    user_message = request.json.get("message", "")
    # Simple echo response
    response = f"Server says: {user_message}"
    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(debug=True)