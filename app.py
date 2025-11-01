from flask import Flask, request, jsonify
from flask_cors import CORS
from documents import rag_chat, rag_retriever, llm

app = Flask(__name__)
CORS(app)

chat_history = []   # <-- ADD THIS

@app.route("/api/chat", methods=["POST"])
def chat():
    global chat_history
    data = request.get_json()
    message = data.get("message", "").strip()
    if not message:
        return jsonify({"response": "Please enter a message."}), 400

    try:
        response = rag_chat(message, chat_history, retriever=rag_retriever, llm=llm)
        chat_history.append((message, response))          # <-- ADD THIS
        if len(chat_history) > 6:                         # optional prune
            chat_history = chat_history[-6:]
        return jsonify({"response": response})
    except Exception as e:
        print("❌ Error:", e)
        return jsonify({"response": f"Server error: {str(e)}"}), 500

if __name__ == "__main__":
    print("✅ Flask backend running at http://127.0.0.1:5000")
    app.run(debug=True, use_reloader=False)
