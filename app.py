from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
import numpy as np
from numpy.linalg import norm 

app = Flask(__name__)
model = SentenceTransformer('all-MiniLM-L6-v2')

faq_data = {
    "How do I contribute?": "You can contribute by forking the repository, making your changes, and opening a pull request.",
    "How do I report bugs?": "Please open an issue and follow the issue template.",
    "Where can I find the documentation?": "See the README and Wiki pages."
}

faq_embeddings = {
    question: model.encode(question)
    for question in faq_data
}

def cos_sim(a, b):
    return np.dot(a, b) / (norm(a) * norm(b))

def match_faq_semantic(incoming_q):
    # Getting embeddings from the incoming questions
    question_embedding = model.encode(incoming_q)
    
    #Compare to each existing FAQ
    best_match = None
    highest_score = 0.0

    for stored_q, stored_embedding in faq_embeddings.items():
        score = cos_sim(question_embedding, stored_embedding)
        print(f"🔎 Similarity score with '{stored_q}' : {score:.3f}")
        if score > highest_score:
            highest_score = score
            best_match = stored_q

    if highest_score > 0.65:
        return faq_data[best_match]
    else:
        return None

@app.route('/webhook', methods=['POST'])
def handle_webhook():
    # This is where you'll handle the webhook payload
    payload = request.json
    
    if 'issue' in payload and payload['action'] == 'opened':
        question = payload['issue']['title']
        print("🐛 New Issue Question:", question)
        
    elif 'discussion' in payload and payload['action'] == 'created':
        question = payload['discussion']['title']
        print("💬 New Discussion Question:", question)

    else:
        print("📭 Unhandled webhook event")
        return jsonify({"status": "ignored"}), 200
        
    answer = match_faq_semantic(question)
    if not answer:
        answer = match_faq(question)

    if answer:
        print("✅ Matched FAQ:", answer)
    else:
        print("❌ No match found.")
    
    return jsonify({"status": "success"}), 200  # Respond back to GitHub to acknowledge receipt

@app.route('/', methods = ['GET'])
def home():
    return 'Webhook Server is Running', 200

def match_faq(question):
    for stored_q, answer in faq_data.items():
        if question.lower() in stored_q.lower() or stored_q.lower() in question.lower():
            return answer
    return None

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)  # Expose the app on port 5000