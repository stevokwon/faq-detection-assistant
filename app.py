from flask import Flask, request, jsonify
import openai
import numpy as np
from numpy.linalg import norm 
import os

openai.api_key = os.getenv("OPENAI_API_KEY")
app = Flask(__name__)

faq_data = {
    "How do I contribute?": "You can contribute by forking the repository, making your changes, and opening a pull request.",
    "How do I report bugs?": "Please open an issue and follow the issue template.",
    "Where can I find the documentation?": "See the README and Wiki pages."
}

faq_embeddings = {}

for question in faq_data:
    response = openai.embeddings.create(
        input = question,
        model = 'text-embedding-3-small'
    )
    faq_embeddings[question] = response.data[0].embedding

def cos_sim(a, b):
    return np.dot(a, b) / (norm(a) * norm(b))

def match_faq_semantic(incoming_q):
    # Getting embeddings from the incoming questions
    for question in incoming_q:
        response = openai.embeddings.create(
            input = question,
            model = 'text-embedding-3-small'
        )
        question_embedding = response.data[0].embedding
    
    #Compare to each existing FAQ
    best_match = None
    highest_score = 0.0

    for stored_q, stored_embedding in faq_embeddings.items():
        score = cos_sim(question_embedding, stored_embedding)
        if score > highest_score:
            highest_score = score
            best_match = stored_q

    if highest_score > 0.80:
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