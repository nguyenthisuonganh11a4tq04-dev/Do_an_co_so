from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

from crawler import crawl_facebook_comments
from model_api import predict_sentiment

app = Flask(__name__)
CORS(app)


# =========================
# HOME PAGE
# =========================
@app.route('/')
def home():

    return render_template("index.html")


# =========================
# FACEBOOK API
# =========================
@app.route('/fetch-facebook', methods=['POST'])
def fetch_facebook():

    try:

        data = request.json

        url = data.get("url")

        if not url:

            return jsonify({
                "error": "Missing URL"
            }), 400

        print("=" * 50)
        print("URL:", url)

        # Crawl comment
        comments = crawl_facebook_comments(url)

        print("COMMENTS:")
        print(comments)

        print("TOTAL COMMENTS:", len(comments))

        if len(comments) == 0:

            return jsonify({
                "error": "Không lấy được comment Facebook",
                "results": []
            })

        results = []

        # Predict sentiment
        for comment in comments:

            try:

                sentiment = predict_sentiment(comment)

                print(comment)
                print("=>", sentiment)

                results.append({
                    "text": comment,
                    "sentiment": sentiment
                })

            except Exception as e:

                print("PREDICT ERROR:", e)

        return jsonify(results)

    except Exception as e:

        print("SERVER ERROR:", e)

        return jsonify({
            "error": str(e)
        }), 500


# =========================
# MAIN
# =========================
if __name__ == '__main__':

    app.run(
        host='0.0.0.0',
        port=5000,
        debug=False,
        use_reloader=False
    )