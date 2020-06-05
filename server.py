from flask import Flask, request, jsonify, render_template
import answerer

app = Flask(__name__)


@app.route('/answer', methods=['GET'])
def results():
    query = request.args.get("query") or ""
    print(f"Recevied query for '{query}'")

    output = answerer.answer(query).head(15).rename(columns={
        "Abstract": "abstract", "Published Year": "year", "Similarity": "similarity", "Title": "title", "URL": "url"}).to_dict("records")

    return jsonify(output)


if __name__ == "__main__":
    answerer = answerer.Answerer()
    print("Building Model!")
    answerer.build_model()
    print("Initializing Done!")
    app.run(debug=False)
