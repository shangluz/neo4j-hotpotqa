import os
from flask import Flask, render_template, request, jsonify
from neo4j import GraphDatabase
import numpy as np
from sklearn.cluster import KMeans

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.environ.get("NEO4J_USER", "neo4j")
NEO4J_PWD = os.environ.get("NEO4J_PWD", "zhu21192119")
MODEL_LOCAL_PATH = os.path.abspath(os.environ.get("LOCAL_MODEL_PATH", "./local_model"))
app = Flask(__name__, static_folder="static", template_folder="templates")

def load_embedding_model(preferred_path=MODEL_LOCAL_PATH):
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(preferred_path)
    return model

model = load_embedding_model()
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PWD), max_connection_lifetime=60)

def cosine(a, b):
    a = np.array(a); b = np.array(b)
    denom = (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10)
    return float(np.dot(a, b) / denom)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/retrieve")
def retrieve():
    q = request.args.get("q", "")
    if not q:
        return jsonify({"error": "missing query parameter q"}), 400
    try:
        top_k = int(request.args.get("top_k", 5))
    except:
        top_k = 5
    try:
        q_emb = model.encode(q, convert_to_numpy=True)
        if isinstance(q_emb, np.ndarray) and q_emb.ndim == 2 and q_emb.shape[0] == 1:
            q_emb = q_emb[0]
    except Exception:
        q_emb = model.encode([q], convert_to_numpy=True)[0]
    candidates = []
    with driver.session() as sess:
        try:
            res = sess.run(
                """
                CALL db.index.fulltext.queryNodes('sentenceIndex', $q) YIELD node, score
                RETURN id(node) as id, node.text as text, node.vec as vec, score
                ORDER BY score DESC
                LIMIT 200
                """, q=q)
            rows = list(res)
            if not rows:
                raise Exception("fulltext returned no rows")
            for r in rows:
                vid = r["id"]; text = r["text"]; vec = r["vec"]
                sim = cosine(q_emb, vec) if vec is not None else 0.0
                candidates.append({"id": int(vid), "text": text, "sim": sim})
        except Exception:
            res2 = sess.run("MATCH (s:Sentence) WHERE s.vec IS NOT NULL RETURN id(s) as id, s.text as text, s.vec as vec LIMIT 1000")
            for r in res2:
                vid = r["id"]; text = r["text"]; vec = r["vec"]
                sim = cosine(q_emb, vec) if vec is not None else 0.0
                candidates.append({"id": int(vid), "text": text, "sim": sim})
    candidates = sorted(candidates, key=lambda x: x["sim"], reverse=True)[:top_k]
    return jsonify({"query": q, "results": candidates})

@app.route("/api/multihop")
def multihop():
    q = request.args.get("q", "")
    if not q:
        return jsonify({"error": "missing query parameter q"}), 400
    try:
        top_k = int(request.args.get("top_k", 3))
    except:
        top_k = 3
    try:
        hops = int(request.args.get("hops", 2))
    except:
        hops = 2
    try:
        q_emb = model.encode(q, convert_to_numpy=True)
        if isinstance(q_emb, np.ndarray) and q_emb.ndim == 2 and q_emb.shape[0] == 1:
            q_emb = q_emb[0]
    except Exception:
        q_emb = model.encode([q], convert_to_numpy=True)[0]

    candidates = []
    seed_ids = []
    nodes = {}
    edges = []
    with driver.session() as sess:
        try:
            res = sess.run(
                """
                CALL db.index.fulltext.queryNodes('sentenceIndex', $q) YIELD node, score
                RETURN id(node) as id, node.text as text, node.vec as vec, score
                ORDER BY score DESC
                LIMIT 200
                """, q=q)
            for r in res:
                vid = r["id"]; text = r["text"]; vec = r["vec"]
                sim = cosine(q_emb, vec) if vec is not None else 0.0
                candidates.append({"id": int(vid), "text": text, "sim": sim})
        except Exception:
            res2 = sess.run("MATCH (s:Sentence) WHERE s.vec IS NOT NULL RETURN id(s) as id, s.text as text, s.vec as vec LIMIT 200")
            for r in res2:
                vid = r["id"]; text = r["text"]; vec = r["vec"]
                sim = cosine(q_emb, vec) if vec is not None else 0.0
                candidates.append({"id": int(vid), "text": text, "sim": sim})

        candidates = sorted(candidates, key=lambda x: x["sim"], reverse=True)[: max(1, top_k * 3)]
        seed_ids = [c["id"] for c in candidates]
        for sid in seed_ids:
            rec = sess.run(
                """
                MATCH (s:Sentence) WHERE id(s)=$sid
                OPTIONAL MATCH (p:Paragraph)-[:HAS_SENTENCE]->(s)
                RETURN s.text as stext, id(s) as sid, id(p) as pid, p.title as ptitle, s.cluster as scluster
                """, sid=sid).single()
            if not rec:
                continue
            stext = rec["stext"]; pid = rec["pid"]; ptitle = rec["ptitle"]
            scluster = rec.get("scluster", None)
            nodes[f"s{sid}"] = {"id": f"s{sid}", "type": "Sentence", "text": stext, "cluster": scluster}
            if pid is not None:
                nodes[f"p{pid}"] = {"id": f"p{pid}", "type": "Paragraph", "title": ptitle}
                edges.append({"source": f"p{pid}", "target": f"s{sid}", "type": "HAS_SENTENCE"})
        for _ in range(hops):
            new_nodes = {}
            for nid, v in list(nodes.items()):
                if v["type"] == "Paragraph":
                    pid = int(nid[1:])
                    recs = sess.run(
                        "MATCH (p:Paragraph)-[:HAS_SENTENCE]->(s:Sentence) WHERE id(p)=$pid RETURN id(s) as sid, s.text as text, s.cluster as cluster LIMIT 50",
                        pid=pid)
                    for r in recs:
                        sid = r["sid"]; text = r["text"]; cluster = r.get("cluster", None)
                        key = f"s{sid}"
                        if key not in nodes and key not in new_nodes:
                            new_nodes[key] = {"id": key, "type": "Sentence", "text": text, "cluster": cluster}
                            edges.append({"source": f"p{pid}", "target": key, "type": "HAS_SENTENCE"})
            nodes.update(new_nodes)

    seed_node_ids = [f"s{sid}" for sid in seed_ids]
    return jsonify({"nodes": list(nodes.values()), "edges": edges, "seed_count": len(seed_ids), "seed_ids": seed_node_ids})

@app.route("/api/sentence")
def get_sentence():
    node_id = request.args.get("id")
    if not node_id:
        return jsonify({"error":"id required"}),400
    try:
        if isinstance(node_id, str) and node_id.startswith("s"):
            nid = int(node_id[1:])
        else:
            nid = int(node_id)
    except:
        return jsonify({"error":"invalid id"}),400
    with driver.session() as sess:
        rec = sess.run("MATCH (s:Sentence) WHERE id(s)=$id RETURN s.text as text", id=nid).single()
        if not rec:
            return jsonify({"error":"not found"}),404
        return jsonify({"text": rec["text"]})

@app.route("/api/paragraph")
def get_paragraph():
    node_id = request.args.get("id")
    if not node_id:
        return jsonify({"error":"id required"}),400
    try:
        if isinstance(node_id, str) and node_id.startswith("p"):
            pid = int(node_id[1:])
        else:
            pid = int(node_id)
    except:
        return jsonify({"error":"invalid id"}),400
    with driver.session() as sess:
        rec = sess.run("MATCH (p:Paragraph) WHERE id(p)=$id RETURN p.title as title, p.text as text", id=pid).single()
        if not rec:
            return jsonify({"error":"not found"}),404
        return jsonify({"id": f"p{pid}", "title": rec["title"], "text": rec["text"]})

@app.route("/api/cluster", methods=["POST"])
def cluster():
    data = request.json or request.form
    paragraph_id = data.get("paragraph_id")
    paragraph_title = data.get("paragraph_title")
    try:
        n_clusters = int(data.get("n_clusters", 3))
    except:
        n_clusters = 3
    if not paragraph_id and not paragraph_title:
        return jsonify({"error": "paragraph_id or paragraph_title required"}), 400

    with driver.session() as sess:
        if paragraph_id:
            try:
                if isinstance(paragraph_id, str) and paragraph_id.startswith("p"):
                    pid_val = int(paragraph_id[1:])
                else:
                    pid_val = int(paragraph_id)
            except:
                return jsonify({"error": "invalid paragraph_id"}), 400
            res = sess.run(
                "MATCH (p:Paragraph) WHERE id(p)=$pid RETURN p.title as title",
                pid=pid_val)
            rec0 = res.single()
            if not rec0:
                return jsonify({"error":"paragraph not found by id"}),404
            title_used = rec0["title"]
            res2 = sess.run("MATCH (p) WHERE id(p)=$pid MATCH (p)-[:HAS_SENTENCE]->(s:Sentence) RETURN id(s) as id, s.vec as vec, s.text as text", pid=pid_val)
            rows = [r for r in res2]
        else:
            title_used = paragraph_title
            res2 = sess.run("MATCH (p:Paragraph {title:$title})-[:HAS_SENTENCE]->(s:Sentence) RETURN id(s) as id, s.vec as vec, s.text as text", title=paragraph_title)
            rows = [r for r in res2]

        if not rows:
            return jsonify({"error": "no paragraph or sentences found"}), 404

        ids = [r["id"] for r in rows]
        embs = [r["vec"] for r in rows]
        texts = [r["text"] for r in rows]

    idxs = [i for i, e in enumerate(embs) if e]
    if not idxs:
        return jsonify({"error": "no embeddings available for sentences in paragraph"}), 400

    X = np.array([embs[i] for i in idxs])
    k = min(n_clusters, len(X))
    kmeans = KMeans(n_clusters=k, random_state=42).fit(X)
    labels = kmeans.labels_
    with driver.session() as sess:
        for li, ii in enumerate(idxs):
            sid = ids[ii]
            lab = int(labels[li])
            sess.run("MATCH (s) WHERE id(s)=$id SET s.cluster=$lab", id=sid, lab=lab)

    return jsonify({"paragraph": title_used, "n_clusters": int(k), "assigned": int(len(labels))})

if __name__ == "__main__":
    host = os.environ.get("FLASK_HOST", "0.0.0.0")
    port = int(os.environ.get("FLASK_PORT", 5000))
    app.run(host=host, port=port, debug=False)
