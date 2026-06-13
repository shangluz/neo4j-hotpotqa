import os
import json
import uuid
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
CONFIG = {
    "infile": "processed_sentences.jsonl",
    "neo4j_uri": "bolt://localhost:7687",
    "user": "neo4j",
    "password": "zhu21192119",
    "embed_model": "all-MiniLM-L6-v2",
    "batch": 64,
}

def check_connection(driver):
    try:
        with driver.session() as s:
            r = s.run("RETURN 1 AS ok").single()
            return r and r["ok"] == 1
    except Exception as e:
        print("Neo4j connection failed:", e)
        return False

def create_constraints(tx):
    tx.run("CREATE CONSTRAINT IF NOT EXISTS FOR (q:Question) REQUIRE q.qid IS UNIQUE")
    tx.run("CREATE CONSTRAINT IF NOT EXISTS FOR (p:Paragraph) REQUIRE p.pid IS UNIQUE")
    tx.run("CREATE CONSTRAINT IF NOT EXISTS FOR (s:Sentence) REQUIRE s.sid IS UNIQUE")

def flush_batch(driver, batch_sentences, batch_embeddings):
    if not batch_sentences:
        return
    with driver.session() as sess:
        rows = []
        for s, e in zip(batch_sentences, batch_embeddings):
            rows.append({
                "sid": s["sid"],
                "text": s["text"],
                "qid": s["qid"],
                "pid": s["pid"],
                "vec": e.tolist() if hasattr(e, 'tolist') else e
            })
        query = """
        UNWIND $rows AS r
        MERGE (s:Sentence {sid: r.sid})
          SET s.text = r.text, s.vec = r.vec
        WITH s, r
        MATCH (q:Question {qid: r.qid})
        MERGE (q)-[:HAS_EVIDENCE]->(s)
        WITH s, r
        MATCH (p:Paragraph {pid: r.pid})
        MERGE (p)-[:HAS_SENTENCE]->(s)
        """
        sess.run(query, rows=rows)

def main():
    infile = CONFIG["infile"]
    uri = CONFIG["neo4j_uri"]
    user = CONFIG["user"]
    password = CONFIG["password"]
    embed_model = CONFIG["embed_model"]
    batch_size = CONFIG["batch"]
    driver = GraphDatabase.driver(uri, auth=(user, password), max_connection_lifetime=60)
    if not check_connection(driver):
        print("无法连接到 Neo4j，请检查 URI/用户名/密码。")
        return
    print(f"加载嵌入模型: {embed_model}")
    model = SentenceTransformer(embed_model)
    with driver.session() as sess:
        sess.execute_write(create_constraints)
    print(f"开始流式处理文件: {infile}")
    questions_seen = set()
    paragraphs_seen = set()
    batch_texts = []
    batch_sentences = []
    total_sentences = 0
    try:
        with open(infile, "r", encoding="utf-8") as f:
            total_lines = sum(1 for _ in f)
        print(f"共 {total_lines} 个句子。")
    except:
        total_lines = None
        print("无法统计行数，将不显示总进度。")
    with open(infile, "r", encoding="utf-8") as f:
        pbar = tqdm(f, desc="处理句子", unit=" lines", total=total_lines)
        for line in pbar:
            it = json.loads(line)
            qid = it.get("question_id") or f"q{total_sentences}"
            pid = str(uuid.uuid5(uuid.NAMESPACE_URL, qid + "::" + (it.get("paragraph_title", "") or "")[:80]))
            sid = f"s{total_sentences}"
            text = it.get("sentence_text", "")
            if qid not in questions_seen:
                questions_seen.add(qid)
                with driver.session() as sess:
                    sess.run("MERGE (q:Question {qid:$qid}) SET q.text=$text",
                             qid=qid, text=it.get("question", ""))
            if pid not in paragraphs_seen:
                paragraphs_seen.add(pid)
                with driver.session() as sess:
                    sess.run("MERGE (p:Paragraph {pid:$pid}) SET p.title=$title, p.text=$text",
                             pid=pid, title=it.get("paragraph_title", ""), text=it.get("paragraph_text", ""))
            batch_texts.append(text)
            batch_sentences.append({
                "sid": sid,
                "text": text,
                "qid": qid,
                "pid": pid
            })
            total_sentences += 1
            if len(batch_texts) >= batch_size:
                embeddings = model.encode(batch_texts, convert_to_numpy=True)
                flush_batch(driver, batch_sentences, embeddings)
                batch_texts = []
                batch_sentences = []
                pbar.set_postfix({"已写入": total_sentences})
        if batch_texts:
            embeddings = model.encode(batch_texts, convert_to_numpy=True)
            flush_batch(driver, batch_sentences, embeddings)
    print(f"导入完成！共处理 {total_sentences} 个句子。")
    driver.close()

if __name__ == "__main__":
    main()