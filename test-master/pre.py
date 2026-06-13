import os
import json
import re
from datasets import load_from_disk, concatenate_datasets
from tqdm import tqdm

def simple_sent_tokenize(text):
    if not text or not isinstance(text, str):
        return []
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s for s in sentences if s]

def find_dataset_dirs(root):
    dirs = []
    if os.path.isdir(root):
        if os.path.exists(os.path.join(root, "dataset_info.json")):
            dirs.append(root)
        else:
            for name in sorted(os.listdir(root)):
                p = os.path.join(root, name)
                if os.path.isdir(p) and os.path.exists(os.path.join(p, "dataset_info.json")):
                    dirs.append(p)
    return dirs

def load_local_hf_dataset(path):
    dataset_dirs = find_dataset_dirs(path)
    if not dataset_dirs:
        raise RuntimeError(f"No HuggingFace dataset directories (dataset_info.json) found under {path}")
    datasets = []
    for d in dataset_dirs:
        print("Loading dataset from:", d)
        ds = load_from_disk(d)
        if isinstance(ds, dict):
            for k, v in ds.items():
                if hasattr(v, "__len__"):
                    datasets.append(v)
        else:
            datasets.append(ds)
    if len(datasets) == 1:
        return datasets[0]
    else:
        print(f"Concatenating {len(datasets)} datasets")
        return concatenate_datasets(datasets)

def extract_contexts(sample):
    if "context" in sample and sample["context"] is not None:
        ctx = sample["context"]
        if isinstance(ctx, dict):
            titles = ctx.get("title", [])
            sentences_list = ctx.get("sentences", [])
            result = []
            for title, sents in zip(titles, sentences_list):
                if isinstance(sents, list):
                    para = " ".join(sents)
                else:
                    para = str(sents)
                result.append([title, para])
            if result:
                return result
        elif isinstance(ctx, list) and len(ctx) > 0:
            if isinstance(ctx[0], (list, tuple)) and len(ctx[0]) >= 2:
                items = []
                for item in ctx:
                    title = item[0] if item[0] is not None else ""
                    para = item[1]
                    if isinstance(para, list):
                        para = " ".join(para)
                    items.append([title, para])
                return items
    for key in ("contexts", "wiki_context", "context_with_answers", "paragraphs"):
        if key in sample and sample[key] is not None:
            val = sample[key]
            if isinstance(val, list) and len(val) > 0 and isinstance(val[0], (list, tuple)) and len(val[0]) >= 2:
                return val
            if isinstance(val, dict):
                items = []
                for t, p in val.items():
                    items.append([t, p])
                return items
            if isinstance(val, list) and all(isinstance(x, str) for x in val):
                return [["", "\n".join(val)]]
    question = sample.get("question", "") or ""
    return [["", question]]

def main():
    input_path = "test"
    output_path = "processed_sentences.jsonl"
    max_samples = None
    print("开始加载数据集", flush=True)
    ds = load_local_hf_dataset(input_path)
    print(f"Loaded dataset, number of samples: {len(ds)}", flush=True)
    outf = open(output_path, "w", encoding="utf-8")
    cnt = 0
    for sample in tqdm(ds, desc="Processing samples"):
        if max_samples is not None and cnt >= max_samples:
            break
        qid = sample.get("id") or sample.get("_id") or f"q{cnt}"
        question = sample.get("question", "") or sample.get("query", "")
        contexts = extract_contexts(sample)
        for title, para in contexts:
            if not isinstance(para, str):
                if isinstance(para, list):
                    para = "\n".join(str(x) for x in para)
                else:
                    para = str(para)
            sentences = simple_sent_tokenize(para)
            for sent in sentences:
                out_obj = {
                    "question_id": qid,
                    "question": question,
                    "paragraph_title": title if title is not None else "",
                    "paragraph_text": para,
                    "sentence_text": sent
                }
                outf.write(json.dumps(out_obj, ensure_ascii=False) + "\n")
        cnt += 1
    outf.close()
    print(f"Wrote {output_path} from {cnt} samples", flush=True)

if __name__ == "__main__":
    main()