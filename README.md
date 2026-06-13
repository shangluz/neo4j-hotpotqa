# Neo4j 语义检索与可视化系统

基于 Neo4j 图数据库，支持多跳过程的查询、检索、简单聚类和可视化。该系统使用 Sentence Transformers 进行文本嵌入，结合 Neo4j 图数据库实现知识图谱的存储与检索。

## 项目简介

本项目是一个完整的文本语义检索与可视化解决方案，主要功能包括：

- **文本预处理**：将 HuggingFace 数据集转换为结构化句子数据
- **图数据库导入**：将句子、段落、问题导入 Neo4j 并生成向量嵌入
- **语义检索**：基于向量相似度的语义搜索
- **多跳查询**：沿图关系进行多跳遍历，发现关联知识
- **聚类分析**：对段落内句子进行 K-Means 聚类
- **交互式可视化**：使用 Cytoscape.js 进行图谱可视化

## 项目结构

```
.
├── pre.py                    
├── import_neo4j.py           
├── assignment.py 
├── templates/
│   └── index.html 
├── local_model/
│   ├── config.json
│   ├── model.safetensors
│   ├── tokenizer.json
│   └── ...
├── test/ 
│   ├── data-00000-of-00001.arrow
│   └── dataset_info.json
└── README.md
```

## 环境要求

- Python 3.8+
- Neo4j 数据库 4.x+
- 所需 Python 包：
  - flask
  - neo4j
  - sentence-transformers
  - numpy
  - scikit-learn
  - datasets
  - tqdm

## 安装与配置

### 1. 安装依赖

```bash
pip install flask neo4j sentence-transformers numpy scikit-learn datasets tqdm
```

### 2. 配置 Neo4j 数据库

确保 Neo4j 数据库已启动，默认配置：
- URI: `bolt://localhost:7687`
- 用户名: `neo4j`
- 密码: `zhu21192119`

可通过环境变量修改：
```bash
export NEO4J_URI="bolt://localhost:7687"
export NEO4J_USER="neo4j"
export NEO4J_PWD="your_password"
```

### 3. 创建全文索引

在 Neo4j 中创建全文索引以支持搜索：

```cypher
CALL db.index.fulltext.createNodeIndex('sentenceIndex', ['Sentence'], ['text'])
```

## 使用方法

### 第一步：数据预处理

将 HuggingFace 格式的数据集转换为结构化 JSONL 文件：

```bash
python pre.py
```

输入：`test/` 目录下的数据集
输出：`processed_sentences.jsonl`

数据格式：
```json
{
  "question_id": "q1",
  "question": "原始问题",
  "paragraph_title": "段落标题",
  "paragraph_text": "段落完整文本",
  "sentence_text": "单个句子文本"
}
```

### 第二步：导入 Neo4j

将预处理的数据导入 Neo4j 图数据库：

```bash
python import_neo4j.py
```

该脚本会：
1. 加载 `all-MiniLM-L6-v2` 嵌入模型
2. 创建 `Question`、`Paragraph`、`Sentence` 节点
3. 生成句子向量嵌入（384 维）
4. 建立节点间的关系：
   - `Question -[:HAS_EVIDENCE]-> Sentence`
   - `Paragraph -[:HAS_SENTENCE]-> Sentence`

### 第三步：启动 Web 服务

```bash
python assignment.py
```

默认访问地址：`http://localhost:5000`

## API 接口

### 1. 语义检索 `/api/retrieve`

查询语义相似的句子：

```
GET /api/retrieve?q=<查询文本>&top_k=5
```

参数：
- `q`: 查询文本（必需）
- `top_k`: 返回结果数量（默认 5）

响应：
```json
{
  "query": "查询文本",
  "results": [
    {
      "id": 123,
      "text": "匹配的句子",
      "sim": 0.85
    }
  ]
}
```

### 2. 多跳查询 `/api/multihop`

执行多跳图遍历，返回节点和边：

```
GET /api/multihop?q=<查询文本>&top_k=3&hops=2
```

参数：
- `q`: 查询文本（必需）
- `top_k`: 初始种子节点数（默认 3）
- `hops`: 跳数（默认 2）

响应：
```json
{
  "nodes": [
    {"id": "s1", "type": "Sentence", "text": "...", "cluster": 0},
    {"id": "p1", "type": "Paragraph", "title": "..."}
  ],
  "edges": [
    {"source": "p1", "target": "s1", "type": "HAS_SENTENCE"}
  ],
  "seed_ids": ["s1", "s2"],
  "seed_count": 2
}
```

### 3. 获取句子详情 `/api/sentence`

```
GET /api/sentence?id=s123
```

### 4. 获取段落详情 `/api/paragraph`

```
GET /api/paragraph?id=p456
```

### 5. 聚类分析 `/api/cluster`

对段落内的句子进行 K-Means 聚类：

```
POST /api/cluster
Content-Type: application/json

{
  "paragraph_id": "p1",
  "n_clusters": 3
}
```

## Web 界面功能

访问 `http://localhost:5000` 可使用图形化界面：

1. **语义搜索**：输入查询文本，点击"搜索并可视化"
2. **多跳遍历**：选择跳数（1-3），发现关联知识
3. **聚类分析**：选择 Paragraph 节点，设置聚类数，进行 K-Means 聚类
4. **交互式图谱**：
   - 点击节点查看详情
   - 鼠标滚轮缩放
   - 拖拽移动节点
   - 点击"重置视图"恢复初始视角

## 数据模型

```
(Question)-[:HAS_EVIDENCE]->(Sentence)
(Paragraph)-[:HAS_SENTENCE]->(Sentence)

Question 属性:
  - qid: 问题唯一标识
  - text: 问题文本

Paragraph 属性:
  - pid: 段落唯一标识
  - title: 段落标题
  - text: 段落文本

Sentence 属性:
  - sid: 句子唯一标识
  - text: 句子文本
  - vec: 384维向量嵌入
  - cluster: 聚类标签（可选）
```

## 模型说明

使用 `all-MiniLM-L6-v2` 句子嵌入模型：
- 输出维度：384
- 最大输入长度：256 tokens
- 支持离线模式（已下载至 `local_model/`）

模型特点：
- 轻量级、高效率
- 适合语义相似度计算
- 支持句子和短段落嵌入

## 环境变量配置

| 变量名 | 默认值 | 说明 |
|--------|--------|------|
| `NEO4J_URI` | `bolt://localhost:7687` | Neo4j 连接地址 |
| `NEO4J_USER` | `neo4j` | Neo4j 用户名 |
| `NEO4J_PWD` | `zhu21192119` | Neo4j 密码 |
| `LOCAL_MODEL_PATH` | `./local_model` | 本地模型路径 |
| `FLASK_HOST` | `0.0.0.0` | Flask 监听地址 |
| `FLASK_PORT` | `5000` | Flask 监听端口 |
| `HF_HUB_OFFLINE` | `1` | 离线模式 |
| `TRANSFORMERS_OFFLINE` | `1` | 离线模式 |
