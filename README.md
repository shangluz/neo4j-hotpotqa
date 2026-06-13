# neo4j-hotpotqa

大数据存储大作业 — 将 HotpotQA（或类似 QA 数据）与 Neo4j 图数据库结合的示例/工程。

## 项目简介

本项目旨在展示如何将问答数据（例如 HotpotQA）存入 Neo4j 图数据库，并通过 Python 脚本进行导入、查询与简单分析。仓库名为 `neo4j-hotpotqa`，代码以 Python 为主，前端或展示页面使用了部分 HTML 文件。

语言组成（按仓库统计）:
- Python: 74.8%
- HTML: 25.2%

## 目录结构（示例）

- scripts/ - 导入与处理数据的 Python 脚本
- data/ - 数据文件（可能包含 HotpotQA 数据子集）
- web/ - 简单的 HTML 展示或交互页面

> 实际目录以仓库当前内容为准，上述为常见约定。

## 环境与依赖

建议使用虚拟环境：

```bash
python -m venv venv
source venv/bin/activate   # macOS / Linux
venv\Scripts\activate     # Windows
pip install -r requirements.txt
```

如果仓库中没有 requirements.txt，常见依赖包括：
- neo4j
- requests
- tqdm
- pandas

另外需要安装并运行 Neo4j 数据库（社区版或桌面版），并在脚本中配置连接 URI、用户名和密码。

## 使用方法（示例）

1. 准备 Neo4j 并记下连接地址（例如 bolt://localhost:7687）及用户名/密码。
2. 将需要导入的数据放入 `data/` 目录（如 hotpotqa.json）。
3. 编辑导入脚本中的配置（URI、认证、数据路径）。
4. 运行导入脚本：

```bash
python scripts/import_to_neo4j.py
```

5. 使用 Neo4j Browser 或脚本执行查询与分析。

## 贡献

欢迎提交 issue 或 pull request。如果你有改进导入流程、增加数据清洗或可视化的想法，欢迎参与。

## 许可证

如需添加许可证，请在仓库中创建 LICENSE 文件（例如 MIT）。

---

作者: shangluz
仓库: https://github.com/shangluz/neo4j-hotpotqa
