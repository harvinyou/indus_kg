# cn_patents_csv_triplet_extraction

介绍：使用lda、tf-idf等无监督算法加停用词表抽取专利主题信息构建企业间关系三元组

## 主要代码：

### script/knowledgeGraphGenerator_tfidf.py:自定义节点和边的筛选条件

#### 配置文件路径可调整参数意义：script/config_knowledgeGraphGenerator_tfidf.json:

1. `input_files`: 一个包含要处理的输入文件路径的列表。这些文件包含了需要用于构建知识图谱的数据。
2. `max_features`: 用于 TF-IDF 计算的最大特征数量，即最终选择的关键词的数量上限。
3. `node_ratio`: 用于配置节点数量的比例，可能会影响节点的筛选。
4. `edge_ratio`: 用于配置边的数量的比例，可能会影响边的筛选。
5. `max_nodes`: 最大节点数量的限制。
6. `min_nodes`: 最小节点数量的限制。
7. `max_edges`: 最大边的数量限制。
8. `min_edges`: 最小边的数量限制。
9. `top_n_keywords`: 从文本中提取的关键词数量限制。
10. `layout`: 知识图谱绘制时的布局方式，可以是 `'spring'`, `'random'`, `'circular'` 或 `'kamada_kawai'` 中的一个。

##### 默认参数：

```
{
    "input_files": ["data/汽车制造业_test.xlsx", "data/专利demo_test.xlsx"],
    "max_features": 1000,
    "node_ratio": 0.1,
    "edge_ratio": 0.1,
    "max_nodes": 5,
    "min_nodes": 2,
    "max_edges": 10,
    "min_edges": 5,
    "top_n_keywords": 10,
    "layout": "spring"
}
```

任何问题联系：bityhw@163.com

项目最新更新：https://github.com/yuriamao/cn_patents_csv_triplet_extraction
