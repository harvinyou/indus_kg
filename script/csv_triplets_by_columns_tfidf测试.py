import os
import pandas as pd
from tqdm import tqdm
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from datetime import datetime
import random

# 专利文档路径
# input_path = 'data/专利demo_test.xlsx'
input_path = 'data/汽车制造业_test.xlsx'
# input_path = 'data/汽车制造业.xlsx'

# 下载中文停用词
nltk.download('stopwords',download_dir='data')
# 下载分词器
nltk.download('punkt',download_dir='data')

# 读取专利表
def read_excel(input_path):
    pd_excel = pd.read_excel(input_path)
    return pd_excel

#####构建图，复杂度很大，会卡顿

def generate_knowledge_graph(data): 
    G = nx.DiGraph()

    tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words=stopwords.words('chinese'))
    tfidf_vectorizer.fit(data['摘要'])  # 拟合 TF-IDF 向量化器

    for idx, row in tqdm(data.iterrows(), total=len(data)):
        enterprise_id = row['企业id']
        patent_title = row['专利名称']
        applicant = row['申请人']
        inventors = row['发明人']
        summary = row['摘要']

        G.add_node(patent_title)
        G.add_node(applicant)
        G.add_node(enterprise_id)

        # 解析发明人
        inventors_list = inventors.split(';')

        for inventor in tqdm(inventors_list, leave=False):  # 添加进度条
            G.add_node(inventor)
            G.add_edge(inventor, patent_title, relation='invented')

        # 添加边，将专利名称作为关系
        G.add_edge(enterprise_id, applicant, relation='chinese_name')
        G.add_edge(applicant, patent_title, relation='applied_for')

        # 使用TF-IDF提取摘要关键词
        keywords_tfidf = extract_keywords_with_tfidf(summary, tfidf_vectorizer)

        for keyword in tqdm(keywords_tfidf, leave=False):  # 添加进度条
            G.add_node(keyword)
            G.add_edge(patent_title, keyword, relation='keywords')
            
    return G


# 使用TF-IDF提取关键词
def extract_keywords_with_tfidf(text, tfidf_vectorizer, top_n=10):
    X = tfidf_vectorizer.transform([text])
    top_keywords = X.toarray().argsort()[0][::-1][:top_n]  # 取前 top_n 个关键词的索引
    feature_names = tfidf_vectorizer.get_feature_names_out()
    return [feature_names[i] for i in top_keywords]

if __name__ == "__main__":
    # 创建以运行时间命名的子文件夹
    current_time = datetime.now().strftime("%Y%m%d%H%M%S")
    output_folder = os.path.join('output', current_time)
    os.makedirs(output_folder, exist_ok=True)
    print('output_path:',output_folder)

    # 以下的代码会把图和文件保存到子文件夹中
    data = read_excel(input_path)

    print('合并列为三元组，tf-idf抽取关键词')
    G = generate_knowledge_graph(data)  

    print('发明人，专利，摘要关键词图节点与关系边建设完成')

    # 仅保留“专利 - keywords - 关键词”的关系
    keyword_edges = [(node1, node2) for node1, node2, data in G.edges(data=True) if data['relation'] == 'keywords']

    G_keyword = G.edge_subgraph(keyword_edges)

    pos = nx.circular_layout(G_keyword)

    plt.figure(figsize=(10, 8))
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False

    # 设置比例参数
    node_ratio = 0.1  # 选择10%的节点
    edge_ratio = 0.1  # 选择10%的关系

    # 计算选择的节点和关系数量
    num_nodes_to_select = int(len(G_keyword.nodes()) * node_ratio)
    num_edges_to_select = int(len(G_keyword.edges()) * edge_ratio)

    # 设置节点和边的上限
    max_nodes = 100
    max_edges = 100

    # 限制节点和边的数量不超过上限
    limited_nodes = random.sample(list(G_keyword.nodes()), min(num_nodes_to_select, max_nodes))
    limited_edges = [(edge[0], edge[1]) for edge in G_keyword.edges() if edge[0] in limited_nodes and edge[1] in limited_nodes][:min(num_edges_to_select, max_edges)]

    G_limited = G_keyword.edge_subgraph(limited_edges)
        # 将节点放在中心位置
    pos = nx.spring_layout(G_limited)


    nx.draw(G_limited, pos, width=1,with_labels=True, node_size=30, font_size=12, font_color='black', font_weight='bold', node_shape='o')

    plt.savefig(os.path.join(output_folder, 'knowledge_graph.png'))  # 保存到子文件夹

    print('保存知识图谱')

    triplets = []
    # 获取边的信息
    for edge in G.edges(data=True):
        entity_1, entity_2, relation_data = edge
        relation = relation_data['relation']
        triplets.append({'entity_1': entity_1, 'relation': relation, 'entity_2': entity_2})

    triplets_df = pd.DataFrame(triplets)
    triplets_df.to_csv(os.path.join(output_folder, 'triplets.csv'), index=False, encoding='utf-8-sig')


    print('三元组已保存')

    print('显示知识图谱')
    plt.show()
    print('程序正常结束')
