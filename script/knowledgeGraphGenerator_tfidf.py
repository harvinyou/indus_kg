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
import json

class KnowledgeGraphGenerator:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        self.output_folder = None

    def create_output_folder(self):
        current_time = datetime.now().strftime("%Y%m%d%H%M%S")
        self.output_folder = os.path.join('output', current_time)
        os.makedirs(self.output_folder, exist_ok=True)
        print('output_path:', self.output_folder)

    def read_excel(self, input_file):
        # 下载中文停用词
        nltk.download('stopwords',download_dir='data')
        # 下载分词器
        nltk.download('punkt',download_dir='data')
        pd_excel = pd.read_excel(input_file)
        return pd_excel


    def generate_knowledge_graph(self, data, node_filter=None, edge_filter=None): 
        G = nx.DiGraph()

        tfidf_vectorizer = TfidfVectorizer(max_features=self.max_features, stop_words=stopwords.words('chinese'))
        tfidf_vectorizer.fit(data['摘要'])  

        relation_counts = {}  # 添加一个字典用于记录关系数量

        for idx, row in tqdm(data.iterrows(), total=len(data)):
            enterprise_id = row['企业id']
            patent_title = row['专利名称']
            applicant = row['申请人']
            inventors = row['发明人']
            summary = row['摘要']

            if node_filter is None or node_filter(patent_title):
                G.add_node(patent_title)
            if node_filter is None or node_filter(applicant):
                G.add_node(applicant)
            if node_filter is None or node_filter(enterprise_id):
                G.add_node(enterprise_id)

            inventors_list = inventors.split(';')

            for inventor in tqdm(inventors_list, leave=False):
                if node_filter is None or node_filter(inventor):
                    G.add_node(inventor)
                    G.add_edge(inventor, patent_title, relation='invented')

            if node_filter is None or node_filter(enterprise_id) and node_filter is None or node_filter(applicant):
                G.add_edge(enterprise_id, applicant, relation='chinese_name')
            if node_filter is None or node_filter(applicant) and node_filter is None or node_filter(patent_title):
                G.add_edge(applicant, patent_title, relation='applied_for')

            keywords_tfidf = self.extract_keywords_with_tfidf(summary, tfidf_vectorizer)

            for keyword in tqdm(keywords_tfidf, leave=False):
                if node_filter is None or node_filter(keyword):
                    G.add_node(keyword)
                    G.add_edge(patent_title, keyword, relation='keywords')

                # 在这里统计关系数量
                relation_counts['invented'] = relation_counts.get('invented', 0) + 1
                relation_counts['chinese_name'] = relation_counts.get('chinese_name', 0) + 1
                relation_counts['applied_for'] = relation_counts.get('applied_for', 0) + 1
                relation_counts['keywords'] = relation_counts.get('keywords', 0) + 1

        print('关系数量统计：', relation_counts)  # 输出关系数量统计

        if edge_filter is not None:
            G = G.edge_subgraph([(node1, node2) for node1, node2, data in G.edges(data=True) if edge_filter(data['relation'])])

        return G

    def extract_keywords_with_tfidf(self, text, tfidf_vectorizer):
        X = tfidf_vectorizer.transform([text])
        top_keywords = X.toarray().argsort()[0][::-1][:self.top_n_keywords]  # 使用 self.top_n_keywords
        feature_names = tfidf_vectorizer.get_feature_names_out()
        return [feature_names[i] for i in top_keywords]

    def filter_keyword_edges(self, G):
        keyword_edges = [(node1, node2) for node1, node2, data in G.edges(data=True) if data['relation'] == 'keywords']
        G_keyword = G.edge_subgraph(keyword_edges)
        return G_keyword
    
    def select_connected_nodes_and_edges(self, G_keyword):
        connected_components = list(nx.connected_components(G_keyword.to_undirected()))

        if not connected_components:
            print('Error: 没有连通的节点和边，请调整参数或者检查输入数据')
            return [], []

        # 找到最大的连通子图
        max_connected_component = max(connected_components, key=len)

        # 将连通子图转换为子图对象
        max_connected_subgraph = G_keyword.subgraph(max_connected_component)

        # 获取最大连通子图的节点和边
        limited_nodes = list(max_connected_subgraph.nodes())
        limited_edges = list(max_connected_subgraph.edges())

        # 限制节点和边的数量，尽量靠近 max_nodes 和 max_edges
        limited_nodes = random.sample(limited_nodes, min(self.max_nodes, max(self.min_nodes, len(limited_nodes))))
        limited_edges = random.sample(limited_edges, min(self.max_edges, max(self.min_edges, len(limited_edges))))

        return limited_nodes, limited_edges


        return limited_nodes, limited_edges

    def generate_limited_graph(self, G_keyword, limited_edges):
        G_limited = G_keyword.edge_subgraph(limited_edges)
        return G_limited

    def draw_and_save_graph(self, G_limited):
        if self.layout == 'spring':
            pos = nx.spring_layout(G_limited)
        elif self.layout == 'random':
            pos = nx.random_layout(G_limited)
        elif self.layout == 'circular':
            pos = nx.circular_layout(G_limited)
        elif self.layout == 'kamada_kawai':
            pos = nx.kamada_kawai_layout(G_limited)
        else:
            pos = nx.spring_layout(G_limited)  # 默认使用 spring layout

        plt.figure(figsize=(10, 8))
        plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
        plt.rcParams['axes.unicode_minus'] = False
        nx.draw(G_limited, pos, width=1, with_labels=True, node_size=30, font_size=12, font_color='black', font_weight='bold', node_shape='o')
        plt.savefig(os.path.join(self.output_folder, 'knowledge_graph.png'))
        print('找到连通的节点和边，保存知识图谱')

    def save_triplets(self, G):
        triplets = []
        for edge in G.edges(data=True):
            entity_1, entity_2, relation_data = edge
            relation = relation_data['relation']
            triplets.append({'entity_1': entity_1, 'relation': relation, 'entity_2': entity_2})

        triplets_df = pd.DataFrame(triplets)
        triplets_df.to_csv(os.path.join(self.output_folder, 'triplets.csv'), index=False, encoding='utf-8-sig')
        print('三元组已保存')

    def display_graph(self, G_limited):
        print('显示知识图谱,请手动关闭图片或者ctrl+c强制退出结束运行')
        plt.show()
    
    def run(self, node_filter=None, edge_filter=None):
        self.create_output_folder()
        for input_file in self.input_files:
            data = self.read_excel(input_file)
            G = self.generate_knowledge_graph(data, node_filter=node_filter, edge_filter=edge_filter)  
            G_keyword = self.filter_keyword_edges(G)
            limited_nodes, limited_edges = self.select_connected_nodes_and_edges(G_keyword)
            G_limited = self.generate_limited_graph(G_keyword, limited_edges)
            self.draw_and_save_graph(G_limited)
            self.save_triplets(G)
            self.display_graph(G_limited)
        print('程序正常结束')

if __name__ == "__main__":
    with open('script/config_knowledgeGraphGenerator_tfidf.json', 'r') as f:
        config = json.load(f)
    # 自定义节点和边的筛选条件
    def custom_node_filter(node):
        return len(node) < 100

    def custom_edge_filter(relation):
        return relation == 'keywords'
    
    graph_generator = KnowledgeGraphGenerator(**config)
    graph_generator.run(node_filter=custom_node_filter, edge_filter=custom_edge_filter)
