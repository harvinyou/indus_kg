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

class KnowledgeGraphGenerator:
    def __init__(self, input_path, max_features=1000, node_ratio=0.1, edge_ratio=0.1, max_nodes=100, max_edges=100):
        self.input_path = input_path
        self.max_features = max_features
        self.node_ratio = node_ratio
        self.edge_ratio = edge_ratio
        self.max_nodes = max_nodes
        self.max_edges = max_edges
        self.output_folder = None

    def create_output_folder(self):
        current_time = datetime.now().strftime("%Y%m%d%H%M%S")
        self.output_folder = os.path.join('output', current_time)
        os.makedirs(self.output_folder, exist_ok=True)
        print('output_path:', self.output_folder)

    def read_excel(self):
        pd_excel = pd.read_excel(self.input_path)
        return pd_excel

    def generate_knowledge_graph(self, data): 
        G = nx.DiGraph()

        tfidf_vectorizer = TfidfVectorizer(max_features=self.max_features, stop_words=stopwords.words('chinese'))
        tfidf_vectorizer.fit(data['摘要'])  

        for idx, row in tqdm(data.iterrows(), total=len(data)):
            enterprise_id = row['企业id']
            patent_title = row['专利名称']
            applicant = row['申请人']
            inventors = row['发明人']
            summary = row['摘要']

            G.add_node(patent_title)
            G.add_node(applicant)
            G.add_node(enterprise_id)

            inventors_list = inventors.split(';')

            for inventor in tqdm(inventors_list, leave=False):
                G.add_node(inventor)
                G.add_edge(inventor, patent_title, relation='invented')

            G.add_edge(enterprise_id, applicant, relation='chinese_name')
            G.add_edge(applicant, patent_title, relation='applied_for')

            keywords_tfidf = self.extract_keywords_with_tfidf(summary, tfidf_vectorizer)

            for keyword in tqdm(keywords_tfidf, leave=False):
                G.add_node(keyword)
                G.add_edge(patent_title, keyword, relation='keywords')
            
        return G

    def extract_keywords_with_tfidf(self, text, tfidf_vectorizer, top_n=10):
        X = tfidf_vectorizer.transform([text])
        top_keywords = X.toarray().argsort()[0][::-1][:top_n]
        feature_names = tfidf_vectorizer.get_feature_names_out()
        return [feature_names[i] for i in top_keywords]

    def filter_keyword_edges(self, G):
        keyword_edges = [(node1, node2) for node1, node2, data in G.edges(data=True) if data['relation'] == 'keywords']
        G_keyword = G.edge_subgraph(keyword_edges)
        return G_keyword
    
    def select_connected_nodes_and_edges(self, G_keyword):
        connected_components = list(nx.connected_components(G_keyword.to_undirected()))

        # 找到最大的连通子图
        max_connected_component = max(connected_components, key=len)

        # 将连通子图转换为子图对象
        max_connected_subgraph = G_keyword.subgraph(max_connected_component)

        # 获取最大连通子图的节点和边
        limited_nodes = list(max_connected_subgraph.nodes())
        limited_edges = list(max_connected_subgraph.edges())

        # 限制节点和边的数量
        limited_nodes = random.sample(limited_nodes, min(len(limited_nodes), self.max_nodes))
        limited_edges = random.sample(limited_edges, min(len(limited_edges), self.max_edges))

        return limited_nodes, limited_edges







    def generate_limited_graph(self, G_keyword, limited_edges):
        G_limited = G_keyword.edge_subgraph(limited_edges)
        return G_limited

    def draw_and_save_graph(self, G_limited):
        pos = nx.spring_layout(G_limited)
        plt.figure(figsize=(10, 8))
        plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
        plt.rcParams['axes.unicode_minus'] = False
        nx.draw(G_limited, pos, width=1, with_labels=True, node_size=30, font_size=12, font_color='black', font_weight='bold', node_shape='o')
        plt.savefig(os.path.join(self.output_folder, 'knowledge_graph.png'))
        print('保存知识图谱')

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
        print('显示知识图谱')
        plt.show()
    
    def run(self):
        self.create_output_folder()
        data = self.read_excel()
        G = self.generate_knowledge_graph(data)  
        G_keyword = self.filter_keyword_edges(G)
        limited_nodes, limited_edges = self.select_connected_nodes_and_edges(G_keyword)
        G_limited = self.generate_limited_graph(G_keyword, limited_edges)
        self.draw_and_save_graph(G_limited)
        self.save_triplets(G)
        self.display_graph(G_limited)
        print('程序正常结束')

if __name__ == "__main__":
    input_path = 'data/汽车制造业_test.xlsx'
    graph_generator = KnowledgeGraphGenerator(input_path)
    graph_generator.run()
