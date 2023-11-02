from TextRank import textRank
import pandas as pd
import os
import jieba
# https://github.com/abner-wong/textrank
text = """欧亚经济委员会执委会一体化与宏观经济委员格拉济耶夫日前接受新华社记者采访时高度评价中国抗击新冠疫情工作，\
并表示期待欧亚经济联盟与中国加强抗疫合作，共同推动地区发展。格拉济耶夫说，中国依靠治理体系与全国人民协同努力，\
在抗疫工作上取得极大成效。中国采取的措施符合全球利益。格拉济耶夫认为，中国经济将会快速恢复，欧亚经济联盟许多企业与中国市场联系紧密，\
应与中国加强合作，采取协调措施降低此次疫情带来的消极影响。格拉济耶夫建议，面对疫情，欧亚经济联盟与中国扩大信息技术应用，\
推进商品清关程序自动化，更广泛地利用相关机制，为对外经济活动参与者建立绿色通道。谈及双方在医学卫生领域的合作时，\
格拉济耶夫说：“我们应从当前考验中汲取经验，在生物安全领域制定共同规划并联合开展生物工程研究。”格拉济耶夫还表示，\
俄罗斯与其他欧亚经济联盟国家金融市场更易受国际投机行为影响。欧亚经济联盟应借鉴中国的人民币国际化经验，加强与中国银行体系和金融市场对接。\
欧亚经济联盟成立于2015年，成员国包括俄罗斯、哈萨克斯坦、白俄罗斯、吉尔吉斯斯坦和亚美尼亚。欧亚经济委员会执委会是欧亚经济联盟最高权力机构。"""

T = textRank.TextRank(text,pr_config={'alpha': 0.85, 'max_iter': 100})

current_path = os.getcwd()  
base_path = os.path.dirname(current_path) 
csv_path = os.path.join(base_path, "indus_kg/data/专利demo_test.xlsx")
df = pd.read_excel(csv_path, engine='openpyxl')
df.head()

text_summary = ' '.join(df['摘要'])

T = textRank.TextRank(text_summary,pr_config={'alpha': 0.85, 'max_iter': 100})
keywords=T.get_n_keywords(100)
# 提取前n个句子作为摘要
summarys=T.get_n_sentences(100)

with open('textrank_keywords.txt', 'w', encoding='utf-8') as f:    
    for keyword in keywords:    
        f.write('   '.join(map(str, keyword)) + '\n')  # 将所有元素转换为字符串并连接
with open('textrank_summarys.txt', 'w', encoding='utf-8') as f:    
    for summary in summarys:    
        f.write('   '.join(map(str, summary)) + '\n')  # 将所有元素转换为字符串并连接