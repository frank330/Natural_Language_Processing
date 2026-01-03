import jieba
import jieba.analyse
import re
from collections import Counter
from typing import List, Tuple, Dict



class LuhnSummarizer:
    """    """
    """    初始化Luhn摘要器
        
        这个函数就像给摘要器"设置参数"，告诉它我们希望生成多长的摘要，以及如何判断哪些词是重要的。
        想象一下，你正在训练一个助手帮你总结文章，你需要告诉它：
         摘要应该包含几句话（summary_length）
         什么样的词才算"重要词"（word_freq_threshold）
         哪些太短的词应该忽略（min_word_length）
         哪些常见的无意义词要过滤掉（stopwords_file）       
  
                          """
    def __init__(self,
                 summary_length: int = 3,
                 word_freq_threshold: float = 0.1,
                 min_word_length: int = 2,
                 stopwords_file: str = '../data/stopwords/stopwords.txt'):

        self.summary_length = summary_length
        self.word_freq_threshold = word_freq_threshold
        self.min_word_length = min_word_length
        # 从文件加载停用词
        self.stopwords = self.load_stopwords(stopwords_file)




    """
            从文件加载停用词列表
            停用词（stopwords）是文本处理中的一个重要概念。在一篇文章中，有些词虽然出现频率很高，
            但它们对理解文章主题没有帮助，比如中文中的"的"、"了"、"是"、"在"等。这些词就像
            文章中的"噪音"，需要被过滤掉，这样才能让真正有意义的词凸显出来。

            这个函数的作用就是从文件中读取这些停用词，存储在一个集合（set）中，方便后续快速查找。
            使用集合而不是列表的好处是，查找某个词是否是停用词时速度会快很多。

            """
    def load_stopwords(self, file_path: str) -> set:

        stopwords = set()
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                word = line.strip()
                if word:  # 忽略空行
                    stopwords.add(word)
        return stopwords









    """
            文本预处理：分词并过滤停用词和短词

            在自然语言处理中，原始文本不能直接使用。

            对于中文文本，预处理包括：
            1. 分词：中文不像英文有空格分隔，需要把连续的字符切分成有意义的词
               比如"我喜欢自然语言处理"需要切分成"我"、"喜欢"、"自然语言"、"处理"
            2. 过滤停用词：去掉"的"、"了"、"是"等无意义的词
            3. 过滤短词：通常单字词意义不明确，只保留2个字以上的词
            4. 字符过滤：只保留中文、英文、数字，去掉标点符号等

            """

    def preprocess_text(self, text: str) -> List[str]:

        words = jieba.cut(text)
        filtered_words = []
        for word in words:
            word = word.strip()
            # 过滤条件：长度>=2，不是停用词，只包含中文、英文、数字
            if (len(word) >= self.min_word_length and 
                word not in self.stopwords and
                re.match(r'^[\u4e00-\u9fa5a-zA-Z0-9]+$', word)):
                filtered_words.append(word)
        
        return filtered_words





    """
            计算所有句子中词的频率

            词频统计是Luhn算法的第一步，也是最关键的一步。这个函数就像在做"词频普查"：
            遍历文章中的所有句子，统计每个词出现了多少次。

            为什么要统计词频？因为Luhn算法的基本假设是：在一篇文章中，出现频率高的词往往
            代表了文章的主题。比如一篇关于"人工智能"的文章，"人工智能"、"机器学习"、"算法"
            这些词会反复出现，而"天气"、"美食"这些无关词出现次数很少。

            这个函数使用Python的Counter类来统计词频，Counter会自动帮我们计算每个词出现的次数，
            非常方便。返回的结果就像一本"词频字典"，可以快速查询任意词的出现次数。

            """
    def calculate_word_frequencies(self, sentences: List[str]) -> Counter:

        all_words = []
        for sentence in sentences:
            words = self.preprocess_text(sentence)
            all_words.extend(words)
        
        return Counter(all_words)









    """
            根据词频阈值筛选重要词汇

            有了词频统计后，我们需要从中筛选出"重要词汇"。但问题是：出现多少次才算"重要"？
            如果设置得太低，几乎所有词都算重要，那就失去了筛选的意义；如果设置得太高，
            可能只有极少数词被选中，可能会遗漏一些重要但出现频率稍低的词。
            Luhn的策略：使用相对阈值而不是绝对阈值。具体来说，它找出
            出现频率最高的词，然后以这个最高频率的一定比例（比如10%）作为阈值。这样，
            即使文章长短不同，也能自动适应，找出相对重要的词。

            举个例子：如果"人工智能"出现20次（最高频），阈值设为0.1，那么出现2次以上的词
            都算重要词。这样既不会漏掉重要词，也不会把无关词误判为重要词。

            """
    def get_important_words(self, word_freq: Counter) -> set:

        if not word_freq:
            return set()
        # 获取最高频词的频率
        max_freq = word_freq.most_common(1)[0][1] if word_freq else 1
        # 计算阈值
        threshold = max_freq * self.word_freq_threshold
        # 筛选重要词汇
        important_words = {word for word, freq in word_freq.items() 
                          if freq >= threshold}
        return important_words





    """
            计算句子的重要性得分

            这是Luhn算法的核心函数，它的作用是给每个句子打分，分数越高说明这个句子越重要。
            Luhn算法的评分公式：得分 = (重要词数量²) / 句子总词数  
   
            """

    def calculate_sentence_score(self, sentence: str, important_words: set) -> float:

        words = self.preprocess_text(sentence)
        
        if not words:
            return 0.0
        # 统计句子中重要词的数量
        important_count = sum(1 for word in words if word in important_words)
        if important_count == 0:
            return 0.0
        # 计算句子的重要性得分：
        score = (important_count ** 2) / len(words)
        
        return score






    """
           生成文本摘要

           这是整个类的核心方法，它整合了前面所有的步骤，最终生成文本摘要。整个过程就像
           一个完整的流水线：

           步骤1：将文本分割成句子
           首先需要把一篇文章切分成一个个句子。中文的句子通常以句号、问号、感叹号等结尾。

           步骤2：计算词频
           统计所有句子中每个词出现的次数，找出哪些词是高频词。

           步骤3：筛选重要词汇
           根据词频阈值，从高频词中筛选出真正重要的词汇。

           步骤4：给每个句子打分
           根据句子中包含的重要词汇数量和密度，计算每个句子的重要性得分。

           步骤5：选择得分最高的句子
           按照得分从高到低排序，选择前N个句子（N由summary_length参数决定）。

           步骤6：按原文顺序排列
           虽然我们是按得分选择的句子，但最终输出时按照它们在原文中的顺序排列，
           这样摘要读起来更自然流畅。

           步骤7：生成摘要
           把选中的句子用句号连接起来，形成最终的摘要。

           """
    def summarize(self, text: str) -> str:

        # 1. 将文本分割成句子
        # 使用中文句号、问号、感叹号等作为句子分隔符
        sentences = re.split(r'[。！？\n]', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if not sentences:
            return ""
        
        if len(sentences) <= self.summary_length:
            return '。'.join(sentences) + '。'
        
        # 2. 计算词频
        word_freq = self.calculate_word_frequencies(sentences)
        
        # 3. 获取重要词汇
        important_words = self.get_important_words(word_freq)
        
        if not important_words:
            # 如果没有重要词，返回前N个句子
            return '。'.join(sentences[:self.summary_length]) + '。'
        
        # 4. 计算每个句子的得分
        sentence_scores = []
        for i, sentence in enumerate(sentences):
            score = self.calculate_sentence_score(sentence, important_words)
            sentence_scores.append((i, sentence, score))
        
        # 5. 按得分排序，选择得分最高的句子
        sentence_scores.sort(key=lambda x: x[2], reverse=True)
        top_sentences = sentence_scores[:self.summary_length]
        
        # 6. 按原文顺序排列选中的句子
        top_sentences.sort(key=lambda x: x[0])
        
        # 7. 生成摘要
        summary = '。'.join([s[1] for s in top_sentences]) + '。'
        
        return summary











"""
    加载数据文件

    在实际应用中，数据是存储在文件中的，而不是直接写在代码里。这个函数
    负责从文件中读取数据。

    这个函数的工作流程：
    1. 打开文件，逐行读取
    2. 对每一行，按制表符分割成两部分：原文和参考摘要
    3. 过滤掉空行和格式不正确的行
    4. 将所有有效数据存储在一个列表中返回
    """
def load_data(file_path: str) -> List[Tuple[str, str]]:

    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split('    ')
                if len(parts) >= 2:
                    original = parts[0].strip()
                    reference = parts[1].strip()
                    if original and reference:
                        data.append((original, reference))
    except FileNotFoundError:
        print(f"错误：找不到文件 {file_path}")
    except Exception as e:
        print(f"读取文件时出错：{e}")
    return data




def main():

    print("=" * 60)
    print("Luhn文本摘要算法演示")
    print()

    # 初始化摘要器
    summarizer = LuhnSummarizer(
        summary_length=1,  # 生成2句摘要
        word_freq_threshold=0.1,  # 词频阈值
        min_word_length=2  # 最小词长度
    )

    # 加载测试数据
    data = load_data('../data/test.txt')
    print(f"成功加载 {len(data)} 条测试数据\n")
    # 测试前3条数据
    print("=" * 60)
    print("示例展示（前3条数据）")
    print("=" * 60)
    for i, (original, reference) in enumerate(data[:3], 1):
        print(f"\n{'='*60}")
        print(f"示例 {i}")
        print(f"{'='*60}")
        print(f"\n原文：\n{original[:200]}..." if len(original) > 200 else f"\n原文：\n{original}")
        # 生成摘要
        summary = summarizer.summarize(original)
        print(f"\n生成的摘要：\n{summary}")



if __name__ == "__main__":
    main()

