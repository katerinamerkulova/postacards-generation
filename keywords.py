import re  # регулярные выражения

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


class Analyzer:
    def __init__(self):
        self.sysblock_df = pd.read_excel("sysblock.xlsx")
        self.corpus = self.get_corpus(self.sysblock_df)

        # стоп-слова
        self.stopwords = open("russian.txt", "r", encoding="UTF-8").read().split("\n")

        # Initializing TF-IDF Vectorizer with stopwords
        self.vectorizer = TfidfVectorizer(stop_words=self.stopwords, smooth_idf=True, use_idf=True)
        # Creating vocab with our corpora
        self.vectorizer.fit_transform([self.corpus])
        # Storing vocab
        self.feature_names = self.vectorizer.get_feature_names_out()

    @staticmethod
    def extract_topn_from_vector(feature_names, sorted_items, topn=10):
        """get the feature names and tf-idf score of top n items"""

        # use only top n items from vector
        sorted_items = sorted_items[:topn]

        score_vals = []
        feature_vals = []

        # word index and corresponding tf-idf score
        for idx, score in sorted_items:
            # keep track of feature name and its corresponding score
            score_vals.append(round(score, 3))
            feature_vals.append(feature_names[idx])

        # create a tuples of feature, score
        results = {}
        for idx in range(len(feature_vals)):
            results[feature_vals[idx]] = score_vals[idx]

        return results

    def get_corpus(self, df):
        corp_df = df[df["Язык текста открытки"] == "русский"]
        corpus = corp_df["Текст открытки"].dropna().apply(self.pre_analysis).tolist()
        corpus = " ".join(corpus)
        return corpus

    def get_keywords(self, doc):
        """Return top k keywords from a doc using TF-IDF method"""

        # generate tf-idf for the given document
        tf_idf_vector = self.vectorizer.transform([doc])

        # sort the tf-idf vectors by descending order of scores
        sorted_items = self.sort_coo(tf_idf_vector.tocoo())

        # extract only top 100 keywords
        keywords = self.extract_topn_from_vector(self.feature_names, sorted_items, 100)

        return list(keywords.keys())

    def get_period(self, start_year, end_year):
        # предобработка дат
        date_sr = self.sysblock_df['Дата написания текста']
        self.sysblock_df["date"] = date_sr.apply(self.get_year)
        self.sysblock_df = self.sysblock_df[self.sysblock_df["date"].str.isnumeric()]
        self.sysblock_df["date"] = self.sysblock_df["date"].apply(lambda x: int(x))

        period_df = self.sysblock_df[(self.sysblock_df["date"] >= start_year) & (self.sysblock_df["date"] <= end_year)]
        return period_df

    @staticmethod
    def get_year(date):
        if type(date) is str:
            if '-' in date:
                return date.split('-')[0]
            elif '.' in date:
                return date.split('.')[2]
        return " "

    def pipeline(self, start_year, end_year):
        period_df = self.get_period(start_year, end_year)
        doc = self.get_corpus(period_df)

        result, df = [], {}
        df['full_text'] = doc
        df['top_keywords'] = self.get_keywords(doc)
        result.append(df)

        final = pd.DataFrame(result)
        final_str = " ".join(final['top_keywords'][0][:10])
        return final_str

    # обрабатываем каждую открытку
    @staticmethod
    def pre_analysis(words):
        words = re.findall(r'\w+', words.lower())
        words = [word for word in words if not re.match(r'\d+', word)]
        words = " ".join(words)
        return words

    @staticmethod
    def sort_coo(coo_matrix):
        """Sort a dict with highest score"""
        tuples = zip(coo_matrix.col, coo_matrix.data)
        return sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)
