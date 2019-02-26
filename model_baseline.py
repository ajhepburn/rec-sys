import scipy, sys
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn import preprocessing
from rank_metrics import mean_reciprocal_rank

class BaselineItemToItemRecommender:
    def __init__(self):
        self.rpath = './data/csv/metadata_clean.csv'

    def csv_to_df(self) -> tuple:
        df = pd.read_csv(self.rpath, sep='\t')
        df_user, df_items, df_interactions = df[['user_id', 'user_location']], df[['item_id', 'item_timestamp', 'item_body','item_titles', 'item_cashtags', 'item_industries', 'item_sectors']], df[['user_id', 'item_id']]
        return (df_user, df_items, df_interactions)

    def build_item_matrix(self, df: pd.DataFrame) -> scipy.sparse.csr_matrix:
        count = CountVectorizer(stop_words='english')
        item_matrix = count.fit_transform(df['soup'])
        return item_matrix

    def get_item_recommendations_by_item(self, indices, df, id, cosine_sim) -> pd.DataFrame:
        idx = indices[id]
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:11]
        item_indices = [i[0] for i in sim_scores]
        return df.iloc[item_indices]

    def build_single_user_profile(self, df, df_items, indices, cosine_sim):
        df['item_id'] = df['item_id'].astype(str)
        df = df.groupby('user_id')['item_id'].apply(' '.join).reset_index()
        user_id = 769010
        user_idx = df.loc[df['user_id'] == user_id]
        user_items = [int(x) for x in user_idx['item_id'].item().split(' ')]
        user_items_cosine = [cosine_sim[indices[x]] for x in user_items]
        user_cosine = sum(user_items_cosine) / len(user_items_cosine)
        sim_scores = list(enumerate(user_cosine))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:11]
        item_indices = [i[0] for i in sim_scores]
        print("\n\nPRINTING TWEETS BY USER {0}".format(user_id))
        user_posts = df_items.loc[df_items['item_id'].isin(user_items)][['item_id', 'item_body', 'item_industries']][:10]
        print(user_posts)
        user_post_cats, recommended_cats = [], []
        for x in user_posts['item_industries'].values.tolist():
            if '|' in x:
                user_post_cats += x.split('|')
            else:
                user_post_cats.append(x)
        user_posts_cats = list(set(user_post_cats))

        print("\n\nPRINTING RECOMMENDATIONS FOR USER {0}".format(user_id))
        recommended_items = df_items.iloc[item_indices][['item_id', 'item_body', 'item_industries']]
        print(recommended_items)
        for x in recommended_items['item_industries'].values.tolist():
            if '|' in x:
                recommended_cats += x.split('|')
            else:
                recommended_cats.append(x)
        recommended_cats = list(set(recommended_cats))
        self.evaluate_recommendations(user_posts_cats, recommended_cats)
        # print(user_items)

    def evaluate_recommendations(self, up_cats, rec_cats):
        lb = preprocessing.LabelBinarizer()
        lb.fit(up_cats)
        
    def run(self):
        df_user, df_items, df_interactions = self.csv_to_df()

        clean_df = lambda xs: [str.lower(x.replace(" ", "")) for x in xs]
        create_soup = lambda x:  x['item_cashtags'].replace("|", " ") + ' ' + x['item_cashtags'].replace("|", " ") + ' ' + x['item_industries'].replace("|", " ") + ' ' + x['item_sectors'].replace("|", " ")

        df_items[['item_body','item_titles', 'item_cashtags', 'item_industries', 'item_sectors']].apply(clean_df, axis=1)
        df_items['soup'] = df_items.apply(create_soup, axis=1)

        item_matrix = self.build_item_matrix(df_items)
        cosine_sim2 = cosine_similarity(item_matrix, item_matrix)

        df_items = df_items.reset_index()
        indices = pd.Series(df_items.index, index=df_items['item_id'])
        self.build_single_user_profile(df_interactions, df_items, indices, cosine_sim2)
        

        #self.get_item_recommendations_by_item(indices=indices, df=df_items, id=70630261, cosine_sim=cosine_sim2))




if __name__ == '__main__':
    bm = BaselineItemToItemRecommender()
    bm.run()