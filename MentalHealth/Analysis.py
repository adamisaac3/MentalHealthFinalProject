import matplotlib.pyplot as plt
import seaborn as sb
from nltk.sentiment import SentimentIntensityAnalyzer
import pandas as pd
from wordcloud import WordCloud
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
import spacy
from gensim import corpora
from gensim.models import LdaModel
from scipy.stats import mannwhitneyu

#r/Anxiety Posts
anxiety_posts = pd.read_json('https://raw.githubusercontent.com/adamisaac3/MentalHealthFinalProject/refs/heads/main/MentalHealth/r/Anxiety')
anxiety_hot = pd.read_json('https://raw.githubusercontent.com/adamisaac3/MentalHealthFinalProject/refs/heads/main/MentalHealth/r/Anxiety_Hot')

#r/Depression Posts
depression_hot = pd.read_json('https://raw.githubusercontent.com/adamisaac3/MentalHealthFinalProject/refs/heads/main/MentalHealth/r/depression_hot')
depression_posts = pd.read_json('https://raw.githubusercontent.com/adamisaac3/MentalHealthFinalProject/refs/heads/main/MentalHealth/r/depression')

#r/SelfImprove Posts
self_improvement_posts = pd.read_json('https://raw.githubusercontent.com/adamisaac3/MentalHealthFinalProject/refs/heads/main/MentalHealth/r/selfimprovement')
self_improvement_hot = pd.read_json('https://github.com/adamisaac3/MentalHealthFinalProject/blob/main/MentalHealth/r/selfimprovement_hot')

a_d_hot = pd.concat([anxiety_hot, depression_hot], ignore_index=True)

anxiety_and_depression = pd.concat([anxiety_posts, depression_posts], ignore_index=True)

self_improvement = pd.concat([self_improvement_posts, self_improvement_hot], ignore_index=True)
combined_posts = pd.concat([a_d_hot, anxiety_and_depression], ignore_index=True)

def wordclouds(dataset):
    def preprocess_text(text):
        if not isinstance(text, str):
            return ""
        text = text.lower()
        text = re.sub(r'[^a-z\s]', '', text)
        tokens = word_tokenize(text)
        stop_words = stopwords.words('english') + ['i', 'm', 'of', 'be', 'to', 'my', 'dont', 'have', 'im', 'ive', 'not', 'go', 'like', 'go', 's', 'know', 'get', 'feel', 'take', 'thing', 'much', 'thats', 'youre', 'make', 'something', 'see','still','things','really','way']
        return " ".join([word for word in tokens if word not in stop_words])

    dataset['title_clean'] = dataset['title'].apply(preprocess_text)
    dataset['text_clean'] = dataset['text'].apply(preprocess_text)

    all_comments = []
    for post in dataset['top_3_comments']:
        if isinstance(post, list):
            all_comments.extend([comment['body'] for comment in post if 'body' in comment])
    comments_text = " ".join(preprocess_text(comment) for comment in all_comments)

    title_text = " ".join(dataset['title_clean'])
    body_text = " ".join(dataset['text_clean'])

    def generate_wordcloud(text, title):
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(title, fontsize=16)
        plt.show()


    generate_wordcloud(title_text, "Word Cloud for Titles")
    generate_wordcloud(body_text, "Word Cloud for Body Text")
    generate_wordcloud(comments_text, "Word Cloud for Comments")

def sentiment(dataset):
    def preprocess_text(text):
        if not isinstance(text, str):
            return ""
        text = text.lower()
        text = re.sub(r'[^a-z\s]', '', text)
        return text

    dataset['title_clean'] = dataset['title'].apply(preprocess_text)
    dataset['text_clean'] = dataset['text'].apply(preprocess_text)

    comments = []
    for post_id, post_comments in zip(dataset['id'], dataset['top_3_comments']):
        if isinstance(post_comments, list):
            for comment in post_comments:
                comments.append({
                    'post_id': post_id,
                    'comment': preprocess_text(comment['body']) if 'body' in comment else None
                })
    comments_df = pd.DataFrame(comments)

    vader = SentimentIntensityAnalyzer()
    dataset['title_vader'] = dataset['title_clean'].apply(lambda x: vader.polarity_scores(x)['compound'])
    dataset['body_vader'] = dataset['text_clean'].apply(lambda x: vader.polarity_scores(x)['compound'])
    comments_df['comment_vader'] = comments_df['comment'].apply( lambda x: vader.polarity_scores(x)['compound'] if x else None)
    dataset['sentiment_category'] = dataset['body_vader'].apply(
        lambda x: 'Positive' if x > 0 else 'Negative' if x < 0 else 'Neutral'
    )

    def plot_sentiment_distribution(sentiments, title):
        plt.figure(figsize=(10, 5))
        plt.hist(sentiments, bins=30, color='skyblue', edgecolor='black')
        plt.title(f"Sentiment Distribution: {title}", fontsize=16)
        plt.xlabel("Sentiment Score")
        plt.ylabel("Frequency")
        plt.show()

    plot_sentiment_distribution(dataset['title_vader'], "Titles")
    plot_sentiment_distribution(dataset['body_vader'], "Body Text")
    plot_sentiment_distribution(comments_df['comment_vader'].dropna(), "Comments")

def time(dataset):
    dataset['date_created'] = pd.to_datetime(dataset['date_created'])
    dataset['month'] = pd.to_datetime(dataset['date_created']).dt.month
    dataset['day_of_week'] = pd.to_datetime(dataset['date_created']).dt.day_name()
    dataset['hour'] = pd.to_datetime(dataset['date_created']).dt.hour
    monthly_posts = dataset.groupby('month').size()
    sb.countplot(x='day_of_week', data=dataset,
                  order=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
    plt.title("Posts by Day of the Week")
    plt.show()

    hourly_posts = dataset.groupby('hour').size()

    plt.figure(figsize=(10, 6))
    sb.barplot(x=hourly_posts.index, y=hourly_posts.values, palette="viridis")

    plt.xlabel("Hour of Day", fontsize=12)
    plt.ylabel("Number of Posts", fontsize=12)
    plt.title("Distribution of Posts by Hour of the Day", fontsize=14)
    plt.xticks(ticks=range(0, 24), labels=[f"{h}:00" for h in range(0, 24)], rotation=45)
    plt.show()

    sentiment(dataset)
    sentiment_by_month = dataset.groupby('month')['body_vader'].mean()
    plt.plot(sentiment_by_month.index, sentiment_by_month.values, marker='o')
    plt.title("Average Sentiment by Month")
    plt.xlabel("Month")
    plt.ylabel("Sentiment Score")
    plt.xticks(range(1, 13))  # Months 1-12
    plt.show()

    hourly_sentiment = dataset.groupby('hour')['body_vader'].mean()
    hourly_sentiment_counts = dataset.groupby(['hour', 'sentiment_category']).size().unstack(fill_value=0)
    plt.figure(figsize=(10, 6))
    hourly_sentiment_counts.plot(kind='bar', stacked=True, color=['red', 'green', 'blue'], width=0.8)

    # Add labels and title
    plt.xlabel("Hour of Day", fontsize=12)
    plt.ylabel("Number of Posts", fontsize=12)
    plt.title("Sentiment Distribution by Hour of the Day", fontsize=14)
    plt.xticks(ticks=range(0, 24), labels=[f"{h}:00" for h in range(0, 24)], rotation=45)
    plt.legend(title="Sentiment", labels=["Negative", "Neutral", "Positive"])
    plt.show()


def LDA(dataset):
    stop_words = stopwords.words('english') + ['i', 'm', 'of', 'be', 'to', 'my', 'dont', 'have', 'im', 'ive', 'not','go', 'like', 'go', 's', 'know', 'get', 'feel', 'take', 'thing', 'much','thats', 'youre', 'make', 'something', 'see', 'still', 'things',                                  'really', 'way']
    nlp = spacy.load('en_core_web_sm')

    def preprocess_text(text):
        text = text.lower()
        text = re.sub(r'[^a-z\s]', '', text)
        tokens = word_tokenize(text)
        preprocessed_tokens = []
        for word in tokens:
            if word not in stop_words:
                preprocessed_tokens.append(word)


        doc = nlp(" ".join(preprocessed_tokens))
        lemmatized_tokens = [token.lemma_ for token in doc]

        return lemmatized_tokens

    dataset['processed_text'] = dataset['title'].fillna('') + ' ' + dataset['text'].fillna('')
    dataset['processed_text'] = dataset['processed_text'].apply(preprocess_text)

    dictionary = corpora.Dictionary(dataset['processed_text'])
    corpus = [dictionary.doc2bow(text) for text in dataset['processed_text']]

    num_topics = 5
    passes = 15
    iterations = 400

    lda_model = LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=passes, iterations=iterations)

    topics = lda_model.print_topics(num_words=5)
    for topic in topics:
        print(topic)

def whitney(dataset1, dataset2):
    dataset1_sentiments = dataset1['body_vader'].dropna()
    dataset2_sentiments = dataset2['body_vader'].dropna()
    stat, p_value = mannwhitneyu(dataset1_sentiments, dataset2_sentiments, alternative='two-sided')
    print(f'U stat: {stat}')
    print(f'p-value: {p_value}')

sentiment(a_d_hot)
sentiment(self_improvement)

whitney(a_d_hot, self_improvement)