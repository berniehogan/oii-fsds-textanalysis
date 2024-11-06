# utils/analysis.py
from collections import Counter
from datetime import datetime
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from utils.text_processor import *
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from sklearn.manifold import TSNE
import seaborn as sns

def analyze_vocabulary(texts, min_freq=2): # same as Day 2
    """
    Analyze vocabulary distribution in a corpus.
    Returns word frequencies and vocabulary statistics.
    Texts is list of strings.
    """
    
    # Preprocess texts
    texts = [preprocess_text(text) for text in texts]
    concatenated_text = ' '.join(texts)
    # Tokenize all texts
    
    vectorizer = TfidfVectorizer(stop_words='english')
    vectorizer.fit(texts)
    words = vectorizer.get_feature_names_out()
    
    # Count word frequencies
    word_freq = Counter(words)
    
    # Calculate vocabulary statistics
    total_words = len(words)
    unique_words = len(word_freq)
    
    # Create frequency distribution DataFrame
    freq_df = pd.DataFrame(list(word_freq.items()), columns=['word', 'frequency'])
    freq_df['percentage'] = freq_df['frequency'] / total_words * 100
    freq_df = freq_df.sort_values('frequency', ascending=False)
    
    # Calculate cumulative coverage
    freq_df['cumulative_percentage'] = freq_df['percentage'].cumsum()
    
    stats = {
        'total_words': total_words,
        'unique_words': unique_words,
        'words_min_freq': sum(1 for freq in word_freq.values() if freq >= min_freq),
        'coverage_top_1000': freq_df.iloc[:1000]['frequency'].sum() / total_words * 100 if len(freq_df) >= 1000 else 100
    }
    
    return freq_df, stats


def tfidf_analyze_subreddit(posts, max_terms=1000, min_doc_freq=2, include_selftext=False): # new
    """
    Generates TF-IDF matrix and vocabulary statistics for a subreddit/ multiple posts.

    """
    # Combine title and optionally selftext
    texts = [
        preprocess_text(post.get('title', '')) + (' ' + preprocess_text(post.get('selftext', '')) if include_selftext else '')
        for post in posts
    ]
    
    # Analyze vocabulary first
    freq_df, vocab_stats = analyze_vocabulary(texts, min_freq=min_doc_freq)
    # Generate TF-IDF matrix and feature names
    tfidf_matrix, feature_names = generate_tfidf_matrix(texts, max_terms, min_doc_freq)
    
    # Create results object from the matrix and feature names
    results = {
        "tfidf_matrix": tfidf_matrix, 
        "feature_names": feature_names, 
        "freq_df":freq_df, 
        "vocab_stats":vocab_stats}
    
    return results


def generate_tfidf_matrix(texts, max_terms=1000, min_doc_freq=2):
    """
    Generate TF-IDF matrix and feature names from texts.
    """

    vectorizer = TfidfVectorizer(
        stop_words=stopwords.words('english'),
        max_features=max_terms,
        min_df=min_doc_freq
    )
    
    tfidf_matrix = vectorizer.fit_transform(texts)
    feature_names = vectorizer.get_feature_names_out()
    
    return tfidf_matrix, feature_names


def create_posts_dataframe(posts): # new
    """
    Create DataFrame from Reddit posts with key metadata.
    Metadate includes title, selftext, url, domain, time, and author.
    """
    df = pd.DataFrame([{
        'title': post.get('title'),
        'selftext': post.get('selftext'),
        'url': post.get('url'),
        'domain': post.get('domain'),
        'time': datetime.fromtimestamp(post.get('created_utc', 0)),
        'author': post.get('author')
    } for post in posts])
    return df

def get_mean_tfidf(tfidf_matrix, feature_names=None, return_df=True):
    """
    Calculate mean TF-IDF score for each term in the matrix.
    """
    
    mean_tfidf = tfidf_matrix.mean(axis=0).tolist()[0]

    tfidf_scores = list(zip(feature_names, mean_tfidf))

    tfidf_scores = sorted(tfidf_scores, key=lambda x: x[1], reverse=True)

    if return_df:
        df = pd.DataFrame(tfidf_scores, columns=['term', 'score'])
        df.set_index('term', inplace=True)
        return df

    return tfidf_scores

def create_report(tfidf_matrix, feature_names, freq_df, vocab_stats):
    """
    Create results object from TF-IDF matrix and feature names.
    """
    
    return {
        'vocab_stats': vocab_stats,
        'freq_distribution': freq_df,
        'tf_idf_scores': get_mean_tfidf(tfidf_matrix, feature_names, return_df=True),
        'vectorizer': None,  # Vectorizer is not needed in the results
        'matrix_shape': tfidf_matrix.shape,
        'matrix_sparsity': 100 * (1 - tfidf_matrix.nnz / (tfidf_matrix.shape[0] * tfidf_matrix.shape[1]))
    }

def get_top_terms(tfidf_results, n_terms=5):
    """
    Get top terms from TF-IDF results.
    
    Args:
        tfidf_results: Dictionary of term-tfidf scores
        n_terms: Number of top terms to return
    Returns:
        list: Top n terms
    """
    
    if isinstance(tfidf_results, pd.DataFrame):
        tfidf_scores_sorted = tfidf_results.sort_values('score', ascending=False)
    elif isinstance(tfidf_results, (pd.Series, dict)):
        tfidf_scores_sorted = pd.Series(tfidf_results).sort_values(ascending=False)
    else:
        raise ValueError("tfidf_results must be DataFrame, Series or dict")
    return tfidf_scores_sorted.head(n_terms).index.tolist()
    
def plot_word_timeseries(df, terms, figsize=(12, 6), include_selftext=False): # better to use my function
    """
    Plot time series for given terms.
    Can only plot daily counts (absolute frequency).
    
    Args:
        df: DataFrame with posts
        terms: List of terms to plot
        figsize: Tuple of figure dimensions
    Returns:
        tuple: (fig, ax) matplotlib objects
    """
    # Prepare data
    df['date'] = pd.to_datetime(df['time']).dt.date
    daily_counts = {term: [] for term in terms}
    dates = sorted(df['date'].unique())
    
    # Get vocabulary from all posts
    if include_selftext:
        all_text = ' '.join(df['title'] + ' ' + df['selftext'])
    else:
        all_text = ' '.join(df['title'])
        
    vocab = set(preprocess_text(all_text).split())
    
    # Validate terms
    invalid_terms = [term for term in terms if term not in vocab]
    if invalid_terms:
        raise ValueError(f"Terms not in vocabulary: {invalid_terms}")
    
    # Count terms per day
    for date in dates:
        day_posts = df[df['date'] == date]
        day_text = ' '.join(day_posts['title'] + ' ' + day_posts['selftext'])
        words = preprocess_text(day_text).split()
        word_counts = Counter(words)
        
        for term in terms:
            daily_counts[term].append(word_counts.get(term, 0))
    
    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    for term in terms:
        ax.plot(dates, daily_counts[term], marker='o', label=term)
    
    ax.set_title('Term Frequency Over Time')
    ax.set_xlabel('Date')
    ax.set_ylabel('Absolute Frequency')
    ax.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    return fig, ax

def plot_word_similarities_mds(tfidf_matrix, feature_names, n_terms=10, similarity_threshold=0.3, title=None):
    """
    Plot word similarities using MDS for a single TF-IDF matrix.
    
    Args:
        tfidf_matrix: scipy sparse matrix from TF-IDF vectorization
        feature_names: list of words corresponding to matrix columns
        n_terms: number of top terms to plot
        similarity_threshold: minimum similarity to draw connections
    
    Returns:
        tuple: (fig, ax) matplotlib objects
    """
    # Get top n terms based on mean TF-IDF scores
    mean_tfidf = tfidf_matrix.mean(axis=0).A1 
    top_indices = mean_tfidf.argsort()[-n_terms:][::-1] # get indices of top terms
    
    # Get vectors for top terms
    term_vectors = tfidf_matrix.T[top_indices].toarray()
    top_terms = feature_names[top_indices]
    
    # Calculate similarities and distances
    similarities = cosine_similarity(term_vectors)
    distances = 1 - similarities
    
    # Use MDS for 2D projection
    mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
    coords = mds.fit_transform(distances)
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.scatter(coords[:, 0], coords[:, 1])
    
    # Add word labels
    for i, term in enumerate(top_terms): # add labels for top terms
        ax.annotate(
            term, 
            (coords[i, 0], coords[i, 1]), 
            fontsize=16,
            bbox=dict(facecolor='white', edgecolor='gray', alpha=0.7),
            ha='center', va='center')
    
    # Draw lines between similar terms
    for i in range(len(top_terms)):
        for j in range(i+1, len(top_terms)): # each pair of terms is considered only once and avoids self-pairing
            if similarities[i,j] > similarity_threshold: # only draw lines for similar terms
                ax.plot([coords[i,0], coords[j,0]], # draw lines
                       [coords[i,1], coords[j,1]], 
                       'gray', alpha=0.3)
    if title: 
        ax.set_title(f'Word Similarities in {title}')
    else:
        ax.set_title('Word Similarities')
    plt.tight_layout()
    return fig, ax

def plot_word_similarities_tsne(tfidf_matrix, feature_names, n_highlight=5, perplexity=30, title=None):
    """
    Plot word similarities using t-SNE with all terms but highlighting top N.
    """
    # Get vectors for all terms
    term_vectors = tfidf_matrix.T.toarray()
    
    # Identify top terms
    mean_tfidf = tfidf_matrix.mean(axis=0).A1
    top_indices = mean_tfidf.argsort()[-n_highlight:][::-1]
    top_terms = feature_names[top_indices]
    
    # Calculate t-SNE for all terms
    tsne = TSNE(n_components=2, 
                perplexity=min(30, len(feature_names)/4), 
                random_state=42,
                metric='cosine')
    coords = tsne.fit_transform(term_vectors)
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Plot all points in light gray
    ax.scatter(coords[:, 0], coords[:, 1], 
              c='lightgray', alpha=0.5, s=30)
    
    # Highlight top terms
    ax.scatter(coords[top_indices, 0], coords[top_indices, 1], 
              c='red', s=100)
    
    # Add labels for top terms
    for i, term in enumerate(top_terms):
        ax.annotate(term, 
                   (coords[top_indices[i], 0], coords[top_indices[i], 1]),
                   fontsize=14,
                   bbox=dict(facecolor='white', edgecolor='gray', alpha=0.7)
        )
    if title:
        ax.set_title(f'Word Similarities in {title} (Top {n_highlight} Terms Highlighted)')
    else:
        ax.set_title(f'Word Similarities (Top {n_highlight} Terms Highlighted)')
    plt.tight_layout()
    return fig, ax

# My own function

def extract_term(term:str,string:pd.DataFrame,df:pd.DataFrame,name:str)->pd.DataFrame:
    """
    Extract the count of a term in a string and add it to a dataframe.
    Needed for frequency_top_terms_ts function.
    String is column of pd.DataFrame.
    """
    string_series=pd.Series(string)
    count=string_series.str.count(rf"\b{term}\b") # to make sure that we are only counting the word and not part of a word
    df[f"count_{term}_{name}"]=count
    return df

def freq_top_terms_ts(df: pd.DataFrame, time_column: str, title_column: str, text_column: str, top_words: dict, resampling: str) -> pd.DataFrame:
    """
    Plot the top terms by TF-IDF score for a subreddit.
    If Reddit data stored as json need to convert to DataFrame first with create_posts_dataframe function.
    Counts relative frequency of top words.
    """

    df[time_column] = pd.to_datetime(df[time_column], unit="s") # convert to datetime
    df = df[[time_column, title_column, text_column]].copy() # keep only relevant columns

    df[f"{text_column}_processed"] = df[text_column].apply(preprocess_text) # preprocess text
    df[f"{title_column}_processed"] = df[title_column].apply(preprocess_text) # preprocess title
    
    # Extract term counts for each word in top_words
    for word in top_words:
        df = extract_term(word, df[f"{title_column}_processed"], df, "title")
        df = extract_term(word, df[f"{text_column}_processed"], df, "text")

    # Calculate total word count for each post
    df["total_title"] = df[f"{title_column}_processed"].str.split().str.len()
    df["total_text"] = df[f"{text_column}_processed"].str.split().str.len()
    df["total_words"] = df["total_title"] + df["total_text"]
     
    # Calculate total count for each word
    for word in top_words:
        df[f"count_{word}_total"] = df[f"count_{word}_title"] + df[f"count_{word}_text"]
        df.drop(columns=[f"count_{word}_title", f"count_{word}_text"], inplace=True)

    # Group by time and sum counts
    df_grouped = df.resample(resampling, on=time_column).sum()

    # Calculate frequency for each word in top_words
    for word in top_words:
        df_grouped[f"frequency_{word}"] = df_grouped[f"count_{word}_total"] / df_grouped["total_words"]

    return df_grouped


def plot_freq_top_terms_ts(df_grouped:pd.DataFrame, top_words:dict, title:str) -> None:
    """
    Plot the frequency of top terms by TF-IDF score in a subreddit as a time series.
    Visualises results from freq_top_terms_ts function.
    """
    # Create axis and plot time series
    fig, ax = plt.subplots(figsize=(15, 10))
    for word in top_words:
        df_grouped[f"frequency_{word}"].plot(ax=ax, label=word)
    
    # Format plot
    ax.set_title(title)
    ax.set_ylabel("Frequency")
    ax.set_xlabel("Date")
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    if len(df_grouped.index)<10:
        ax.set_xticks(df_grouped.index)
        ax.set_xticklabels(df_grouped.index.strftime('%Y-%m-%d'))
    
    plt.legend(fontsize=14)
    plt.show()

def plot_similarities(tfidf_matrix, labels, 
                      title="term document plot", 
                        method='tsne', is_documents=True, label_color=False,
                      top_terms=None, figsize=(12, 8)):
    """
    Create projection visualization of document or term similarities
    
    Parameters:
    - tfidf_matrix: scipy sparse matrix
    - labels: list of labels (document texts or terms)
    - title: plot title
    - method: 'tsne' or 'mds' for dimensionality reduction
    - top_terms: if int, only annotate top n terms
    - is_documents: if True, plot documents, else plot terms
    - figsize: tuple for figure size
    """

    # Convert to dense array and transpose if visualizing terms
    matrix = tfidf_matrix.toarray()
    if not is_documents:
        matrix = matrix.T
    
    # Dimensionality reduction method
    if method == 'tsne':
        tsne = TSNE(n_components=2, 
                    perplexity=min(30, len(labels)-1),
                    random_state=42,
                    metric='cosine')
        coords = tsne.fit_transform(matrix)
    elif method == 'mds':
        mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
        distances = 1 - cosine_similarity(matrix)
        coords = mds.fit_transform(distances)
    else:
        raise ValueError("Method must be 'tsne' or 'mds'") 
    
    # Create visualization
    fig, ax = plt.subplots(figsize=figsize)
    if is_documents:
        sns.scatterplot(x=coords[:, 0], y=coords[:, 1], alpha=0.6, hue=labels)
    else:
        ax.scatter(coords[:, 0], coords[:, 1], alpha=0.6)
    
    # Add labels
    if top_terms and isinstance(top_terms, int):
        mean_tfidf = tfidf_matrix.mean(axis=0).A1 if is_documents else tfidf_matrix.mean(axis=1).A1
        top_indices = mean_tfidf.argsort()[-top_terms:][::-1]
        labels_to_annotate = [labels[i] for i in top_indices]
        coords_to_annotate = coords[top_indices]
    else:
        labels_to_annotate = labels
        coords_to_annotate = coords

    if label_color:
        unique_labels = list(set(labels_to_annotate))
        color_map = {label: color for label, color in zip(unique_labels, plt.cm.rainbow(np.linspace(0, 1, len(unique_labels))))}
        colors = [color_map[label] for label in labels_to_annotate]
    else:
        colors = ['black'] * len(labels_to_annotate)
    
    for i, (label, color) in enumerate(zip(labels_to_annotate, colors)):
        # Split long labels for documents
        if is_documents:
            label = split_label(label, 20)
    if  label_color:       
        ax.annotate(label, (coords_to_annotate[i, 0], coords_to_annotate[i, 1]),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=8 if is_documents else 12, alpha=0.7, color=color)
    
    
    ax.set_title(title)
    ax.grid(True, linestyle='--', alpha=0.3)
    return fig, ax

def most_informative_features(vectorizer, classifier, n=20):
    feature_names = vectorizer.get_feature_names_out()
    class_labels = classifier.classes_
    top_features = {}

    for i, class_label in enumerate(class_labels):
        top_indices = classifier.feature_log_prob_[i].argsort()[-n:][::-1]
        top_features[class_label] = [(feature_names[j], classifier.feature_log_prob_[i][j]) for j in top_indices]

    return top_features

def naive_bayes(minimum_occurrences_word: int, corpus_text: list, corpus_label: list, random: int, testsize: float, n=10, tfidf=True, report=True, tokenpattern=r"(?u)\b\w\w+\b"):
    """
    Function to perform Naive Bayes Classification on a given dataset and return the classification report.
    - Corpus_text is a list of strings
    - Corpus_label is a list of labels
    - Random is the random state for the train_test_split
    - Testsize is the size of the test set
    - Tfidf is a boolean to determine if tfidf should be used
    - Report is a boolean to determine if the classification report should be returned or the most informative features
    - N is the number of most informative features to return
    """
    if tfidf:
        vectorizer = TfidfVectorizer(min_df=minimum_occurrences_word, token_pattern=tokenpattern)
        tfidf_matrix = vectorizer.fit_transform(corpus_text)
    else:
        vectorizer = CountVectorizer(min_df=minimum_occurrences_word)
        tfidf_matrix = vectorizer.fit_transform(corpus_text)

    # Split data for NBC
    X_train, X_test, y_train, y_test = train_test_split(tfidf_matrix, corpus_label, test_size=testsize, random_state=random)

    # Naive Bayes Classification
    nbc = MultinomialNB()
    nbc.fit(X_train, y_train)
    nbc_pred = nbc.predict(X_test)
    
    if report:
        return classification_report(y_test, nbc_pred)
    
    else:
        top_features = most_informative_features(vectorizer, nbc, n)
        top_features_df = pd.DataFrame()
        for class_label, features in top_features.items(): 
            top_features_df[class_label] = [(feature, f'{np.exp(score):.4f}') for feature, score in features]

        return top_features_df
    
def k_means(minimum_occurrences_word: int, corpus_text: list, corpus_label: int, clusters: int, tokenpattern=r"(?u)\b\w\w+\b"):
    
    """Function to perform K-means clustering on a given dataset and plot the results"""
    
    vectorizer = TfidfVectorizer(min_df=minimum_occurrences_word, token_pattern=tokenpattern)
    tfidf_matrix = vectorizer.fit_transform(corpus_text)
    feature_names = vectorizer.get_feature_names_out()

    # Create and fit the k-means model
    kmeans = KMeans(n_clusters=clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(tfidf_matrix)

    # Reduce dimensionality for plotting
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(feature_names)/4), metric='cosine')
    tfidf_matrix_2d = tsne.fit_transform(tfidf_matrix.toarray())

    # Shilouette score
    silhouette_avg = silhouette_score(tfidf_matrix, cluster_labels)

    # Plot the results
    plt.figure(figsize=(10, 6))

    # Plot the data points, colored by their cluster assignments
    scatter = plt.scatter(tfidf_matrix_2d[:, 0], tfidf_matrix_2d[:, 1], c=cluster_labels, cmap=custom_cmap, alpha=0.7)
    
    # Plot the cluster centers
    # cluster_centers_2d = tsne.fit_transform(kmeans.cluster_centers_)
    # plt.scatter(cluster_centers_2d[:, 0], 
                # cluster_centers_2d[:, 1], 
                # c='red', 
                # marker='x', 
                # s=200, 
                # linewidth=3, 
                # label='Centroids')

    plt.title(f'K-means Clustering (k={clusters})')
    
    # Add a legend
    legend_labels = [f'Cluster {i}' for i in range(clusters)]
    legend_handles = scatter.legend_elements()[0]  # Get the handles for the legend
    plt.legend(legend_handles, legend_labels, title="Clusters", loc='upper right')

    plt.show()

     # Determine the majority label for each cluster
    cluster_label_counts = {}
    for cluster_label, true_label in zip(cluster_labels, corpus_label):
        if cluster_label not in cluster_label_counts:
            cluster_label_counts[cluster_label] = {}
        if true_label not in cluster_label_counts[cluster_label]:
            cluster_label_counts[cluster_label][true_label] = 0
        cluster_label_counts[cluster_label][true_label] += 1

    # Map clusters to majority labels
    cluster_to_label = {
      cluster: max(counts.items(), key=lambda x: x[1])[0]
        for cluster, counts in cluster_label_counts.items()
    }

    # Convert cluster numbers to predicted labels
    kmeans_pred = [cluster_to_label[label] for label in cluster_labels]
    
    return kmeans_pred, cluster_label_counts, cluster_to_label, silhouette_avg
