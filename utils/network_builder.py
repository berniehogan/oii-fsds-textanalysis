# utils/network_builder.py
import networkx as nx
import pandas as pd

def create_comment_tree(comments_df, include_root=True):
    """
    Create directed network of comments with optional root nodes.
    """
    G = nx.DiGraph()
    
    # Add all comments as nodes
    for _, row in comments_df.iterrows():
        G.add_node(row['comment_id'], 
                  author=row['author'],
                  type='comment')
        
        # Handle root comments
        if include_root and pd.isna(row['parent_id']):
            G.add_node(row['post_id'], type='post')
            G.add_edge(row['post_id'], row['comment_id'])
        else:
            parent = row['parent_id']
            if parent in G:
                G.add_edge(parent, row['comment_id'])
    
    return G

def create_user_interaction_network(comments_df):
    """
    Create undirected network of user interactions.
    """
    G = nx.Graph()
    
    # Group comments by post to find interactions
    for post_id, post_comments in comments_df.groupby('post_id'):
        # Create mapping of comments to authors
        comment_authors = post_comments.set_index('comment_id')['author'].to_dict()
        
        # Find interactions through replies
        for _, comment in post_comments.iterrows():
            author = comment['author']
            parent_id = comment['parent_id']
            
            # Skip deleted/None authors
            if author in ['[deleted]', None]:
                continue
                
            # If parent exists, create edge between authors
            if parent_id in comment_authors:
                parent_author = comment_authors[parent_id]
                if parent_author not in ['[deleted]', None]:
                    G.add_edge(author, parent_author)
    
    return G

def create_user_post_network(comments_df):
    """
    Create bipartite network of users and posts.
    """
    G = nx.Graph()
    
    # Add post nodes
    posts = comments_df['post_id'].unique()
    for post_id in posts:
        G.add_node(post_id, bipartite=0)
    
    # Add user nodes and edges
    for _, comment in comments_df.iterrows():
        author = comment['author']
        if author not in ['[deleted]', None]:
            G.add_node(author, bipartite=1)
            G.add_edge(author, comment['post_id'])
    
    return G

def find_similar_users(user_network, giant_component=True, top_n=None, metric='cosine'):
    """Find most similar users based on network structure."""
    
    if giant_component:
        # Get giant component
        giant = max(nx.connected_components(user_network), key=len)
        user_network = user_network.subgraph(giant).copy()
    
    # Get adjacency matrix
    adj_matrix = nx.to_numpy_array(user_network)
    
    # Calculate cosine similarity
    if metric == 'cosine':
        from sklearn.metrics.pairwise import cosine_similarity  
        user_similarities = cosine_similarity(adj_matrix)

    elif metric == 'jaccard':
        from sklearn.metrics.pairwise import pairwise_distances
        def jaccard_similarity(x, y):
            intersection = len(set(x).intersection(set(y)))
            union = len(set(x).union(set(y)))
            return intersection / union
        adj_matrix = adj_matrix.astype(bool).astype(int)
        user_similarities = pairwise_distances(adj_matrix, metric=jaccard_similarity)
    elif metric == 'euclidean':
        from sklearn.metrics.pairwise import euclidean_distances
        user_similarities = euclidean_distances(adj_matrix)

    
    # Get user names
    users = list(user_network.nodes())
    
    # Find top similar pairs (excluding self-similarity)
    similar_pairs = []
    for i in range(len(users)):
        for j in range(i+1, len(users)):
            similar_pairs.append((
                users[i], 
                users[j], 
                user_similarities[i,j]
            ))
    
    # Sort by similarity
    similar_pairs.sort(key=lambda x: x[2], reverse=True)
    
    if top_n is None:
        return similar_pairs
    else:
        return similar_pairs[:top_n]
    
def get_network_stats(G):
    """Calculate basic network metrics."""
    return {
        'nodes': len(G.nodes()),
        'edges': len(G.edges()),
        'density': nx.density(G),
        'components': nx.number_connected_components(G)
    }