# utils/network_builder.py
import networkx as nx
import pandas as pd
from models.reddit_scraper import RedditScraper
import time

def usercomment_tree(comments_df, include_root=True):
    """
    Create directed network of comments with optional root nodes.
    Add comments and replies of comments as nodes and connect them with edges.
    If include_root is True, add post nodes and connects posts with comments.
    """
    G = nx.DiGraph() # Directed graph
    
    # Add all comments as nodes
    for _, row in comments_df.iterrows(): # Iterate over rows
        G.add_node(row['comment_id'], # Add node with comment_id
                  author=row['author'], # Add author attribute
                  type='comment') # Add type attribute
        
        # Handle root comments
        if include_root and pd.isna(row['parent_id']): # if include root and parent_id is NaN
            G.add_node(row['post_id'], type='post') # Add post node
            G.add_edge(row['post_id'], row['comment_id']) # Add edge between post and comment
        else:
            parent = row['parent_id'] #if parent_id is not NaN, set parent to parent_id
            if parent in G: # if parent is in G
                G.add_edge(parent, row['comment_id']) # Add edge between parent and comment
    
    return G

def create_user_interaction_network(comments_df):
    """
    Create undirected network of user interactions. 
    Does not take into account the post author only builds network for comment and reply authors.
    """
    G = nx.Graph() # Undirected graph
    
    # Group comments by post to find interactions
    for post_id, post_comments in comments_df.groupby('post_id'): # group comments by posts
        # Create mapping of comments to authors
        comment_authors = post_comments.set_index('comment_id')['author'].to_dict() # get dictionary where key is comment_id and value is author
        
        # Find interactions through replies
        for _, comment in post_comments.iterrows(): # iterate over comments
            author = comment['author'] # get author
            parent_id = comment['parent_id'] # get parent_id
            
            # Skip deleted/None authors
            if author in ['[deleted]', None]: # skip comment if from deleted author or none
                continue
                
            # If parent exists, create edge between authors
            if parent_id in comment_authors: # if parent_id is in comment_authors
                parent_author = comment_authors[parent_id] # get parent author
                if parent_author not in ['[deleted]', None]: # if parent author is not deleted or none
                    G.add_edge(author, parent_author) # add edge between author and parent author
    
    return G

def create_user_post_network(comments_df):
    """
    Create bipartite network of users and posts.
    Connects authors of comments and replies to posts.
    Use bipartite_layout to visualize the network.
    """
    G = nx.Graph()
    
    # Add post nodes
    posts = comments_df['post_id'].unique() # get unique post_ids
    for post_id in posts: # iterate over posts
        G.add_node(post_id, bipartite=0) # Add post node
    
    # Add user nodes and edges
    for _, comment in comments_df.iterrows(): # iterate over comments
        author = comment['author'] # get author 
        if author not in ['[deleted]', None]: # if author is not deleted or none
            G.add_node(author, bipartite=1) # Add user node
            G.add_edge(author, comment['post_id']) # Add edge between user and post
    
    return G


def jaccard_similarity(x, y):
    """
    Calcuate Jaccard similarity between two sets.
    """
            
    intersection = len(set(x).intersection(set(y)))
    union = len(set(x).union(set(y)))
    return intersection / union

def find_similar_users(user_network, giant_component=True, top_n=None, metric='cosine'):
    """
    Find most similar users based on connections.
    - user_network: NetworkX graph of user interactions -> created with create_user_interaction_network
    - if giant_component is True, only consider the giant component of the network to find most similar users
    - if I use euclidean distance, I will get the user with the highest number of shared neighbors
    - if I use jaccard similarity, I will get the user with the highest proportion
    - if I use cosine similarity, I will get the user with the most similar connection patterns (connections are in similiar proportions) 


    """
    
    if giant_component: 
        # Get giant component
        giant = max(nx.connected_components(user_network), key=len)
        user_network = user_network.subgraph(giant).copy()
    
    # Get adjacency matrix
    adj_matrix = nx.to_numpy_array(user_network) # convert network to numpy array
    
    # Calculate cosine similarity - two users similiar if connected to the same set of users regardless of the number of connections
    if metric == 'cosine':
        from sklearn.metrics.pairwise import cosine_similarity  
        user_similarities = cosine_similarity(adj_matrix) # calculate cosine similarity between users
    # Calculate jaccard similarity - two users similiar if share large proportion of their neighbours
    elif metric == 'jaccard':
        from sklearn.metrics.pairwise import pairwise_distances
        adj_matrix = adj_matrix.astype(bool).astype(int) # 1 if edge exists, 0 otherwise
        user_similarities = pairwise_distances(adj_matrix, metric=jaccard_similarity) # calculate jaccard similarity between sets of each users neighbors
    # Calculate euclidean distance - two users similiar if have high number of shared neigbours
    elif metric == 'euclidean':
        from sklearn.metrics.pairwise import euclidean_distances
        user_similarities = euclidean_distances(adj_matrix)

    
    # Get user names
    users = list(user_network.nodes()) # get list of users
    
    # Find top similar pairs (excluding self-similarity)
    similar_pairs = [] # list to store similar pairs
    for i in range(len(users)): # iterate over users
        for j in range(i+1, len(users)): # iterate over users starting from i+1 to avoid calculating the same pair twice
            similar_pairs.append(( # append user pair and similarity score
                users[i], 
                users[j], 
                user_similarities[i,j]
            ))
    
    # Sort by similarity
    similar_pairs.sort(key=lambda x: x[2], reverse=True) # sort pairs by similarity score
    
    if top_n is None: # if top_n
        return similar_pairs # return all pairs
    else: # if top_n is not None
        return similar_pairs[:top_n] # return top_n pairs
    
def get_network_stats(G):
    """
    Calculate basic network metrics.
    - Nodes = total number of nodes in the network
    - Edges = total number of edges in the network
    - Density = density of the network = number of edges / number of possible edges
    - Components = number of connected components in the network
    
    """
    return {
        'nodes': len(G.nodes()),
        'edges': len(G.edges()),
        'density': nx.density(G),
        'components': nx.number_connected_components(G)
    }


def calculate_tree_width(tree):
    """
    Calculate width of a comment tree.
    - tree: NetworkX DiGraph representing the comment tree from usercomment_tree.
    """
    # Find the root node (node with no incoming edges)
    root = [node for node, degree in tree.in_degree() if degree == 0][0]
    
    # Get the shortest path lengths from the root (essentially the level of each node)
    levels = nx.single_source_shortest_path_length(tree, root)
    
    # Count the nodes at each level (i.e., the width at each level)
    level_counts = {}
    for level in levels.values():
        if level not in level_counts:
            level_counts[level] = 0
        level_counts[level] += 1
    
    # The width of the tree is the maximum number of nodes at any level
    return level_counts