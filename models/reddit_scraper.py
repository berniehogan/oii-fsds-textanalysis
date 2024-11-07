# models/reddit_scraper.py
import requests
import time
import json
import os
from datetime import datetime, timedelta

def cache_results(func):
    def wrapper(self, subreddit, limit=100, cache=False, cache_duration_hours=24, sort="new"):
        cache_dir = '.cache'
        cache_file = os.path.join(cache_dir, f'{subreddit}_{limit}.json')
        
        if cache:
            os.makedirs(cache_dir, exist_ok=True) # creates a "cache" directory if it doesn't exist
            if os.path.exists(cache_file): # checks if the cache file exists
                modified_time = datetime.fromtimestamp(os.path.getmtime(cache_file)) # gets the last modified time of the cache file
                if datetime.now() - modified_time < timedelta(hours=cache_duration_hours): # check if the cached results are still valid based on a specified duration
                    with open(cache_file, 'r') as f: # reads the cache file
                        return json.load(f) # returns the cached results as json
        
        results = func(self, subreddit, limit, sort) # if the cache is not valid or doesn't exist, the function is called to get the results
        
        if cache:
            with open(cache_file, 'w') as f: # writes the results to the cache file
                json.dump(results, f) # writes the results as json
        
        return results
    return wrapper


class RedditScraper:
    def __init__(self, user_agent):
        self.headers = {'User-Agent': user_agent}
        self.base_url = "https://api.reddit.com"
    
    @cache_results # indicates that the results of this method should be cached
    def get_subreddit_posts(self, subreddit, limit=1000, sort="new", cache=False, cache_duration_hours=24):
        posts = []
        after = None
        
        while len(posts) < limit:
            url = f"{self.base_url}/r/{subreddit}/{sort}"
            params = {
                'limit': min(100, limit - len(posts)),
                'after': after
            }
            
            response = requests.get(url, headers=self.headers, params=params)
            data = response.json()
            
            if 'data' not in data:
                break
                
            new_posts = data['data']['children']
            if not new_posts:
                break
                
            posts.extend([post['data'] for post in new_posts])
            after = new_posts[-1]['data']['name']
            
            time.sleep(2)  # Rate limiting
            
        return posts[:limit]
    
    
    def get_post_comments(self, post_id):
        """Get comments for a specific post."""
        url = f"{self.base_url}/comments/{post_id}" # URL for fetching comments
        
        print(f"Fetching comments from {url}")
        
        response = requests.get(url, headers=self.headers) # Send a GET request to the URL
        if response.status_code == 200: # Check if the response is successful
        
            comments_data = response.json() # Parse the response as JSON
            if len(comments_data) > 1 and 'data' in comments_data[1]: # Check if the response contains comments data
                return self.parse_comments(comments_data[1]['data']['children'], post_id) # Parse the comments
        return []


    def parse_comments(self, comments, post_id, parent_id=None):
        parsed_comments = [] # List to store parsed comments
        for comment in comments: # Iterate through the comments
            if 'data' in comment: # Check if the comment contains data
                comment_data = comment['data'] # Get the comment data
                parsed_comments.append({ # Append the parsed comment to the list
                    'comment_id': comment_data.get('id'), # Get the comment ID
                    'parent_id': parent_id, # Get the parent ID 
                    'post_id': post_id, # Get the post ID
                    'author': comment_data.get('author'), # Get the author of the comment
                    'body': comment_data.get('body') # Get the body of the comment
                })
                if 'replies' in comment_data and comment_data['replies']: # Check if the comment has replies
                    parsed_comments.extend(self.parse_comments(comment_data['replies']['data']['children'], post_id, comment_data.get('id'))) # Recursively parse the replies
        return parsed_comments