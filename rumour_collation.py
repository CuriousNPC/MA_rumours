import praw
import pandas as pd
from textblob import TextBlob
import matplotlib.pyplot as plt
from collections import Counter
import configparser
import re
import schedule
import time
from datetime import datetime
import logging
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Set up logging
logging.basicConfig(filename='reddit_analysis.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Read configuration
config = configparser.ConfigParser()
config.read('config.ini')

# Initialize Reddit API client
reddit = praw.Reddit(
    client_id=config['Reddit']['client_id'],
    client_secret=config['Reddit']['client_secret'],
    user_agent=config['Reddit']['user_agent']
)

# Expanded keywords and phrases
MERGER_KEYWORDS = [
    'merger', 'acquisition', 'buyout', 'takeover', 'consolidation',
    'join forces', 'strategic partnership', 'corporate restructuring',
    'company integration', 'synergy', 'hostile bid', 'tender offer',
    'acquiring stake', 'majority shareholder'
]

LAYOFF_KEYWORDS = [
    'layoff', 'downsizing', 'restructuring', 'job cuts', 'redundancies',
    'workforce reduction', 'pink slip', 'let go', 'fired', 'termination',
    'cost-cutting measures', 'headcount reduction'
]

REASSIGNMENT_KEYWORDS = [
    'reassignment', 'reallocation', 'new role', 'position change',
    'department transfer', 'job rotation', 'shifting responsibilities',
    'organizational changes', 'new manager', 'reporting structure change'
]

REDUCED_WORKLOAD_KEYWORDS = [
    'reduced workload', 'less work', 'slow period', 'downtime',
    'bench time', 'low utilization', 'project cancellation', 'on hold',
    'delayed start', 'reduced hours', 'forced vacation'
]

def analyze_subreddit(subreddit_name, num_posts=100, num_comments=100):
    subreddit = reddit.subreddit(subreddit_name)
    posts = []
    comments = []
    
    for post in subreddit.hot(limit=num_posts):
        posts.append({
            'title': post.title,
            'score': post.score,
            'id': post.id,
            'url': post.url,
            'comms_num': post.num_comments,
            'created': pd.to_datetime(post.created_utc, unit='s'),
            'body': post.selftext
        })
        
        post.comments.replace_more(limit=0)
        for comment in post.comments.list()[:num_comments]:
            comments.append({
                'post_id': post.id,
                'comment_id': comment.id,
                'comment_parent_id': comment.parent_id,
                'comment_body': comment.body,
                'comment_score': comment.score,
                'created': pd.to_datetime(comment.created_utc, unit='s')
            })
    
    posts_df = pd.DataFrame(posts)
    comments_df = pd.DataFrame(comments)
    
    return posts_df, comments_df

def sentiment_analysis(df, text_column):
    df['sentiment'] = df[text_column].apply(lambda x: TextBlob(x).sentiment.polarity)
    return df

def keyword_analysis(df, text_column, keywords, column_prefix):
    for keyword in keywords:
        df[f'{column_prefix}_{keyword}'] = df[text_column].apply(lambda x: 1 if keyword.lower() in x.lower() else 0)
    df[f'{column_prefix}_score'] = df[[f'{column_prefix}_{keyword}' for keyword in keywords]].sum(axis=1)
    return df

def plot_sentiment_over_time(df, title, output_dir):
    plt.figure(figsize=(10, 6))
    plt.scatter(df['created'], df['sentiment'], alpha=0.5)
    plt.title(f'Sentiment Over Time - {title}')
    plt.xlabel('Date')
    plt.ylabel('Sentiment')
    plt.savefig(f'{output_dir}/sentiment_{title.lower()}.png')
    plt.close()

def plot_keyword_frequency(df, keywords, title, column_prefix, output_dir):
    keyword_counts = {keyword: df[f'{column_prefix}_{keyword}'].sum() for keyword in keywords}
    plt.figure(figsize=(10, 6))
    plt.bar(keyword_counts.keys(), keyword_counts.values())
    plt.title(f'Keyword Frequency - {title}')
    plt.xlabel('Keywords')
    plt.ylabel('Frequency')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/keyword_frequency_{title.lower()}.png')
    plt.close()

def extract_companies(text):
    # This is a simplified approach. For better results, consider using Named Entity Recognition.
    company_pattern = r'\b[A-Z][a-z]+ (?:Inc|Corp|Co|Ltd)\b'
    return re.findall(company_pattern, text)

def calculate_signal_strength(df, column_prefix):
    total_mentions = df[f'{column_prefix}_score'].sum()
    total_posts = len(df)
    if total_posts > 0:
        return (total_mentions / total_posts) * 100
    return 0

def analyze_company(subreddit_name):
    output_dir = f'output/{subreddit_name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    import os
    os.makedirs(output_dir, exist_ok=True)

    logging.info(f"Starting analysis for r/{subreddit_name}")
    posts_df, comments_df = analyze_subreddit(subreddit_name)
    
    # Combine post titles and bodies
    posts_df['full_text'] = posts_df['title'] + ' ' + posts_df['body']
    
    # Sentiment analysis
    posts_df = sentiment_analysis(posts_df, 'full_text')
    comments_df = sentiment_analysis(comments_df, 'comment_body')
    
    # Keyword analysis
    posts_df = keyword_analysis(posts_df, 'full_text', MERGER_KEYWORDS, 'merger')
    posts_df = keyword_analysis(posts_df, 'full_text', LAYOFF_KEYWORDS, 'layoff')
    posts_df = keyword_analysis(posts_df, 'full_text', REASSIGNMENT_KEYWORDS, 'reassign')
    posts_df = keyword_analysis(posts_df, 'full_text', REDUCED_WORKLOAD_KEYWORDS, 'reduced_work')
    
    comments_df = keyword_analysis(comments_df, 'comment_body', MERGER_KEYWORDS, 'merger')
    comments_df = keyword_analysis(comments_df, 'comment_body', LAYOFF_KEYWORDS, 'layoff')
    comments_df = keyword_analysis(comments_df, 'comment_body', REASSIGNMENT_KEYWORDS, 'reassign')
    comments_df = keyword_analysis(comments_df, 'comment_body', REDUCED_WORKLOAD_KEYWORDS, 'reduced_work')
    
    # Plot results
    plot_sentiment_over_time(posts_df, 'Posts', output_dir)
    plot_sentiment_over_time(comments_df, 'Comments', output_dir)
    plot_keyword_frequency(posts_df, MERGER_KEYWORDS, 'Merger Keywords (Posts)', 'merger', output_dir)
    plot_keyword_frequency(comments_df, MERGER_KEYWORDS, 'Merger Keywords (Comments)', 'merger', output_dir)
    
    # Extract and count mentioned companies
    all_text = ' '.join(posts_df['full_text']) + ' ' + ' '.join(comments_df['comment_body'])
    companies = extract_companies(all_text)
    company_counts = Counter(companies).most_common(10)
    
    # Calculate signal strengths
    merger_signal = calculate_signal_strength(posts_df, 'merger') + calculate_signal_strength(comments_df, 'merger')
    layoff_signal = calculate_signal_strength(posts_df, 'layoff') + calculate_signal_strength(comments_df, 'layoff')
    reassign_signal = calculate_signal_strength(posts_df, 'reassign') + calculate_signal_strength(comments_df, 'reassign')
    reduced_work_signal = calculate_signal_strength(posts_df, 'reduced_work') + calculate_signal_strength(comments_df, 'reduced_work')
    
    # Prepare and save the report
    report = f"""
    Analysis Report for r/{subreddit_name}
    Date: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

    Signal Strengths:
    M&A Activity: {merger_signal:.2f}%
    Layoffs: {layoff_signal:.2f}%
    Reassignments: {reassign_signal:.2f}%
    Reduced Workload: {reduced_work_signal:.2f}%

    Top Mentioned Companies:
    {chr(10).join([f"{company}: {count}" for company, count in company_counts])}

    Sentiment Analysis:
    Posts: Average sentiment = {posts_df['sentiment'].mean():.2f}
    Comments: Average sentiment = {comments_df['sentiment'].mean():.2f}
    """

    with open(f'{output_dir}/report.txt', 'w') as f:
        f.write(report)
    
    logging.info(f"Analysis complete for r/{subreddit_name}. Report saved to {output_dir}/report.txt")
    
    # Save detailed results
    posts_df.to_csv(f'{output_dir}/posts.csv', index=False)
    comments_df.to_csv(f'{output_dir}/comments.csv', index=False)

def main():
    subreddit_names = config['Subreddits']['names'].split(',')
    for subreddit in subreddit_names:
        analyze_company(subreddit.strip())

if __name__ == "__main__":
    schedule.every().day.at("00:00").do(main)
    while True:
        schedule.run_pending()
        time.sleep(60)
