# src/eda_and_analyze.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_channel_stats(channel_stats):
    print("Channel Stats Info:")
    print(channel_stats.info())
    print(channel_stats.describe(include='all'))
    return channel_stats.describe(include='all')

def analyze_video_stats(video_stats):
    print("\nVideo Stats Info:")
    print(video_stats.info())
    print(video_stats.describe(include='all'))
    return video_stats.describe(include='all')

def investigate_bad_videos(video_stats):
    bad_videos = video_stats[(video_stats['like_count'] == 0) |
                             (video_stats['view_count'] == 0) |
                             (video_stats['view_count'] == 1) |
                             (video_stats['comment_count'] == 0) |
                             (video_stats['comment_count'] == 1)]
    print(f"Videos with bad number of comments or views: {bad_videos.shape[0]}")
    return bad_videos

def check_duplicates(video_stats):
    duplicates = video_stats[video_stats.duplicated('title', keep=False)]
    print(f"Duplicate titles found: {duplicates.shape[0]}")
    return duplicates

def print_cleaned_video_stats(video_stats):
    print(f"Number of videos remaining after cleaning: {video_stats.shape[0]}")
    print("Cleaned Video Stats:")
    print(video_stats.sort_values(by='view_count', ascending=False))
    return video_stats

def analyze_outliers(outliers_df, column_name):
    print(f"\n{column_name} Outliers (Sorted):")
    print(outliers_df)
    return outliers_df

def calculate_outliers(column, name):
    quartiles = column.quantile([0.25, 0.5, 0.75]).to_dict()
    IQR = quartiles[0.75] - quartiles[0.25]
    lower_bound = quartiles[0.25] - 1.5 * IQR
    upper_bound = quartiles[0.75] + 1.5 * IQR
    outliers = column[(column < lower_bound) | (column > upper_bound)]
    outlier_df = pd.DataFrame(outliers, columns=[name]).sort_values(by=name, ascending=False)
    return outliers, outlier_df

def eda_plots(df, output_dir="plots"):
    plt.figure(figsize=(10, 6))
    sns.histplot(df['view_count'], bins=50)
    plt.title('View Count Distribution')
    plt.savefig(f"{output_dir}/view_count_distribution.png")
    plt.close()

    plt.figure(figsize=(10, 6))
    sns.histplot(df['like_count'], bins=50)
    plt.title('Like Count Distribution')
    plt.savefig(f"{output_dir}/like_count_distribution.png")
    plt.close()

    plt.figure(figsize=(10, 6))
    sns.histplot(df['comment_count'], bins=50)
    plt.title('Comment Count Distribution')
    plt.savefig(f"{output_dir}/comment_count_distribution.png")
    plt.close()

    correlation_matrix = df.corr()
    plt.figure(figsize=(12, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Matrix')
    plt.savefig(f"{output_dir}/correlation_matrix.png")
    plt.close()

def full_analysis(channel_stats, video_stats):
    channel_stats_summary = analyze_channel_stats(channel_stats)
    video_stats_summary = analyze_video_stats(video_stats)
    bad_videos = investigate_bad_videos(video_stats)
    duplicates = check_duplicates(video_stats)
    cleaned_video_stats = print_cleaned_video_stats(video_stats)

    outliers_view_count, outliers_view_count_df = calculate_outliers(video_stats['view_count'], 'view_count')
    outliers_comment_count, outliers_comment_count_df = calculate_outliers(video_stats['comment_count'], 'comment_count')
    outliers_like_count, outliers_like_count_df = calculate_outliers(video_stats['like_count'], 'like_count')
    outliers_engagement_rate, outliers_engagement_rate_df = calculate_outliers(video_stats['engagement_rate'], 'engagement_rate')

    eda_plots(video_stats)

    return {
        "channel_stats_summary": channel_stats_summary,
        "video_stats_summary": video_stats_summary,
        "bad_videos": bad_videos,
        "duplicates": duplicates,
        "cleaned_video_stats": cleaned_video_stats,
        "outliers_view_count": outliers_view_count_df,
        "outliers_comment_count": outliers_comment_count_df,
        "outliers_like_count": outliers_like_count_df,
        "outliers_engagement_rate": outliers_engagement_rate_df
    }
