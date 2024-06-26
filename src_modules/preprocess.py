# src/preprocess.py

import numpy as np
import pandas as pd
from datetime import datetime
from dateutil.parser import parse


def parse_datetime(dt_str):
    try:
        return datetime.strptime(dt_str, "%Y-%m-%dT%H:%M:%S.%fZ")
    except ValueError:
        return datetime.strptime(dt_str, "%Y-%m-%dT%H:%M:%SZ")


def parse_dates(date_str):
    try:
        return pd.to_datetime(date_str, format='%Y-%m-%dT%H:%M:%S.%fZ')
    except ValueError:
        try:
            return pd.to_datetime(date_str, format='%Y-%m-%dT%H:%M:%SZ')
        except ValueError:
            return pd.NaT


def preprocess_channel_stats(channel_stats):
    channel_stats.drop_duplicates(inplace=True)

    channel_stats['description'].fillna("No description provided", inplace=True)
    channel_stats['country'].fillna("Unknown", inplace=True)
    channel_stats['published_at'] = channel_stats['published_at'].apply(parse_dates)
    channel_stats['published_year'] = channel_stats['published_at'].dt.year

    channel_stats['country'].replace('', 'Unknown', inplace=True)
    channel_stats['default_country'].replace('', 'Unknown', inplace=True)

    # Rename columns
    channel_stats.rename(columns={
        'subscribers': 'channel_subscribers',
        'total_views': 'channel_total_views',
        'total_videos': 'channel_total_videos',
        'description': 'channel_description',
        'country': 'channel_country',
        'published_at': 'channel_creation_year',
        'audience_age': 'channel_audience_age',
        'educational_topic': 'channel_educational_topic',
        'language': 'channel_language',
        'published_year': 'channel_published_year'
    }, inplace=True)

    columns_to_remove = ['banner_image_url', 'custom_url', 'default_country', 'default_language']
    channel_stats.drop(columns=columns_to_remove, inplace=True)

    channel_stats['views_per_video'] = channel_stats['channel_total_views'] / channel_stats['channel_total_videos']
    channel_stats['engagement_rate'] = (channel_stats['channel_total_likes'] + channel_stats[
        'channel_total_comments']) / channel_stats['channel_subscribers']
    channel_stats['conversation_rate'] = (channel_stats['channel_total_comments'] / channel_stats[
        'channel_total_videos']) / channel_stats['channel_subscribers']

    return channel_stats


def preprocess_video_stats(video_stats):
    video_stats.drop_duplicates(inplace=True)
    video_stats['description'].replace('', 'No description available', inplace=True)
    video_stats['published_at'] = pd.to_datetime(video_stats['published_at'])

    video_stats['like_count'] = video_stats['like_count'].fillna(0)
    video_stats['comment_count'] = video_stats['comment_count'].fillna(0)
    video_stats['view_count'] = video_stats['view_count'].replace(0, np.nan).fillna(1)  # Prevent division by zero

    video_stats['engagement_rate'] = ((video_stats['like_count'].astype(float) +
                                       video_stats['comment_count'].astype(float)) /
                                      video_stats['view_count'].astype(float)) * 100
    video_stats['engagement_rate'].fillna(video_stats['engagement_rate'].mean(), inplace=True)

    if video_stats['engagement_rate'].std() == 0:
        video_stats['engagement_rate'] += np.random.normal(0, 0.1, len(video_stats))

    percentiles = video_stats['engagement_rate'].quantile([0.33, 0.66]).values
    if len(set(percentiles)) < len(percentiles):
        percentiles = np.unique(percentiles)
        if len(percentiles) == 1:
            percentiles = [percentiles[0] - 0.01, percentiles[0] + 0.01]

    video_stats['engagement_category'] = pd.cut(video_stats['engagement_rate'],
                                                bins=[-np.inf, percentiles[0], percentiles[1], np.inf],
                                                labels=['low', 'medium', 'high'],
                                                duplicates='drop')

    return video_stats


def calculate_outliers(column, name):
    quartiles = column.quantile([0.25, 0.5, 0.75]).to_dict()
    IQR = quartiles[0.75] - quartiles[0.25]
    lower_bound = quartiles[0.25] - 1.5 * IQR
    upper_bound = quartiles[0.75] + 1.5 * IQR
    outliers = column[(column < lower_bound) | (column > upper_bound)]
    outlier_df = pd.DataFrame(outliers, columns=[name]).sort_values(by=name, ascending=False)
    return outliers, outlier_df


def remove_outliers(video_stats):
    video_stats = video_stats[(video_stats['like_count'] != 0) &
                              (video_stats['view_count'] != 0) &
                              (video_stats['view_count'] != 1) &
                              (video_stats['comment_count'] != 0) &
                              (video_stats['comment_count'] != 1)]

    video_stats = video_stats[video_stats['engagement_rate'] <= 10]
    video_stats = video_stats[video_stats['comment_count'] > 0]

    return video_stats


def combine_outliers(video_stats):
    outliers_view_count, outliers_view_count_df = calculate_outliers(video_stats['view_count'], 'view_count')
    outliers_comment_count, outliers_comment_count_df = calculate_outliers(video_stats['comment_count'],
                                                                           'comment_count')
    outliers_like_count, outliers_like_count_df = calculate_outliers(video_stats['like_count'], 'like_count')
    outliers_engagement_rate, outliers_engagement_rate_df = calculate_outliers(video_stats['engagement_rate'],
                                                                               'engagement_rate')

    combined_outliers = pd.concat([outliers_view_count_df, outliers_comment_count_df, outliers_like_count_df,
                                   outliers_engagement_rate_df]).drop_duplicates()

    return combined_outliers


def preprocess_data(channel_stats, video_stats):
    channel_stats = preprocess_channel_stats(channel_stats)
    video_stats = preprocess_video_stats(video_stats)
    video_stats = remove_outliers(video_stats)
    combined_outliers = combine_outliers(video_stats)

    return channel_stats, video_stats, combined_outliers
