#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author = Morteza Emadi
# Copyright 2024

import pandas as pd
from src.data_load import load_data
from src.preprocess import preprocess_data
from src.EDA_and_analyze import perform_eda, analyze_data
from src.visualize import plot_correlation_matrix, plot_distribution, plot_count, plot_box, plot_scatter
from src.stacking_ensemble_model import train_model
from src.evaluate import evaluate_model


def main():
    # Load data
    video_stats, channel_stats = load_data('video_stats.csv', 'channel_stats.csv')

    # Preprocess data
    video_stats_preprocessed, channel_stats_preprocessed = preprocess_data(video_stats, channel_stats)

    # Perform EDA and analysis
    perform_eda(video_stats_preprocessed, channel_stats_preprocessed)
    analyze_data(video_stats_preprocessed, channel_stats_preprocessed)

    # Visualizations
    features = ['view_count', 'like_count', 'dislike_count', 'comment_count', 'channel_subscribers',
                'channel_total_views',
                'time_diff', 'conversation_rate', 'description_length', 'uploads_per_year', 'like_to_view_ratio',
                'comment_to_view_ratio', 'dislike_to_view_ratio']

    plot_correlation_matrix(video_stats_preprocessed, features, output_path='correlation_matrix.png')
    plot_distribution(video_stats_preprocessed, 'view_count', output_path='view_count_distribution.png')
    plot_count(video_stats_preprocessed, 'engagement_category', output_path='engagement_category_count.png')
    plot_box(video_stats_preprocessed, 'like_count', output_path='like_count_box.png')
    plot_scatter(video_stats_preprocessed, 'view_count', 'like_count', output_path='view_like_scatter.png')

    # Train model
    model = train_model(video_stats_preprocessed, features, 'engagement_category')

    # Evaluate model
    evaluate_model(model, video_stats_preprocessed, features, 'engagement_category')


if __name__ == "__main__":
    main()
