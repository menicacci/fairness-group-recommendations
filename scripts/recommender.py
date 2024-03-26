import math
import heapq
import numpy as np
import statistics
import random

'''
  Takes as input a pandas df and two row indexes.
  For a given column c, if df[row_idx1] and df[row_idx2] are both populated, the pair will be included in the output array.
  If the indexes are the same, it returns an empty array
'''
def find_non_null_column_pairs(df, row_idx1, row_idx2):
    if row_idx1 == row_idx2:
        return []

    row1 = df.iloc[row_idx1].values
    row2 = df.iloc[row_idx2].values

    non_null_mask = ~np.isnan(row1) & ~np.isnan(row2)
    return [(val1, val2) for val1, val2 in zip(row1[non_null_mask], row2[non_null_mask])]


'''
  Calculates the average value of a row (excluding NaN values)
'''
def average_value(df, row_idx):
    return np.nanmean(df.iloc[row_idx].to_numpy())


'''
  Takes as input a pandas df and two row indexes.
  Calculates the peason correlation between two items.
'''
def pearson_correlation(df, row_idx1, row_idx2):
    common_items = find_non_null_column_pairs(df, row_idx1, row_idx2)
    if not common_items:
        return 0

    mean_1 = average_value(df, row_idx1)
    mean_2 = average_value(df, row_idx2)

    n = sum((item[0] - mean_1) * (item[1] - mean_2) for item in common_items)
    d1 = math.sqrt(sum((item[0] - mean_1) ** 2 for item in common_items))
    d2 = math.sqrt(sum((item[1] - mean_2) ** 2 for item in common_items))

    return n / (d1 * d2) if (d1 != 0 and d2 != 0) else 0


'''
  Takes as input a pandas df and two row indexes.
  Calculates the cosine similarity between two items.
'''
def cosine_similarity(df, row_idx1, row_idx2):
    common_items = find_non_null_column_pairs(df, row_idx1, row_idx2)
    if not common_items:
        return 0

    dot_product = sum(item[0] * item[1] for item in common_items)
    magnitude1 = math.sqrt(sum(item[0] ** 2 for item in common_items))
    magnitude2 = math.sqrt(sum(item[1] ** 2 for item in common_items))

    return dot_product / (magnitude1 * magnitude2) if (magnitude1 != 0 and magnitude2 != 0) else 0


'''
  Takes as input a pandas df, the index of the target user, the size of the neighbourhood and a score function.
  Returns the neighbourhood that maximizes the score function ordered by the score itself.
  Output type: [(a, b), ...] -> a: item score, b: item index.
'''
def get_neighborhood(df, target_idx, size, score_function=pearson_correlation):
    top_scores_heap = []

    for row_idx in range(len(df)):
        score = score_function(df, target_idx, row_idx)
        heapq.heappush(top_scores_heap, (score, row_idx))

        if len(top_scores_heap) > size:
            heapq.heappop(top_scores_heap)

    return sorted(top_scores_heap, key=lambda x: x[0], reverse=True)


'''
  Takes as input a pandas df, a column value, a list of neighbours and the mean of the target user's row values.
  Returns a prediction score for the target user.
  Neighbour's list structure: [(a, b, c), ...] -> a: score, b: index, c: mean
'''
def get_prediction_score(df_values, valid_indexes, column, similar_items, target_mean):
    item_similarities = similar_items[:, 0]
    item_means = similar_items[:, 2]

    n = np.sum(item_similarities[valid_indexes] * (df_values - item_means[valid_indexes]))
    d = np.sum(np.abs(item_similarities[valid_indexes]))

    return target_mean + (n / d) if d != 0 else 0


def refactor_similarities(df, similar_users):
    similar_users = [(similar_user[0], similar_user[1], average_value(df, similar_user[1])) for similar_user in
                     similar_users]
    similar_users = np.array(similar_users, dtype=float)
    similar_users[:, 1] = similar_users[:, 1].astype(int)

    return similar_users


'''
  Takes as input a pandas df, a target user index, a list of neighbours and a list of items to predict the rating.
  Returns a list of items and their predicted rating.
  Output type: [(a, b), ...] -> a: predicted rating, b: item index.
'''
def get_items_predictions_based_on_similarity(df, target_user, similar_users, columns):
    predictions = []

    similar_users = refactor_similarities(df, similar_users)
    target_user_mean = average_value(df, target_user)

    df_values = df.iloc[similar_users[:, 1]][columns].values
    nan_indexes = np.isnan(df_values)
    for i, column in enumerate(columns):
        valid_indexes = ~nan_indexes[:, i]
        valid_values = df_values[valid_indexes, i]

        prediction_score = get_prediction_score(valid_values, valid_indexes, column, similar_users, target_user_mean)
        predictions.append((prediction_score, column))

    return predictions


def get_top_k_predictions(predictions, k):
    top_predictions = []
    for prediction in predictions[:k]:
        heapq.heappush(top_predictions, prediction)

    for prediction in predictions[k:]:
        if prediction > top_predictions[0]:
            heapq.heappop(top_predictions)
            heapq.heappush(top_predictions, prediction)

    return sorted(top_predictions, key=lambda x: x[0], reverse=True)


'''
  Takes as input a pandas df, a target user index, a list of neighbours and the desired output list length.
  Returns a list of items with the highest predicted rating, with the rating as well.
  Output type: [(a, b), ...] -> a: predicted rating, b: item index.
'''
def get_recommendations_based_on_similarity(df, target_user, similar_users, recommendation_size):
    user_row = df.iloc[target_user]
    user_nan_columns = user_row[user_row.isna()].index.tolist()

    top_predictions = []
    predictions = get_items_predictions_based_on_similarity(df, target_user, similar_users, user_nan_columns)
    return get_top_k_predictions(predictions, recommendation_size)


'''
  Returns a list of predictions for a user
'''
def get_predictions(df, target_user, neighbourhood_size=20, predictions_size=10, score_function=pearson_correlation):
    similar_users = get_neighborhood(df, target_user, neighbourhood_size, score_function)
    predictions = get_recommendations_based_on_similarity(df, target_user, similar_users, predictions_size)

    return predictions


def find_nan_columns_in_group(df, group):
    group_df = df.iloc[group]
    nan_mask = group_df.isna()

    nan_counts = nan_mask.sum(axis=0)
    return nan_counts[nan_counts == len(group)].index.tolist()


def random_distinct_indexes(df, num):
    distinct_indexes = df.index.tolist()
    random.shuffle(distinct_indexes)

    return distinct_indexes[:num]


def average_aggregation(prediction_scores):
    return statistics.mean(prediction_scores)


def least_misery(prediction_scores):
    return min(prediction_scores)


def get_combined_predictions(group_predictions):
    first_user_predictions = group_predictions[0]
    combined_predictions = [[prediction[1], []] for prediction in first_user_predictions]

    for user_predictions in group_predictions:
        for i in range(len(user_predictions)):
            combined_predictions[i][1].append(user_predictions[i][0])

    return combined_predictions


def get_group_recommendation(df, group, neighbourhood_size=50, predictions_size=10,
                             group_score_function=average_aggregation, score_function=pearson_correlation):
    group_neighbours = [get_neighborhood(df, target_user, neighbourhood_size, score_function) for target_user in group]

    nan_columns = find_nan_columns_in_group(df, group)

    group_predictions = [get_items_predictions_based_on_similarity(df, target_user, similar_users, nan_columns)
                         for target_user, similar_users in zip(group, group_neighbours)
                         ]

    combined_predictions = get_combined_predictions(group_predictions)
    for combined_prediction in combined_predictions:
        combined_prediction.insert(0, group_score_function(combined_prediction[1]))

    return get_top_k_predictions(combined_predictions, predictions_size)


def apply_disagreement_detection(users_predictions, predicted_score):
    st_dev = statistics.stdev(users_predictions)
    return ((1 / st_dev) ** (1 / len(users_predictions))) * predicted_score if st_dev > 0 else 0


def get_group_recommendation_with_disagreements(df, group, neighbourhood_size=50, predictions_size=10,
                                                group_predictions_size=300, group_score_function=average_aggregation):
    predictions = get_group_recommendation(df, group, neighbourhood_size, group_predictions_size,
                                           group_score_function=group_score_function)

    for prediction in predictions:
        predicted_score = prediction[0]
        users_predictions = prediction[2]

        prediction[0] = apply_disagreement_detection(users_predictions, predicted_score)

    return get_top_k_predictions(predictions, predictions_size)


# Print functions
def float_approx(num):
    return "{:.2f}".format(num)


def float_list_approx(nums):
    approximations = [float_approx(num) for num in nums]
    return ', '.join(approximations)


def print_group_predictions(group_predictions):
    max_id_length = max(len(str(prediction[1])) for prediction in group_predictions)
    scores = []

    for prediction in group_predictions:
        movie_id = prediction[1]
        predicted_score = float(float_approx(prediction[0]))
        user_predictions = float_list_approx(prediction[2])
        print(
            f"Movie ID: {movie_id:<{max_id_length}}\tScore: [{predicted_score}]\t Users' Predictions: [{user_predictions}]")

        scores.append(predicted_score)

    print(f"AVG: [{float_approx(statistics.mean(scores))}]\tST-DEV: [{float_approx(statistics.stdev(scores))}]\tDIFF: [{float_approx(max(scores) - min(scores))}]")


def print_user_predictions(user_predictions):
    max_id_length = max(len(str(prediction[1])) for prediction in user_predictions)

    for prediction in user_predictions:
        movie_id = prediction[1]
        predicted_score = "{:.2f}".format(prediction[0])
        print(f"Movie ID: {movie_id:<{max_id_length}}\tPredicted Rating: [{predicted_score}]")
