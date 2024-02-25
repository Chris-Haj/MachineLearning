import pandas as pd
import numpy as np


def repeated_movie_titles(
        fn_movies: str,
) -> pd.DataFrame:
    """
    Identify the movie titles (original_title) that appear more than once in the movies dataset.

    Args:
        fn_movies: filename of the movies dataset (CSV file)

    Returns:
        DataFrame with columns 'original_title', 'n_movies', sorted by 'n_movies' in descending order.
    """
    data = pd.read_csv(fn_movies)
    counts = data['original_title'].value_counts()
    return counts[counts > 1]


def actors_in_top_movies(
        fn_movies: str,
        fn_actors: str,
        n_movies: int = 2,
        revenue_weight: float = 0.5,
        vote_weight: float = 0.5,
        budget_weight: float = 0.5,
) -> pd.DataFrame:
    """
    Use the weights to calculate the score for each movie and identify the top n_movies.
    The score is calculated as follows:
    score = revenue * revenue_weight + vote * vote_weight + budget * budget_weight
    Then, identify the actors who appeared in those movies.
    Return a DataFrame with the following columns:
     - Movie name
     - Movie score (rounded to 1 decimal place)
     - Actor name
    The DataFrame should be sorted by movie score in descending order, and then by actor name in ascending order.

    Args:
        fn_movies: filename of the movies dataset (CSV file)
        fn_actors: filename of the actors dataset (CSV file)
        n_movies: number of top movies to return

    Returns:
        DataFrame with columns 'movie_name', 'movie_score', 'actor_name'
    """
    actors = pd.read_csv(fn_actors)
    movies = pd.read_csv(fn_movies)
    RevWeights = (movies['revenue'] * revenue_weight).astype(int)
    VoteWeights = (movies['vote_average'] * vote_weight).astype(int)
    BudgetWeights = (movies['budget'] * budget_weight).astype(int)

    movies['score'] = RevWeights + VoteWeights + BudgetWeights
    topMovies = movies.nlargest(n_movies, 'score')
    top_movie_actors = pd.merge(topMovies, actors, left_on='original_title', right_on='movie_title')

    result = top_movie_actors[['original_title', 'score', 'actor_name']]
    result.columns = ['movie_name', 'movie_score', 'actor_name']
    result['movie_score'] = result['movie_score'].round(1)

    result = result.sort_values(by=['movie_score', 'actor_name'], ascending=[False, True])

    return result


def actors_with_most_collaborations(
        fn_actors: str,
        n_actors: int = 5,
) -> pd.DataFrame:
    """
    Identify the actors who have played alongside the greatest number of unique actors.

    Tip: use merge to join the actors DataFrame with itself, and then count the number of unique actors for each pair.

    Args:
        fn_actors: filename of the actors dataset (CSV file)
        n_actors: number of top actors to return

    Returns:
        DataFrame with columns 'actor_name', 'n_movies'

    """
    actors = pd.read_csv(fn_actors)
    actor_pairs = pd.merge(actors, actors, on='movie_title')
    different_pairs = actor_pairs[actor_pairs['actor_id_x'] != actor_pairs['actor_id_y']]
    collaboration_counts = different_pairs.groupby('actor_name_x')['actor_name_y'].nunique()
    top_actors = collaboration_counts.sort_values(ascending=False).head(n_actors).reset_index()
    top_actors.columns = ['actor_name', 'n_collaborations']
    return top_actors


def highest_grossing_movies_by_year(
        fn_movies: str,
        n_years: int = 5,
) -> pd.DataFrame:
    """
    Identify the highest-grossing movie for each of the top n_years based on total revenue.
    The function should return a DataFrame with the following columns:
     - Year
     - Movie Name
     - Total Revenue
     - Average Revenue
     - Standard Deviation of Revenue
     - Number of Movies
    The DataFrame should be sorted by Year in ascending order.
    **Note: In this assignment the column names should be as specified above, not `movie_name`, `total_revenue`, etc.**

    Args:
        fn_movies: filename of the movies dataset (CSV file)
        n_years: number of years to return

    Returns:
        DataFrame with columns as specified above

    """
    movies = pd.read_csv(fn_movies)


## Extra Credit

def actors_with_highest_median_score(
        fn_movies: str,
        fn_actors: str,
        n_actors: int = 5,
) -> pd.DataFrame:
    """

    Identify the actors with the highest median vote_average score.
    To do so, you will need to merge the movies and actors datasets, and then calculate the median score for each actor.


    The function should return a DataFrame with the following columns:
     - Actor Name
     - Median Vote Average
    The DataFrame should be sorted by Median Vote Average in descending order.

    Args:
        fn_movies: filename of the movies dataset (CSV file)
        fn_actors: filename of the actors dataset (CSV file)
        n_actors: number of top actors to return

    Returns:
        DataFrame with columns as specified above

    """

    actors = pd.read_csv('actors.csv')
    movies = pd.read_csv('tmdb_5000_movies.csv')


if __name__ == '__main__':
    # actors = pd.read_csv('actors.csv').head(10)
    actors = 'actors.csv'
    movies = 'tmdb_5000_movies.csv'
    res = actors_with_most_collaborations(actors,5)
    print(res)
