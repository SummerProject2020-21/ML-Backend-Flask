from operator import le
from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
import requests
import datetime
from scipy.spatial import distance

app = Flask(__name__)


@app.route("/getmovies", methods=["POST"])
def get_movies():
    movies = []
    for i in range(1, 11):
        link = (
            "https://api.themoviedb.org/3/trending/all/day?api_key=7af3ff745ee1860685136e9e80d7ebce&page="
            + str(i)
        )
        response = requests.get(link)
        # print(response.json())
        results = response.json()["results"]
        for j in range(0, len(results)):
            movies.append(results[j])
        # print("##################")
        # print("##################")
        # print("##################")
        # print("##################")
        # print("##################")
        # print("##################")
        # print("##################")
    # print(movies)
    cols = [
        "title",
        "vote_average",
        "release_date",
        "adult",
        "vote_count",
        "popularity",
    ]
    df = pd.DataFrame(columns=cols)

    # print("##################")
    # print("##################")
    # print("##################")
    # print("##################")
    # print("##################")
    # print("##################")
    # print("##################")

    # print(df)

    for movie in movies:
        # print("***********")
        # print(movie)
        hasDate = (
            movie["first_air_date"] != ""
            if "first_air_date" in movie.keys()
            else movie["release_date"] != ""
        )
        if hasDate == True:
            df = df.append(
                {
                    "title": movie["name"]
                    if "title" not in movie.keys()
                    else movie["title"],
                    "vote_average": movie["vote_average"],
                    "release_date": (
                        datetime.date.today()
                        - datetime.datetime.strptime(
                            movie["first_air_date"], "%Y-%M-%d"
                        ).date()
                    ).days
                    if "first_air_date" in movie.keys()
                    else (
                        datetime.date.today()
                        - datetime.datetime.strptime(
                            movie["release_date"], "%Y-%M-%d"
                        ).date()
                    ).days,
                    "adult": 0
                    if "adult" not in movie.keys()
                    else 1
                    if movie["adult"] == True
                    else 0,
                    "vote_count": movie["vote_count"],
                    "popularity": movie["popularity"],
                },
                ignore_index=True,
            )

    # print("##################")
    # print("##################")
    # print("##################")
    # print("##################")
    # print("##################")
    # print("##################")
    # print("##################")

    # GETTING ALL RELEVANT PARAMETERS WITHIN THE RANGE 0-10
    min_date = df["release_date"].min()
    max_date = df["release_date"].max()
    df["release_date"] = df["release_date"].map(
        lambda release_date: ((release_date - min_date) / (max_date - min_date)) * 10
    )

    min_vote = df["vote_count"].min()
    max_vote = df["vote_count"].max()
    df["vote_count"] = df["vote_count"].map(
        lambda vote: ((vote - min_vote) / (max_vote - min_vote)) * 10
    )

    min_popularity = df["popularity"].min()
    max_popularity = df["popularity"].max()
    df["popularity"] = df["popularity"].map(
        lambda popularity: (
            (popularity - min_popularity) / (max_popularity - min_popularity)
        )
        * 10
    )

    # print(df)

    movie_query = request.json["movies"]

    movie_recommendation_data = []

    for index, row in df.iterrows():
        movie_recommendation_data.append(
            [
                row["vote_average"],
                row["release_date"],
                row["adult"],
                row["vote_count"],
                row["popularity"],
            ]
        )

    # print(movie_recommendation_data)

    similarity_matrix = []

    for i in range(len(movie_recommendation_data)):
        similarity = []
        for j in range(len(movie_recommendation_data)):
            similarity.append(
                (
                    1
                    - distance.cosine(
                        movie_recommendation_data[i], movie_recommendation_data[j]
                    ),
                    j,
                )
            )
        similarity_matrix.append(similarity)

    # print(similarity_matrix)

    # print(df)

    response_movies = []

    for movie in movie_query:
        indexes = df.index[df["title"] == movie].tolist()
        if len(indexes) != 0:
            index = indexes[0]
            movie_details = df.iloc[index]
            sim_matrix = similarity_matrix[index]
            sim_matrix.sort(key=lambda x: x[0], reverse=True)
            # print(sim_matrix[1:6])
            # print(movie_details["title"])
            similar_movies = []
            for i in sim_matrix[1:6]:
                sim_movie_details = df.iloc[i[1]]
                similar_movies.append(sim_movie_details["title"])
            # print(similar_movies)
            response_movies.append(
                {"name": movie_details["title"], "similar_movies": similar_movies}
            )

    return jsonify({"message": "Success", "movie_list": response_movies})


@app.route("/getsalary", methods=["POST"])
def get_salary():
    print(request.json["experience"])
    input = request.json["experience"]
    # Importing the dataset
    dataset = pd.read_csv("Salary_Data.csv")
    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, 1].values
    # Splitting the dataset into the Training set and Test set
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=1 / 3, random_state=0
    )
    X_test = np.append(X_test, [input])
    # # Feature Scaling
    # from sklearn.preprocessing import StandardScaler

    # sc_X = StandardScaler()
    # X_train = sc_X.fit_transform(X_train)
    # X_test = sc_X.transform(X_test)
    # sc_y = StandardScaler()
    # y_train = sc_y.fit_transform(y_train)

    # Fitting Simple Linear Regression to the Training set
    from sklearn.linear_model import LinearRegression

    regressor = LinearRegression()
    regressor.fit(X_train, y_train)
    # Predicting the Test set results
    y_pred = regressor.predict(X_test.reshape(-1, 1))
    print(int(y_pred[len(y_pred) - 1]))
    # y_pred = pd.Series(y_pred).to_json(orient="values")
    return jsonify(y_pred[len(y_pred) - 1])


@app.route("/", methods=["GET"])
def health_check():
    return "API Up And Running"


# Default port:
if __name__ == "__main__":
    app.run(use_reloader=True)
