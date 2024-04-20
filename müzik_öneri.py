import pandas as pd
from sklearn.metrics.pairwise import sigmoid_kernel, cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier

df = pd.read_csv("data.csv")
df.info()
df.head(5)

feature_cols = ['acousticness', 'danceability', 'duration_ms', 'energy',
                'instrumentalness', 'key', 'liveness', 'loudness', 'mode',
                'speechiness', 'tempo', 'time_signature', 'valence']

# Normalize the data
scaler = MinMaxScaler()
normalized_df = scaler.fit_transform(df[feature_cols])

indices = pd.Series(df.index, index=df['song_title']).drop_duplicates()

cosine = cosine_similarity(normalized_df)

sig_kernel = sigmoid_kernel(normalized_df)

def generate_recommendation(song_title, model_type=cosine):
    if song_title not in indices:
        print(f"Error: The song '{song_title}' is not in the dataset.")
        return []

    index = indices[song_title]
    score = list(enumerate(model_type[index]))
    similarity_score = sorted(score, key=lambda x: x[1], reverse=True)
    similarity_score = similarity_score[1:11]
    top_songs_index = [i[0] for i in similarity_score]
    top_songs = df['song_title'].iloc[top_songs_index].values
    return top_songs


def print_recommendations(song_title):

    print(f"\nRecommended Songs for '{song_title}' based on Cosine Similarity:")
    recommended_songs_cosine = generate_recommendation(song_title, cosine)
    for i, song in enumerate(recommended_songs_cosine, 1):
        print(f"{i}. {song}")


    print(f"\nRecommended Songs for '{song_title}' based on Sigmoid Kernel:")
    recommended_songs_sigmoid = generate_recommendation(song_title, sig_kernel)
    for i, song in enumerate(recommended_songs_sigmoid, 1):
        print(f"{i}. {song}")


user_input = input("Enter a song title: ")
print_recommendations(user_input)