"""Author : Vineet """

# In[46]:


import numpy as np
import pandas as pd

df = pd.read_csv("C:/Users/My PC/PycharmProjects/azurePythonBot/tmdb_5000_movies.csv")
df.head(10)


# In[12]:


print(df.shape, '\n', df.columns)

#new_cols=['Title']
df=df.rename(columns = {'title':'Title'})
df=df.rename(columns = {'runtime':'Runtime'})

#creating a year column
df['release_date'] = pd.to_datetime(df['release_date'], format ="%Y-%m-%d",errors='coerce')
df['Year'] = df['release_date'].dt.year
df['Year']  = df['Year'].fillna(0).astype(np.int64)


# In[14]:


#Concat the title and overview

df["desc"] = df["original_title"] +' '+ df['overview']
#df["desc"]


# In[16]:


#Import TfIdfVectorizer from scikit-learn
from sklearn.feature_extraction.text import TfidfVectorizer

#Define a TF-IDF Vectorizer Object. Remove all english stop words such as 'the', 'a'
tfidf = TfidfVectorizer(stop_words='english')

#Replace NaN with an empty string
df['desc'] = df['desc'].fillna('')

#Construct the required TF-IDF matrix by fitting and transforming the data
tfidf_matrix = tfidf.fit_transform(df['desc'])

#Output the shape of tfidf_matrix
tfidf_matrix.shape


# In[17]:


# Import linear_kernel
from sklearn.metrics.pairwise import linear_kernel

# Compute the cosine similarity matrix
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)


# In[19]:


#Construct a reverse map of indices and movie titles
indices = pd.Series(df.index, index=df['Title']).drop_duplicates()


class film:
    def __init__(self,title):
        self.title = title


    # Function that takes in movie title as input and outputs most similar movies
    def get_recommendations(self):
        # Get the index of the movie that matches the title

        idx = indices[self.title]

        # Get the pairwsie similarity scores of all movies with that movie
        sim_scores = list(enumerate(cosine_sim[idx]))

        # Sort the movies based on the similarity scores
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

        # Get the scores of the 10 most similar movies
        sim_scores = sim_scores[1:11]

        # Get the movie indices
        movie_indices = [i[0] for i in sim_scores]

        # Return the top 10 most similar movies

        self.bot_says =  df[['Title']].iloc[movie_indices]

        return self.bot_says


aa =film("9")
print(aa.get_recommendations())

"""
film.get_recommendations('The Dark Knight Rises')



film.get_recommendations("Schindler's List")



# Saving model to disk
#pickle.dump(regressor, open('model1.pkl','wb'))
pickle.dump(film, open('model2.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('model2.pkl','rb'))
#print(model.predict([[2, 9, 6]]))
print(model.get_recommendations("A Beautiful Mind"))

"""




