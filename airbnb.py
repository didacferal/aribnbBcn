import streamlit as st
import pandas as pd

pkmDataPic = pd.read_csv('https://raw.githubusercontent.com/didacferal/Pokemon-Speed-regrerssion-model/main/pokemon_data.csv')
modifiedPic = pd.read_csv('https://raw.githubusercontent.com/didacferal/Pokemon-Speed-regrerssion-model/main/modified.csv')
shuffledPic = pd.read_csv('https://raw.githubusercontent.com/didacferal/Pokemon-Speed-regrerssion-model/main/shuffled.csv')
summary1 = pd.read_csv('https://raw.githubusercontent.com/didacferal/Pokemon-Speed-regrerssion-model/main/summary1.csv')
summary2 = pd.read_csv('https://raw.githubusercontent.com/didacferal/Pokemon-Speed-regrerssion-model/main/summary2.csv')


st.set_page_config(page_title='Pokémon Speed Regression Model', layout='centered')


# Comença la app 
st.title("Pokémon Speed Regression Model")
st.text("by Dídac Fernández Alsina")
st.image('https://www.smogon.com/xy/statspread.png', caption= 'Image from smogon.com')

st.subheader("Pokémon statistics: brief introduction")
st.markdown('In 1996 the Pokémon franchise began as Pocket Monsters: Red and Green (later released outside of Japan as Pokémon Red and Blue), a pair of video games for the original Game Boy handheld system that were developed by Game Freak and published by Nintendo')
st.markdown('Every Pokémon has 6 base statistics categories: HP (health points), attack, defense, special attack, special defense and speed.')
st.markdown('In this project we will compare different models to find the most accurate speed using the other 5 base statistics.')


# Loading the dataset
#---------------------------------------
st.subheader("Loading and selecting features")

with st.expander("Loading dataset"):
    st.write("""
            We load the a dataset with all the differents Pokémons.
            """)
    st.write("""
            This dataset contains every name, number (position in the general list), types, base statistics and other information.
            """)
    st.dataframe(pkmDataPic)

code= """
    original_df = pd.read_csv('Pokemon.csv')
"""
st.code(code, language='python')
#---------------------------------------

# Select features
#---------------------------------------
with st.expander("Selecting features"):
    st.write("""
            We create a new dataset with including only the 6 base statistics that we are going to need.
            """)
    st.write("""
            HP - Attack - Defense - Sp.Atk - Sp. Defense - Speed
            """)
    st.dataframe(modifiedPic)

code= """
    modified_df = original_df[['HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed']]
"""
st.code(code, language='python')
#---------------------------------------

#---------------------------------------
st.subheader("Splitting data: Train, Val & Test")

with st.expander("Shuffle data"):
    st.write("""
            The data must be shuffled before splitting the data to normalize it.
            """)
    st.dataframe(shuffledPic)

code= """
    shuffled_df = modified_df.sample(frac=1)
"""
st.code(code, language='python')

with st.expander("Splitting the data"):
    st.write("""
            The data is separated in 3 different datasets.
            """)
    st.write("""
            Train - Validation - Test
            """)    

code= """
    train_df = shuffled_df[:500]
val_df = shuffled_df[500:650]
test_df = shuffled_df[650:]
"""
st.code(code, language='python')
#---------------------------------------

# Preprocessing the inputs
#---------------------------------------
st.subheader("Preprocessing the inputs")
with st.expander("Preprocess the inputs"):
    st.write("""
            Data is preprocessed to improve the quality and efficiency of the algorithms that learn from them.
            """)
    st.write("""
            0=HP - 1=Attack - 2=Deffense - 3=Sp.Atk - 4=Sp.Def
            """)
    st.image('preprocess.png')

code= """
    scaler = MinMaxScaler().fit(X_train)
X_train_scaled, X_val_scaled, X_test_scaled = scaler.transform(X_train), scaler.transform(X_val), scaler.transform(X_test)
pd.DataFrame(X_train_scaled).hist()
"""
st.code(code, language='python')
#---------------------------------------


# Pick the best model
#---------------------------------------
st.subheader("Comparing and picking models")
with st.expander("Base line model"):
    st.write("""
            With the first model we obtain:
            """)
    st.write("""
            22.66112
            """)

code= """
    average_speed = y_train.mean()
mean_absolute_error(y_val, [average_speed]*len(y_val))
"""
st.code(code, language='python')


with st.expander("Linear regression"):
    st.write("""
            With the second model we obtain:
            """)
    st.write("""
            17.915843060903303
            """)

code= """
    linear_model = LinearRegression().fit(X_train, y_train)
mean_absolute_error(y_val, linear_model.predict(X_val))
"""
st.code(code, language='python')


with st.expander("Random forest regression"):
    st.write("""
            With the third model we obtain:
            """)
    st.write("""
            18.285974126984126
            """)

code= """
    random_forest = RandomForestRegressor().fit(X_train, y_train)
mean_absolute_error(y_val, random_forest.predict(X_val))
"""
st.code(code, language='python')


with st.expander("1st neural network"):
    st.dataframe(summary1)

code= """
    model_1 = Sequential([layers.Input((5,)), layers.Dense(1)])
model_1.compile(loss='mse', optimizer=Adam(learning_rate=0.01), metrics=['mean_absolute_error'])
model_1.fit(x=X_train, y=y_train, validation_data=(X_val, y_val), epochs=100)
"""
st.code(code, language='python')


with st.expander("2nd neural network"):
    st.dataframe(summary2)

code= """
    model_2 = Sequential([layers.Input((5,)), layers.Dense(32), layers.Dense(32), layers.Dense(1)])
model_2.compile(loss='mse', optimizer=Adam(learning_rate=0.01), metrics=['mean_absolute_error'])
model_2.fit(x=X_train, y=y_train, validation_data=(X_val, y_val), epochs=100)
"""
st.code(code, language='python')
#---------------------------------------


# Evaluate the chosen model
#---------------------------------------
st.subheader("Evaluate the chosen model")

with st.expander("Linear regression"):
    st.write("""
            Testing linear regression we obtain:
            """)
    st.write("""
            19.08714434236981
            """)

code= """
    mean_absolute_error(y_test, linear_model.predict(X_test))
"""
st.code(code, language='python')
#---------------------------------------


# Input data
#---------------------------------------
st.subheader("Working wirth the model via inputs")

st.write("""
        With the chosen model we try an imput of: (80,82,83,100,100)
        """)
st.write("""
        The result is a speed of: 79.39148736
        """)

code= """
    import numpy as np
input_data = (80,82,83,100,100)
scaler = MinMaxScaler()
input_data_as_numpy_array = np.asarray(input_data)
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = linear_model.predict(input_data_reshaped)
print(prediction)
"""
st.code(code, language='python')
#---------------------------------------