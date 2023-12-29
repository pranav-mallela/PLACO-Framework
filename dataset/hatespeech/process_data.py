import pandas as pd

df = pd.read_csv('../lstm_hate_speech_data.csv')

# Define the set of characters to be removed
characters_to_remove = set(""""~!@#$%^&*()-=+~\|]}[{';: /?.>,<\n""")

# Create a translation table with None as replacement for the specified characters
translation_table = str.maketrans('', '', ''.join(characters_to_remove))

# Apply the translation to the 'tweet' column
df.iloc[:, 6] = df.iloc[:, 6].apply(lambda x: x.translate(translation_table))

# save back to csv
df.to_csv('lstm_hate_speech_data_clean.csv', index=False)

