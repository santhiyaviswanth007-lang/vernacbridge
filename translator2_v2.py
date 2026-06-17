import pandas as pd
from rapidfuzz import process

data = pd.read_csv("vernacbridge_datset.csv")

sentences = data["tanglish_input"].tolist()
while True:


    user_input = input("Enter Tanglish sentence: ")

    best_match = process.extractOne(user_input, sentences)
    print(best_match)
    matched_sentence = best_match[0]
    print(matched_sentence)
    score = best_match[1]
    match = data[
        data["tanglish_input"]
        == best_match[0]
        ]

    if score >= 80:
        print(match.iloc[0]["english_output"])
    else:
        print("Translation confidence too low.")

