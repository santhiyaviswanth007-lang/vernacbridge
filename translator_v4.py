import pandas as pd
from rapidfuzz import process

data = pd.read_csv("vernacbridge_datset.csv")

sentences = data["tanglish_input"].tolist()


def translate(text):

    best_match = process.extractOne(text, sentences)

    score = best_match[1]

    match = data[
        data["tanglish_input"] == best_match[0]
    ]

    if score >= 80:

        return (
            best_match[0],
            score,
            match.iloc[0]["category"],
            match.iloc[0]["english_output"]
        )

    else:
        return None


while True:

    user_input = input("Please enter a sentence: ")

    if user_input.lower() == "exit":
        print("Thank you for using translator")
        break

    result = translate(user_input)

    if result is None:
        print("Translation confidence too low.")

    else:

        matched_sentence, score, category, translation = result

        print("\nMatched Sentence:")
        print(matched_sentence)

        print("\nConfidence Score:")
        print(score)

        print("\nCategory:")
        print(category)

        print("\nEnglish Translation:")
        print(translation)