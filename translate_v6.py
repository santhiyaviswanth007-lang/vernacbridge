import pandas as pd
from rapidfuzz import process


data = pd.read_csv("vernacbridge_datset.csv")



sentences = data["tanglish_input"].tolist()

def translate(text):


# Get Top 3 matches
    matches = process.extract(text, sentences, limit=3)
    return matches

while True:


    user_input = input("Please enter a sentence: ")

    if user_input.lower() == "exit":
        print("Thank you for using VernacBridge!")
        break

    matches = translate(user_input)

    print("\nTop 3 Matches:")

    for i, item in enumerate(matches, start=1):
        print(f"{i}. {item[0]} ({item[1]:.2f}%)")

    choice = int(input("\nChoose a match (1-3): "))

    if choice < 1 or choice > 3:
        print("Invalid choice!")
        continue

    selected_match = matches[choice - 1]

    matched_sentence = selected_match[0]
    score = selected_match[1]

    match = data[data["tanglish_input"] == matched_sentence]

    category = match.iloc[0]["category"]
    translation = match.iloc[0]["english_output"]

    print("\nMatched Sentence:")
    print(matched_sentence)

    print("\nConfidence Score:")
    print(f"{score:.2f}%")

    print("\nCategory:")
    print(category)

    print("\nEnglish Translation:")
    print(translation)