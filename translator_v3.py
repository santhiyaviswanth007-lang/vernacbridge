import pandas as pd
from rapidfuzz import process

data=pd.read_csv("vernacbridge_datset.csv")
sentences=data["tanglish_input"].tolist()
def translate(text):
    best_match = process.extractOne(text, sentences)
    print(best_match)
    score=best_match[1]
    match=data[data["tanglish_input"] == best_match[0]]
    if score >= 80:
        return match.iloc[0]["english_output"]
    else:
        return "translation_error"
while True:
    user_input=input("Please enter a sentence: ")
    if user_input.lower()=="exit":
        print("Thank you for using translator")

        break
    result=translate(user_input)
    print(result)