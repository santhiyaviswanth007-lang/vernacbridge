import pandas as pd
data=pd.read_csv("vernacbridge_datset.csv")
print(data.head())
import pandas as pd

# Load dataset
data = pd.read_csv("vernacbridge_datset.csv")

# Translation function
def translate(text):
    match = data[data["tanglish_input"] == text]

    if len(match) > 0:
        return match.iloc[0]["english_output"]

    return "Translation not found."

# Test
while True:
    user_input = input("Enter Tanglish sentence: ")
    result = translate(user_input)
    print("\nEnglish Output:")
    print(result)