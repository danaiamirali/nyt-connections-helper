# NYT Connections Helper
A basic CLI helper for the NYT Connections game.

## How To Use
Create a file called `input.txt` in the main directory, and load it with the connections words in a comma separated format (e.g., "egg, brick, cheese, ...")

In a virtual environment, run:
```
pip install -r requirements.txt
```

Then, run:
```
python3 main.py
```

## How Does It Work?
Provided a list of the remaining words in `words.txt`, a LLM agent (GPT-4), augmented with the ability to use functions returning word clusterings with embedding-based K-Means clustering, chats with the user and provides hints.
