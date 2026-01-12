# Ballpark estimate. Later here would be proper tokenizer call
# GPT 3-4 is roughly 4 bytes per token

def calc_token_count(data: str):
    return (len(data) + 3) // 4

