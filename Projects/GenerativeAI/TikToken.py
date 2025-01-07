import tiktoken

encoding = tiktoken.get_encoding("cl100k_base")
encoding.encode("Generative AI is great!")
encoding.decode([5648, 1413, 15592, 374, 2294, 0])


def num_tokens_from_string(string: str, encoding_name: str) -> int:
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


num_tokens_from_string(string="Generative AI is great!", encoding_name="cl100k_base")


def compare_encodings(example_string: str) -> None:
    print(f'\nExample string: "{example_string}"')
    for encoding_name in ["r50k_base", "p50k_base", "cl100k_base"]:
        encoding = tiktoken.get_encoding(encoding_name)
        token_integers = encoding.encode(example_string)
        num_tokens = len(token_integers)
        token_bytes = [encoding.decode_single_token_bytes(token) for token in token_integers]
        print()
        print(f"{encoding_name}: {num_tokens} tokens")
        print(f"token integers: {token_integers}")
        print(f"token bytes: {token_bytes}")


compare_encodings("antidisestablishmentarianism")
compare_encodings("3 + 3 = 6")
compare_encodings("Bugün hava çok yağmurlu.")
