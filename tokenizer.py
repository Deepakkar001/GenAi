import tiktoken  # pyright: ignore[reportMissingImports]

encode = tiktoken.encoding_for_model('gpt-4o')

print("Vocab Size", encode.n_vocab)

text = 'the cat sat on  the mat'
 
tokens = encode.encode(text)
print("Tokens",tokens)