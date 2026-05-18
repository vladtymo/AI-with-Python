from gensim.models import Word2Vec

# Приклад корпусу
sentences = [
    ["king", "is", "man"],
    ["queen", "is", "woman"],
    ["man", "and", "woman"],
    ["parliament", "enacts", "laws"],
]

# Створення моделі Word2Vec
model = Word2Vec(sentences, vector_size=50, window=2, min_count=1, workers=4)

# Вектор для слова "king"
vector = model.wv["king"]
print("Вектор для слова 'king':\n", vector)

# Знайдемо слова, подібні до "king"
similar = model.wv.most_similar("king")
print("Схожі слова до 'king':\n", similar)
