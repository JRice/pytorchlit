import gensim.downloader
model = gensim.downloader.load("glove-wiki-gigaword-50")
# print("Tower:")
# print(model["tower"])

print(f"Queen - King (similarity: {model.similarity('queen', 'king')}):")
# print(model["queen"] - model["king"])
# print("woman - man:")
print(f"Woman - man (similarity: {model.similarity('woman', 'man')}):")
# print(model["woman"] - model["man"])

def closest_to_this_minus_that_versus(this, that, versus):
    print(f"{this} - {that} =~ {versus}")
    for name in model.most_similar(positive=[this, versus], negative=[that], topn=5):
        print(f"Similar word: {name}")

closest_to_this_minus_that_versus("queen", "king", "man")
closest_to_this_minus_that_versus("sushi", "japan", "germany")
