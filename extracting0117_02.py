import sys
import io
# 저장 -> open('r') -> 변수 할당 -> 파싱 -> 저장
# 파싱 필요 없을 때

sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.detach(), encoding='utf-8')

from gensim.models import word2vec
#from models.word_eval import WordEmbeddingEvaluator

model = word2vec.Word2Vec.load('embedding/mix_addata_0117_02.model')
model1 = word2vec.Word2Vec.load('embedding/mix_addata_0117_03.model')
#model0 = word2vec.Word2vec.WordEmbeddingEvaluator('embedding/mix_addata_0117_02.model', method='word2vec', dim=100, tokenizer_name = 'mecab')
#유사한 단어 출력
#print(model.most_similar(positive=["영화"]))
#print(model.most_similar(positive=["회사"]))
#print(model.most_similar(positive=["웹툰"]))
#print(model.most_similar(positive=["고양이"]))
#print(model.most_similar(positive=["학교"]))
#print(model.most_similar(positive=["구두"]))

word1 = '캘린더'
word2 = '달력'
#list1 = model.most_similar(negative=[word1])
list1 = model.most_similar(positive=[word1], topn=10)
print("<<<", word1, ">>> size=200 \n", list1)
list1 = model1.most_similar(positive=[word1], topn=10)
print("<<<", word1, ">>> size=500\n", list1)

list2 = model.most_similar(positive=[word2])
print("<<<", word2, ">>> size=200 \n", list2)

list2 = model1.most_similar(positive=[word2], topn=10)
print("<<<", word2, ">>> size=500\n", list2)
