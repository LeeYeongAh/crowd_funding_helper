from konlpy.tag import Okt
from gensim.models import word2vec
from smart_open import smart_open

import sys
import csv
maxparse = sys.maxsize
while True:
    try:
        csv.field_size_limit(maxparse)
        break
    except OverflowError:
        maxparse = int(maxparse/10)

 #네이버 영화 코퍼스를 읽는다.
f = open('C://Users//vivid//바탕 화면//word2vec_03//crowd_funding_helper//mix_oktorigin_1_adddata.txt', 'r', encoding='utf-8')
rdr = csv.reader(f, delimiter='\t')
rdw = list(rdr)
f.close()

 #트위터 형태소 분석기를 로드한다. Twiter가 KoNLPy v0.4.5 부터 Okt로 변경 되었다.
twitter = Okt()

 #텍스트를 한줄씩 처리합니다.
result = []
for line in rdw:
     #형태소 분석하기, 단어 기본형 사용
    malist = twitter.pos( line[0], norm=True, stem=True)
    r = []
    for word in malist:
         #Josa”, “Eomi”, “'Punctuation” 는 제외하고 처리
        if not word[1] in ["Josa","Eomi","Punctuation"]:
            r.append(word[0])
     #형태소 사이에 공백 " "  을 넣습니다. 그리고 양쪽 공백을 지웁니다.
    rl = (" ".join(r)).strip()
    result.append(rl)
     #print(rl)

 #형태소들을 별도의 파일로 저장 합니다.
with open("embedding//mix_adddata_0117_02.nlp",'w', encoding='utf-8') as fp:
    fp.write("\n".join(result))

 #Word2Vec 모델 만들기
wData = word2vec.LineSentence("embedding//mix_adddata_0117_02.nlp")
wModel =word2vec.Word2Vec(wData, size=200, window=10, hs=1, min_count=2, sg=1)
wModel.save("embedding//mix_addata_0117_02.model")
print("Word2Vec Modeling finished")

with open("embedding//mix_adddata_0117_03.nlp",'w', encoding='utf-8') as fp:
    fp.write("\n".join(result))
#임베딩 차원수 늘리기.
wData2 = word2vec.LineSentence("embedding//mix_adddata_0117_03.nlp")
wModel2 =word2vec.Word2Vec(wData, size=500, window=10, hs=1, min_count=2, sg=1)
wModel2.save("embedding//mix_addata_0117_03.model")

print("Word2Vec Modeling finished")
