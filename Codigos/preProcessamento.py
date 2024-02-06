import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import nltk
import re
from unicodedata import normalize
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
import string
from nltk import word_tokenize


#Pré-processamento
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

EMOTICONS = {
    u":‑\)":"Happy face or smiley",
    u":\)":"Happy face or smiley",
    u":-\]":"Happy face or smiley",
    u":\]":"Happy face or smiley",
    u":-3":"Happy face smiley",
    u":3":"Happy face smiley",
    u":->":"Happy face smiley",
    u":>":"Happy face smiley",
    u"8-\)":"Happy face smiley",
    u":o\)":"Happy face smiley",
    u":-\}":"Happy face smiley",
    u":\}":"Happy face smiley",
    u":-\)":"Happy face smiley",
    u":c\)":"Happy face smiley",
    u":\^\)":"Happy face smiley",
    u"=\]":"Happy face smiley",
    u"=\)":"Happy face smiley",
    u":‑D":"Laughing, big grin or laugh with glasses",
    u":D":"Laughing, big grin or laugh with glasses",
    u"8‑D":"Laughing, big grin or laugh with glasses",
    u"8D":"Laughing, big grin or laugh with glasses",
    u"X‑D":"Laughing, big grin or laugh with glasses",
    u"XD":"Laughing, big grin or laugh with glasses",
    u"=D":"Laughing, big grin or laugh with glasses",
    u"=3":"Laughing, big grin or laugh with glasses",
    u"B\^D":"Laughing, big grin or laugh with glasses",
    u":-\)\)":"Very happy",
    u":‑\(":"Frown, sad, andry or pouting",
    u":-\(":"Frown, sad, andry or pouting",
    u":\(":"Frown, sad, andry or pouting",
    u":‑c":"Frown, sad, andry or pouting",
    u":c":"Frown, sad, andry or pouting",
    u":‑<":"Frown, sad, andry or pouting",
    u":<":"Frown, sad, andry or pouting",
    u":‑\[":"Frown, sad, andry or pouting",
    u":\[":"Frown, sad, andry or pouting",
    u":-\|\|":"Frown, sad, andry or pouting",
    u">:\[":"Frown, sad, andry or pouting",
    u":\{":"Frown, sad, andry or pouting",
    u":@":"Frown, sad, andry or pouting",
    u">:\(":"Frown, sad, andry or pouting",
    u":'‑\(":"Crying",
    u":'\(":"Crying",
    u":'‑\)":"Tears of happiness",
    u":'\)":"Tears of happiness",
    u"D‑':":"Horror",
    u"D:<":"Disgust",
    u"D:":"Sadness",
    u"D8":"Great dismay",
    u"D;":"Great dismay",
    u"D=":"Great dismay",
    u"DX":"Great dismay",
    u":‑O":"Surprise",
    u":O":"Surprise",
    u":‑o":"Surprise",
    u":o":"Surprise",
    u":-0":"Shock",
    u"8‑0":"Yawn",
    u">:O":"Yawn",
    u":-\*":"Kiss",
    u":\*":"Kiss",
    u":X":"Kiss",
    u";‑\)":"Wink or smirk",
    u";\)":"Wink or smirk",
    u"\*-\)":"Wink or smirk",
    u"\*\)":"Wink or smirk",
    u";‑\]":"Wink or smirk",
    u";\]":"Wink or smirk",
    u";\^\)":"Wink or smirk",
    u":‑,":"Wink or smirk",
    u";D":"Wink or smirk",
    u":‑P":"Tongue sticking out, cheeky, playful or blowing a raspberry",
    u":P":"Tongue sticking out, cheeky, playful or blowing a raspberry",
    u"X‑P":"Tongue sticking out, cheeky, playful or blowing a raspberry",
    u"XP":"Tongue sticking out, cheeky, playful or blowing a raspberry",
    u":‑Þ":"Tongue sticking out, cheeky, playful or blowing a raspberry",
    u":Þ":"Tongue sticking out, cheeky, playful or blowing a raspberry",
    u":b":"Tongue sticking out, cheeky, playful or blowing a raspberry",
    u"d:":"Tongue sticking out, cheeky, playful or blowing a raspberry",
    u"=p":"Tongue sticking out, cheeky, playful or blowing a raspberry",
    u">:P":"Tongue sticking out, cheeky, playful or blowing a raspberry",
    u":‑/":"Skeptical, annoyed, undecided, uneasy or hesitant",
    u":/":"Skeptical, annoyed, undecided, uneasy or hesitant",
    u":-[.]":"Skeptical, annoyed, undecided, uneasy or hesitant",
    u">:[(\\\)]":"Skeptical, annoyed, undecided, uneasy or hesitant",
    u">:/":"Skeptical, annoyed, undecided, uneasy or hesitant",
    u":[(\\\)]":"Skeptical, annoyed, undecided, uneasy or hesitant",
    u"=/":"Skeptical, annoyed, undecided, uneasy or hesitant",
    u"=[(\\\)]":"Skeptical, annoyed, undecided, uneasy or hesitant",
    u":L":"Skeptical, annoyed, undecided, uneasy or hesitant",
    u"=L":"Skeptical, annoyed, undecided, uneasy or hesitant",
    u":S":"Skeptical, annoyed, undecided, uneasy or hesitant",
    u":‑\|":"Straight face",
    u":\|":"Straight face",
    u":$":"Embarrassed or blushing",
    u":‑x":"Sealed lips or wearing braces or tongue-tied",
    u":x":"Sealed lips or wearing braces or tongue-tied",
    u":‑#":"Sealed lips or wearing braces or tongue-tied",
    u":#":"Sealed lips or wearing braces or tongue-tied",
    u":‑&":"Sealed lips or wearing braces or tongue-tied",
    u":&":"Sealed lips or wearing braces or tongue-tied",
    u"O:‑\)":"Angel, saint or innocent",
    u"O:\)":"Angel, saint or innocent",
    u"0:‑3":"Angel, saint or innocent",
    u"0:3":"Angel, saint or innocent",
    u"0:‑\)":"Angel, saint or innocent",
    u"0:\)":"Angel, saint or innocent",
    u":‑b":"Tongue sticking out, cheeky, playful or blowing a raspberry",
    u"0;\^\)":"Angel, saint or innocent",
    u">:‑\)":"Evil or devilish",
    u">:\)":"Evil or devilish",
    u"\}:‑\)":"Evil or devilish",
    u"\}:\)":"Evil or devilish",
    u"3:‑\)":"Evil or devilish",
    u"3:\)":"Evil or devilish",
    u">;\)":"Evil or devilish",
    u"\|;‑\)":"Cool",
    u"\|‑O":"Bored",
    u":‑J":"Tongue-in-cheek",
    u"#‑\)":"Party all night",
    u"%‑\)":"Drunk or confused",
    u"%\)":"Drunk or confused",
    u":-###..":"Being sick",
    u":###..":"Being sick",
    u"<:‑\|":"Dump",
    u"\(>_<\)":"Troubled",
    u"\(>_<\)>":"Troubled",
    u"\(';'\)":"Baby",
    u"\(\^\^>``":"Nervous or Embarrassed or Troubled or Shy or Sweat drop",
    u"\(\^_\^;\)":"Nervous or Embarrassed or Troubled or Shy or Sweat drop",
    u"\(-_-;\)":"Nervous or Embarrassed or Troubled or Shy or Sweat drop",
    u"\(~_~;\) \(・\.・;\)":"Nervous or Embarrassed or Troubled or Shy or Sweat drop",
    u"\(-_-\)zzz":"Sleeping",
    u"\(\^_-\)":"Wink",
    u"\(\(\+_\+\)\)":"Confused",
    u"\(\+o\+\)":"Confused",
    u"\(o\|o\)":"Ultraman",
    u"\^_\^":"Joyful",
    u"\(\^_\^\)/":"Joyful",
    u"\(\^O\^\)／":"Joyful",
    u"\(\^o\^\)／":"Joyful",
    u"\(__\)":"Kowtow as a sign of respect, or dogeza for apology",
    u"_\(\._\.\)_":"Kowtow as a sign of respect, or dogeza for apology",
    u"<\(_ _\)>":"Kowtow as a sign of respect, or dogeza for apology",
    u"<m\(__\)m>":"Kowtow as a sign of respect, or dogeza for apology",
    u"m\(__\)m":"Kowtow as a sign of respect, or dogeza for apology",
    u"m\(_ _\)m":"Kowtow as a sign of respect, or dogeza for apology",
    u"\('_'\)":"Sad or Crying",
    u"\(/_;\)":"Sad or Crying",
    u"\(T_T\) \(;_;\)":"Sad or Crying",
    u"\(;_;":"Sad of Crying",
    u"\(;_:\)":"Sad or Crying",
    u"\(;O;\)":"Sad or Crying",
    u"\(:_;\)":"Sad or Crying",
    u"\(ToT\)":"Sad or Crying",
    u";_;":"Sad or Crying",
    u";-;":"Sad or Crying",
    u";n;":"Sad or Crying",
    u";;":"Sad or Crying",
    u"Q\.Q":"Sad or Crying",
    u"T\.T":"Sad or Crying",
    u"QQ":"Sad or Crying",
    u"Q_Q":"Sad or Crying",
    u"\(-\.-\)":"Shame",
    u"\(-_-\)":"Shame",
    u"\(一一\)":"Shame",
    u"\(；一_一\)":"Shame",
    u"\(=_=\)":"Tired",
    u"\(=\^\·\^=\)":"cat",
    u"\(=\^\·\·\^=\)":"cat",
    u"=_\^=	":"cat",
    u"\(\.\.\)":"Looking down",
    u"\(\._\.\)":"Looking down",
    u"\^m\^":"Giggling with hand covering mouth",
    u"\(\・\・?":"Confusion",
    u"\(?_?\)":"Confusion",
    u">\^_\^<":"Normal Laugh",
    u"<\^!\^>":"Normal Laugh",
    u"\^/\^":"Normal Laugh",
    u"\（\*\^_\^\*）" :"Normal Laugh",
    u"\(\^<\^\) \(\^\.\^\)":"Normal Laugh",
    u"\(^\^\)":"Normal Laugh",
    u"\(\^\.\^\)":"Normal Laugh",
    u"\(\^_\^\.\)":"Normal Laugh",
    u"\(\^_\^\)":"Normal Laugh",
    u"\(\^\^\)":"Normal Laugh",
    u"\(\^J\^\)":"Normal Laugh",
    u"\(\*\^\.\^\*\)":"Normal Laugh",
    u"\(\^—\^\）":"Normal Laugh",
    u"\(#\^\.\^#\)":"Normal Laugh",
    u"\（\^—\^\）":"Waving",
    u"\(;_;\)/~~~":"Waving",
    u"\(\^\.\^\)/~~~":"Waving",
    u"\(-_-\)/~~~ \($\·\·\)/~~~":"Waving",
    u"\(T_T\)/~~~":"Waving",
    u"\(ToT\)/~~~":"Waving",
    u"\(\*\^0\^\*\)":"Excited",
    u"\(\*_\*\)":"Amazed",
    u"\(\*_\*;":"Amazed",
    u"\(\+_\+\) \(@_@\)":"Amazed",
    u"\(\*\^\^\)v":"Laughing,Cheerful",
    u"\(\^_\^\)v":"Laughing,Cheerful",
    u"\(\(d[-_-]b\)\)":"Headphones,Listening to music",
    u'\(-"-\)':"Worried",
    u"\(ーー;\)":"Worried",
    u"\(\^0_0\^\)":"Eyeglasses",
    u"\(\＾ｖ\＾\)":"Happy",
    u"\(\＾ｕ\＾\)":"Happy",
    u"\(\^\)o\(\^\)":"Happy",
    u"\(\^O\^\)":"Happy",
    u"\(\^o\^\)":"Happy",
    u"\)\^o\^\(":"Happy",
    u":O o_O":"Surprised",
    u"o_0":"Surprised",
    u"o\.O":"Surpised",
    u"\(o\.o\)":"Surprised",
    u"oO":"Surprised",
    u"\(\*￣m￣\)":"Dissatisfied",
    u"\(‘A`\)":"Snubbed or Deflated"
}

def lowercase(Text):
    lower_text = Text.lower()
    return lower_text

def Clean(text):
    new_text = re.sub(r"http\S+", " ", text) # Removendo URLS
    new_text = re.sub('RT @[\w_]+: ', ' ', new_text)# Removendo RT
    new_text = re.sub(r"@\S+", " ", new_text) # Removendo tags
    new_text = normalize('NFKD', new_text).encode('ASCII', 'ignore').decode('ASCII') # Removendo caracteres especiais
    new_text = re.sub('[0-9]', ' ', str(new_text))
    new_text = re.sub('\s+', ' ', new_text)
    return new_text

def remove_punctuation(Text):
    punctuationfree="".join([i for i in Text if i not in string.punctuation])
    return punctuationfree

def remove_whitespace(Text):
    whitespacefree = Text.strip()
    return whitespacefree

def remove_stopwords(Text):
    stopwords = nltk.corpus.stopwords.words('english')
    clean_text = [word for word in Text if word not in stopwords]
    return clean_text

def stemmer(Text):
    stemmer = PorterStemmer()
    stem_text = [stemmer.stem(w) for w in Text]
    return stem_text

def lemmatizer(Text):
  lemmatizer = WordNetLemmatizer()
  lemm_text = [lemmatizer.lemmatize(w) for w in Text]
  return lemm_text

def remove_emoji(Text):
  emoji_pattern = re.compile("["
                          u"\U0001F600-\U0001F64F"  # Emoticons
                          u"\U0001F300-\U0001F5FF"  # Simbolos e pictogramas
                          u"\U0001F680-\U0001F6FF"  # Simbolos de transporte e mapa
                          u"\U0001F1E0-\U0001F1FF"  # Bandeiras (ios)
                          u"\U00002702-\U000027B0"
                          u"\U000024C2-\U0001F251"
                          "]+", flags=re.UNICODE)
  return emoji_pattern.sub(r'', Text)

def remove_emoticons(Text):
  emoticon_pattern = re.compile(u'(' + u'|'.join(k for k in EMOTICONS) + u')')
  return emoticon_pattern.sub(r'', Text)


#Conjunto de dados Liar
dataset_liar_test = pd.read_table('/Liar_dataset/test.tsv')
dataset_liar_train = pd.read_table('/Liar_dataset/train.tsv')
dataset_liar_valid = pd.read_table('/Liar_dataset/valid.tsv')

dataset_liar_test = dataset_liar_test.drop(columns=['id', 'subject', 'speaker', "speaker's job", 'state', 'party filiation', 'barely true counts', 'false counts', 'half true counts', 'mostly true counts', 'pants on fire counts', 'the context'])
dataset_liar_train = dataset_liar_train.drop(columns=['id', 'subject', 'speaker', "speaker's job", 'state', 'party filiation', 'barely true counts', 'false counts', 'half true counts', 'mostly true counts', 'pants on fire counts', 'the context'])
dataset_liar_valid = dataset_liar_valid.drop(columns=['id', 'subject', 'speaker', "speaker's job", 'state', 'party filiation', 'barely true counts', 'false counts', 'half true counts', 'mostly true counts', 'pants on fire counts', 'the context'])

dataset_liar_test.rename(columns={'label':'Label', 'statement':'Text'}, inplace = True)
dataset_liar_train.rename(columns={'label':'Label', 'statement':'Text'}, inplace = True)
dataset_liar_valid.rename(columns={'label':'Label', 'statement':'Text'}, inplace = True)

dataset_liar = pd.concat([dataset_liar_test, dataset_liar_train, dataset_liar_valid], ignore_index = True)

dataset_liar = dataset_liar.reindex(['Text', 'Label'], axis=1)

print("Dimensão do Conjunto de dados: ", dataset_liar.shape)
print("Nome das colunas: ", dataset_liar.columns.format())
print("Balanceamento dos dados: \n", dataset_liar['Label'].value_counts())
print("Tamanho médio dos textos: ", dataset_liar['Text'].str.len().mean())

#Nuvem de palavras
texto = " ".join(t for t in dataset_liar.Text)
word_cloud = WordCloud(collocations = False, background_color = 'black').generate(texto)
plt.imshow(word_cloud, interpolation='bilinear')
plt.axis("off")
plt.savefig('./WordClouds/word_cloud_liar.pdf', dpi=1000)
plt.show()

dataset_liar['Text'] = dataset_liar['Text'].apply(lambda x: lowercase(x))
dataset_liar['Text'] = dataset_liar['Text'].apply(lambda x: Clean(x))
dataset_liar['Text'] = dataset_liar['Text'].apply(lambda x: remove_punctuation(x))
dataset_liar['Text'] = dataset_liar['Text'].apply(lambda x: remove_whitespace(x))
dataset_liar['Text'] = dataset_liar['Text'].apply(lambda x: remove_emoji(x))
dataset_liar['Text'] = dataset_liar['Text'].apply(lambda x: remove_emoticons(x))
dataset_liar['Token'] = dataset_liar['Text'].apply(word_tokenize)
dataset_liar['Token'] = dataset_liar['Token'].apply(lambda x: remove_stopwords(x))
dataset_liar['Token'] = dataset_liar['Token'].apply(lambda x: stemmer(x))
dataset_liar['Token'] = dataset_liar['Token'].apply(lambda x: lemmatizer(x))
print("Conjunto de dados Liar pré-processados!")

dataset_liar.to_csv("./dataframes/dataframe_liar.csv", encoding = 'utf-8')



#Conjunto de dados Covid
dataset_covid_test = pd.read_csv('/Covid_dataset/english_test_with_labels.csv')
dataset_covid_train = pd.read_csv('/Covid_dataset/Constraint_Train.csv')
dataset_covid_valid = pd.read_csv('/Covid_dataset/Constraint_Val.csv')

dataset_covid_test = dataset_covid_test.drop(columns=['id'])
dataset_covid_train = dataset_covid_train.drop(columns=['id'])
dataset_covid_valid = dataset_covid_valid.drop(columns=['id'])

dataset_covid_test.rename(columns={'label':'Label', 'tweet':'Text'}, inplace = True)
dataset_covid_train.rename(columns={'label':'Label', 'tweet':'Text'}, inplace = True)
dataset_covid_valid.rename(columns={'label':'Label', 'tweet':'Text'}, inplace = True)

dataset_covid = pd.concat([dataset_covid_test, dataset_covid_train, dataset_covid_valid], ignore_index = True)

print("Dimensão do Conjunto de dados: ", dataset_covid.shape)
print("Nome das colunas: ", dataset_covid.columns.format())
print("Balanceamento dos dados: \n", dataset_covid['Label'].value_counts())
print("Tamanho médio dos textos: ", dataset_covid['Text'].str.len().mean())

#Nuvem de palavras
texto = " ".join(t for t in dataset_covid.Text)
word_cloud = WordCloud(collocations = False, background_color = 'black').generate(texto)
plt.imshow(word_cloud, interpolation='bilinear')
plt.axis("off")
plt.savefig('./WordClouds/word_cloud_covid.pdf', dpi=1000)
plt.show()

dataset_covid['Text'] = dataset_covid['Text'].apply(lambda x: lowercase(x))
dataset_covid['Text'] = dataset_covid['Text'].apply(lambda x: Clean(x))
dataset_covid['Text'] = dataset_covid['Text'].apply(lambda x: remove_punctuation(x))
dataset_covid['Text'] = dataset_covid['Text'].apply(lambda x: remove_whitespace(x))
dataset_covid['Text'] = dataset_covid['Text'].apply(lambda x: remove_emoji(x))
dataset_covid['Text'] = dataset_covid['Text'].apply(lambda x: remove_emoticons(x))
dataset_covid['Token'] = dataset_covid['Text'].apply(word_tokenize)
dataset_covid['Token'] = dataset_covid['Token'].apply(lambda x: remove_stopwords(x))
dataset_covid['Token'] = dataset_covid['Token'].apply(lambda x: stemmer(x))
dataset_covid['Token'] = dataset_covid['Token'].apply(lambda x: lemmatizer(x))
print("Conjunto de dados Covid pré-processados!")

dataset_covid.to_csv("./dataframes/dataframe_covid.csv", encoding = 'utf-8')


##Conjunto de dados George McIntire (GM)

dataset_gm = pd.read_csv('/GM_dataset/fake_or_real_news.csv')

dataset_gm = dataset_gm.drop(columns=['Unnamed: 0', 'title'])

dataset_gm.rename(columns={'text':'Text', 'label':'Label'}, inplace = True)

print("Dimensão do Conjunto de dados: ", dataset_gm.shape)
print("Nome das colunas: ", dataset_gm.columns.format())
print("Balanceamento dos dados: \n", dataset_gm['Label'].value_counts())
print("Tamanho médio dos textos: ", dataset_gm['Text'].str.len().mean())

#Nuvem de palavras
texto = " ".join(t for t in dataset_gm.Text)
word_cloud = WordCloud(collocations = False, background_color = 'black').generate(texto)
plt.imshow(word_cloud, interpolation='bilinear')
plt.axis("off")
plt.savefig('./WordClouds/word_cloud_gm.pdf', dpi=1000)
plt.show()

dataset_gm['Text'] = dataset_gm['Text'].apply(lambda x: lowercase(x))
dataset_gm['Text'] = dataset_gm['Text'].apply(lambda x: Clean(x))
dataset_gm['Text'] = dataset_gm['Text'].apply(lambda x: remove_punctuation(x))
dataset_gm['Text'] = dataset_gm['Text'].apply(lambda x: remove_whitespace(x))
dataset_gm['Text'] = dataset_gm['Text'].apply(lambda x: remove_emoji(x))
dataset_gm['Text'] = dataset_gm['Text'].apply(lambda x: remove_emoticons(x))
dataset_gm['Token'] = dataset_gm['Text'].apply(word_tokenize)
dataset_gm['Token'] = dataset_gm['Token'].apply(lambda x: remove_stopwords(x))
dataset_gm['Token'] = dataset_gm['Token'].apply(lambda x: stemmer(x))
dataset_gm['Token'] = dataset_gm['Token'].apply(lambda x: lemmatizer(x))
print("Conjunto de dados GM pré-processados!")

dataset_gm.to_csv("./dataframes/dataframe_gm.csv", encoding = 'utf-8')