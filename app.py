from flask import Flask, render_template, send_from_directory, request
from tensorflow import keras
from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import tensorflow_hub as hub
import wget
import tokenization
import tensorflow as tf
import joblib
import requests
import numpy as np
import re
from tensorflow import keras
#import keras
import tensorflow_hub as hub
import re
import string
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
#import contractions
#from nltk.stem import PorterStemmer
#stemming = PorterStemmer()
#from tqdm.notebook import tqdm
import unicodedata
#from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer
# try:
#     import cPickle as pickle
# except ImportError: 
#     import pickle
from emot.emo_unicode import UNICODE_EMO, EMOTICONS
#from bert import tokenization

#url= "https://raw.githubusercontent.com/tensorflow/models/master/official/nlp/bert/tokenization.py"
#filename = wget.download(url)



stops = list(set(stopwords.words("english")).union(set(['br', 'the', 'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've",\
            "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', \
            'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their',\
            'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', \
            'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', \
            'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', \
            'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after',\
            'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further',\
            'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more',\
            'most', 'other', 'some', 'such', 'only', 'own', 'same', 'so', 'than', 'too', 'very', \
            's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', \
            've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn',\
            "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn',\
            "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", \
            'won', "won't", 'wouldn', "wouldn't"])))

new_stops=['i',
 'me',
 'my',
 'myself',
 'we',
 'our',
 'ours',
 'ourselves',
 'you',
 "you're",
 "you've",
 "you'll",
 "you'd",
 'your',
 'yours',
 'yourself',
 'yourselves',
 'he',
 'him',
 'his',
 'himself',
 'she',
 "she's",
 'her',
 'hers',
 'herself',
 'it',
 "it's",
 'its',
 'itself',
 'they',
 'them',
 'their',
 'theirs',
 'themselves',
 'which',
 'who',
 'whom',
 'this',
 'that',
 "that'll",
 'these',
 'those',
 'am',
 'is',
 'are',
 'was',
 'were',
 'be',
 'been',
 'being',
 'have',
 'has',
 'had',
 'having',
 'do',
 'does',
 'did',
 'doing',
 'a',
 'an',
 'the',
 'and',
 'but',
 'if',
 'or',
 'because',
 'as',
 'until',
 'while',
 'of',
 'at',
 'by',
 'for',
 'with',
 'about',
 'between',
 'into',
 'through',
 'during',
 'before',
 'after',
 'above',
 'below',
 'to',
 'from',
 'in',
 'on',
 'under',
 'again',
 'further',
 'then',
 'once',
 'here',
 'there',
 'how',
 'all',
 'any',
 'both',
 'each',
 'few',
 'more',
 'most',
 'other',
 'some',
 'such',
 'only',
 'own',
 'same',
 'so',
 'than',
 'too',
 'very',
 's',
 'just',
 'should',
 "should've",
 'now',
 'll',
 're',
 've']

def unicode_to_ascii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn')

conts = { 
    "ain't": "are not",
    "aren't": "are not",
    "can't": "can not",
    "cannot": "can not",
    "can't've": "can not have",
    "'cause": "because",
    "could've": "could have",
    "couldn't": "could not",
    "couldn't've": "could not have",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hadn't've": "had not have",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he would",
    "he'd've": "he would have",
    "he'll": "he will",
    "he'll've": "he will have",
    "he's": "he is",
    "how'd": "how did",
    "how'd'y": "how do you",
    "how'll": "how will",
    "how's": "how is",
    "i'd": "i would",
    "i'd've": "i would have",
    "i'll": "i will",
    "i'll've": "i will have",
    "i'm": "i am",
    "i've": "i have",
    "isn't": "is not",
    "it'd": "it would",
    "it'd've": "it would have",
    "it'll": "it will",
    "it'll've": "it will have",
    "it's": "it has / it is",
    "let's": "let us",
    "ma'am": "madam",
    "mayn't": "may not",
    "might've": "might have",
    "mightn't": "might not",
    "mightn't've": "might not have",
    "must've": "must have",
    "mustn't": "must not",
    "mustn't've": "must not have",
    "needn't": "need not",
    "needn't've": "need not have",
    "o'clock": "of the clock",
    "oughtn't": "ought not",
    "oughtn't've": "ought not have",
    "shan't": "shall not",
    "sha'n't": "shall not",
    "shan't've": "shall not have",
    "she'd": "she would",
    "she'd've": "she would have",
    "she'll": "she will",
    "she'll've": "she will have",
    "she's": "she is",
    "should've": "should have",
    "shouldn't": "should not",
    "shouldn't've": "should not have",
    "so've": "so have",
    "so's": "so as",
    "that'd": "that would",
    "that'd've": "that would have",
    "that's": "that is",
    "there'd": "there would",
    "there'd've": "there would have",
    "there's": "there is",
    "they'd": "they would",
    "they'd've": "they would have",
    "they'll": "they will",
    "they'll've": "they will have",
    "they're": "they are",
    "they've": "they have",
    "to've": "to have",
    "wasn't": "was not",
    "we'd": "we would",
    "we'd've": "we would have",
    "we'll": "we will",
    "we'll've": "we will have",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "what'll": "what will",
    "what'll've": "what will have",
    "what're": "what are",
    "what's": "what is",
    "what've": "what have",
    "when's": "when is",
    "when've": "when have",
    "where'd": "where did",
    "where's": "where is",
    "where've": "where have",
    "who'll": "who will",
    "who'll've": "who will have",
    "who's": "who is",
    "who've": "who have",
    "why's": "why has",
    "why've": "why have",
    "will've": "will have",
    "won't": "will not",
    "won't've": "will not have",
    "would've": "would have",
    "wouldn't": "would not",
    "wouldn't've": "would not have",
    "y'all": "you all",
    "y'all'd": "you all would",
    "y'all'd've": "you all would have",
    "y'all're": "you all are",
    "y'all've": "you all have",
    "you'd": "you would",
    "you'd've": "you would have",
    "you'll": "you will",
    "you'll've": "you will have",
    "you're": "you are",
    "you've": "you have",
    "'cause": "because",
    "'d": " would",
    "'ll": " will",
    "'re": " are",
    "'em": " them",
    "doin'": "doing",
    "goin'": "going",
    "nothin'": "nothing",
    "somethin'": "something",
    "havin'": "having",
    "lovin'": "loving",
    "'coz": "because",
    "thats": "that is",
    "whats": "what is",
    "ima": "I am going to",
    "gonna": "going to",
    "gotta": "got to",
    "wanna": "want to",
    "woulda": "would have",
    "gimme": "give me",
    "asap": "as soon as possible",
    "u":"you",
    "&":"and",
    "%":"percent",
    "pct":"percent",
    "£":"pounds",
    "$":"dollars",
    "₽":"rubles",
    "₹":"rupees",
    "¥":"yen",
    "₿":"bitcoin",
    "fyi" : "for your information",
    "faq" : "frequently asked questions",
    "eg" : "example",
    "$" : " dollar ",
    "€" : " euro ",
    "4ao" : "for adults only",
    "a.m" : "before midday",
    "a3" : "anytime anywhere anyplace",
    "aamof" : "as a matter of fact",
    "acct" : "account",
    "adih" : "another day in hell",
    "afaic" : "as far as i am concerned",
    "afaict" : "as far as i can tell",
    "afaik" : "as far as i know",
    "afair" : "as far as i remember",
    "afk" : "away from keyboard",
    "app" : "application",
    "approx" : "approximately",
    "apps" : "applications",
    "asap" : "as soon as possible",
    "asl" : "age, sex, location",
    "atk" : "at the keyboard",
    "ave." : "avenue",
    "aymm" : "are you my mother",
    "ayor" : "at your own risk", 
    "b&b" : "bed and breakfast",
    "b+b" : "bed and breakfast",
    "b.c" : "before christ",
    "b2b" : "business to business",
    "b2c" : "business to customer",
    "b4" : "before",
    "b4n" : "bye for now",
    "b@u" : "back at you",
    "bae" : "before anyone else",
    "bak" : "back at keyboard",
    "bbbg" : "bye bye be good",
    "bbc" : "british broadcasting corporation",
    "bbias" : "be back in a second",
    "bbl" : "be back later",
    "bbs" : "be back soon",
    "be4" : "before",
    "bfn" : "bye for now",
    "blvd" : "boulevard",
    "bout" : "about",
    "brb" : "be right back",
    "bros" : "brothers",
    "bro" : "brother",
    "brt" : "be right there",
    "bsaaw" : "big smile and a wink",
    "btw" : "by the way",
    "bwl" : "bursting with laughter",
    "c/o" : "care of",
    "cet" : "central european time",
    "cf" : "compare",
    "cia" : "central intelligence agency",
    "csl" : "can not stop laughing",
    "cu" : "see you",
    "cul8r" : "see you later",
    "cv" : "curriculum vitae",
    "cwot" : "complete waste of time",
    "cya" : "see you",
    "cyt" : "see you tomorrow",
    "dae" : "does anyone else",
    "dbmib" : "do not bother me i am busy",
    "diy" : "do it yourself",
    "dm" : "direct message",
    "dwh" : "during work hours",
    "e123" : "easy as one two three",
    "eet" : "eastern european time",
    "eg" : "example",
    "embm" : "early morning business meeting",
    "encl" : "enclosed",
    "encl." : "enclosed",
    "faq" : "frequently asked questions",
    "fawc" : "for anyone who cares",
    "fb" : "facebook",
    "fc" : "fingers crossed",
    "fig" : "figure",
    "fimh" : "forever in my heart", 
    "ft." : "feet",
    "ft" : "featuring",
    "ftl" : "for the loss",
    "ftw" : "for the win",
    "fwiw" : "for what it is worth",
    "fyi" : "for your information",
    "g9" : "genius",
    "gahoy" : "get a hold of yourself",
    "gal" : "get a life",
    "gcse" : "general certificate of secondary education",
    "gfn" : "gone for now",
    "gg" : "good game",
    "gl" : "good luck",
    "glhf" : "good luck have fun",
    "gmt" : "greenwich mean time",
    "gmta" : "great minds think alike",
    "gn" : "good night",
    "g.o.a.t" : "greatest of all time",
    "goat" : "greatest of all time",
    "goi" : "get over it",
    "gps" : "global positioning system",
    "gr8" : "great",
    "gratz" : "congratulations",
    "gyal" : "girl",
    "h&c" : "hot and cold",
    "hp" : "horsepower",
    "hr" : "hour",
    "hrh" : "his royal highness",
    "ht" : "height",
    "ibrb" : "i will be right back",
    "ic" : "i see",
    "icq" : "i seek you",
    "icymi" : "in case you missed it",
    "idc" : "i do not care",
    "idgadf" : "i do not give a damn fuck",
    "idgaf" : "i do not give a fuck",
    "idk" : "i do not know",
    "ie" : "that is",
    "i.e" : "that is",
    "ifyp" : "i feel your pain",
    "IG" : "instagram",
    "iirc" : "if i remember correctly",
    "ilu" : "i love you",
    "ily" : "i love you",
    "imho" : "in my humble opinion",
    "imo" : "in my opinion",
    "imu" : "i miss you",
    "iow" : "in other words",
    "irl" : "in real life",
    "j4f" : "just for fun",
    "jic" : "just in case",
    "jk" : "just kidding",
    "jsyk" : "just so you know",
    "l8r" : "later",
    "lb" : "pound",
    "lbs" : "pounds",
    "ldr" : "long distance relationship",
    "lmao" : "laugh my ass off",
    "lmfao" : "laugh my fucking ass off",
    "lol" : "laughing out loud",
    "ltd" : "limited",
    "ltns" : "long time no see",
    "m8" : "mate",
    "mf" : "motherfucker",
    "mfs" : "motherfuckers",
    "mfw" : "my face when",
    "mofo" : "motherfucker",
    "mph" : "miles per hour",
    "mr" : "mister",
    "mrw" : "my reaction when",
    "ms" : "miss",
    "mte" : "my thoughts exactly",
    "nagi" : "not a good idea",
    "nbc" : "national broadcasting company",
    "nbd" : "not big deal",
    "nfs" : "not for sale",
    "ngl" : "not going to lie",
    "nhs" : "national health service",
    "nrn" : "no reply necessary",
    "nsfl" : "not safe for life",
    "nsfw" : "not safe for work",
    "nth" : "nice to have",
    "nvr" : "never",
    "nyc" : "new york city",
    "oc" : "original content",
    "og" : "original",
    "ohp" : "overhead projector",
    "oic" : "oh i see",
    "omdb" : "over my dead body",
    "omg" : "oh my god",
    "omw" : "on my way",
    "p.a" : "per annum",
    "p.m" : "after midday",
    "pm" : "prime minister",
    "poc" : "people of color",
    "pov" : "point of view",
    "pp" : "pages",
    "ppl" : "people",
    "prw" : "parents are watching",
    "ps" : "postscript",
    "pt" : "point",
    "ptb" : "please text back",
    "pto" : "please turn over",
    "qpsa" : "what happens", #"que pasa",
    "ratchet" : "rude",
    "rbtl" : "read between the lines",
    "rlrt" : "real life retweet", 
    "rofl" : "rolling on the floor laughing",
    "roflol" : "rolling on the floor laughing out loud",
    "rotflmao" : "rolling on the floor laughing my ass off",
    "rt" : "retweet",
    "ruok" : "are you ok",
    "sfw" : "safe for work",
    "sk8" : "skate",
    "smh" : "shake my head",
    "sq" : "square",
    "srsly" : "seriously", 
    "ssdd" : "same stuff different day",
    "tbh" : "to be honest",
    "tbs" : "tablespooful",
    "tbsp" : "tablespooful",
    "tfw" : "that feeling when",
    "thks" : "thank you",
    "tho" : "though",
    "thx" : "thank you",
    "tia" : "thanks in advance",
    "til" : "today i learned",
    "tl;dr" : "too long i did not read",
    "tldr" : "too long i did not read",
    "tmb" : "tweet me back",
    "tntl" : "trying not to laugh",
    "ttyl" : "talk to you later",
    "u" : "you",
    "u2" : "you too",
    "u4e" : "yours for ever",
    "utc" : "coordinated universal time",
    "w/" : "with",
    "w/o" : "without",
    "w8" : "wait",
    "wassup" : "what is up",
    "wb" : "welcome back",
    "wtf" : "what the fuck",
    "wtg" : "way to go",
    "wtpa" : "where the party at",
    "wuf" : "where are you from",
    "wuzup" : "what is up",
    "wywh" : "wish you were here",
    "yd" : "yard",
    "ygtr" : "you got that right",
    "ynk" : "you never know",
    "zzz" : "sleeping bored and tired"
}

def convert_emojis(text):
    for emot in UNICODE_EMO:
        text = text.replace(emot, "_".join(UNICODE_EMO[emot].replace(",","").replace(":","").split()))
        text=re.sub('face','',text)
    return text

EMOTICONS_NEW = {
    u":‑\)":"Happy",
    u":\)":"Happy",
    u":-\]":"Happy",
    u":\]":"Happy",
    u":-3":"Happy",
    u":3":"Happy",
    u":->":"Happy",
    u":>":"Happy",
    u"8-\)":"Happy",
    u":o\)":"Happy",
    u":-\}":"Happy",
    u":\}":"Happy",
    u":-\)":"Happy",
    u":c\)":"Happy",
    u":\^\)":"Happy",
    u"=\]":"Happy",
    u"=\)":"Happy",
    u":‑D":"Laughing",
    u":D":"Laughing",
    u"8‑D":"Laughing",
    u"8D":"Laughing",
    u"X‑D":"Laughing",
    u"XD":"Laughing",
    u"=D":"Laughing",
    u"=3":"Laughing",
    u"B\^D":"Laughing",
    u":-\)\)":"Very happy",
    u":‑\(":"Frown",
    u":-\(":"Frown",
    u":\(":"Frown",
    u":‑c":"Frown",
    u":c":"Frown",
    u":‑<":"Frown",
    u":<":"Frown",
    u":‑\[":"Frown",
    u":\[":"Frown",
    u":-\|\|":"Frown",
    u">:\[":"Frown",
    u":\{":"Frown",
    u":@":"Frown",
    u">:\(":"Frown",
    u":'‑\(":"Crying",
    u":'\(":"Crying",
    u":'‑\)":"very happy",
    u":'\)":"very happy",
    u"D‑':":"Horror",
    u"D:<":"Disgust",
    u"D:":"Sadness",
    u"D8":"dismay",
    u"D;":"dismay",
    u"D=":"dismay",
    u"DX":"dismay",
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
    u";‑\)":"Wink",
    u";\)":"Wink",
    u"\*-\)":"Wink",
    u"\*\)":"Wink",
    u";‑\]":"Wink",
    u";\]":"Wink",
    u";\^\)":"Wink",
    u":‑,":"Wink",
    u";D":"Wink",
    u":‑P":"playful",
    u":P":"playful",
    u"X‑P":"playful",
    u"XP":"playful",
    u":‑Þ":"playful",
    u":Þ":"playful",
    u":b":"playful",
    u"d:":"playful",
    u"=p":"playful",
    u">:P":"playful",
    u":‑/":"skeptical",
    u":/":"skeptical",
    u":-[.]":"skeptical",
    u">:[(\\\)]":"skeptical",
    u">:/":"skeptical",
    u":[(\\\)]":"skeptical",
    u"=/":"skeptical",
    u"=[(\\\)]":"skeptical",
    u":L":"skeptical",
    u"=L":"skeptical",
    u":S":"skeptical",
    u":‑\|":"unimpressed",
    u":\|":"unimpressed ",
    u":$":"Embarrassed",
    u":‑x":"tounge tied",
    u":x":"tounge tied",
    u":‑#":"tounge tied",
    u":#":"tounge tied",
    u":‑&":"tounge tied",
    u":&":"tounge tied",
    u"O:‑\)":"innocent",
    u"O:\)":"innocent",
    u"0:‑3":"innocent",
    u"0:3":"innocent",
    u"0:‑\)":"innocent",
    u"0:\)":"innocent",
    u":‑b":"playful",
    u"0;\^\)":"innocent",
    u">:‑\)":"evil",
    u">:\)":"evil",
    u"\}:‑\)":"evil",
    u"\}:\)":"evil",
    u"3:‑\)":"evil",
    u"3:\)":"evil",
    u">;\)":"evil",
    u"\|;‑\)":"Cool",
    u"\|‑O":"Bored",
    u":‑J":"Tongue-in-cheek",
    u"#‑\)":"Party",
    u"%‑\)":"confused",
    u"%\)":"confused",
    u":-###..":"sick",
    u":###..":"sick",
    u"<:‑\|":"Dump",
    u"\(>_<\)":"Troubled",
    u"\(>_<\)>":"Troubled",
    u"\(';'\)":"Baby",
    u"\(\^\^>``":"Nervous",
    u"\(\^_\^;\)":"Nervous",
    u"\(-_-;\)":"Nervous",
    u"\(~_~;\) \(・\.・;\)":"Nervous",
    u"\(-_-\)zzz":"Sleeping",
    u"\(\^_-\)":"Wink",
    u"\(\(\+_\+\)\)":"Confused",
    u"\(\+o\+\)":"Confused",
    u"\(o\|o\)":"Ultraman",
    u"\^_\^":"Joyful",
    u"\(\^_\^\)/":"Joyful",
    u"\(\^O\^\)／":"Joyful",
    u"\(\^o\^\)／":"Joyful",
    u"\(__\)":"respect",
    u"_\(\._\.\)_":"respect",
    u"<\(_ _\)>":"respect",
    u"<m\(__\)m>":"respect",
    u"m\(__\)m":"respect",
    u"m\(_ _\)m":"respect",
    u"\('_'\)":"sad",
    u"\(/_;\)":"sad",
    u"\(T_T\) \(;_;\)":"sad",
    u"\(;_;":"Sad of Crying",
    u"\(;_:\)":"sad",
    u"\(;O;\)":"sad",
    u"\(:_;\)":"sad",
    u"\(ToT\)":"sad",
    u";_;":"sad",
    u";-;":"sad",
    u";n;":"sad",
    u";;":"sad",
    u"Q\.Q":"sad",
    u"T\.T":"sad",
    u"QQ":"sad",
    u"Q_Q":"sad",
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
    u"\^m\^":"laugh",
#     u"\(\・\・?":"Confusion",
#     u"\(?_?\)":"Confusion",
    u">\^_\^<":"laugh",
    u"<\^!\^>":"laugh",
    u"\^/\^":"laugh",
    u"\（\*\^_\^\*）" :"laugh",
    u"\(\^<\^\) \(\^\.\^\)":"laugh",
    u"\(^\^\)":"laugh",
    u"\(\^\.\^\)":"laugh",
    u"\(\^_\^\.\)":"laugh",
    u"\(\^_\^\)":"laugh",
    u"\(\^\^\)":"laugh",
    u"\(\^J\^\)":"laugh",
    u"\(\*\^\.\^\*\)":"laugh",
    u"\(\^—\^\）":"laugh",
    u"\(#\^\.\^#\)":"laugh",
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
    u"\(\*\^\^\)v":"cheerful",
    u"\(\^_\^\)v":"cheerful",
    u"\(\(d[-_-]b\)\)":"listening music",
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
    u"\(‘A`\)":"Snubbed"
}

def convert_emoticons(text):
    for emot in EMOTICONS_NEW:
        text = re.sub(u'('+emot+')', "_".join(EMOTICONS_NEW[emot].replace(","," ").split()), text)
    return text

def rem_stops(text):
    l=''
    for i in text.split():
        if(i not in stops):
            l+=' ' + i
    return l

def clean(text):
    # convert to lower case
    text = text.lower()
#     print('lower',text)
    
    # replace contractions and then remove 's
    pattern = re.compile(r'(?<!\w)(' + '|'.join(re.escape(key) for key in conts.keys()) + r')(?!\w)')
    text = pattern.sub(lambda x: conts[x.group()], text)
    
#     print('own contractions',text)
    #remaining contractions and slang
    #text = contractions.fix(text)
    
#     print('cont lib',text)
    
    # replace apostrophe with a space
    text = re.sub("'",' ', text)
    
#     print('replace apostrophe',text)
    
    # remove single lettered alphabetical words
    text =  re.sub(r"\b[a-zA-Z]\b", "", text)
    
#     print('single lettered',text)
    
    #   space instead of \n
    text = re.sub('\n', ' ', text)
    
#     print('removing \\n',text)
    
    #replacing email addresses with blank space
    text = re.sub(r"[a-zA-Z0-9_\-\.]+@[a-zA-Z0-9_\-\.]+\.[a-zA-Z]{2,5}"," ",text)
    
#     print('rem mail',text)
    
    #replacing urls with blank space
    text = re.sub("^(http:\/\/www\.|https:\/\/www\.|http:\/\/|https:\/\/)?[a-z0-9]+([\-\.]{1}[a-z0-9]+)*\.[a-z]{2,5}(:[0-9]{1,5})?(\/.*)?$"," ",text)
#     print('rem url1',text)
    text = re.sub(r"(https?:\/\/)(\s)*(www\.)?(\s)*((\w|\s)+\.)*([\w\-\s]+\/)*([\w\-]+)((\?)?[\w\s]*=\s*[\w\%&]*)*"," ",text)
    
#     print('rem url2',text)
    
    # replacing html tags with space
    text = re.sub(r"<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});", " ", text)
    
#     print('rem html',text)
    
    # replacing emojis with text
    text = convert_emojis(text)
#     print('replace emoji',text)
    
    # replacing emoticons with text
#     text = convert_emoticons(text)
    
#     print('replace emoticon',text)
   
    # space instead of all punctuations and non alphanumeric data
#     text= re.sub('[()]', ' ', text)
#     text = re.sub(r'([?.!,¿#\-:*/;<=_>^"{|}`+~])', r" ", text)
    
    puncts = [',', '.', '"', ':', ')', '(', '-', '!', '?', '|', ';', "'", '$', '&', '/', '[', ']',
          '>', '%', '=', '#', '*', '+', '\\', '•', '~', '@', '£', '·', '_', '{', '}', '©', '^',
          '®', '`', '<', '→', '°', '€', '™', '›', '♥', '←', '×', '§', '″', '′', 'Â', '█',
          '½', 'à', '…', '“', '★', '”', '–', '●', 'â', '►', '−', '¢', '²', '¬', '░', '¶',
          '↑', '±', '¿', '▾', '═', '¦', '║', '―', '¥', '▓', '—', '‹', '─', '▒', '：', '¼',
          '⊕', '▼', '▪', '†', '■', '’', '▀', '¨', '▄', '♫', '☆', 'é', '¯', '♦', '¤', '▲',
          'è', '¸', '¾', 'Ã', '⋅', '‘', '∞', '∙', '）', '↓', '、', '│', '（', '»', '，', '♪',
          '╩', '╚', '³', '・', '╦', '╣', '╔', '╗', '▬', '❤', 'ï', 'Ø', '¹', '≤', '‡', '√',
          '\t','\n','❖','«','✉','❽','♪♫','☆','ψ']

    #replace each symbol with space
    for punctuation in puncts:
        text = text.replace(punctuation, ' ')
        
#     print('rem puncs',text)
    
    # remove stopwords
    text = rem_stops(text)
#     print('rem stops',text)
    
    # remove multiple white spaces
    text = re.sub(r'\s+', ' ', text)
    
#     print('rem whitespace',text)
    return text.lower()
clean_text_vect = np.vectorize(clean)

def chunk_clean(array,chunk_size=256):
    cleaned_array = []
    
    for i in tqdm(range(0, len(array), chunk_size)):
        text_chunk = clean_text_vect(array[i:i+chunk_size])
        cleaned_array.extend(text_chunk)

    return np.array(cleaned_array)

def bert_encode(texts, tokenizer, max_len=512):
    all_tokens = []
    all_masks = []
    all_segments = []
    
    for text in texts:
        text = tokenizer.tokenize(text)
        text = text[:max_len-2]
        input_sequence = ["[CLS]"] + text + ["[SEP]"]
        pad_len = max_len - len(input_sequence)
        tokens = tokenizer.convert_tokens_to_ids(input_sequence)
        tokens += [0] * pad_len
        pad_masks = [1] * len(input_sequence) + [0] * pad_len
        segment_ids = [0] * max_len
        all_tokens.append(tokens)
        all_masks.append(pad_masks)
        all_segments.append(segment_ids)
    return np.array(all_tokens), np.array(all_masks), np.array(all_segments)




app = Flask(__name__)

max_len=512
import tensorflow as tf
input_word_ids = Input(shape=(max_len,), dtype=tf.int32, name="input_word_ids")
input_mask = Input(shape=(max_len,), dtype=tf.int32, name="input_mask")
segment_ids = Input(shape=(max_len,), dtype=tf.int32, name="segment_ids")
module_url = r"bert_multi_from_tfhub"
bert_layer = hub.KerasLayer(module_url, trainable=True)
pooled_output, sequence_output = bert_layer([input_word_ids, input_mask, segment_ids])
vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
tokenizer = tokenization.FullTokenizer(vocab_file, do_lower_case)

_, sequence_output = bert_layer([input_word_ids, input_mask, segment_ids])
clf_output = sequence_output[:, 0, :]
dense_layer1 = Dense(units=256,activation='relu')(clf_output)
dense_layer1 = Dropout(0.4)(dense_layer1)
dense_layer2 = Dense(units=128, activation='relu')(dense_layer1)
dense_layer2 = Dropout(0.4)(dense_layer2)
out = Dense(1, activation='sigmoid')(dense_layer2)
#out = Dense(1,activation='softmax')(clf_output)

model = Model(inputs=[input_word_ids, input_mask, segment_ids], outputs=out)
model.compile(Adam(lr=5e-5), loss='binary_crossentropy', metrics=['accuracy'])


def load_model():
    return model.load_weights(r"model_5nal_new_2.h5")
    return model1

model1=load_model()

@app.route('/assets/<path:path>')
def send_assert_file(path):
    return send_from_directory('assets/', path)

@app.route('/index.html')
def index_page():
    return render_template("index.html")

@app.route('/inner-page1.html')
def visualizations():
    return render_template("inner-page1.html")

@app.route('/art.html')
def hello_world():
    return render_template("art.html")

@app.route('/model.html')
def home():
	return render_template('model.html')

@app.route('/predict',methods=['POST'])
def predict():
    m=[]
    # read h5, build model
    if request.method == 'POST':
        message=request.form['message']
        if message==":(":
            message="Depressed"
        elif message==":)":
            message="happy"
        message=clean(message)
        m.append(message)
        encoded_message=bert_encode(m, tokenizer, max_len=512)
        #call cleaning function
        #predict score
        score = model.predict(encoded_message)
        print(score)
    return render_template('result.html',prediction = score)



app.run(debug=True)