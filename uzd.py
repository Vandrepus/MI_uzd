# Imports
import re 
from collections import Counter 
import time 
from langdetect import detect 
from transformers import AutoTokenizer, AutoModelForSequenceClassification,pipeline,MarianMTModel, MarianTokenizer
import torch 
import os

def first_uzd():
    print("\nUzdevuma nosacijumi:")
    print("Izveidot programmu, kas nosaka, cik bieži katrs vārds atkārtojas tekstā.")
    print("Ignorēt lielos burtus, lai \"kaķis\" un \"Kaķis\" tiktu uzskatīti par vienādiem.")
    print("\nIevade:")
    print("\"Mākoņainā dienā kaķis sēdēja uz palodzes. Kaķis domāja, kāpēc debesis ir pelēkas. Kaķis gribēja redzēt sauli, bet saule slēpās aiz mākoņiem.\"")
    time.sleep(5)

    print("\nRezultats:")
    text = "Mākoņainā dienā kaķis sēdēja uz palodzes. Kaķis domāja, kāpēc debesis ir pelēkas. Kaķis gribēja redzēt sauli, bet saule slēpās aiz mākoņiem."
    text_lowercase = text.lower()
    clean_text= re.sub(r'[^\w\s]', '', text_lowercase)
    words = clean_text.split()
    word_counter = Counter(words)

    for vards, skaits in word_counter.items():
        if skaits > 1:
            print(f"{vards}: {skaits}")

def second_uzd():
    print("\nUzdevuma nosacijumi:")
    print("Izveidot programmu, kas nosaka, kurā valodā ir katrs teksts. Izmanto bibliotēku vai algoritmu, kas ļauj identificēt valodu.")
    print("\nIevade:")
    print("\"Šodien ir saulaina diena.\"")
    print("\"Today is a sunny day.\"")
    print("\"Сегодня солнечный день.\"")
    time.sleep(5)

    print("\nRezultats:")
    texts = [
        "Šodien ir saulaina diena.",
        "Today is a sunny day.",  # Detects as so language i dont know why... D:
        "Сегодня солнечный день."
    ]

    for x in texts:
        lang= detect(x)
        print(f"Teksts: \"{x}\" -> Valoda: {lang}")

def third_uzd():
    print("\nUzdevuma nosacijumi:")
    print("Identificēt vārdu sakritību starp abiem tekstiem")
    print("Aprēķinat, cik procentuāli liels ir sakritības līmenis.")
    print("\nIevade:")
    print("\"Rudens lapas ir dzeltenas un oranžas. Lapas klāj zemi un padara to krāsainu.\"")
    print("\"Krāsainas rudens lapas krīt zemē. Lapas ir oranžas un dzeltenas.\"")
    time.sleep(5)

    print("\nRezultats:")

    text1 = "Rudens lapas ir dzeltenas un oranžas. Lapas klāj zemi un padara to krāsainu."
    text2 = "Krāsainas rudens lapas krīt zemē. Lapas ir oranžas un dzeltenas."

    def update_text(text):
        text_lowercase = text.lower()
        clean_text = re.sub(r'[^\w\s]', '', text_lowercase)
        words = clean_text.split()
        return set(words)

    words1 = update_text(text1)
    words2 = update_text(text2)

    common_words = words1 & words2

    unique_words = words1 | words2
    total = len(unique_words)

    percents = (len(common_words) / total) * 100

    print(f"Kopīgie vārdi: {common_words}")
    print(f"Sakritības līmenis: {percents:.2f}%")

def forth_uzd():
    print("\nUzdevuma nosacijumi:")
    print("Izveidot programmu, kas nosaka katra teikuma emocionālo noskaņojumu: pozitīvs, negatīvs vai neitrāls.")
    print("\nIevade:")
    print("\"Šis produkts ir lielisks, esmu ļoti apmierināts!\"")
    print("\"Esmu vīlies, produkts neatbilst aprakstam.\"")
    print("\"Neitrāls produkts, nekas īpašs.\"")
    time.sleep(5)

    print("\nRezultats:")

    model_name = "cardiffnlp/twitter-xlm-roberta-base-sentiment"  
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    texts_lv = [
        "Šis produkts ir lielisks, esmu ļoti apmierināts!",
        "Esmu vīlies, produkts neatbilst aprakstam.",
        "Neitrāls produkts, nekas īpašs."
    ]


    for text in texts_lv:
        inputs = tokenizer(text, return_tensors="pt",  truncation=True, max_length=128)
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)
        sentiment = torch.argmax(probs).item()

        if sentiment == 0:
            label = "negatīvs"
        elif sentiment == 1:
            label = "neitrāls"
        else:
            label = "pozitīvs"

        print(f"Teikums: \"{text}\" -> Noskaņojums: {label}")

def fifth_uzd():
    print("\nUzdevuma nosacijumi:")
    print("Noņem liekos simbolus (piemēram, @, #, !!!, URL).")
    print("Pārveido tekstu uz mazajiem burtiem.")
    print("Izveidot tīru un viegli lasāmu tekstu.")
    print("\nIevade:")
    print("\"@John: Šis ir lielisks produkts!!! Vai ne? 👏👏👏 http://example.com\"")
   
    time.sleep(5)

    print("\nRezultats:")
    txt = "@John: Šis ir lielisks produkts!!! Vai ne? 👏👏👏 http://example.com"

    txt = re.sub(r'@\w+', '', txt)  
    txt = re.sub(r'http\S+', '', txt)  
    txt = re.sub(r'[^\w\s.,!?]', '', txt)  
    txt = re.sub(r'\s+', ' ', txt) 
    txt = txt.strip()  

    txt = txt.lower()

    print(txt)

def sixth_uzd():
    print("\nUzdevuma nosacijumi:")
    print("Izveidot programmu, kas automātiski rezumē rakstu un izceļ galvenās idejas.")
    print("\nIevade:")
    print("\"Latvija ir valsts Baltijas reģionā. Tās galvaspilsēta ir Rīga, kas ir slavena ar savu vēsturisko centru un skaistajām ēkām. Latvija robežojas ar Lietuvu, Igauniju un Krieviju, kā arī tai ir piekļuve Baltijas jūrai. Tā ir viena no Eiropas Savienības dalībvalstīm.\"")
    time.sleep(5)
    print("\nRezultats:")

    summarizer = pipeline("summarization", model="Falconsai/text_summarization") # modelis neatbalsta Lv valodu,līdz ar to tas nekorekti strāda
    teksts = """Latvija ir valsts Baltijas reģionā. Tās galvaspilsēta ir Rīga, kas ir slavena ar savu vēsturisko centru un skaistajām ēkām. Latvija robežojas ar Lietuvu, Igauniju un Krieviju, kā arī tai ir piekļuve Baltijas jūrai. Tā ir viena no Eiropas Savienības dalībvalstīm."""
    print(summarizer(teksts, max_length=1000, min_length=30, do_sample=False)) # output [{'summary_text': 'Latvija robeojas ar Lietuvu, Igauniju un Krieviju, k ar tai ir piekuve Baltijas jrai.'}]

def seventh_uzd(): # Nav gatavs
    print("\nUzdevuma nosacijumi:")
    print("Izmanto vārdu iegulšanas modeli, lai:")
    print("1.Iegūtu katra vārda vektoru.")
    print("2.Salīdzinātu, kuri vārdi ir semantiski līdzīgāki.")
    print("\nIevade:")
    print("Vārdi: māja ; dzīvoklis ; jūra")
    time.sleep(5)
    print("\nRezultats:")
    
def eight_uzd():


    ner_pipeline = pipeline(
        "ner",
        model="Davlan/bert-base-multilingual-cased-ner-hrl",
        tokenizer="Davlan/bert-base-multilingual-cased-ner-hrl",
        grouped_entities=True
    )

    
    text = "Valsts prezidents Egils Levits piedalījās pasākumā, ko organizēja Latvijas Universitāte."

    
    entities = ner_pipeline(text)

    for entity in entities:
        if entity['entity_group'] in ['PER', 'ORG']:
            print(f"{entity['word']}: {entity['entity_group']}")

def nineth_uzd(): # Nav gatavs
    print()

def ten_uzd():
    modelis_nosaukums = 'Helsinki-NLP/opus-mt-lv-en'
    tokenizer = MarianTokenizer.from_pretrained(modelis_nosaukums)
    model = MarianMTModel.from_pretrained(modelis_nosaukums)

    teksti = [
        "Labdien! Kā jums klājas?",
        "Es šodien lasīju interesantu grāmatu."
    ]

    for teksts in teksti:
        tokens = tokenizer.prepare_seq2seq_batch([teksts], return_tensors='pt')
        tulkojums = model.generate(**tokens)
        tulkots_teksts = tokenizer.decode(tulkojums[0], skip_special_tokens=True)
        print(f"Latviski: \"{teksts}\" -> Angliski: \"{tulkots_teksts}\"")



while True:
    try:
        # Main console
        print("\nIzvēlieties vienu no uzdevumiem")
        print("Uzdevums 1: Vārdu biežuma analīze tekstā")
        print("Uzdevums 2: Teksta valodas noteikšana")
        print("Uzdevums 3: Vārdu sakritību pārbaude divos tekstos")
        print("Uzdevums 4: Noskaņojuma analīze")
        print("Uzdevums 5: Teksta tīrīšana un normalizēšana")
        print("Uzdevums 6: Automātiska rezumēšana - strāda,bet ne tā kā vajag")
        print("Uzdevums 7: Vārdu iegulšana (word embeddings) - Nav gatavs")
        print("Uzdevums 8: Frāžu atpazīšana (NER) ")
        print("Uzdevums 9: Teksta ģenerēšana - Nav gatavs")
        print("Uzdevums 10: Tulkotāja izveide")
        print("11 izslegt programmu")
        uzd = int(input("Jūsu izvele: "))


        if uzd == 1:
            first_uzd()
        elif uzd == 2:
            second_uzd()
        elif uzd == 3:
            third_uzd()
        elif uzd == 4:
            forth_uzd()
        elif uzd == 5:
            fifth_uzd()
        elif uzd == 6:
            sixth_uzd()
        elif uzd == 7:
            seventh_uzd()
        elif uzd == 8:
            eight_uzd()
        elif uzd == 9:
            nineth_uzd()
        elif uzd == 10:
            ten_uzd()
        elif uzd == 11:
            break
    except:
        print("\n\n\nIevadiet ciparu!")