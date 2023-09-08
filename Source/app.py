import nltk
from flask import Flask, request, render_template, jsonify
from dotenv import load_dotenv
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from nltk.tokenize import word_tokenize
import arabicstopwords.arabicstopwords as stp
from nltk.stem import ISRIStemmer
nltk.download('punkt')
import openai
import os

app = Flask(__name__)
load_dotenv()

openai.api_key = os.environ.get("OPENAI_API_KEY")

loader = TextLoader("formatted_Labor_rules_v0.1.txt", encoding='utf-8')
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=2000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)
embeddings = OpenAIEmbeddings()

embed_and_save = False

if embed_and_save:
    print("Run embedding and save ..")
    store = Chroma.from_documents(texts, embeddings, persist_directory="./chroma_db", collection_name="work-system")
    print("Run embedding and save ..")
else:
    print("Load embeddings ..")
    store = Chroma(embedding_function=embeddings, persist_directory="./chroma_db", collection_name="work-system")
    #print(len(store.get()['ids']))



@app.route('/')
def index():
    return render_template('index.html')

@app.route('/gpt', methods=['POST'])
def gpt():
    question = request.json['question']
    question = question.strip()
    words = word_tokenize(question)
    filtered_words = [word for word in words if not stp.is_stop(word)]
    stemmer = ISRIStemmer()
    stemmed_words = [stemmer.stem(word) for word in filtered_words]
    preprocessed_question = (' '.join(stemmed_words))

    relevant_chunk = store.similarity_search_with_score(preprocessed_question, k=6)

    relevant_chunk = relevant_chunk[0][0].page_content

    #print(relevant_chunk)

    try:
        completion = openai.ChatCompletion.create(
            model='gpt-3.5-turbo-16k',
            messages=[
                {"role": "system",
                 "content": "استخدم هذا النص في الإجابة على السؤال. إذا كنت لا تستطيع الإجابة لا تحاول تصنيع إجابة فقط قل اعد صياغه السوال.\nالنص: " + relevant_chunk},
                {"role": "user", "content": "السؤال: " + question},
            ]
        )

        #print(completion)

        return jsonify(answer=completion.choices[0].message.content)
    except Exception as e:
        return jsonify(answer=str(e))


if __name__ == '__main__':
    app.run()
