#Import libraries
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

#init flask
app = Flask(__name__)
load_dotenv()

#init OPENAI_API_KEY
os.environ['OPENAI_API_KEY']='yourkey'

#Get the openai api key
openai.api_key = os.environ.get("OPENAI_API_KEY")

#loading the Dataset
loader = TextLoader("Dataset.txt", encoding='utf-8')
documents = loader.load()
#Spliting the dataset to chunks with 2000 size for each chunk
text_splitter = CharacterTextSplitter(chunk_size=3500, chunk_overlap=0)
texts = text_splitter.split_documents(documents)
embeddings = OpenAIEmbeddings()
'''
False value of embed_and_save means the dataset already embedded to Chroma database,
otherwise the dataset will embed and save in chroma database
'''
embed_and_save = False


if embed_and_save:
    print("Run embedding and save ..")
    store = Chroma.from_documents(texts, embeddings, persist_directory="./chroma_db", collection_name="work-system")
else:
    print("Load embeddings ..")
    store = Chroma(embedding_function=embeddings, persist_directory="./chroma_db", collection_name="work-system")
    print(len(store.get()['ids']))


#Running Home page
@app.route('/')
def index():
    return render_template('index.html')

#Running the function for retrive question from user and answer
'''  
gpt function: retrieve the question from user and tokenize it to pass the question to model.
Also the function calculate the similarity between the question and similar chunk to the question
'''
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
    try:
        #Ask the chatgpt a question
        completion = openai.ChatCompletion.create(
            model='gpt-3.5-turbo-16k',
            messages=[
                {"role": "system",
                 "content": "استخدم هذا النص المتعلق بنظام العمل السعودي في الإجابة على السؤال. إذا كنت لا تستطيع الإجابة لا تحاول تصنيع إجابة فقط اسئل من السائل اعادة صياغه السوال.\nالنص: " + relevant_chunk},
                {"role": "user", "content": "السؤال: " + question},
            ]
        )
        #return the answer to the website
        return jsonify(answer='شكرًا لك على سؤالك، '+completion.choices[0].message.content)
    except Exception as e:
        return jsonify(answer=str(e))


if __name__ == '__main__':
    app.run()