from fastapi.testclient import TestClient
from main import app
import gen_quest
import time

def gen_ask_api(ntest):
    # print(question)
    with TestClient(app) as client:
        for i in range(ntest):
            time.sleep(60)

            gen_quest.make_quest()
            question = [

            ]

            file_path = "question.txt"
            with open(file_path, "r", encoding = 'utf-8') as file:
                question_file = file.read().split("\n")
                for quest in question_file:
                    question.append(quest)

            cnt = 0

            for quest in question:
                response = client.post("/ask", json = {
                    "query": quest
                })
                cnt += 1
                print(cnt)
                # print("User : ", quest)
                # print(response.status_code)
                # print("Chatbot : ", response.json()["answer"])
                # print("-" * 120)

def test_ask_api():
    # print(question)
    with TestClient(app) as client:
        gen_quest.make_quest()
        question = [

        ]

        file_path = "question.txt"
        with open(file_path, "r", encoding = 'utf-8') as file:
            question_file = file.read().split("\n")
            for quest in question_file:
                question.append(quest)

        cnt = 0

        for quest in question:
            response = client.post("/ask", json = {
                "query": quest
            })

            cnt += 1
            print("User : ", quest)
            print(response.status_code)
            print("Chatbot : ", response.json()["answer"])
            print("-" * 120)
        
if __name__ == "__main__":
    test_ask_api()