from fastapi.testclient import TestClient
from main import app
import time

def test_ask_api():
    with TestClient(app) as client:
        question = [
            'Ba con bảo con phải dọn phòng ăn giờ con cần làm những bước gì ạ?',
            'Con muốn chăm sóc cây cảnh nhưng không biết bắt đầu từ đâu ạ?',
            'Con muốn chăm sóc chuột hamster thì phải làm gì ạ?',
            'Máy rửa chén xài như thế nào ạ?',
            'Làm sao để sử dụng kéo cắt cành ạ?',
            'Làm sao để giặt áo vest',
            'Cách phơi đồ thể thao đúng cách',
            'Vệ sinh máy ép trái cây như thế nào',
            'Làm sao để lau dọn phòng khách',
            'Cách bảo quản các loại đồ hộp',
            'Nên sử dụng loại nước rửa đồ dùng học tập nào',
            'Làm sao để giặt khăn trải bàn',
            'Cách phơi đồ lụa đúng cách',
        ]

        cnt = 0

        for quest in question:
            time.sleep(5)
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