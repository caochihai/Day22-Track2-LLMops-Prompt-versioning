# Hướng dẫn chi tiết Bài Lab Day 22: LangSmith + Prompt Versioning

Tài liệu này phân tích chi tiết các yêu cầu, cấu trúc và các bước thực hiện bài lab Day 22 trong lộ trình Track 2.

---

## 1. Bài lab này làm về gì?
Bài lab này tập trung vào việc xây dựng và quản lý một **RAG (Retrieval-Augmented Generation) pipeline** ở cấp độ production. Các nội dung chính bao gồm:
- **Observability**: Sử dụng **LangSmith** để giám sát (tracing) các luồng chạy của LLM.
- **Prompt Management**: Quản lý phiên bản prompt thông qua **LangSmith Prompt Hub**.
- **A/B Testing**: Thực hiện điều hướng (routing) yêu cầu giữa các phiên bản prompt khác nhau.
- **Automated Evaluation**: Đánh giá chất lượng câu trả lời bằng khung đánh giá **RAGAS**.
- **AI Safety/Guardrails**: Sử dụng **Guardrails AI** để kiểm soát đầu ra (phát hiện thông tin nhạy cảm PII và định dạng JSON).

---

## 2. Pipeline thực hiện và Luồng chạy
Luồng chạy của bài lab được chia thành 4 giai đoạn chính (Step):

1.  **Step 1 (Base RAG + Tracing)**:
    - Load dữ liệu -> Cắt nhỏ (Chunking) -> Tạo Vector Store (FAISS).
    - Thiết lập RAG Chain cơ bản.
    - Tích hợp LangSmith để ghi lại vết (trace) của từng câu hỏi.
2.  **Step 2 (Prompt Hub & A/B Routing)**:
    - Đẩy các phiên bản prompt (V1, V2) lên Prompt Hub.
    - Kéo prompt về và thực hiện A/B Routing dựa trên mã băm (hash) của request ID (chia 50/50).
3.  **Step 3 (RAGAS Evaluation)**:
    - Chạy 50 câu hỏi qua cả 2 phiên bản prompt.
    - Sử dụng RAGAS để tính toán các chỉ số: `faithfulness`, `answer_relevancy`, `context_recall`, `context_precision`.
4.  **Step 4 (Guardrails)**:
    - Áp dụng các bộ lọc để làm sạch dữ liệu đầu ra (Redact email, phone... và sửa lỗi JSON).

---

## 3. Các file có tác dụng gì?
Dựa trên cấu trúc dự án trong `pseudocode/`, các file có vai trò như sau:

-   **`01_langsmith_rag_pipeline.py`**: Xây dựng pipeline RAG cơ bản và cấu hình LangSmith tracing.
-   **`02_prompt_hub_ab_routing.py`**: Xử lý việc đẩy/kéo prompt từ Hub và logic chia tải A/B.
-   **`03_ragas_evaluation.py`**: Chứa bộ dữ liệu 50 câu hỏi mẫu và logic đánh giá chất lượng bằng RAGAS.
-   **`04_guardrails_validator.py`**: Định nghĩa các custom validator để bảo vệ đầu ra của LLM.
-   **`config.py`** (cần tạo): Chứa các cấu hình chung như API Keys, Base URL.
-   **`qa_pairs.py`** (cần tạo): Chứa danh sách các cặp câu hỏi - câu trả lời mẫu (ground truth).
-   **`run_all.py`**: Script chạy tuần tự cả 4 bước trên.

---

## 4. Các hàm quan trọng trong file
### `01_langsmith_rag_pipeline.py`
-   `build_vectorstore()`: Đọc file văn bản, chia nhỏ và tạo index FAISS.
-   `build_rag_chain(vectorstore)`: Xây dựng chuỗi LangChain (retriever -> prompt -> llm -> parser).
-   `ask(chain, question)`: Hàm thực hiện truy vấn, được đánh dấu bằng `@traceable` để gửi log về LangSmith.

### `02_prompt_hub_ab_routing.py`
-   `push_prompts_to_hub(client)`: Đẩy prompt lên hệ thống quản lý tập trung.
-   `pull_prompts_from_hub(client)`: Lấy prompt về code để sử dụng.
-   `get_prompt_version(request_id)`: Logic xác định phiên bản prompt (V1 hay V2) dựa trên MD5 hash.
-   `ask_ab(...)`: Thực hiện truy vấn với phiên bản prompt cụ thể.

### `03_ragas_evaluation.py`
-   `run_rag(...)`: Chạy RAG và trả về cả câu trả lời lẫn các đoạn context đã dùng (để RAGAS đánh giá).
-   `run_ragas_eval(...)`: Gọi thư viện RAGAS để tính toán điểm số trung bình.

### `04_guardrails_validator.py`
-   `PIIDetector.validate()`: Tìm kiếm và ẩn (redact) các thông tin như Email, SĐT, SSN.
-   `JSONFormatter.validate()`: Sửa các lỗi định dạng JSON phổ biến (như thiếu ngoặc, dùng nháy đơn).

---

## 5. Có cần phải viết thêm code hay không?
**CÓ.** Bạn không chỉ chạy các file có sẵn. Các file trong thư mục `pseudocode/` chỉ là **bản nháp (template)**.
- Bạn cần copy các file này ra thư mục gốc của project.
- Tìm các dòng có chữ `# TODO` để bỏ comment và bổ sung logic code hoàn chỉnh.
- Cấu hình file `.env` với các API Key cần thiết (OpenAI/Gemini, LangSmith).

---

## 6. Có cần chụp ảnh hay không?
**CÓ.** Bạn cần thu thập bằng chứng (evidence) để nộp bài.
- **Thư mục lưu trữ**: `evidence/`
- **Danh sách ảnh cần có**:
    1.  `01_langsmith_traces.png`: Chụp giao diện LangSmith Dashboard hiển thị danh sách >= 50 traces.
    2.  `02_prompt_hub.png`: Chụp giao diện Prompt Hub hiển thị 2 phiên bản prompt đã push lên thành công.
    3.  `03_ragas_scores.png`: Chụp màn hình terminal hiển thị bảng so sánh điểm số giữa V1 và V2.

---

## 7. Chi tiết từng bước làm bài lab

### Bước 1: Thiết lập môi trường
- Cài đặt thư viện: `pip install -r requirements.txt`.
- Tạo file `.env` chứa `LANGCHAIN_API_KEY`, `OPENAI_API_KEY`, `LANGCHAIN_PROJECT`.

### Bước 2: Hoàn thiện Step 1 (Tracing)
- Thực hiện logic trong `01_langsmith_rag_pipeline.py`.
- Đảm bảo khi chạy, terminal in ra kết quả của 50 câu hỏi.
- **Chụp ảnh**: Kiểm tra trên web LangSmith và chụp lại danh sách traces -> lưu vào `evidence/01_langsmith_traces.png`.

### Bước 3: Hoàn thiện Step 2 (Prompt Hub & A/B)
- Viết 2 system prompt khác nhau (một cái ngắn gọn, một cái chi tiết).
- Hoàn thiện logic đẩy/kéo prompt trong `02_prompt_hub_ab_routing.py`.
- **Chụp ảnh**: Chụp giao diện Prompt Hub trên web -> lưu vào `evidence/02_prompt_hub.png`.
- **Lưu log**: Chạy script và lưu output vào `evidence/02_ab_routing_log.txt`.

### Bước 4: Hoàn thiện Step 3 (Evaluation)
- Chạy đánh giá RAGAS (quá trình này mất khoảng 15-20 phút vì gọi LLM nhiều lần).
- Đảm bảo ít nhất một phiên bản có điểm `faithfulness >= 0.8`.
- **Chụp ảnh**: Bảng so sánh kết quả trên terminal -> lưu vào `evidence/03_ragas_scores.png`.
- **Lưu file**: Copy file kết quả `data/ragas_report.json` sang `evidence/03_ragas_report.json`.

### Bước 5: Hoàn thiện Step 4 (Guardrails)
- Cài đặt các validator cho PII và JSON trong `04_guardrails_validator.py`.
- Chạy demo và kiểm tra xem các chuỗi vi phạm có được "sửa" (FIX) hay không.
- **Lưu log**: Lưu output demo vào `evidence/04_pii_demo_log.txt` và `evidence/04_json_demo_log.txt`.

---

## 8. Kiểm tra thành công của bài lab
Bài lab được coi là thành công nếu:
- [ ] Có ít nhất **100 traces** trong dự án LangSmith (50 từ Step 1, 50 từ Step 2).
- [ ] Có **2 prompts** xuất hiện trên Prompt Hub.
- [ ] Điểm **Faithfulness >= 0.8** (đảm bảo LLM không "nói dối" so với context).
- [ ] Các bộ lọc Guardrails hoạt động chính xác (ẩn được email/phone mẫu).
- [ ] Thư mục `evidence/` chứa đầy đủ các file đã liệt kê ở trên.

---

## 9. Báo cáo bài lab
Bạn cần nộp các thông tin sau:
1.  **Link GitHub Repository**: Chứa toàn bộ code đã hoàn thiện và thư mục `evidence/`.
2.  **Link LangSmith Project**: Link trực tiếp dẫn đến project của bạn trên LangChain Smith.
3.  **Thông tin trong báo cáo**:
    - Mô tả ngắn gọn về 2 phiên bản prompt bạn đã thử nghiệm.
    - Nhận xét về phiên bản nào tốt hơn dựa trên kết quả RAGAS.
    - Giải thích cách bạn đã xử lý khi điểm số không đạt yêu cầu (ví dụ: thay đổi `chunk_size` hoặc sửa prompt).

> **Lưu ý quan trọng**: Tuyệt đối không commit file `.env` hoặc để lộ API Key trong code khi push lên GitHub (Sẽ bị trừ 10 điểm nếu vi phạm).
