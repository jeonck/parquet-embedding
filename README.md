# parquet-embedding

## 영화 데이터에 대한 임베딩을 생성하고, 사용자가 입력한 쿼리와 유사한 영화를 검색하는 기능을 제공


### **코드 설명**

1. **라이브러리 임포트**
    
    ```python
    import polars as pl
    import numpy as np
    from sentence_transformers import SentenceTransformer
    
    ```
    
    - `polars`: 데이터프레임을 다루기 위한 라이브러리.
    - `numpy`: 수치 계산을 위한 라이브러리.
    - `sentence_transformers`: 문장 임베딩을 생성하기 위한 라이브러리.
2. **임베딩 모델 로드**
    
    ```python
    model = SentenceTransformer('Alibaba-NLP/gte-modernbert-base')
    
    ```
    
    - 사전 훈련된 문장 임베딩 모델을 로드합니다.
3. **샘플 영화 데이터 준비**
    
    ```python
    data = {
        "title": ["The Matrix", "Inception", "Interstellar", "The Dark Knight"],
        "genre": ["Sci-Fi, Action", "Sci-Fi, Thriller", "Sci-Fi, Drama", "Action, Crime"],
        "description": [
            "A computer hacker learns about the true nature of reality and fights against machines.",
            "A skilled thief enters dreams to steal secrets using groundbreaking technology.",
            "A team of explorers travels through a wormhole to find a new home for humanity.",
            "Batman confronts the Joker, a chaotic criminal mastermind terrorizing Gotham."
        ]
    }
    
    ```
    
    - 영화 제목, 장르, 설명을 포함하는 샘플 데이터를 준비합니다.
4. **임베딩 생성**
    
    ```python
    descriptions = data["description"]
    embeddings = model.encode(descriptions, convert_to_numpy=True, normalize_embeddings=True)
    data["embedding"] = [emb.astype(np.float32) for emb in embeddings]
    
    ```
    
    - 영화 설명을 임베딩하여 각 설명에 대한 벡터를 생성하고, 이를 데이터에 추가합니다.
5. **Polars DataFrame 생성 및 Parquet 파일로 저장**
    
    ```python
    df = pl.DataFrame(data)
    df.write_parquet("movie_embeddings.parquet")
    
    ```
    
    - 준비한 데이터를 Polars DataFrame으로 변환하고, Parquet 파일 형식으로 저장합니다.
6. **Parquet 파일 로드**
    
    ```python
    df = pl.read_parquet("movie_embeddings.parquet", columns=["title", "genre", "embedding"])
    
    ```
    
    - 저장된 Parquet 파일을 읽어와서 영화 제목, 장르, 임베딩을 포함하는 데이터프레임을 생성합니다.
7. **임베딩을 2D numpy 배열로 변환**
    
    ```python
    embeddings_matrix = np.stack(df["embedding"].to_numpy(allow_copy=False))
    
    ```
    
    - 임베딩을 2D numpy 배열로 변환하여 유사도 계산에 사용합니다.
8. **유사도 검색 함수 정의**
    
    ```python
    def fast_dot_product(query, matrix, k=3):
        ...
    
    ```
    
    - 쿼리와 임베딩 매트릭스 간의 코사인 유사도를 계산하여 상위 k개의 결과를 반환하는 함수를 정의합니다.
9. **사용자 쿼리 처리 및 검색**
    
    ```python
    def search_similar_movies(query_text, k=2, filter_genre=None):
        ...
    
    ```
    
    - 사용자가 입력한 쿼리를 임베딩으로 변환하고, 필터링된 장르가 있을 경우 해당 장르에서 유사한 영화를 검색합니다.
10. **메인 함수**
    
    ```python
    def main():
        ...
    
    ```
    
    - 사용자 쿼리 예시를 통해 전체 영화와 특정 장르에서 유사한 영화를 검색하는 메인 함수를 정의합니다.
11. **실행**
    
    ```python
    if __name__ == "__main__":
        main()
    
    ```
    
    - 스크립트가 직접 실행될 때 메인 함수를 호출하여 프로그램을 시작합니다.
