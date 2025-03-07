# Portable Parquet Embedding Similarity Search

import polars as pl
import numpy as np
from sentence_transformers import SentenceTransformer
 
# 1. 임베딩 모델 로드
model = SentenceTransformer('Alibaba-NLP/gte-modernbert-base')
 
# 2. 샘플 영화 데이터 준비
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
 
# 3. 임베딩 생성
descriptions = data["description"]
embeddings = model.encode(descriptions, convert_to_numpy=True, normalize_embeddings=True)
data["embedding"] = [emb.astype(np.float32) for emb in embeddings]
 
# 4. Polars DataFrame 생성 및 Parquet 파일로 저장
df = pl.DataFrame(data)
df.write_parquet("movie_embeddings.parquet")
 
# 5. Parquet 파일 로드
df = pl.read_parquet("movie_embeddings.parquet", columns=["title", "genre", "embedding"])
 
# 6. 임베딩을 2D numpy 배열로 변환
embeddings_matrix = np.stack(df["embedding"].to_numpy(allow_copy=False))
 
# 7. 유사도 검색 함수 정의
def fast_dot_product(query, matrix, k=3):
    """
    쿼리와 매트릭스 간 코사인 유사도를 계산하여 상위 k개의 결과를 반환
    """
    dot_products = query @ matrix.T
    idx = np.argpartition(dot_products, -k)[-k:]
    idx = idx[np.argsort(dot_products[idx])[::-1]]
    scores = dot_products[idx]
    return idx, scores
 
# 8. 사용자 쿼리 처리 및 검색
def search_similar_movies(query_text, k=2, filter_genre=None):
    # 쿼리 임베딩 생성
    query_embed = model.encode([query_text], convert_to_numpy=True, normalize_embeddings=True)[0]
     
    if filter_genre:
        # 필터링 적용
        df_filter = df.filter(pl.col("genre").str.contains(filter_genre))
        if df_filter.is_empty():
            print(f"'{filter_genre}' 장르를 포함하는 영화를 찾을 수 없습니다.")
            return
        embeddings_filter = np.stack(df_filter["embedding"].to_numpy(allow_copy=False))
        idx, scores = fast_dot_product(query_embed, embeddings_filter, k=k)
        related_movies = df_filter[idx]
    else:
        # 전체 데이터에서 검색
        idx, scores = fast_dot_product(query_embed, embeddings_matrix, k=k)
        related_movies = df[idx]
     
    # 결과 출력
    for row, score in zip(related_movies.rows(named=True), scores):
        print(f"제목: {row['title']}, 장르: {row['genre']}, 유사도: {score:.4f}")
 
# 9. 메인 함수
def main():
    # 사용자 쿼리 예시
    query_text = "우주 여행과 인류 생존에 관한 영화"
     
    print("=== 전체 영화에서 검색 ===")
    search_similar_movies(query_text, k=2)
     
    print("\n=== Sci-Fi 장르에서 검색 ===")
    search_similar_movies(query_text, k=2, filter_genre="Sci-Fi")
 
if __name__ == "__main__":
    main()
