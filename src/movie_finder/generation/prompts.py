from movie_finder.models import MovieResult

SYSTEM_PROMPT = """You help a user identify a movie from their description.
You receive the description and numbered candidate movies, each with numbered plot passages.

Rules:
- Choose only from the candidates. Do not use your own knowledge of movies, even if you think you know the answer.
- If no candidate clearly matches the description, set "candidate" and "passage" to null.
- Otherwise set "candidate" to the number of the matching movie and "passage" to the number of its passage that best supports the match.
- In "explanation", say in one or two sentences which details match, or why none of the candidates fit. Base it only on the passages."""


def build_user_prompt(query: str, candidates: list[MovieResult], passages_per_movie: int) -> str:
    lines = ["Description:", query, "", "Candidates:"]
    for i, movie in enumerate(candidates, start=1):
        lines.append(f"[{i}] {movie.title} ({movie.year})")
        for j, passage in enumerate(movie.passages[:passages_per_movie], start=1):
            lines.append(f"  Passage {j}: {passage.chunk.text}")
    return "\n".join(lines)
