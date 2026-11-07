from pydantic import BaseModel, ConfigDict, Field



class SearchRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    query: str = Field(min_length=1, max_length=1000)
    top_k: int = Field(default=5, ge=1, le=20)


class PassageResponse(BaseModel):
    text: str
    score: float


class MovieResponse(BaseModel):
    movie_id: str
    title: str
    year: int
    score: float
    passages: list[PassageResponse]


class SearchResponse(BaseModel):
    results: list[MovieResponse]
