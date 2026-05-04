import json
import os
import re
from dataclasses import dataclass
from typing import Any

import pandas as pd
from dotenv import load_dotenv
from langchain_community.utilities import SQLDatabase
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import Engine, URL

try:
    from langchain_anthropic import ChatAnthropic
except ImportError as exc:
    raise ImportError(
        "langchain-anthropic 패키지가 필요합니다. "
        "`pip install langchain-anthropic`로 설치한 뒤 다시 실행하세요."
    ) from exc


# 자주 쓰는 예시 질문과 키워드를 상수로 모아 둔다.
K_ALL_LECTURES = "모든 강의를 보여줘"
K_TUESDAY_ONLY = "화요일 과목만 보여줘"
K_LIBERAL_ARTS = "교양 과목들을 찾아줘"
K_LIBERAL_ARTS_SOYANG = "교양(소양) 과목들을 찾아줘"
K_NO_FRIDAY_LIBERAL_ARTS = "금요일 강의를 제외한 교양 과목들을 찾아줘"
K_CS_DEPT = "컴퓨터공학부에서 개설한 강의를 보여줘"
K_REMAINING_SEATS = "여유석이 남아있는 과목을 보여줘"
K_EE_REMAINING_SEATS = "여유석이 남아있는 전기공학과 과목들을 찾아줘"
K_ELECTRICAL_ENGINEERING = "전기공학과"
K_DEPARTMENT = "학과"
K_COLLEGE = "단과대"
K_WEEKDAYS = [
    "월요일",
    "화요일",
    "수요일",
    "목요일",
    "금요일",
    "토요일",
    "일요일",
]
K_WEEKDAY_TO_DB = {
    "월요일": "월",
    "화요일": "화",
    "수요일": "수",
    "목요일": "목",
    "금요일": "금",
    "토요일": "토",
    "일요일": "일",
}
CATEGORY_PREFIX_TERMS = ["교양", "전공", "일반선택"]


@dataclass(frozen=True)
class QueryExample:
    question: str
    sql: str


DEFAULT_TOP_K = 5
VALUE_LOOKUP_LIMIT = 30
MAX_REPAIR_ATTEMPTS = 1

# 모델이 SQL 패턴을 안정적으로 따라오도록 대표 예시를 제공한다.
QUERY_EXAMPLES = [
    QueryExample(
        question=K_ALL_LECTURES,
        sql="SELECT c.* FROM cnu_courses AS c LIMIT {top_k};",
    ),
    QueryExample(
        question=K_TUESDAY_ONLY,
        sql=(
            "SELECT DISTINCT c.* "
            "FROM cnu_courses AS c "
            "JOIN course_schedule AS cs "
            "ON c.subject_code = cs.subject_code AND c.section = cs.section "
            "WHERE cs.day_of_week = '화' "
            "LIMIT {top_k};"
        ),
    ),
    QueryExample(
        question=K_LIBERAL_ARTS,
        sql=(
            "SELECT DISTINCT c.* "
            "FROM cnu_courses AS c "
            "JOIN subject AS s ON c.subject_code = s.subject_code "
            "WHERE s.category LIKE '교양%' "
            "LIMIT {top_k};"
        ),
    ),
    QueryExample(
        question=K_LIBERAL_ARTS_SOYANG,
        sql=(
            "SELECT DISTINCT c.* "
            "FROM cnu_courses AS c "
            "JOIN subject AS s ON c.subject_code = s.subject_code "
            "WHERE s.category = '교양(소양)' "
            "LIMIT {top_k};"
        ),
    ),
    QueryExample(
        question=K_NO_FRIDAY_LIBERAL_ARTS,
        sql=(
            "SELECT DISTINCT c.* "
            "FROM cnu_courses AS c "
            "JOIN subject AS s ON c.subject_code = s.subject_code "
            "WHERE s.category LIKE '교양%' "
            "AND NOT EXISTS ("
            "SELECT 1 FROM course_schedule AS cs "
            "WHERE cs.subject_code = c.subject_code "
            "AND cs.section = c.section "
            "AND cs.day_of_week = '금'"
            ") "
            "LIMIT {top_k};"
        ),
    ),
    QueryExample(
        question=K_CS_DEPT,
        sql=(
            "SELECT DISTINCT c.* "
            "FROM cnu_courses AS c "
            "JOIN department AS d ON c.dept_code = d.dept_code "
            "WHERE d.dept_name LIKE '%컴퓨터공학부%' "
            "LIMIT {top_k};"
        ),
    ),
    QueryExample(
        question=K_REMAINING_SEATS,
        sql="SELECT c.* FROM cnu_courses AS c WHERE c.capacity > c.enrolled LIMIT {top_k};",
    ),
    QueryExample(
        question=K_EE_REMAINING_SEATS,
        sql=(
            "SELECT DISTINCT c.* "
            "FROM cnu_courses AS c "
            "JOIN department AS d ON c.dept_code = d.dept_code "
            "WHERE d.dept_name LIKE '%전기공학과%' AND c.capacity > c.enrolled "
            "LIMIT {top_k};"
        ),
    ),
]


def build_postgres_url() -> str:
    """환경변수 값을 사용해 PostgreSQL 연결 문자열을 만든다."""
    url = URL.create(
        drivername="postgresql+psycopg2",
        username=os.getenv("POSTGRES_USER", "postgres"),
        password=os.getenv("POSTGRES_PASSWORD", "postgres"),
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=int(os.getenv("POSTGRES_PORT", "5432")),
        database=os.getenv("POSTGRES_DB", "postgres"),
    )
    return url.render_as_string(hide_password=False)


def normalize_whitespace(text_value: str) -> str:
    """연속 공백을 하나로 줄여 질문 파싱을 안정화한다."""
    return re.sub(r"\s+", " ", text_value).strip()


def extract_category_semantics(question: str) -> dict[str, Any] | None:
    """
    질문에 포함된 과목 구분 키워드를 해석한다.
    예: 교양 -> broad prefix, 교양(소양) -> exact match
    """
    for prefix in CATEGORY_PREFIX_TERMS:
        if f"{prefix}(" in question:
            match = re.search(rf"({re.escape(prefix)}\([^)]+\))", question)
            if match:
                return {
                    "raw_text": match.group(1),
                    "column": "subject.category",
                    "broad_category": False,
                    "match_mode": "exact",
                    "normalized_value": match.group(1),
                }

        if prefix in question:
            return {
                "raw_text": prefix,
                "column": "subject.category",
                "broad_category": True,
                "match_mode": "prefix",
                "normalized_value": prefix,
            }

    return None


def normalize_question(question: str) -> str:
    """
    원문 질문은 유지하되, 모델이 놓치기 쉬운 도메인 규칙을 보조 힌트로 덧붙인다.
    """
    question = normalize_whitespace(question)
    hints: list[str] = []

    if "여유석" in question or "잔여석" in question:
        hints.append(
            "Use the actual remaining-seat condition from the schema, such as "
            "cnu_courses.capacity > cnu_courses.enrolled, instead of matching the literal text."
        )

    if K_ELECTRICAL_ENGINEERING in question or K_DEPARTMENT in question or K_COLLEGE in question:
        hints.append(
            "A department or major name may require a JOIN from cnu_courses to department "
            "using dept_code and filtering on department.dept_name."
        )

    if any(day in question for day in K_WEEKDAYS):
        weekday_map = ", ".join(f"{k} -> {v}" for k, v in K_WEEKDAY_TO_DB.items())
        hints.append(
            "Time information is stored in course_schedule, not in cnu_courses. "
            "Join course_schedule by (subject_code, section) and use course_schedule.day_of_week. "
            f"Weekday mapping: {weekday_map}."
        )

    category_semantics = extract_category_semantics(question)
    if category_semantics:
        if category_semantics["broad_category"]:
            hints.append(
                f"When the user says {category_semantics['raw_text']} without a parenthesized subtype, "
                f"interpret it as a broad category and prefer subject.category LIKE "
                f"'{category_semantics['normalized_value']}%'."
            )
        else:
            hints.append(
                f"When the user explicitly says {category_semantics['raw_text']}, "
                "treat it as an exact category value for subject.category."
            )

    return question if not hints else question + "\n\nDomain hints:\n- " + "\n- ".join(hints)


def normalize_sql(raw_sql: str) -> str:
    """모델 응답에서 SQL 본문만 추출해 실행 가능한 형태로 정리한다."""
    sql = raw_sql.strip()
    sql = re.sub(r"^SQLQuery:\s*", "", sql, flags=re.IGNORECASE)

    fenced = re.search(r"```(?:sql)?\s*(.*?)```", sql, flags=re.IGNORECASE | re.DOTALL)
    if fenced:
        sql = fenced.group(1).strip()

    return sql.strip()


def run_query(engine: Engine, sql: str) -> pd.DataFrame:
    """생성된 SQL을 실행하고 결과를 데이터프레임으로 반환한다."""
    with engine.connect() as conn:
        return pd.read_sql_query(text(sql), conn)


def safe_fetch_values(
    engine: Engine,
    table: str,
    column: str,
    tokens: list[str],
    limit: int = VALUE_LOOKUP_LIMIT,
) -> list[str]:
    """
    질문 토큰으로 실제 DB 값을 미리 조회해 모델 프롬프트에 넣는다.
    조회 실패 시 전체 흐름을 막지 않도록 빈 리스트를 반환한다.
    """
    if not tokens:
        return []

    try:
        with engine.connect() as conn:
            conditions = []
            params: dict[str, Any] = {"limit": limit}

            for idx, token in enumerate(tokens):
                key = f"token_{idx}"
                conditions.append(f"{column} ILIKE :{key}")
                params[key] = f"%{token}%"

            sql = text(
                f"""
                SELECT DISTINCT {column}
                FROM {table}
                WHERE {" OR ".join(conditions)}
                ORDER BY {column}
                LIMIT :limit
                """
            )
            return [row[0] for row in conn.execute(sql, params).fetchall() if row[0]]
    except Exception:
        return []


def extract_korean_tokens(question: str) -> list[str]:
    """질문에서 한글, 영문, 숫자 기반 토큰만 추려 중복 없이 반환한다."""
    tokens = re.findall(r"[가-힣A-Za-z0-9()]+", question)
    filtered = [token for token in tokens if len(token) >= 2]
    return list(dict.fromkeys(filtered))


def build_value_context(engine: Engine, question: str) -> str:
    """
    질문과 매칭되는 실제 값 후보를 DB에서 찾아 프롬프트에 공급한다.
    이 단계가 있으면 모델이 임의 값을 지어낼 가능성이 줄어든다.
    """
    inspector = inspect(engine)
    tables = set(inspector.get_table_names())
    tokens = extract_korean_tokens(question)
    lines: list[str] = []

    lookups = [
        ("department", "dept_name"),
        ("subject", "subject_name"),
        ("subject", "category"),
        ("cnu_courses", "professor"),
        ("course_schedule", "day_of_week"),
        ("course_schedule", "classroom"),
    ]

    for table, column in lookups:
        if table not in tables:
            continue

        values = safe_fetch_values(engine, table, column, tokens)
        if values:
            joined = ", ".join(values[:VALUE_LOOKUP_LIMIT])
            lines.append(f"- {table}.{column}: {joined}")

    return "\n".join(lines) if lines else "- No matched values were retrieved from the database."


def build_examples(top_k: int) -> str:
    """Few-shot 예시를 하나의 프롬프트 블록으로 합친다."""
    blocks = []
    for idx, example in enumerate(QUERY_EXAMPLES, start=1):
        blocks.append(
            f"Example {idx}\n"
            f"Question: {example.question}\n"
            f"SQLQuery: {example.sql.format(top_k=top_k)}"
        )
    return "\n\n".join(blocks)


def extract_json_object(text_value: str) -> dict[str, Any]:
    """모델 응답에서 JSON 객체 부분만 잘라 파싱한다."""
    match = re.search(r"\{.*\}", text_value, flags=re.DOTALL)
    if not match:
        raise ValueError("No JSON object found in model response.")
    return json.loads(match.group(0))


def build_analysis_seed(question: str) -> dict[str, Any]:
    """질문에서 미리 해석한 카테고리 힌트를 분석 프롬프트에 주입한다."""
    category_semantics = extract_category_semantics(question)
    seed: dict[str, Any] = {}
    if category_semantics:
        seed["category_filter"] = category_semantics
    return seed


def analyze_question(
    llm: ChatAnthropic,
    ddl_info: str,
    question: str,
    value_context: str,
) -> dict[str, Any]:
    """
    SQL 생성 전에 모델에게 의도, 조인, 필터 구조를 JSON으로 먼저 계획하게 한다.
    이렇게 분리하면 최종 SQL 생성 품질이 안정적이다.
    """
    analysis_seed = build_analysis_seed(question)
    seed_json = json.dumps(analysis_seed, ensure_ascii=False, indent=2)

    prompt = f"""
You are planning a text-to-SQL query for PostgreSQL.
Return exactly one JSON object and nothing else.

Schema:
{ddl_info}

Matched database values:
{value_context}

Precomputed hints:
{seed_json}

Question:
{question}

Rules:
- Copy the category_filter semantics from Precomputed hints when they are present.
- If category_filter.broad_category is true, keep match_mode as prefix and normalized_value as the broad prefix.
- If category_filter.broad_category is false, keep match_mode as exact.
- Do not collapse a broad category such as '교양' into one specific value such as '교양(소양)' unless the user explicitly said that exact subtype.

JSON format:
{{
  "intent": "one-sentence summary",
  "target_table": "primary table name",
  "joins": ["table join reasons"],
  "filters": ["schema-grounded filter descriptions"],
  "aggregations": ["aggregation requirements, if any"],
  "sort": "sorting requirement or empty string",
  "limit": "requested row count or {DEFAULT_TOP_K}",
  "category_filter": {{
    "raw_text": "original category mention",
    "column": "subject.category",
    "broad_category": true,
    "match_mode": "prefix",
    "normalized_value": "교양"
  }},
  "notes": ["important mappings such as remaining seats = capacity > enrolled or weekday filters using course_schedule.day_of_week"]
}}
""".strip()

    response = llm.invoke(prompt).content
    analysis = extract_json_object(response)

    if "category_filter" not in analysis and "category_filter" in analysis_seed:
        analysis["category_filter"] = analysis_seed["category_filter"]

    return analysis


def build_sql_prompt(
    ddl_info: str,
    question: str,
    analysis: dict[str, Any],
    value_context: str,
    top_k: int,
) -> str:
    """분석 결과와 few-shot 예시를 합쳐 최종 SQL 생성 프롬프트를 만든다."""
    examples = build_examples(top_k)
    analysis_json = json.dumps(analysis, ensure_ascii=False, indent=2)

    return f"""
You are an expert PostgreSQL SQL generator for a Korean university course database.
Return only executable SQL.
Do not output explanations, markdown fences, or the SQLQuery prefix.
Use only tables and columns that exist in the schema.
Limit results to at most {top_k} rows unless the user explicitly asks for a different count.



Schema:
{ddl_info}

Matched database values:
{value_context}

Question analysis:
{analysis_json}

{examples}

Question:
{question}

SQLQuery:
""".strip()


def repair_sql(
    llm: ChatAnthropic,
    ddl_info: str,
    question: str,
    bad_sql: str,
    error_message: str,
    value_context: str,
    top_k: int,
    analysis: dict[str, Any],
) -> str:
    """실행 실패 시 DB 에러를 바탕으로 SQL을 한 번 더 수정한다."""
    analysis_json = json.dumps(analysis, ensure_ascii=False, indent=2)
    prompt = f"""
You are repairing a PostgreSQL query for a Korean university course database.
Return only corrected executable SQL.

Schema:
{ddl_info}

Matched database values:
{value_context}

Question analysis:
{analysis_json}

Original question:
{question}

Broken SQL:
{bad_sql}

Database error:
{error_message}

Rules:
- Keep the original user intent.
- Use only real tables and columns from the schema.
- Limit results to at most {top_k} rows unless the question asks for another count.
- If the question asks for remaining seats, prefer cnu_courses.capacity > cnu_courses.enrolled.
- If the question asks for weekday or time conditions, use course_schedule instead of a removed lecture_time field.
- If analysis.category_filter.broad_category is true, keep a prefix category predicate such as subject.category LIKE '교양%'.
- If analysis.category_filter.match_mode is exact, keep an exact category predicate.
""".strip()

    return normalize_sql(llm.invoke(prompt).content)


def generate_sql(
    engine: Engine,
    db: SQLDatabase,
    llm: ChatAnthropic,
    question: str,
    top_k: int = DEFAULT_TOP_K,
) -> str:
    """질문 하나를 받아 실행 가능한 SQL만 생성한다."""
    ddl_info = db.get_table_info()
    normalized_question = normalize_question(question)
    value_context = build_value_context(engine, normalized_question)
    analysis = analyze_question(llm, ddl_info, normalized_question, value_context)
    sql_prompt = build_sql_prompt(ddl_info, normalized_question, analysis, value_context, top_k)
    return normalize_sql(llm.invoke(sql_prompt).content)


def execute_with_repair(
    engine: Engine,
    db: SQLDatabase,
    llm: ChatAnthropic,
    question: str,
    top_k: int = DEFAULT_TOP_K,
) -> tuple[str, pd.DataFrame]:
    """SQL 생성, 실행, 실패 시 1회 복구까지 한 번에 수행한다."""
    ddl_info = db.get_table_info()
    normalized_question = normalize_question(question)
    value_context = build_value_context(engine, normalized_question)
    analysis = analyze_question(llm, ddl_info, normalized_question, value_context)
    sql_prompt = build_sql_prompt(ddl_info, normalized_question, analysis, value_context, top_k)
    sql = normalize_sql(llm.invoke(sql_prompt).content)

    for attempt in range(MAX_REPAIR_ATTEMPTS + 1):
        try:
            return sql, run_query(engine, sql)
        except Exception as exc:
            if attempt >= MAX_REPAIR_ATTEMPTS:
                raise

            sql = repair_sql(
                llm=llm,
                ddl_info=ddl_info,
                question=normalized_question,
                bad_sql=sql,
                error_message=str(exc),
                value_context=value_context,
                top_k=top_k,
                analysis=analysis,
            )

    raise RuntimeError("Failed to generate executable SQL.")


def main() -> None:
    """환경을 초기화하고 예시 질문 몇 개를 실행한다."""
    load_dotenv()

    connect_url = build_postgres_url()
    engine = create_engine(connect_url)
    db = SQLDatabase.from_uri(connect_url)
    llm = ChatAnthropic(
        model=os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6"),
        api_key=os.getenv("ANTHROPIC_API_KEY"),
        temperature=0,
    )

    questions = [
        "잔여석이 남아있는 화요일 기계공학과 전공 과목들을 찾아줘",
        "원격 교양 과목들을 찾아줘",
    ]

    print("Tables:", list(db.get_usable_table_names()))

    for question in questions:
        sql, result_df = execute_with_repair(engine, db, llm, question)
        print(f"\nQuestion: {question}")
        print(f"Executable SQL:\n{sql}")
        print(f"Result:\n{result_df}\n")


if __name__ == "__main__":
    main()
