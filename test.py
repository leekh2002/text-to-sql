import json
import os
import re
from dataclasses import dataclass
from typing import Any

import pandas as pd
from dotenv import load_dotenv
from langchain_community.utilities import SQLDatabase
from langchain_openai import ChatOpenAI
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import Engine, URL


K_ALL_LECTURES = "모든 강의를 보여줘"
K_WEDNESDAY_ONLY = "수요일 과목만 보여줘"
K_LIBERAL_ARTS = "교양 과목들을 찾아줘"
K_LIBERAL_ARTS_SOYANG = "교양(소양) 과목들을 찾아줘"
K_NO_FRIDAY_LIBERAL_ARTS = "금요일 강의를 제외한 교양 과목들을 찾아줘"
K_CS_DEPT = "컴퓨터인공지능학부에서 개설한 강의를 보여줘"
K_REMAINING_SEATS = "잔여석이 남아있는 과목을 보여줘"
K_EE_REMAINING_SEATS = "잔여석이 남아있는 전기공학과 과목들을 찾아줘"
K_ELECTRICAL_ENGINEERING = "전기공학과"
K_DEPARTMENT = "학과"
K_COLLEGE = "학부"
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

QUERY_EXAMPLES = [
    QueryExample(
        question=K_ALL_LECTURES,
        sql="SELECT c.* FROM cnu_courses AS c LIMIT {top_k};",
    ),
    QueryExample(
        question=K_WEDNESDAY_ONLY,
        sql=(
            "SELECT DISTINCT c.* "
            "FROM cnu_courses AS c "
            "JOIN course_schedule AS cs "
            "ON c.subject_code = cs.subject_code AND c.section = cs.section "
            "WHERE cs.day_of_week = '수' "
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
            "WHERE d.dept_name LIKE '%컴퓨터인공지능학부%' "
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
    return re.sub(r"\s+", " ", text_value).strip()


def extract_category_semantics(question: str) -> dict[str, Any] | None:
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
    question = normalize_whitespace(question)
    hints: list[str] = []

    if "잔여석" in question:
        hints.append(
            "Use the actual remaining-seat condition from the schema, such as cnu_courses.capacity > cnu_courses.enrolled, instead of matching the literal text."
        )

    if K_ELECTRICAL_ENGINEERING in question or K_DEPARTMENT in question or K_COLLEGE in question:
        hints.append(
            "A department or major name may require a JOIN from cnu_courses to department using dept_code and filtering on department.dept_name."
        )

    if any(day in question for day in K_WEEKDAYS):
        weekday_map = ", ".join(f"{k} -> {v}" for k, v in K_WEEKDAY_TO_DB.items())
        hints.append(
            "Time information is stored in course_schedule, not in cnu_courses. Join course_schedule by (subject_code, section) and use course_schedule.day_of_week. "
            f"Weekday mapping: {weekday_map}."
        )

    category_semantics = extract_category_semantics(question)
    if category_semantics:
        if category_semantics["broad_category"]:
            hints.append(
                f"When the user says {category_semantics['raw_text']} without a parenthesized subtype, interpret it as a broad category and prefer subject.category LIKE '{category_semantics['normalized_value']}%'."
            )
        else:
            hints.append(
                f"When the user explicitly says {category_semantics['raw_text']}, treat it as an exact category value for subject.category."
            )

    return question if not hints else question + "\n\nDomain hints:\n- " + "\n- ".join(hints)


def normalize_sql(raw_sql: str) -> str:
    sql = raw_sql.strip()
    sql = re.sub(r"^SQLQuery:\s*", "", sql, flags=re.IGNORECASE)

    fenced = re.search(r"```(?:sql)?\s*(.*?)```", sql, flags=re.IGNORECASE | re.DOTALL)
    if fenced:
        sql = fenced.group(1).strip()

    return sql.strip()


def run_query(engine: Engine, sql: str) -> pd.DataFrame:
    with engine.connect() as conn:
        return pd.read_sql_query(text(sql), conn)


def safe_fetch_values(engine: Engine, table: str, column: str, tokens: list[str], limit: int = VALUE_LOOKUP_LIMIT) -> list[str]:
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
    tokens = re.findall(r"[가-힣A-Za-z0-9()]+", question)
    filtered = [token for token in tokens if len(token) >= 2]
    return list(dict.fromkeys(filtered))


def build_value_context(engine: Engine, question: str) -> str:
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
    blocks = []
    for idx, example in enumerate(QUERY_EXAMPLES, start=1):
        blocks.append(
            f"Example {idx}\n"
            f"Question: {example.question}\n"
            f"SQLQuery: {example.sql.format(top_k=top_k)}"
        )
    return "\n\n".join(blocks)


def extract_json_object(text_value: str) -> dict[str, Any]:
    match = re.search(r"\{.*\}", text_value, flags=re.DOTALL)
    if not match:
        raise ValueError("No JSON object found in model response.")
    return json.loads(match.group(0))


def build_analysis_seed(question: str) -> dict[str, Any]:
    category_semantics = extract_category_semantics(question)
    seed: dict[str, Any] = {}
    if category_semantics:
        seed["category_filter"] = category_semantics
    return seed


def analyze_question(llm: ChatOpenAI, ddl_info: str, question: str, value_context: str) -> dict[str, Any]:
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
    examples = build_examples(top_k)
    analysis_json = json.dumps(analysis, ensure_ascii=False, indent=2)

    return f"""
You are an expert PostgreSQL SQL generator for a Korean university course database.
Return only executable SQL.
Do not output explanations, markdown fences, or the SQLQuery prefix.
Use only tables and columns that exist in the schema.
Limit results to at most {top_k} rows unless the user explicitly asks for a different count.

Generation rules:
- Prefer schema-grounded predicates over literal text matching.
- If the question mentions a department, major, college, or program name, join department when needed and filter on department.dept_name.
- If the question asks for remaining seats, use cnu_courses.capacity > cnu_courses.enrolled.
- If the question asks about weekdays or class times, join course_schedule on (subject_code, section).
- course_schedule can contain multiple rows per lecture, so use DISTINCT when joining it unless the user explicitly asks for each meeting row.
- If the question asks for a count, generate COUNT(*).
- If analysis.category_filter.broad_category is true, join subject and use a prefix predicate such as subject.category LIKE '교양%'. Never replace it with one narrower exact value.
- If analysis.category_filter.match_mode is exact, join subject and use an exact predicate such as subject.category = '교양(소양)'.
- Preserve user literals unless the matched database values show a better grounded equivalent.

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
    llm: ChatOpenAI,
    ddl_info: str,
    question: str,
    bad_sql: str,
    error_message: str,
    value_context: str,
    top_k: int,
    analysis: dict[str, Any],
) -> str:
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


def generate_sql(engine: Engine, db: SQLDatabase, llm: ChatOpenAI, question: str, top_k: int = DEFAULT_TOP_K) -> str:
    ddl_info = db.get_table_info()
    normalized_question = normalize_question(question)
    value_context = build_value_context(engine, normalized_question)
    analysis = analyze_question(llm, ddl_info, normalized_question, value_context)
    sql_prompt = build_sql_prompt(ddl_info, normalized_question, analysis, value_context, top_k)
    return normalize_sql(llm.invoke(sql_prompt).content)


def execute_with_repair(
    engine: Engine,
    db: SQLDatabase,
    llm: ChatOpenAI,
    question: str,
    top_k: int = DEFAULT_TOP_K,
) -> tuple[str, pd.DataFrame]:
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
    load_dotenv()

    connect_url = build_postgres_url()
    engine = create_engine(connect_url)
    db = SQLDatabase.from_uri(connect_url)
    llm = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"), temperature=0)
    #print(db.get_table_info())

    questions = [
        "화요일 전공기초 과목들을 찾아줘",
        "교양 과목들을 찾아줘",
    ]

    print("Tables:", list(db.get_usable_table_names()))

    for question in questions:
        sql, result_df = execute_with_repair(engine, db, llm, question)
        print(f"\nQuestion: {question}")
        print(f"Executable SQL:\n{sql}")
        print(f"Result:\n{result_df}\n")


if __name__ == "__main__":
    main()
